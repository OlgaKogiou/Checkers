#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "RAJA/RAJA.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/QuickPool.hpp"

#if defined(RAJA_ENABLE_CUDA)
#include <cuda_runtime.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#endif

#if defined(RAJA_ENABLE_CUDA)
using ExecutionPolicy = RAJA::cuda_exec<256>;
#else
using ExecutionPolicy = RAJA::seq_exec;
#endif

namespace {

namespace fs = std::filesystem;

constexpr double k_megabyte_divisor = 1024.0 * 1024.0;
constexpr std::size_t k_default_max_rules = 50;
constexpr std::size_t k_min_rule_frequency = 2;
constexpr bool k_trace_umpire_allocations = false;

using Symbol = std::uint64_t;

struct PairToken {
    Symbol lhs {0};
    Symbol rhs {0};

    RAJA_HOST_DEVICE bool operator<(const PairToken& other) const
    {
        if (lhs != other.lhs) {
            return lhs < other.lhs;
        }
        return rhs < other.rhs;
    }

    RAJA_HOST_DEVICE bool operator==(const PairToken& other) const
    {
        return lhs == other.lhs && rhs == other.rhs;
    }
};

struct PairEqual {
    RAJA_HOST_DEVICE bool operator()(const PairToken& lhs, const PairToken& rhs) const
    {
        return lhs == rhs;
    }
};

struct PairTokenHash {
    std::size_t operator()(const PairToken& pair) const noexcept
    {
        const auto lhs_hash = std::hash<Symbol> {}(pair.lhs);
        const auto rhs_hash = std::hash<Symbol> {}(pair.rhs);
        return lhs_hash ^ (rhs_hash + 0x9e3779b97f4a7c15ULL + (lhs_hash << 6) + (lhs_hash >> 2));
    }
};

struct TensorEntry {
    std::string tensor_key;
    std::string state_type;
    std::string category;
    int layer {0};
    std::size_t numel {0};
    fs::path file_path;
};

struct GroupDefinition {
    std::string state_type;
    std::string category;
    int group_id {0};
    std::vector<int> layers;
    int representative_layer {0};
};

struct GrammarRule {
    Symbol replacement_symbol {0};
    Symbol lhs {0};
    Symbol rhs {0};
    std::size_t expansion_length {0};
    std::size_t frequency {0};
};

struct GrammarLearningResult {
    std::vector<GrammarRule> rules;
    std::size_t compressed_token_count {0};
    double runtime_ms {0.0};
    std::size_t grammar_bytes {0};
    double average_rule_expansion_length {0.0};
    std::size_t unique_symbol_count {0};
    std::size_t repeated_symbol_count {0};
    std::size_t max_symbol_frequency {0};
    std::size_t unique_pair_count {0};
    std::size_t repeated_pair_count {0};
    std::size_t max_pair_frequency {0};
    std::string stop_reason {"not_started"};
};

struct TensorCompressionResult {
    std::string state_type;
    std::string category;
    int group_id {-1};
    int layer {0};
    bool is_representative {false};
    std::size_t original_bytes {0};
    std::size_t compressed_stream_bytes {0};
    std::size_t grammar_bytes_charged {0};
    std::size_t permutation_bytes {0};
    std::size_t exact_reconstruction_bytes {0};
    double runtime_ms {0.0};
    double transform_only_ratio {1.0};
    double exact_reconstruction_ratio {1.0};
};

struct GroupCompressionSummary {
    std::string state_type;
    std::string category;
    int group_id {0};
    std::vector<int> layers;
    int representative_layer {0};
    std::size_t rule_count {0};
    double average_rule_expansion_length {0.0};
    std::size_t grammar_bytes {0};
    std::size_t original_bytes {0};
    std::size_t transform_only_bytes {0};
    std::size_t permutation_bytes {0};
    std::size_t exact_reconstruction_bytes {0};
    double learning_time_ms {0.0};
    double apply_time_ms {0.0};
    double transform_only_ratio {1.0};
    double exact_reconstruction_ratio {1.0};
    std::size_t unique_symbol_count {0};
    std::size_t repeated_symbol_count {0};
    std::size_t max_symbol_frequency {0};
    std::size_t unique_pair_count {0};
    std::size_t repeated_pair_count {0};
    std::size_t max_pair_frequency {0};
    std::string stop_reason;
};

struct CheckpointCompressionSummary {
    std::size_t original_bytes {0};
    std::size_t transform_only_bytes {0};
    std::size_t permutation_bytes {0};
    std::size_t exact_reconstruction_bytes {0};
    std::size_t grouped_tensor_count {0};
    std::size_t uncompressed_tensor_count {0};
    std::size_t grammar_bytes {0};
    double transform_only_ratio {1.0};
    double exact_reconstruction_ratio {1.0};
    std::size_t total_rule_count {0};
    std::size_t zero_rule_group_count {0};
    std::string symbolization_mode {"exact_float32_bit_patterns"};
};

struct CompressionConfig {
    fs::path export_dir;
    std::size_t max_rules {k_default_max_rules};
    std::optional<std::string> state_type_filter;
    std::optional<std::string> category_filter;
    bool sort_before_repair {false};
};

double safe_divide(double numerator, double denominator)
{
    if (std::abs(denominator) <= std::numeric_limits<double>::epsilon()) {
        return 0.0;
    }

    return numerator / denominator;
}

double bytes_to_mb(std::size_t bytes)
{
    return static_cast<double>(bytes) / k_megabyte_divisor;
}

void synchronize_device()
{
#if defined(RAJA_ENABLE_CUDA)
    cudaDeviceSynchronize();
#endif
}

void check_cuda(cudaError_t status, const char* operation)
{
#if defined(RAJA_ENABLE_CUDA)
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(status));
    }
#else
    static_cast<void>(status);
    static_cast<void>(operation);
#endif
}

template <typename T>
T* allocate_buffer(umpire::Allocator& allocator, std::size_t count, const char* label)
{
    auto* pointer = static_cast<T*>(allocator.allocate(count * sizeof(T)));
    if (k_trace_umpire_allocations) {
        std::cerr << "[umpire alloc] " << label
                  << " ptr=" << static_cast<const void*>(pointer)
                  << " count=" << count
                  << " bytes=" << (count * sizeof(T))
                  << '\n';
    }
    return pointer;
}

template <typename T>
void deallocate_buffer(umpire::Allocator& allocator, T* pointer, const char* label)
{
    if (pointer == nullptr) {
        return;
    }

    if (k_trace_umpire_allocations) {
        std::cerr << "[umpire free] " << label
                  << " ptr=" << static_cast<const void*>(pointer)
                  << '\n';
    }
    allocator.deallocate(pointer);
}

std::vector<std::string> parse_csv_row(const std::string& line)
{
    std::vector<std::string> columns;
    std::string current;
    bool inside_quotes = false;

    for (char character : line) {
        if (character == '"') {
            inside_quotes = !inside_quotes;
            continue;
        }

        if (character == ',' && !inside_quotes) {
            columns.push_back(current);
            current.clear();
            continue;
        }

        current.push_back(character);
    }

    columns.push_back(current);

    for (auto& column : columns) {
        const auto first = column.find_first_not_of(" \t\r");
        const auto last = column.find_last_not_of(" \t\r");
        if (first == std::string::npos) {
            column.clear();
            continue;
        }
        column = column.substr(first, last - first + 1);
    }

    return columns;
}

std::vector<int> parse_group_layers(const std::string& text)
{
    std::vector<int> layers;
    std::stringstream stream(text);
    int value = 0;
    while (stream >> value) {
        layers.push_back(value);
    }
    return layers;
}

std::vector<float> load_tensor_file(const fs::path& file_path, std::size_t numel)
{
    std::ifstream stream(file_path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Failed to open tensor payload: " + file_path.string());
    }

    std::vector<float> tensor(numel, 0.0f);
    stream.read(reinterpret_cast<char*>(tensor.data()), static_cast<std::streamsize>(numel * sizeof(float)));
    if (stream.gcount() != static_cast<std::streamsize>(numel * sizeof(float))) {
        throw std::runtime_error("Tensor payload size mismatch: " + file_path.string());
    }

    return tensor;
}

std::vector<Symbol> tensor_to_symbols(const std::vector<float>& tensor)
{
    std::vector<Symbol> symbols(tensor.size(), 0);
    for (std::size_t index = 0; index < tensor.size(); ++index) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &tensor[index], sizeof(float));
        symbols[index] = static_cast<Symbol>(bits);
    }
    return symbols;
}

std::uint32_t float_bits_to_order_key(std::uint32_t bits)
{
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

std::vector<Symbol> sort_exact_float_symbols(const std::vector<Symbol>& symbols)
{
    std::vector<Symbol> sorted_symbols = symbols;
    std::stable_sort(
        sorted_symbols.begin(),
        sorted_symbols.end(),
        [](Symbol lhs, Symbol rhs) {
            const auto lhs_bits = static_cast<std::uint32_t>(lhs);
            const auto rhs_bits = static_cast<std::uint32_t>(rhs);
            return float_bits_to_order_key(lhs_bits) < float_bits_to_order_key(rhs_bits);
        });
    return sorted_symbols;
}

void collect_exact_pair_statistics(const std::vector<Symbol>& symbols, GrammarLearningResult& result)
{
    std::unordered_map<Symbol, std::size_t> symbol_counts;
    symbol_counts.reserve(symbols.size());
    for (Symbol symbol : symbols) {
        ++symbol_counts[symbol];
    }

    result.unique_symbol_count = symbol_counts.size();
    for (const auto& [symbol, count] : symbol_counts) {
        static_cast<void>(symbol);
        if (count >= 2) {
            ++result.repeated_symbol_count;
        }
        result.max_symbol_frequency = std::max(result.max_symbol_frequency, count);
    }

    if (symbols.size() < 2) {
        result.stop_reason = "fewer than two symbols";
        return;
    }

    std::unordered_map<PairToken, std::size_t, PairTokenHash> pair_counts;
    pair_counts.reserve(symbols.size() - 1);
    for (std::size_t index = 0; index + 1 < symbols.size(); ++index) {
        ++pair_counts[PairToken {symbols[index], symbols[index + 1]}];
    }

    result.unique_pair_count = pair_counts.size();
    for (const auto& [pair, count] : pair_counts) {
        static_cast<void>(pair);
        if (count >= 2) {
            ++result.repeated_pair_count;
        }
        result.max_pair_frequency = std::max(result.max_pair_frequency, count);
    }

    if (result.repeated_pair_count == 0) {
        result.stop_reason = "no repeated exact adjacent pairs in representative tensor";
    }
}

std::map<std::tuple<std::string, std::string, int>, TensorEntry> load_manifest(const fs::path& manifest_path)
{
    std::ifstream stream(manifest_path);
    if (!stream) {
        throw std::runtime_error("Failed to open manifest: " + manifest_path.string());
    }

    std::string line;
    std::getline(stream, line);

    std::map<std::tuple<std::string, std::string, int>, TensorEntry> entries;
    while (std::getline(stream, line)) {
        if (line.empty()) {
            continue;
        }

        const auto columns = parse_csv_row(line);
        if (columns.size() != 7) {
            throw std::runtime_error("Malformed manifest row: " + line);
        }

        TensorEntry entry;
        entry.tensor_key = columns[0];
        entry.layer = std::stoi(columns[1]);
        entry.category = columns[2];
        entry.state_type = columns[3];
        entry.numel = static_cast<std::size_t>(std::stoull(columns[4]));
        entry.file_path = manifest_path.parent_path() / columns[6];
        entries[{entry.state_type, entry.category, entry.layer}] = entry;
    }

    return entries;
}

std::vector<GroupDefinition> load_groups(const fs::path& grouping_summary_path,
                                         const CompressionConfig& config)
{
    std::ifstream stream(grouping_summary_path);
    if (!stream) {
        throw std::runtime_error("Failed to open grouping summary: " + grouping_summary_path.string());
    }

    std::string line;
    std::getline(stream, line);

    std::vector<GroupDefinition> groups;
    while (std::getline(stream, line)) {
        if (line.empty()) {
            continue;
        }

        const auto columns = parse_csv_row(line);
        if (columns.size() < 10) {
            throw std::runtime_error("Malformed grouping summary row: " + line);
        }

        GroupDefinition group;
        group.state_type = columns[0];
        group.category = columns[1];
        group.group_id = std::stoi(columns[2]);
        group.layers = parse_group_layers(columns[4]);
        group.representative_layer = std::stoi(columns[5]);

        if (config.state_type_filter && group.state_type != *config.state_type_filter) {
            continue;
        }
        if (config.category_filter && group.category != *config.category_filter) {
            continue;
        }

        groups.push_back(group);
    }

    return groups;
}

template <typename T>
void copy_to_device(umpire::ResourceManager& rm, T* destination, const std::vector<T>& source)
{
    static_cast<void>(rm);
#if defined(RAJA_ENABLE_CUDA)
    check_cuda(
        cudaMemcpy(destination, source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice),
        "cudaMemcpyHostToDevice");
#else
    std::copy(source.begin(), source.end(), destination);
#endif
}

template <typename T>
void copy_from_device(umpire::ResourceManager& rm, std::vector<T>& destination, const T* source)
{
    static_cast<void>(rm);
#if defined(RAJA_ENABLE_CUDA)
    check_cuda(
        cudaMemcpy(destination.data(), source, destination.size() * sizeof(T), cudaMemcpyDeviceToHost),
        "cudaMemcpyDeviceToHost");
#else
    std::copy(source, source + destination.size(), destination.begin());
#endif
}

std::size_t apply_rule_on_device(const GrammarRule& rule,
                                 Symbol* input_tokens,
                                 Symbol* output_tokens,
                                 std::size_t current_length,
                                 umpire::ResourceManager& rm,
                                 umpire::Allocator& device_allocator)
{
    if (current_length == 0) {
        return 0;
    }

    auto* match_flags = allocate_buffer<unsigned int>(device_allocator, current_length, "apply_rule.match_flags");
    auto* replace_flags = allocate_buffer<unsigned int>(device_allocator, current_length, "apply_rule.replace_flags");
    auto* emit_counts = allocate_buffer<unsigned int>(device_allocator, current_length, "apply_rule.emit_counts");
    auto* output_positions = allocate_buffer<unsigned int>(device_allocator, current_length, "apply_rule.output_positions");

    RAJA::forall<ExecutionPolicy>(RAJA::TypedRangeSegment<std::size_t>(0, current_length),
        [=] RAJA_HOST_DEVICE(std::size_t index) {
            unsigned int matched = 0;
            if (index + 1 < current_length && input_tokens[index] == rule.lhs && input_tokens[index + 1] == rule.rhs) {
                matched = 1;
            }
            match_flags[index] = matched;
        });

    RAJA::forall<ExecutionPolicy>(RAJA::TypedRangeSegment<std::size_t>(0, current_length),
        [=] RAJA_HOST_DEVICE(std::size_t index) {
            const unsigned int current_match = match_flags[index];
            const unsigned int previous_match = (index > 0) ? match_flags[index - 1] : 0u;
            const unsigned int replace_here = current_match && !previous_match;
            replace_flags[index] = replace_here;
        });

    RAJA::forall<ExecutionPolicy>(RAJA::TypedRangeSegment<std::size_t>(0, current_length),
        [=] RAJA_HOST_DEVICE(std::size_t index) {
            const unsigned int consumed_by_previous = (index > 0) ? replace_flags[index - 1] : 0u;
            emit_counts[index] = replace_flags[index] ? 1u : (consumed_by_previous ? 0u : 1u);
        });

#if defined(RAJA_ENABLE_CUDA)
    thrust::device_ptr<unsigned int> emit_begin(emit_counts);
    thrust::device_ptr<unsigned int> pos_begin(output_positions);
    thrust::exclusive_scan(thrust::device, emit_begin, emit_begin + current_length, pos_begin);
    const auto total_emitted = thrust::reduce(thrust::device, emit_begin, emit_begin + current_length, 0u, thrust::plus<unsigned int>());
#else
    std::vector<unsigned int> host_emit(current_length, 0u);
    std::vector<unsigned int> host_positions(current_length, 0u);
    copy_from_device(rm, host_emit, emit_counts);
    unsigned int running_sum = 0u;
    for (std::size_t index = 0; index < current_length; ++index) {
        host_positions[index] = running_sum;
        running_sum += host_emit[index];
    }
    rm.copy(output_positions, host_positions.data(), current_length * sizeof(unsigned int));
    const auto total_emitted = running_sum;
#endif

    RAJA::forall<ExecutionPolicy>(RAJA::TypedRangeSegment<std::size_t>(0, current_length),
        [=] RAJA_HOST_DEVICE(std::size_t index) {
            if (emit_counts[index] == 0u) {
                return;
            }

            const auto destination_index = static_cast<std::size_t>(output_positions[index]);
            output_tokens[destination_index] = replace_flags[index] ? rule.replacement_symbol : input_tokens[index];
        });

    synchronize_device();
    deallocate_buffer(device_allocator, match_flags, "apply_rule.match_flags");
    deallocate_buffer(device_allocator, replace_flags, "apply_rule.replace_flags");
    deallocate_buffer(device_allocator, emit_counts, "apply_rule.emit_counts");
    deallocate_buffer(device_allocator, output_positions, "apply_rule.output_positions");
    return static_cast<std::size_t>(total_emitted);
}

std::optional<std::pair<PairToken, std::size_t>> find_best_pair_on_device(Symbol* tokens,
                                                                          std::size_t token_count,
                                                                          umpire::ResourceManager& rm,
                                                                          umpire::Allocator& device_allocator)
{
    if (token_count < 2) {
        return std::nullopt;
    }

    const std::size_t pair_count = token_count - 1;
    auto* pair_tokens = allocate_buffer<PairToken>(device_allocator, pair_count, "find_best_pair.pair_tokens");
    auto* unique_pairs = allocate_buffer<PairToken>(device_allocator, pair_count, "find_best_pair.unique_pairs");
    auto* unique_counts = allocate_buffer<unsigned int>(device_allocator, pair_count, "find_best_pair.unique_counts");

    RAJA::forall<ExecutionPolicy>(RAJA::TypedRangeSegment<std::size_t>(0, pair_count),
        [=] RAJA_HOST_DEVICE(std::size_t index) {
            pair_tokens[index] = PairToken {tokens[index], tokens[index + 1]};
        });

#if defined(RAJA_ENABLE_CUDA)
    thrust::device_ptr<PairToken> pair_begin(pair_tokens);
    thrust::device_ptr<PairToken> unique_begin(unique_pairs);
    thrust::device_ptr<unsigned int> counts_begin(unique_counts);
    thrust::sort(thrust::device, pair_begin, pair_begin + pair_count);
    auto reduce_end = thrust::reduce_by_key(
        thrust::device,
        pair_begin,
        pair_begin + pair_count,
        thrust::constant_iterator<unsigned int>(1u),
        unique_begin,
        counts_begin,
        PairEqual());

    const std::size_t unique_count = static_cast<std::size_t>(reduce_end.first - unique_begin);
    if (unique_count == 0) {
        deallocate_buffer(device_allocator, pair_tokens, "find_best_pair.pair_tokens");
        deallocate_buffer(device_allocator, unique_pairs, "find_best_pair.unique_pairs");
        deallocate_buffer(device_allocator, unique_counts, "find_best_pair.unique_counts");
        return std::nullopt;
    }

    auto max_iter = thrust::max_element(thrust::device, counts_begin, counts_begin + unique_count);
    const std::size_t best_index = static_cast<std::size_t>(max_iter - counts_begin);

    PairToken best_pair {};
    unsigned int best_count = 0u;
    thrust::copy_n(unique_begin + best_index, 1, &best_pair);
    thrust::copy_n(counts_begin + best_index, 1, &best_count);

    synchronize_device();
    deallocate_buffer(device_allocator, pair_tokens, "find_best_pair.pair_tokens");
    deallocate_buffer(device_allocator, unique_pairs, "find_best_pair.unique_pairs");
    deallocate_buffer(device_allocator, unique_counts, "find_best_pair.unique_counts");

    if (best_count < k_min_rule_frequency) {
        return std::nullopt;
    }

    return std::make_pair(best_pair, static_cast<std::size_t>(best_count));
#else
    std::vector<PairToken> host_pairs(pair_count);
    copy_from_device(rm, host_pairs, pair_tokens);
    std::sort(host_pairs.begin(), host_pairs.end());
    PairToken best_pair {};
    std::size_t best_count = 0;
    std::size_t current_count = 0;
    PairToken current_pair {};
    for (std::size_t index = 0; index < host_pairs.size(); ++index) {
        if (index == 0 || !(host_pairs[index] == current_pair)) {
            current_pair = host_pairs[index];
            current_count = 1;
        } else {
            ++current_count;
        }
        if (current_count > best_count) {
            best_count = current_count;
            best_pair = current_pair;
        }
    }

    deallocate_buffer(device_allocator, pair_tokens, "find_best_pair.pair_tokens");
    deallocate_buffer(device_allocator, unique_pairs, "find_best_pair.unique_pairs");
    deallocate_buffer(device_allocator, unique_counts, "find_best_pair.unique_counts");
    if (best_count < k_min_rule_frequency) {
        return std::nullopt;
    }
    return std::make_pair(best_pair, best_count);
#endif
}

GrammarLearningResult learn_grammar_on_device(const std::vector<Symbol>& host_symbols,
                                              std::size_t max_rules,
                                              umpire::ResourceManager& rm,
                                              umpire::Allocator& device_allocator)
{
    GrammarLearningResult result {};
    if (host_symbols.empty()) {
        result.stop_reason = "empty tensor";
        return result;
    }

    collect_exact_pair_statistics(host_symbols, result);

    auto start_time = std::chrono::high_resolution_clock::now();
    auto* token_buffer_a = allocate_buffer<Symbol>(device_allocator, host_symbols.size(), "learn_grammar.token_buffer_a");
    auto* token_buffer_b = allocate_buffer<Symbol>(device_allocator, host_symbols.size(), "learn_grammar.token_buffer_b");
    copy_to_device(rm, token_buffer_a, host_symbols);

    Symbol* current_tokens = token_buffer_a;
    Symbol* scratch_tokens = token_buffer_b;
    std::size_t current_length = host_symbols.size();

    std::unordered_map<Symbol, std::size_t> expansion_lengths;
    expansion_lengths.reserve(host_symbols.size() + max_rules);
    for (Symbol symbol : host_symbols) {
        expansion_lengths.emplace(symbol, 1);
    }

    Symbol next_symbol = static_cast<Symbol>(1ull << 32);
    for (std::size_t rule_index = 0; rule_index < max_rules; ++rule_index) {
        const auto best_pair = find_best_pair_on_device(current_tokens, current_length, rm, device_allocator);
        if (!best_pair) {
            if (result.stop_reason == "not_started") {
                result.stop_reason = "no repeated exact adjacent pairs after compression step";
            }
            break;
        }

        GrammarRule rule {};
        rule.replacement_symbol = next_symbol++;
        rule.lhs = best_pair->first.lhs;
        rule.rhs = best_pair->first.rhs;
        rule.frequency = best_pair->second;
        rule.expansion_length = expansion_lengths[rule.lhs] + expansion_lengths[rule.rhs];
        expansion_lengths[rule.replacement_symbol] = rule.expansion_length;

        const std::size_t new_length = apply_rule_on_device(
            rule,
            current_tokens,
            scratch_tokens,
            current_length,
            rm,
            device_allocator);
        if (new_length >= current_length) {
            result.stop_reason = "best rule did not reduce token stream";
            break;
        }

        result.rules.push_back(rule);
        current_length = new_length;
        std::swap(current_tokens, scratch_tokens);
    }

    if (result.rules.size() == max_rules && max_rules > 0) {
        result.stop_reason = "reached max_rules";
    } else if (result.rules.empty() && result.stop_reason == "not_started") {
        result.stop_reason = "no rules learned";
    } else if (!result.rules.empty() && result.stop_reason == "not_started") {
        result.stop_reason = "no repeated exact adjacent pairs after final compression step";
    }

    result.compressed_token_count = current_length;
    result.grammar_bytes = result.rules.size() * (2 * sizeof(std::uint32_t));
    if (!result.rules.empty()) {
        double expansion_sum = 0.0;
        for (const auto& rule : result.rules) {
            expansion_sum += static_cast<double>(rule.expansion_length);
        }
        result.average_rule_expansion_length = expansion_sum / static_cast<double>(result.rules.size());
    }
    result.runtime_ms =
        std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count();

    synchronize_device();
    deallocate_buffer(device_allocator, token_buffer_a, "learn_grammar.token_buffer_a");
    deallocate_buffer(device_allocator, token_buffer_b, "learn_grammar.token_buffer_b");
    return result;
}

TensorCompressionResult apply_grammar_to_tensor(const TensorEntry& tensor,
                                                const std::vector<GrammarRule>& rules,
                                                bool is_representative,
                                                int group_id,
                                                std::size_t grammar_bytes_charged,
                                                bool sort_before_repair,
                                                umpire::ResourceManager& rm,
                                                umpire::Allocator& device_allocator)
{
    const auto host_tensor = load_tensor_file(tensor.file_path, tensor.numel);
    auto host_symbols = tensor_to_symbols(host_tensor);
    if (sort_before_repair) {
        host_symbols = sort_exact_float_symbols(host_symbols);
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    auto* token_buffer_a = allocate_buffer<Symbol>(device_allocator, host_symbols.size(), "apply_grammar.token_buffer_a");
    auto* token_buffer_b = allocate_buffer<Symbol>(device_allocator, host_symbols.size(), "apply_grammar.token_buffer_b");
    copy_to_device(rm, token_buffer_a, host_symbols);

    Symbol* current_tokens = token_buffer_a;
    Symbol* scratch_tokens = token_buffer_b;
    std::size_t current_length = host_symbols.size();

    for (const auto& rule : rules) {
        const std::size_t new_length = apply_rule_on_device(
            rule,
            current_tokens,
            scratch_tokens,
            current_length,
            rm,
            device_allocator);
        current_length = new_length;
        std::swap(current_tokens, scratch_tokens);
    }

    TensorCompressionResult result {};
    result.state_type = tensor.state_type;
    result.category = tensor.category;
    result.group_id = group_id;
    result.layer = tensor.layer;
    result.is_representative = is_representative;
    result.original_bytes = tensor.numel * sizeof(float);
    result.compressed_stream_bytes = current_length * sizeof(std::uint32_t);
    result.grammar_bytes_charged = grammar_bytes_charged;
    result.permutation_bytes = sort_before_repair ? (tensor.numel * sizeof(std::uint32_t)) : 0u;
    result.exact_reconstruction_bytes =
        result.compressed_stream_bytes + result.grammar_bytes_charged + result.permutation_bytes;
    result.runtime_ms =
        std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count();
    result.transform_only_ratio = safe_divide(
        static_cast<double>(result.original_bytes),
        static_cast<double>(result.compressed_stream_bytes + result.grammar_bytes_charged));
    result.exact_reconstruction_ratio = safe_divide(
        static_cast<double>(result.original_bytes),
        static_cast<double>(result.exact_reconstruction_bytes));

    synchronize_device();
    deallocate_buffer(device_allocator, token_buffer_a, "apply_grammar.token_buffer_a");
    deallocate_buffer(device_allocator, token_buffer_b, "apply_grammar.token_buffer_b");
    return result;
}

void write_tensor_report(const fs::path& output_path, const std::vector<TensorCompressionResult>& rows)
{
    std::ofstream stream(output_path);
    stream << "state_type,category,group_id,layer,is_representative,original_size_mb,compressed_stream_size_mb,grammar_size_mb_charged,permutation_size_mb,transform_only_size_mb,exact_reconstruction_size_mb,transform_only_ratio,exact_reconstruction_ratio,time_ms\n";
    for (const auto& row : rows) {
        const std::size_t transform_only_bytes = row.compressed_stream_bytes + row.grammar_bytes_charged;
        stream << row.state_type << ','
               << row.category << ','
               << row.group_id << ','
               << row.layer << ','
               << (row.is_representative ? "true" : "false") << ','
               << bytes_to_mb(row.original_bytes) << ','
               << bytes_to_mb(row.compressed_stream_bytes) << ','
               << bytes_to_mb(row.grammar_bytes_charged) << ','
               << bytes_to_mb(row.permutation_bytes) << ','
               << bytes_to_mb(transform_only_bytes) << ','
               << bytes_to_mb(row.exact_reconstruction_bytes) << ','
               << row.transform_only_ratio << ','
               << row.exact_reconstruction_ratio << ','
               << row.runtime_ms << '\n';
    }
}

void write_group_report(const fs::path& output_path, const std::vector<GroupCompressionSummary>& rows)
{
    std::ofstream stream(output_path);
    stream << "state_type,category,group_id,layers,representative_layer,rule_count,average_rule_expansion_length,grammar_size_mb,permutation_size_mb,original_group_size_mb,transform_only_group_size_mb,exact_reconstruction_group_size_mb,transform_only_ratio,exact_reconstruction_ratio,unique_symbol_count,repeated_symbol_count,max_symbol_frequency,unique_pair_count,repeated_exact_adjacent_pair_count,max_exact_adjacent_pair_frequency,stop_reason,learning_time_ms,apply_time_ms,total_time_ms\n";
    for (const auto& row : rows) {
        std::stringstream layers_stream;
        for (std::size_t index = 0; index < row.layers.size(); ++index) {
            if (index > 0) {
                layers_stream << ' ';
            }
            layers_stream << row.layers[index];
        }

        stream << row.state_type << ','
               << row.category << ','
               << row.group_id << ','
               << '"' << layers_stream.str() << '"' << ','
               << row.representative_layer << ','
               << row.rule_count << ','
               << row.average_rule_expansion_length << ','
               << bytes_to_mb(row.grammar_bytes) << ','
               << bytes_to_mb(row.permutation_bytes) << ','
               << bytes_to_mb(row.original_bytes) << ','
               << bytes_to_mb(row.transform_only_bytes) << ','
               << bytes_to_mb(row.exact_reconstruction_bytes) << ','
               << row.transform_only_ratio << ','
               << row.exact_reconstruction_ratio << ','
               << row.unique_symbol_count << ','
               << row.repeated_symbol_count << ','
               << row.max_symbol_frequency << ','
               << row.unique_pair_count << ','
               << row.repeated_pair_count << ','
               << row.max_pair_frequency << ','
               << '"' << row.stop_reason << '"' << ','
               << row.learning_time_ms << ','
               << row.apply_time_ms << ','
               << (row.learning_time_ms + row.apply_time_ms) << '\n';
    }
}

void write_checkpoint_summary(const fs::path& output_path,
                              const CheckpointCompressionSummary& summary,
                              const std::map<std::string, std::pair<std::size_t, std::size_t>>& state_type_totals)
{
    std::ofstream stream(output_path);
    stream << std::fixed << std::setprecision(6);
    stream << "checkpoint_original_size_mb: " << bytes_to_mb(summary.original_bytes) << '\n';
    stream << "checkpoint_transform_only_size_mb: " << bytes_to_mb(summary.transform_only_bytes) << '\n';
    stream << "checkpoint_exact_reconstruction_size_mb: " << bytes_to_mb(summary.exact_reconstruction_bytes) << '\n';
    stream << "checkpoint_transform_only_ratio: " << summary.transform_only_ratio << '\n';
    stream << "checkpoint_exact_reconstruction_ratio: " << summary.exact_reconstruction_ratio << '\n';
    stream << "grouped_tensor_count: " << summary.grouped_tensor_count << '\n';
    stream << "uncompressed_tensor_count: " << summary.uncompressed_tensor_count << '\n';
    stream << "grammar_size_mb_total: " << bytes_to_mb(summary.grammar_bytes) << "\n\n";
    stream << "symbolization_mode: " << summary.symbolization_mode << '\n';
    stream << "permutation_size_mb_total: " << bytes_to_mb(summary.permutation_bytes) << '\n';
    stream << "total_rule_count: " << summary.total_rule_count << '\n';
    stream << "zero_rule_group_count: " << summary.zero_rule_group_count << "\n\n";

    for (const auto& [state_type, totals] : state_type_totals) {
        stream << state_type << ": original_size_mb=" << bytes_to_mb(totals.first)
               << ", exact_reconstruction_size_mb=" << bytes_to_mb(totals.second)
               << ", exact_reconstruction_ratio=" << safe_divide(static_cast<double>(totals.first), static_cast<double>(totals.second))
               << '\n';
    }
}

CompressionConfig parse_args(int argc, char** argv)
{
    if (argc < 2) {
        throw std::runtime_error(
            "Usage: tensor_group_repair <export_dir> [--max-rules N] [--state-type NAME] [--category NAME]");
    }

    CompressionConfig config {};
    config.export_dir = fs::path(argv[1]);
    for (int index = 2; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--max-rules" && index + 1 < argc) {
            config.max_rules = static_cast<std::size_t>(std::stoull(argv[++index]));
        } else if (argument == "--state-type" && index + 1 < argc) {
            config.state_type_filter = std::string(argv[++index]);
        } else if (argument == "--category" && index + 1 < argc) {
            config.category_filter = std::string(argv[++index]);
        } else if (argument == "--sort-before-repair") {
            config.sort_before_repair = true;
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + argument);
        }
    }

    return config;
}

void run_group_repair(const CompressionConfig& config)
{
    auto& rm = umpire::ResourceManager::getInstance();
#if defined(RAJA_ENABLE_CUDA)
    auto device_allocator = rm.getAllocator("DEVICE");
#else
    auto device_allocator = rm.getAllocator("HOST");
#endif

    const auto manifest_entries = load_manifest(config.export_dir / "manifest.csv");
    const auto groups = load_groups(config.export_dir / "analysis" / "plots" / "grouping_summary.csv", config);
    const fs::path output_dir = config.export_dir / (config.sort_before_repair ? "repair_sorted" : "repair");
    fs::create_directories(output_dir);

    std::set<std::tuple<std::string, std::string, int>> grouped_keys;
    std::vector<TensorCompressionResult> tensor_rows;
    std::vector<GroupCompressionSummary> group_rows;
    CheckpointCompressionSummary checkpoint_summary {};
    checkpoint_summary.symbolization_mode =
        config.sort_before_repair ? "sorted_exact_float32_bit_patterns" : "exact_float32_bit_patterns";
    std::map<std::string, std::pair<std::size_t, std::size_t>> state_type_totals;

    for (const auto& group : groups) {
        const auto representative_it = manifest_entries.find({group.state_type, group.category, group.representative_layer});
        if (representative_it == manifest_entries.end()) {
            throw std::runtime_error("Missing representative tensor in manifest for group");
        }

        const auto representative_tensor = load_tensor_file(representative_it->second.file_path, representative_it->second.numel);
        auto representative_symbols = tensor_to_symbols(representative_tensor);
        if (config.sort_before_repair) {
            representative_symbols = sort_exact_float_symbols(representative_symbols);
        }
        const auto grammar_result = learn_grammar_on_device(
            representative_symbols,
            config.max_rules,
            rm,
            device_allocator);

        GroupCompressionSummary group_summary {};
        group_summary.state_type = group.state_type;
        group_summary.category = group.category;
        group_summary.group_id = group.group_id;
        group_summary.layers = group.layers;
        group_summary.representative_layer = group.representative_layer;
        group_summary.rule_count = grammar_result.rules.size();
        group_summary.average_rule_expansion_length = grammar_result.average_rule_expansion_length;
        group_summary.grammar_bytes = grammar_result.grammar_bytes;
        group_summary.learning_time_ms = grammar_result.runtime_ms;
        group_summary.unique_symbol_count = grammar_result.unique_symbol_count;
        group_summary.repeated_symbol_count = grammar_result.repeated_symbol_count;
        group_summary.max_symbol_frequency = grammar_result.max_symbol_frequency;
        group_summary.unique_pair_count = grammar_result.unique_pair_count;
        group_summary.repeated_pair_count = grammar_result.repeated_pair_count;
        group_summary.max_pair_frequency = grammar_result.max_pair_frequency;
        group_summary.stop_reason = grammar_result.stop_reason;
        checkpoint_summary.total_rule_count += grammar_result.rules.size();
        if (grammar_result.rules.empty()) {
            ++checkpoint_summary.zero_rule_group_count;
        }

        bool grammar_charged = false;
        for (int layer : group.layers) {
            const auto tensor_it = manifest_entries.find({group.state_type, group.category, layer});
            if (tensor_it == manifest_entries.end()) {
                throw std::runtime_error("Missing grouped tensor in manifest");
            }

            grouped_keys.insert({group.state_type, group.category, layer});
            const std::size_t grammar_bytes = grammar_charged ? 0u : grammar_result.grammar_bytes;
            auto tensor_result = apply_grammar_to_tensor(
                tensor_it->second,
                grammar_result.rules,
                layer == group.representative_layer,
                group.group_id,
                grammar_bytes,
                config.sort_before_repair,
                rm,
                device_allocator);
            grammar_charged = true;

            group_summary.original_bytes += tensor_result.original_bytes;
            group_summary.transform_only_bytes += tensor_result.compressed_stream_bytes + tensor_result.grammar_bytes_charged;
            group_summary.permutation_bytes += tensor_result.permutation_bytes;
            group_summary.exact_reconstruction_bytes += tensor_result.exact_reconstruction_bytes;
            group_summary.apply_time_ms += tensor_result.runtime_ms;
            tensor_rows.push_back(tensor_result);
        }

        group_summary.transform_only_ratio = safe_divide(
            static_cast<double>(group_summary.original_bytes),
            static_cast<double>(group_summary.transform_only_bytes));
        group_summary.exact_reconstruction_ratio = safe_divide(
            static_cast<double>(group_summary.original_bytes),
            static_cast<double>(group_summary.exact_reconstruction_bytes));
        group_rows.push_back(group_summary);

        checkpoint_summary.original_bytes += group_summary.original_bytes;
        checkpoint_summary.transform_only_bytes += group_summary.transform_only_bytes;
        checkpoint_summary.permutation_bytes += group_summary.permutation_bytes;
        checkpoint_summary.exact_reconstruction_bytes += group_summary.exact_reconstruction_bytes;
        checkpoint_summary.grouped_tensor_count += group.layers.size();
        checkpoint_summary.grammar_bytes += group_summary.grammar_bytes;
        state_type_totals[group.state_type].first += group_summary.original_bytes;
        state_type_totals[group.state_type].second += group_summary.exact_reconstruction_bytes;
    }

    for (const auto& [key, entry] : manifest_entries) {
        if (config.state_type_filter && entry.state_type != *config.state_type_filter) {
            continue;
        }
        if (config.category_filter && entry.category != *config.category_filter) {
            continue;
        }
        if (grouped_keys.contains(key)) {
            continue;
        }

        const std::size_t original_bytes = entry.numel * sizeof(float);
        checkpoint_summary.original_bytes += original_bytes;
        checkpoint_summary.transform_only_bytes += original_bytes;
        checkpoint_summary.exact_reconstruction_bytes += original_bytes;
        checkpoint_summary.uncompressed_tensor_count += 1;
        state_type_totals[entry.state_type].first += original_bytes;
        state_type_totals[entry.state_type].second += original_bytes;
    }

    checkpoint_summary.transform_only_ratio = safe_divide(
        static_cast<double>(checkpoint_summary.original_bytes),
        static_cast<double>(checkpoint_summary.transform_only_bytes));
    checkpoint_summary.exact_reconstruction_ratio = safe_divide(
        static_cast<double>(checkpoint_summary.original_bytes),
        static_cast<double>(checkpoint_summary.exact_reconstruction_bytes));

    write_tensor_report(output_dir / "tensor_compression.csv", tensor_rows);
    write_group_report(output_dir / "group_compression.csv", group_rows);
    write_checkpoint_summary(output_dir / "checkpoint_compression_summary.txt", checkpoint_summary, state_type_totals);

    std::cout << "Wrote tensor compression report to " << (output_dir / "tensor_compression.csv") << std::endl;
    std::cout << "Wrote group compression report to " << (output_dir / "group_compression.csv") << std::endl;
    std::cout << "Wrote checkpoint summary to " << (output_dir / "checkpoint_compression_summary.txt") << std::endl;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const auto config = parse_args(argc, argv);
        run_group_repair(config);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tensor_group_repair failed: " << error.what() << std::endl;
        return 1;
    }
}