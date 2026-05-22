#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

constexpr double k_megabyte_divisor = 1024.0 * 1024.0;
constexpr std::size_t k_default_max_rules = 50;
constexpr std::size_t k_min_rule_frequency = 2;
constexpr std::size_t k_plane_metadata_bytes = 16;
constexpr std::size_t k_tensor_metadata_bytes = 16;

struct TensorEntry {
    std::string tensor_key;
    std::string state_type;
    std::string category;
    int layer {0};
    std::size_t numel {0};
    std::string dtype;
    fs::path file_path;
};

struct CompressionConfig {
    fs::path export_dir;
    int layer {0};
    std::string category {"key"};
    std::size_t max_rules {k_default_max_rules};
};

struct DTypeLayout {
    std::string name;
    std::size_t storage_bytes {0};
    std::size_t exponent_bits {0};
    std::size_t mantissa_bits {0};
};

struct GrammarRule {
    std::uint32_t replacement_symbol {0};
    std::uint32_t lhs {0};
    std::uint32_t rhs {0};
    std::size_t frequency {0};
};

struct PlaneCompressionResult {
    std::string plane_name;
    std::string encoding_mode;
    std::size_t symbol_count {0};
    std::size_t unique_symbol_count {0};
    std::size_t stream_bytes {0};
    std::size_t grammar_bytes {0};
    std::size_t metadata_bytes {k_plane_metadata_bytes};
    std::size_t compressed_bytes {0};
    std::size_t rule_count {0};
    std::size_t max_symbol_frequency {0};
    std::size_t max_pair_frequency {0};
};

struct TensorBitplaneResult {
    std::string state_type;
    std::string category;
    int layer {0};
    std::string dtype;
    std::size_t numel {0};
    std::size_t original_bytes {0};
    std::size_t compressed_bytes {0};
    std::size_t grammar_bytes {0};
    std::size_t metadata_bytes {0};
    double compression_ratio {1.0};
    double runtime_ms {0.0};
    PlaneCompressionResult sign_plane;
    PlaneCompressionResult exponent_plane;
    PlaneCompressionResult mantissa_plane;
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

DTypeLayout parse_dtype_layout(const std::string& dtype)
{
    if (dtype == "float32") {
        return DTypeLayout {dtype, 4, 8, 23};
    }
    if (dtype == "float16" || dtype == "fp16") {
        return DTypeLayout {dtype, 2, 5, 10};
    }
    if (dtype == "bfloat16" || dtype == "bf16") {
        return DTypeLayout {dtype, 2, 8, 7};
    }
    throw std::runtime_error("Unsupported dtype for bit-plane decomposition: " + dtype);
}

std::size_t choose_symbol_width(std::uint32_t max_symbol)
{
    if (max_symbol < 256u) {
        return 1;
    }
    if (max_symbol < 65536u) {
        return 2;
    }
    return 4;
}

std::uint64_t encode_pair(std::uint32_t lhs, std::uint32_t rhs)
{
    return (static_cast<std::uint64_t>(lhs) << 32) | rhs;
}

std::optional<std::pair<std::pair<std::uint32_t, std::uint32_t>, std::size_t>> find_best_pair(
    const std::vector<std::uint32_t>& tokens)
{
    if (tokens.size() < 2) {
        return std::nullopt;
    }

    std::unordered_map<std::uint64_t, std::size_t> counts;
    counts.reserve(tokens.size());
    for (std::size_t index = 0; index + 1 < tokens.size(); ++index) {
        ++counts[encode_pair(tokens[index], tokens[index + 1])];
    }

    std::size_t best_count = 0;
    std::uint64_t best_pair = 0;
    for (const auto& [pair_key, count] : counts) {
        if (count > best_count) {
            best_count = count;
            best_pair = pair_key;
        }
    }

    if (best_count < k_min_rule_frequency) {
        return std::nullopt;
    }

    return std::make_pair(
        std::make_pair(static_cast<std::uint32_t>(best_pair >> 32), static_cast<std::uint32_t>(best_pair & 0xffffffffu)),
        best_count);
}

std::vector<std::uint32_t> apply_rule(const std::vector<std::uint32_t>& tokens,
                                      const GrammarRule& rule,
                                      std::size_t& replaced_count)
{
    std::vector<std::uint32_t> output;
    output.reserve(tokens.size());
    replaced_count = 0;

    std::size_t index = 0;
    while (index < tokens.size()) {
        if (index + 1 < tokens.size() && tokens[index] == rule.lhs && tokens[index + 1] == rule.rhs) {
            output.push_back(rule.replacement_symbol);
            index += 2;
            ++replaced_count;
        } else {
            output.push_back(tokens[index]);
            ++index;
        }
    }

    return output;
}

PlaneCompressionResult compress_plane_with_repair(const std::string& plane_name,
                                                  const std::vector<std::uint32_t>& symbols,
                                                  std::size_t max_rules)
{
    PlaneCompressionResult result {};
    result.plane_name = plane_name;
    result.symbol_count = symbols.size();
    if (symbols.empty()) {
        result.encoding_mode = "empty";
        result.compressed_bytes = result.metadata_bytes;
        return result;
    }

    std::unordered_map<std::uint32_t, std::size_t> symbol_counts;
    symbol_counts.reserve(symbols.size());
    std::uint32_t max_symbol = 0;
    for (auto symbol : symbols) {
        ++symbol_counts[symbol];
        max_symbol = std::max(max_symbol, symbol);
    }
    result.unique_symbol_count = symbol_counts.size();
    for (const auto& [symbol, count] : symbol_counts) {
        static_cast<void>(symbol);
        result.max_symbol_frequency = std::max(result.max_symbol_frequency, count);
    }

    if (result.unique_symbol_count == 1) {
        result.encoding_mode = "constant";
        result.stream_bytes = choose_symbol_width(max_symbol);
        result.compressed_bytes = result.stream_bytes + result.metadata_bytes;
        return result;
    }

    result.encoding_mode = "repair";
    std::vector<std::uint32_t> current_tokens = symbols;
    std::vector<GrammarRule> rules;
    rules.reserve(max_rules);
    std::uint32_t next_symbol = max_symbol + 1u;

    for (std::size_t rule_index = 0; rule_index < max_rules; ++rule_index) {
        const auto best_pair = find_best_pair(current_tokens);
        if (!best_pair) {
            break;
        }

        GrammarRule rule {};
        rule.replacement_symbol = next_symbol++;
        rule.lhs = best_pair->first.first;
        rule.rhs = best_pair->first.second;
        rule.frequency = best_pair->second;
        result.max_pair_frequency = std::max(result.max_pair_frequency, rule.frequency);

        std::size_t replaced_count = 0;
        auto updated_tokens = apply_rule(current_tokens, rule, replaced_count);
        if (updated_tokens.size() >= current_tokens.size() || replaced_count == 0) {
            break;
        }

        current_tokens = std::move(updated_tokens);
        rules.push_back(rule);
        max_symbol = std::max(max_symbol, rule.replacement_symbol);
    }

    const auto symbol_width = choose_symbol_width(max_symbol);
    result.rule_count = rules.size();
    result.stream_bytes = current_tokens.size() * symbol_width;
    result.grammar_bytes = rules.size() * (2 * symbol_width);
    result.compressed_bytes = result.stream_bytes + result.grammar_bytes + result.metadata_bytes;
    return result;
}

std::vector<std::uint32_t> load_raw_words(const TensorEntry& entry, const DTypeLayout& layout)
{
    std::ifstream stream(entry.file_path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Failed to open tensor payload: " + entry.file_path.string());
    }

    const std::size_t total_bytes = entry.numel * layout.storage_bytes;
    std::vector<std::uint8_t> raw(total_bytes, 0u);
    stream.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(total_bytes));
    if (stream.gcount() != static_cast<std::streamsize>(total_bytes)) {
        throw std::runtime_error("Tensor payload size mismatch: " + entry.file_path.string());
    }

    std::vector<std::uint32_t> words(entry.numel, 0u);
    for (std::size_t index = 0; index < entry.numel; ++index) {
        std::uint32_t word = 0u;
        for (std::size_t byte = 0; byte < layout.storage_bytes; ++byte) {
            word |= static_cast<std::uint32_t>(raw[index * layout.storage_bytes + byte]) << (8 * byte);
        }
        words[index] = word;
    }
    return words;
}

struct DecomposedPlanes {
    std::vector<std::uint32_t> sign_symbols;
    std::vector<std::uint32_t> exponent_symbols;
    std::vector<std::uint32_t> mantissa_symbols;
};

DecomposedPlanes decompose_words(const std::vector<std::uint32_t>& words, const DTypeLayout& layout)
{
    DecomposedPlanes planes;
    planes.sign_symbols.reserve(words.size());
    planes.exponent_symbols.reserve(words.size());

    const std::uint32_t exponent_mask = (1u << layout.exponent_bits) - 1u;
    const std::uint32_t mantissa_mask = (layout.mantissa_bits == 32)
        ? 0xffffffffu
        : ((1u << layout.mantissa_bits) - 1u);
    const std::size_t mantissa_nibbles = (layout.mantissa_bits + 3u) / 4u;
    planes.mantissa_symbols.reserve(words.size() * mantissa_nibbles);

    for (auto word : words) {
        const auto sign = (word >> (layout.exponent_bits + layout.mantissa_bits)) & 0x1u;
        const auto exponent = (word >> layout.mantissa_bits) & exponent_mask;
        const auto mantissa = word & mantissa_mask;

        planes.sign_symbols.push_back(sign);
        planes.exponent_symbols.push_back(exponent);
        for (std::size_t nibble_index = 0; nibble_index < mantissa_nibbles; ++nibble_index) {
            const auto shift = 4u * (mantissa_nibbles - 1u - nibble_index);
            planes.mantissa_symbols.push_back((mantissa >> shift) & 0xFu);
        }
    }

    return planes;
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
        entry.dtype = columns[5];
        entry.file_path = manifest_path.parent_path() / columns[6];
        entries[{entry.state_type, entry.category, entry.layer}] = entry;
    }
    return entries;
}

TensorBitplaneResult compress_tensor_with_bitplane_repair(const TensorEntry& entry, std::size_t max_rules)
{
    const auto start_time = std::chrono::high_resolution_clock::now();
    const auto layout = parse_dtype_layout(entry.dtype);
    const auto words = load_raw_words(entry, layout);
    const auto planes = decompose_words(words, layout);

    TensorBitplaneResult result {};
    result.state_type = entry.state_type;
    result.category = entry.category;
    result.layer = entry.layer;
    result.dtype = entry.dtype;
    result.numel = entry.numel;
    result.original_bytes = entry.numel * layout.storage_bytes;

    result.sign_plane = compress_plane_with_repair("sign", planes.sign_symbols, max_rules);
    result.exponent_plane = compress_plane_with_repair("exponent", planes.exponent_symbols, max_rules);
    result.mantissa_plane = compress_plane_with_repair("mantissa", planes.mantissa_symbols, max_rules);

    result.grammar_bytes =
        result.sign_plane.grammar_bytes + result.exponent_plane.grammar_bytes + result.mantissa_plane.grammar_bytes;
    result.metadata_bytes =
        k_tensor_metadata_bytes + result.sign_plane.metadata_bytes + result.exponent_plane.metadata_bytes +
        result.mantissa_plane.metadata_bytes;
    result.compressed_bytes =
        k_tensor_metadata_bytes + result.sign_plane.compressed_bytes + result.exponent_plane.compressed_bytes +
        result.mantissa_plane.compressed_bytes;
    result.compression_ratio = safe_divide(
        static_cast<double>(result.original_bytes),
        static_cast<double>(result.compressed_bytes));
    result.runtime_ms =
        std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count();
    return result;
}

void write_tensor_report(const fs::path& output_path, const std::vector<TensorBitplaneResult>& rows)
{
    std::ofstream stream(output_path);
    stream << "state_type,category,layer,dtype,numel,original_size_mb,compressed_size_mb,grammar_size_mb,metadata_size_mb,compression_ratio,sign_mode,sign_rule_count,sign_compressed_kb,exponent_mode,exponent_rule_count,exponent_compressed_kb,mantissa_mode,mantissa_rule_count,mantissa_compressed_kb,time_ms\n";
    for (const auto& row : rows) {
        stream << row.state_type << ','
               << row.category << ','
               << row.layer << ','
               << row.dtype << ','
               << row.numel << ','
               << bytes_to_mb(row.original_bytes) << ','
               << bytes_to_mb(row.compressed_bytes) << ','
               << bytes_to_mb(row.grammar_bytes) << ','
               << bytes_to_mb(row.metadata_bytes) << ','
               << row.compression_ratio << ','
               << row.sign_plane.encoding_mode << ','
               << row.sign_plane.rule_count << ','
               << (static_cast<double>(row.sign_plane.compressed_bytes) / 1024.0) << ','
               << row.exponent_plane.encoding_mode << ','
               << row.exponent_plane.rule_count << ','
               << (static_cast<double>(row.exponent_plane.compressed_bytes) / 1024.0) << ','
               << row.mantissa_plane.encoding_mode << ','
               << row.mantissa_plane.rule_count << ','
               << (static_cast<double>(row.mantissa_plane.compressed_bytes) / 1024.0) << ','
               << row.runtime_ms << '\n';
    }
}

void write_summary(const fs::path& output_path, const std::vector<TensorBitplaneResult>& rows, const CompressionConfig& config)
{
    std::ofstream stream(output_path);
    stream << std::fixed << std::setprecision(6);
    stream << "transform_mode: dtype_aware_bitplane_repair\n";
    stream << "tensor_selection: category=" << config.category << ", layer=" << config.layer << "\n";
    stream << "max_rules_per_plane: " << config.max_rules << "\n\n";

    for (const auto& row : rows) {
        stream << row.state_type << ": dtype=" << row.dtype
               << ", original_size_mb=" << bytes_to_mb(row.original_bytes)
               << ", compressed_size_mb=" << bytes_to_mb(row.compressed_bytes)
               << ", grammar_size_mb=" << bytes_to_mb(row.grammar_bytes)
               << ", metadata_size_mb=" << bytes_to_mb(row.metadata_bytes)
               << ", compression_ratio=" << row.compression_ratio
               << ", sign_mode=" << row.sign_plane.encoding_mode
               << ", exponent_mode=" << row.exponent_plane.encoding_mode
               << ", mantissa_mode=" << row.mantissa_plane.encoding_mode
               << '\n';
    }
}

CompressionConfig parse_args(int argc, char** argv)
{
    if (argc < 2) {
        throw std::runtime_error(
            "Usage: tensor_group_bitplane_repair <export_dir> [--layer N] [--category NAME] [--max-rules N]");
    }

    CompressionConfig config {};
    config.export_dir = fs::path(argv[1]);
    for (int index = 2; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--layer" && index + 1 < argc) {
            config.layer = std::stoi(argv[++index]);
        } else if (argument == "--category" && index + 1 < argc) {
            config.category = std::string(argv[++index]);
        } else if (argument == "--max-rules" && index + 1 < argc) {
            config.max_rules = static_cast<std::size_t>(std::stoull(argv[++index]));
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + argument);
        }
    }
    return config;
}

void run_bitplane_repair(const CompressionConfig& config)
{
    const auto manifest_entries = load_manifest(config.export_dir / "manifest.csv");
    const fs::path output_dir = config.export_dir / "bitplane_repair";
    fs::create_directories(output_dir);

    const std::vector<std::string> state_types = {"model_state", "master_weight", "exp_avg", "exp_avg_sq"};
    std::vector<TensorBitplaneResult> rows;
    rows.reserve(state_types.size());

    for (const auto& state_type : state_types) {
        const auto entry_it = manifest_entries.find({state_type, config.category, config.layer});
        if (entry_it == manifest_entries.end()) {
            throw std::runtime_error(
                "Missing tensor for state_type=" + state_type + ", category=" + config.category + ", layer=" +
                std::to_string(config.layer));
        }
        rows.push_back(compress_tensor_with_bitplane_repair(entry_it->second, config.max_rules));
    }

    write_tensor_report(output_dir / "tensor_bitplane_repair.csv", rows);
    write_summary(output_dir / "checkpoint_bitplane_repair_summary.txt", rows, config);

    std::cout << "Wrote tensor bit-plane report to " << (output_dir / "tensor_bitplane_repair.csv") << std::endl;
    std::cout << "Wrote bit-plane summary to " << (output_dir / "checkpoint_bitplane_repair_summary.txt") << std::endl;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const auto config = parse_args(argc, argv);
        run_bitplane_repair(config);
    } catch (const std::exception& error) {
        std::cerr << "tensor_group_bitplane_repair failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
