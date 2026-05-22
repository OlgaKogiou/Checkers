#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

// Core RAJA and Umpire Headers
#include "RAJA/RAJA.hpp"
#include "async_quantile_benchmark.hpp"
#include "quantile_strategies.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/QuickPool.hpp"

#if defined(RAJA_ENABLE_CUDA)
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#endif

// --- Fix 1: Explicitly map to correct modern policy namespaces ---
#if defined(RAJA_ENABLE_CUDA)
using ExecutionPolicy = RAJA::cuda_exec<256>;
using ReducePolicy = RAJA::cuda_reduce;
#elif defined(RAJA_ENABLE_OPENMP)
using ExecutionPolicy = RAJA::omp::omp_parallel_for_exec;
using ReducePolicy = RAJA::omp::omp_reduce;
#else
using ExecutionPolicy = RAJA::seq_exec;
using ReducePolicy = RAJA::seq_reduce;
#endif

namespace {

namespace fs = std::filesystem;

constexpr double k_epsilon = 1.0e-12;
const std::vector<std::string> k_state_types {
    "model_state",
    "master_weight",
    "exp_avg",
    "exp_avg_sq",
};

const std::vector<std::string> k_categories {
    "key",
    "value",
    "query",
    "att_dense",
    "mlp_up",
    "mlp_down",
};

struct MetricRuntime {
    double time_ms {0.0};
    size_t scratch_bytes {0};
    size_t result_bytes {0};
    double effective_bandwidth_gbps {0.0};
};

/*
 * Importance and normalization are intentionally kept separate.
 *
 * Each component metric is first converted into a normalized similarity score
 * on a common 0-100 scale. Only after that normalization step do we combine
 * the metrics into a baseline overall similarity.
 *
 * The baseline weights are uniform on purpose. They are not learned from the
 * observed data because variance is not the same thing as statistical
 * importance. The accompanying sensitivity study can then vary these weights
 * explicitly to show which metrics deserve more emphasis in a future,
 * user-chosen scoring policy.
 */
struct MetricWeights {
    double mean_weight {0.0};
    double std_weight {0.0};
    double skew_weight {0.0};
    double kurt_weight {0.0};
    double median_weight {0.0};
    double iqr_weight {0.0};
    double ks_weight {0.0};
};

/*
 * This fingerprint stays entirely in the raw-data domain.
 *
 * The first four entries are moment-based shape descriptors derived from raw
 * floating-point values. The last four entries are quantile descriptors derived
 * from a sorted copy of the same raw tensor values. None of these metrics rely
 * on histogram bins or density estimation.
 */
struct TensorFingerprint {
    size_t element_count {0};
    double mean {0.0};
    double variance {0.0};
    double stddev {0.0};
    double skewness {0.0};
    double excess_kurtosis {0.0};
    double median {0.0};
    double q1 {0.0};
    double q3 {0.0};
    double iqr {0.0};
};

/*
 * The runtime structure keeps the numerical fingerprint together with the
 * timing and memory profile required to produce it. This lets the analyzer
 * answer both “what does this tensor look like?” and “how expensive was it to
 * compute that answer?” using the same record.
 */
struct FingerprintRun {
    TensorFingerprint fingerprint {};
    size_t input_bytes {0};
    size_t peak_live_device_bytes {0};
    size_t reserved_device_bytes {0};
    size_t allocator_high_watermark_bytes {0};
    MetricRuntime device_allocation_runtime {};
    MetricRuntime pinned_allocation_runtime {};
    MetricRuntime host_to_device_runtime {};
    MetricRuntime moments_runtime {};
    MetricRuntime mean_runtime {};
    MetricRuntime std_runtime {};
    MetricRuntime skewness_runtime {};
    MetricRuntime kurtosis_runtime {};
    MetricRuntime sort_runtime {};
    MetricRuntime quantile_runtime {};
    size_t fingerprint_result_bytes {sizeof(TensorFingerprint)};
    double total_time_ms {0.0};
};

struct TensorRecord {
    std::string tensor_key;
    int layer {0};
    std::string category;
    std::string state_type;
    size_t numel {0};
    fs::path file_path;
};

struct SimilarityResult {
    double similarity_pct {0.0};
    double mean_similarity_pct {0.0};
    double std_similarity_pct {0.0};
    double skew_similarity_pct {0.0};
    double kurt_similarity_pct {0.0};
    double median_similarity_pct {0.0};
    double iqr_similarity_pct {0.0};
    double ks_similarity_pct {0.0};
    MetricWeights weights {};
    MetricRuntime ks_sort_runtime {};
    MetricRuntime ks_gap_runtime {};
    double comparison_time_ms {0.0};
};

struct PairwiseComparisonInput {
    TensorRecord anchor_record;
    TensorRecord compare_record;
    double ks_similarity_pct {0.0};
    MetricRuntime ks_sort_runtime {};
    MetricRuntime ks_gap_runtime {};
};

using LayerMap = std::map<int, TensorRecord>;
using CategoryMap = std::map<std::string, LayerMap>;
using StateMap = std::map<std::string, CategoryMap>;

double safe_divide(double numerator, double denominator)
{
    if (std::abs(denominator) <= k_epsilon) {
        return 0.0;
    }

    return numerator / denominator;
}

double bytes_to_gib(size_t bytes)
{
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

std::vector<std::string> split_csv_row(const std::string& line)
{
    std::vector<std::string> columns;
    std::stringstream stream(line);
    std::string column;

    while (std::getline(stream, column, ',')) {
        const auto first = column.find_first_not_of(" \t\r");
        const auto last = column.find_last_not_of(" \t\r");

        if (first == std::string::npos) {
            columns.push_back("");
            continue;
        }

        column = column.substr(first, last - first + 1);
        columns.push_back(column);
    }

    return columns;
}

std::vector<float> load_tensor_file(const fs::path& file_path, size_t numel)
{
    std::ifstream stream(file_path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Failed to open tensor payload: " + file_path.string());
    }

    std::vector<float> tensor(numel, 0.0f);
    stream.read(reinterpret_cast<char*>(tensor.data()), static_cast<std::streamsize>(numel * sizeof(float)));

    if (stream.gcount() != static_cast<std::streamsize>(numel * sizeof(float))) {
        throw std::runtime_error("Tensor payload size mismatch for: " + file_path.string());
    }

    return tensor;
}

StateMap load_manifest(const fs::path& export_dir)
{
    const fs::path manifest_path = export_dir / "manifest.csv";
    std::ifstream stream(manifest_path);
    if (!stream) {
        throw std::runtime_error("Failed to open manifest: " + manifest_path.string());
    }

    std::string line;
    std::getline(stream, line);

    StateMap state_map;
    while (std::getline(stream, line)) {
        if (line.empty()) {
            continue;
        }

        const auto columns = split_csv_row(line);
        if (columns.size() != 7) {
            throw std::runtime_error("Malformed manifest row: " + line);
        }

        TensorRecord record;
        record.tensor_key = columns[0];
        record.layer = std::stoi(columns[1]);
        record.category = columns[2];
        record.state_type = columns[3];
        record.numel = static_cast<size_t>(std::stoull(columns[4]));
        record.file_path = export_dir / columns[6];

        state_map[record.state_type][record.category][record.layer] = record;
    }

    return state_map;
}

MetricWeights make_uniform_metric_weights()
{
    constexpr double uniform_weight = 1.0 / 7.0;

    MetricWeights weights {};
    weights.mean_weight = uniform_weight;
    weights.std_weight = uniform_weight;
    weights.skew_weight = uniform_weight;
    weights.kurt_weight = uniform_weight;
    weights.median_weight = uniform_weight;
    weights.iqr_weight = uniform_weight;
    weights.ks_weight = uniform_weight;
    return weights;
}

#if !defined(RAJA_ENABLE_CUDA)
double compute_ks_statistic_from_sorted(const std::vector<float>& lhs,
                                        const std::vector<float>& rhs,
                                        MetricRuntime& gap_runtime)
{
    auto gap_start = std::chrono::high_resolution_clock::now();

    size_t lhs_index = 0;
    size_t rhs_index = 0;
    double lhs_cdf = 0.0;
    double rhs_cdf = 0.0;
    double max_gap = 0.0;

    while (lhs_index < lhs.size() && rhs_index < rhs.size()) {
        const float lhs_value = lhs[lhs_index];
        const float rhs_value = rhs[rhs_index];

        if (lhs_value <= rhs_value) {
            while (lhs_index < lhs.size() && lhs[lhs_index] == lhs_value) {
                ++lhs_index;
            }
            lhs_cdf = static_cast<double>(lhs_index) / static_cast<double>(lhs.size());
        }

        if (rhs_value <= lhs_value) {
            while (rhs_index < rhs.size() && rhs[rhs_index] == rhs_value) {
                ++rhs_index;
            }
            rhs_cdf = static_cast<double>(rhs_index) / static_cast<double>(rhs.size());
        }

        max_gap = std::max(max_gap, std::abs(lhs_cdf - rhs_cdf));
    }

    while (lhs_index < lhs.size()) {
        ++lhs_index;
        lhs_cdf = static_cast<double>(lhs_index) / static_cast<double>(lhs.size());
        max_gap = std::max(max_gap, std::abs(lhs_cdf - rhs_cdf));
    }

    while (rhs_index < rhs.size()) {
        ++rhs_index;
        rhs_cdf = static_cast<double>(rhs_index) / static_cast<double>(rhs.size());
        max_gap = std::max(max_gap, std::abs(lhs_cdf - rhs_cdf));
    }

    auto gap_end = std::chrono::high_resolution_clock::now();
    gap_runtime.time_ms =
        std::chrono::duration<double, std::milli>(gap_end - gap_start).count();
    gap_runtime.result_bytes = sizeof(double);

    return max_gap;
}
#endif

RAJA_HOST_DEVICE size_t upper_bound_sorted_device(const float* values,
                                                  size_t count,
                                                  float target)
{
    size_t first = 0;
    size_t last = count;

    while (first < last) {
        const size_t middle = first + ((last - first) / 2);
        if (values[middle] <= target) {
            first = middle + 1;
        } else {
            last = middle;
        }
    }

    return first;
}

double compute_ks_statistic_on_device(const float* lhs,
                                      size_t lhs_count,
                                      const float* rhs,
                                      size_t rhs_count,
                                      MetricRuntime& gap_runtime)
{
    auto gap_start = std::chrono::high_resolution_clock::now();

    RAJA::ReduceMax<ReducePolicy, double> lhs_max_gap(0.0);
    RAJA::ReduceMax<ReducePolicy, double> rhs_max_gap(0.0);

    RAJA::forall<ExecutionPolicy>(RAJA::TypedRangeSegment<size_t>(0, lhs_count),
        [=] RAJA_HOST_DEVICE (size_t i) mutable {
            const float value = lhs[i];
            const size_t rhs_rank = upper_bound_sorted_device(rhs, rhs_count, value);
            const double lhs_cdf = static_cast<double>(i + 1) / static_cast<double>(lhs_count);
            const double rhs_cdf = static_cast<double>(rhs_rank) / static_cast<double>(rhs_count);
            lhs_max_gap.max(std::abs(lhs_cdf - rhs_cdf));
        }
    );

    RAJA::forall<ExecutionPolicy>(RAJA::TypedRangeSegment<size_t>(0, rhs_count),
        [=] RAJA_HOST_DEVICE (size_t i) mutable {
            const float value = rhs[i];
            const size_t lhs_rank = upper_bound_sorted_device(lhs, lhs_count, value);
            const double lhs_cdf = static_cast<double>(lhs_rank) / static_cast<double>(lhs_count);
            const double rhs_cdf = static_cast<double>(i + 1) / static_cast<double>(rhs_count);
            rhs_max_gap.max(std::abs(lhs_cdf - rhs_cdf));
        }
    );

    auto gap_end = std::chrono::high_resolution_clock::now();
    gap_runtime.time_ms =
        std::chrono::duration<double, std::milli>(gap_end - gap_start).count();
    gap_runtime.result_bytes = sizeof(double);
    gap_runtime.effective_bandwidth_gbps =
        safe_divide(bytes_to_gib((lhs_count + rhs_count) * sizeof(float)), gap_runtime.time_ms / 1000.0);

    return std::max(lhs_max_gap.get(), rhs_max_gap.get());
}

/*
 * The device fingerprint path is intentionally split into separate timed stages:
 *
 * 1. allocate device memory for the raw tensor and any scratch buffers
 * 2. allocate pinned host memory so host/device copies use Umpire-managed pages
 * 3. transfer the raw tensor from host to device
 * 4. moments pass: one GPU reduction that accumulates the first four raw moments
 * 5. shape pass: one GPU reduction that accumulates the centered third and
 *    fourth moments used by skewness and kurtosis
 * 6. selection pass: use the cheapest backend-supported selection path to
 *    extract quartiles. The CPU fallback uses nth_element. The CUDA path uses
 *    thrust::sort because this CUDA 12.0 Thrust install does not ship
 *    nth_element.
 * 7. quantile finalize: pull Q1/median/Q3 back to host and derive IQR
 *
 * This gives the exact “what took how much time?” breakdown requested for the
 * final checkpoint-analysis workflow.
 */
FingerprintRun run_fingerprint_analysis(const std::vector<float>& host_tensor,
                                        umpire::ResourceManager& rm,
                                        umpire::Allocator& device_allocator)
{
    FingerprintRun run {};
    run.fingerprint.element_count = host_tensor.size();
    run.input_bytes = host_tensor.size() * sizeof(float);

#if defined(RAJA_ENABLE_CUDA)
    umpire::Allocator host_allocator = rm.getAllocator("PINNED");
#else
    umpire::Allocator host_allocator = rm.getAllocator("HOST");
#endif

    auto total_start = std::chrono::high_resolution_clock::now();

    auto pinned_alloc_start = std::chrono::high_resolution_clock::now();
    float* h_staging = static_cast<float*>(host_allocator.allocate(run.input_bytes));
    auto pinned_alloc_end = std::chrono::high_resolution_clock::now();
    run.pinned_allocation_runtime.time_ms =
        std::chrono::duration<double, std::milli>(pinned_alloc_end - pinned_alloc_start).count();
    run.pinned_allocation_runtime.result_bytes = run.input_bytes;
    std::copy(host_tensor.begin(), host_tensor.end(), h_staging);

    auto device_alloc_start = std::chrono::high_resolution_clock::now();
    float* d_tensor = static_cast<float*>(device_allocator.allocate(run.input_bytes));
    auto device_alloc_end = std::chrono::high_resolution_clock::now();
    run.device_allocation_runtime.time_ms =
        std::chrono::duration<double, std::milli>(device_alloc_end - device_alloc_start).count();
    run.device_allocation_runtime.result_bytes = run.input_bytes;

    auto h2d_start = std::chrono::high_resolution_clock::now();
    rm.copy(d_tensor, h_staging, run.input_bytes);
    auto h2d_end = std::chrono::high_resolution_clock::now();
    run.host_to_device_runtime.time_ms =
        std::chrono::duration<double, std::milli>(h2d_end - h2d_start).count();
    run.host_to_device_runtime.result_bytes = run.input_bytes;
    run.host_to_device_runtime.effective_bandwidth_gbps =
        safe_divide(bytes_to_gib(run.input_bytes), run.host_to_device_runtime.time_ms / 1000.0);
    run.peak_live_device_bytes = std::max(run.peak_live_device_bytes, device_allocator.getCurrentSize());
    host_allocator.deallocate(h_staging);

    RAJA::ReduceSum<ReducePolicy, double> sum1(0.0);
    RAJA::ReduceSum<ReducePolicy, double> sum2(0.0);
    RAJA::ReduceSum<ReducePolicy, double> sum3(0.0);
    RAJA::ReduceSum<ReducePolicy, double> sum4(0.0);
    auto moments_start = std::chrono::high_resolution_clock::now();
    /*
     * This RAJA reduction is already a tiled GPU kernel. The CUDA backend
     * breaks the input range into block-sized chunks, reduces each chunk with
     * block-local partial sums, and then combines those block results.
     */
    RAJA::forall<ExecutionPolicy>(RAJA::TypedRangeSegment<size_t>(0, host_tensor.size()),
        [=] RAJA_HOST_DEVICE (size_t i) mutable {
            const double value = static_cast<double>(d_tensor[i]);
            const double square = value * value;

            sum1 += value;
            sum2 += square;
            sum3 += square * value;
            sum4 += square * square;
        }
    );
    auto moments_end = std::chrono::high_resolution_clock::now();

    const double raw_m1 = sum1.get() / static_cast<double>(host_tensor.size());
    const double raw_m2 = sum2.get() / static_cast<double>(host_tensor.size());
    const double raw_m3 = sum3.get() / static_cast<double>(host_tensor.size());
    const double raw_m4 = sum4.get() / static_cast<double>(host_tensor.size());

    run.moments_runtime.time_ms =
        std::chrono::duration<double, std::milli>(moments_end - moments_start).count();
    run.moments_runtime.result_bytes = 4 * sizeof(double);
    run.moments_runtime.effective_bandwidth_gbps =
        safe_divide(bytes_to_gib(run.input_bytes), run.moments_runtime.time_ms / 1000.0);

    auto mean_start = std::chrono::high_resolution_clock::now();
    run.fingerprint.mean = raw_m1;
    auto mean_end = std::chrono::high_resolution_clock::now();
    run.mean_runtime.time_ms =
        std::chrono::duration<double, std::milli>(mean_end - mean_start).count();
    run.mean_runtime.result_bytes = sizeof(double);

    auto std_start = std::chrono::high_resolution_clock::now();
    run.fingerprint.variance = std::max(0.0, raw_m2 - (raw_m1 * raw_m1));
    run.fingerprint.stddev = std::sqrt(run.fingerprint.variance);
    auto std_end = std::chrono::high_resolution_clock::now();
    run.std_runtime.time_ms =
        std::chrono::duration<double, std::milli>(std_end - std_start).count();
    run.std_runtime.result_bytes = 2 * sizeof(double);

    auto shape_start = std::chrono::high_resolution_clock::now();
    if (run.fingerprint.variance > k_epsilon) {
        /*
         * This is the main practical optimization available here: fuse the
         * centered third and fourth moments into one additional GPU pass so we
         * only stream the tensor from global memory once for both skewness and
         * kurtosis. RAJA still handles the block tiling and shared-memory
         * partial reductions under the hood.
         */
        RAJA::ReduceSum<ReducePolicy, double> centered_sum3(0.0);
        RAJA::ReduceSum<ReducePolicy, double> centered_sum4(0.0);
        const double mean = run.fingerprint.mean;
        RAJA::forall<ExecutionPolicy>(RAJA::TypedRangeSegment<size_t>(0, host_tensor.size()),
            [=] RAJA_HOST_DEVICE (size_t i) mutable {
                const double centered = static_cast<double>(d_tensor[i]) - mean;
                const double centered_square = centered * centered;
                centered_sum3 += centered_square * centered;
                centered_sum4 += centered_square * centered_square;
            }
        );

        if (run.fingerprint.stddev > k_epsilon) {
            const double centered_mu3 = centered_sum3.get() / static_cast<double>(host_tensor.size());
            const double skew_denominator = run.fingerprint.variance * run.fingerprint.stddev;
            run.fingerprint.skewness = safe_divide(centered_mu3, skew_denominator);
        }

        const double centered_mu4 = centered_sum4.get() / static_cast<double>(host_tensor.size());
        const double kurtosis_denominator = run.fingerprint.variance * run.fingerprint.variance;
        if (kurtosis_denominator > 0.0) {
            const double kurtosis = centered_mu4 / kurtosis_denominator;
            run.fingerprint.excess_kurtosis = kurtosis - 3.0;
        }
    }
    auto shape_end = std::chrono::high_resolution_clock::now();

    const double shape_time_ms =
        std::chrono::duration<double, std::milli>(shape_end - shape_start).count();
    const double shape_bandwidth_gbps =
        safe_divide(bytes_to_gib(run.input_bytes), shape_time_ms / 1000.0);
    run.skewness_runtime.time_ms = shape_time_ms;
    run.skewness_runtime.result_bytes = sizeof(double);
    run.skewness_runtime.effective_bandwidth_gbps = shape_bandwidth_gbps;
    run.kurtosis_runtime.time_ms = shape_time_ms;
    run.kurtosis_runtime.result_bytes = sizeof(double);
    run.kurtosis_runtime.effective_bandwidth_gbps = shape_bandwidth_gbps;

    const auto quantile_result = tensor_analyzer_quantiles::extract_quantiles_with_sort(
        host_tensor,
        d_tensor,
        run.input_bytes,
        rm,
        device_allocator,
        host_allocator);

    run.device_allocation_runtime.time_ms += quantile_result.device_allocation_time_ms;
    run.device_allocation_runtime.result_bytes += quantile_result.device_allocation_bytes;
    run.sort_runtime.time_ms = quantile_result.runtime_ms;
    run.sort_runtime.scratch_bytes = quantile_result.scratch_bytes;
    run.sort_runtime.effective_bandwidth_gbps = quantile_result.effective_bandwidth_gbps;
    run.quantile_runtime.time_ms = quantile_result.runtime_ms;
    run.quantile_runtime.result_bytes = quantile_result.result_bytes;
    run.pinned_allocation_runtime.result_bytes += quantile_result.result_bytes;
    run.peak_live_device_bytes = std::max(run.peak_live_device_bytes, quantile_result.peak_live_device_bytes);

    run.fingerprint.q1 = quantile_result.values.q1;
    run.fingerprint.median = quantile_result.values.median;
    run.fingerprint.q3 = quantile_result.values.q3;
    run.fingerprint.iqr = run.fingerprint.q3 - run.fingerprint.q1;
    device_allocator.deallocate(d_tensor);

    run.reserved_device_bytes = device_allocator.getActualSize();
    run.allocator_high_watermark_bytes = device_allocator.getHighWatermark();
    run.total_time_ms =
        std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - total_start).count();

    return run;
}

double component_similarity(double anchor, double candidate)
{
    const double relative_delta = safe_divide(
        std::abs(anchor - candidate),
        std::abs(anchor) + std::abs(candidate) + k_epsilon);
    return std::max(0.0, 100.0 * (1.0 - std::min(1.0, relative_delta)));
}

double weighted_group_similarity_pct(const std::vector<std::pair<double, double>>& weighted_scores)
{
    double weight_sum = 0.0;
    double weighted_sum = 0.0;
    for (const auto& [score, weight] : weighted_scores) {
        weight_sum += weight;
        weighted_sum += score * weight;
    }

    if (weight_sum <= k_epsilon) {
        return 0.0;
    }

    return weighted_sum / weight_sum;
}

double aggregate_similarity_pct(const SimilarityResult& result)
{
    /*
     * The baseline overall similarity is a plain weighted average of already
     * normalized component similarities. With uniform weights, this is a clean
     * neutral reference point: raw metric values are compared first, each
     * comparison is scaled to the same 0-100 range, and only then are the
     * normalized scores averaged.
     *
     * This avoids smuggling data-driven importance into the baseline score.
     * Sensitivity analysis is where we study alternative weightings, rather
     * than hard-coding them here.
     */
    return weighted_group_similarity_pct({
        {result.mean_similarity_pct, result.weights.mean_weight},
        {result.std_similarity_pct, result.weights.std_weight},
        {result.skew_similarity_pct, result.weights.skew_weight},
        {result.kurt_similarity_pct, result.weights.kurt_weight},
        {result.median_similarity_pct, result.weights.median_weight},
        {result.iqr_similarity_pct, result.weights.iqr_weight},
        {result.ks_similarity_pct, result.weights.ks_weight},
    });
}

/*
 * This is a raw-fingerprint similarity score rather than a histogram-based
 * divergence. It is designed to stay sensitive to spread, asymmetry, tail
 * weight, and robust quantile structure without introducing bins.
 */
SimilarityResult compare_fingerprints(const TensorFingerprint& anchor,
                                      const TensorFingerprint& candidate,
                                      const MetricWeights& weights,
                                      const PairwiseComparisonInput& pairwise_input)
{
    auto start = std::chrono::high_resolution_clock::now();

    SimilarityResult result {};
    result.mean_similarity_pct = component_similarity(anchor.mean, candidate.mean);
    result.std_similarity_pct = component_similarity(anchor.stddev, candidate.stddev);
    result.skew_similarity_pct = component_similarity(anchor.skewness, candidate.skewness);
    result.kurt_similarity_pct = component_similarity(anchor.excess_kurtosis, candidate.excess_kurtosis);
    result.median_similarity_pct = component_similarity(anchor.median, candidate.median);
    result.iqr_similarity_pct = component_similarity(anchor.iqr, candidate.iqr);
    result.ks_similarity_pct = pairwise_input.ks_similarity_pct;
    result.weights = weights;
    result.ks_sort_runtime = pairwise_input.ks_sort_runtime;
    result.ks_gap_runtime = pairwise_input.ks_gap_runtime;

    result.similarity_pct = aggregate_similarity_pct(result);

    result.comparison_time_ms =
        std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count();
    return result;
}

PairwiseComparisonInput compute_pairwise_ks(const TensorRecord& anchor_record,
                                            const TensorRecord& compare_record,
                                            const std::vector<float>& anchor_tensor,
                                            const std::vector<float>& compare_tensor,
                                            umpire::ResourceManager& rm,
                                            umpire::Allocator& device_allocator)
{
    PairwiseComparisonInput pairwise {};
    pairwise.anchor_record = anchor_record;
    pairwise.compare_record = compare_record;

#if defined(RAJA_ENABLE_CUDA)
    umpire::Allocator host_allocator = rm.getAllocator("PINNED");
#else
    umpire::Allocator host_allocator = rm.getAllocator("HOST");
#endif

    const size_t anchor_bytes = anchor_tensor.size() * sizeof(float);
    const size_t compare_bytes = compare_tensor.size() * sizeof(float);

    float* h_anchor = static_cast<float*>(host_allocator.allocate(anchor_bytes));
    float* h_compare = static_cast<float*>(host_allocator.allocate(compare_bytes));
    std::copy(anchor_tensor.begin(), anchor_tensor.end(), h_anchor);
    std::copy(compare_tensor.begin(), compare_tensor.end(), h_compare);

    float* d_anchor = static_cast<float*>(device_allocator.allocate(anchor_bytes));
    float* d_compare = static_cast<float*>(device_allocator.allocate(compare_bytes));
    rm.copy(d_anchor, h_anchor, anchor_bytes);
    rm.copy(d_compare, h_compare, compare_bytes);

    auto sort_start = std::chrono::high_resolution_clock::now();
#if defined(RAJA_ENABLE_CUDA)
    thrust::device_ptr<float> anchor_begin(d_anchor);
    thrust::device_ptr<float> compare_begin(d_compare);
    thrust::sort(thrust::device, anchor_begin, anchor_begin + anchor_tensor.size());
    thrust::sort(thrust::device, compare_begin, compare_begin + compare_tensor.size());
#else
    std::sort(h_anchor, h_anchor + anchor_tensor.size());
    std::sort(h_compare, h_compare + compare_tensor.size());
#endif
    auto sort_end = std::chrono::high_resolution_clock::now();

    pairwise.ks_sort_runtime.time_ms =
        std::chrono::duration<double, std::milli>(sort_end - sort_start).count();
    pairwise.ks_sort_runtime.scratch_bytes = anchor_bytes + compare_bytes;
    pairwise.ks_sort_runtime.effective_bandwidth_gbps =
        safe_divide(bytes_to_gib(anchor_bytes + compare_bytes), pairwise.ks_sort_runtime.time_ms / 1000.0);

#if defined(RAJA_ENABLE_CUDA)
    host_allocator.deallocate(h_anchor);
    host_allocator.deallocate(h_compare);

    const double ks_statistic = compute_ks_statistic_on_device(
        d_anchor,
        anchor_tensor.size(),
        d_compare,
        compare_tensor.size(),
        pairwise.ks_gap_runtime);
#else
    std::vector<float> sorted_anchor(anchor_tensor.size());
    std::vector<float> sorted_compare(compare_tensor.size());
    std::copy(h_anchor, h_anchor + anchor_tensor.size(), sorted_anchor.begin());
    std::copy(h_compare, h_compare + compare_tensor.size(), sorted_compare.begin());

    const double ks_statistic = compute_ks_statistic_from_sorted(
        sorted_anchor,
        sorted_compare,
        pairwise.ks_gap_runtime);

    host_allocator.deallocate(h_anchor);
    host_allocator.deallocate(h_compare);
#endif
    pairwise.ks_similarity_pct = std::max(0.0, 100.0 * (1.0 - ks_statistic));

    device_allocator.deallocate(d_anchor);
    device_allocator.deallocate(d_compare);

    return pairwise;
}

MetricWeights compute_group_weights(
    const std::vector<std::pair<TensorRecord, FingerprintRun>>& fingerprint_rows,
    const std::vector<PairwiseComparisonInput>& pairwise_rows,
    const std::string& state_type,
    const std::string& category)
{
    (void)fingerprint_rows;
    (void)pairwise_rows;
    (void)state_type;
    (void)category;
    return make_uniform_metric_weights();
}

void write_fingerprint_csv(const fs::path& output_path,
                           const std::vector<std::pair<TensorRecord, FingerprintRun>>& rows)
{
    std::ofstream stream(output_path);
    stream << "tensor_key,layer,category,state_type,numel,mean,stddev,skewness,excess_kurtosis,median,q1,q3,iqr,input_bytes,peak_live_device_bytes,reserved_device_bytes,allocator_high_watermark_bytes,device_allocation_time_ms,pinned_allocation_time_ms,host_to_device_time_ms,moments_time_ms,mean_time_ms,std_time_ms,skewness_time_ms,kurtosis_time_ms,sort_time_ms,quantile_time_ms,total_time_ms,moments_result_bytes,sort_scratch_bytes,quantile_result_bytes,fingerprint_result_bytes\n";

    for (const auto& row : rows) {
        const auto& record = row.first;
        const auto& run = row.second;
        const auto& fp = run.fingerprint;

        stream << record.tensor_key << ','
               << record.layer << ','
               << record.category << ','
               << record.state_type << ','
               << fp.element_count << ','
               << fp.mean << ','
               << fp.stddev << ','
               << fp.skewness << ','
               << fp.excess_kurtosis << ','
               << fp.median << ','
               << fp.q1 << ','
               << fp.q3 << ','
               << fp.iqr << ','
               << run.input_bytes << ','
               << run.peak_live_device_bytes << ','
               << run.reserved_device_bytes << ','
               << run.allocator_high_watermark_bytes << ','
               << run.device_allocation_runtime.time_ms << ','
               << run.pinned_allocation_runtime.time_ms << ','
               << run.host_to_device_runtime.time_ms << ','
               << run.moments_runtime.time_ms << ','
               << run.mean_runtime.time_ms << ','
               << run.std_runtime.time_ms << ','
               << run.skewness_runtime.time_ms << ','
               << run.kurtosis_runtime.time_ms << ','
               << run.sort_runtime.time_ms << ','
               << run.quantile_runtime.time_ms << ','
               << run.total_time_ms << ','
               << run.moments_runtime.result_bytes << ','
               << run.sort_runtime.scratch_bytes << ','
               << run.quantile_runtime.result_bytes << ','
               << run.fingerprint_result_bytes << '\n';
    }
}

void write_similarity_csv(
    const fs::path& output_path,
    const std::vector<std::tuple<TensorRecord, TensorRecord, SimilarityResult>>& rows)
{
    std::ofstream stream(output_path);
    stream << "state_type,category,anchor_layer,compare_layer,anchor_tensor_key,compare_tensor_key,similarity_pct,mean_similarity_pct,std_similarity_pct,skew_similarity_pct,kurt_similarity_pct,median_similarity_pct,iqr_similarity_pct,ks_similarity_pct,mean_weight,std_weight,skew_weight,kurt_weight,median_weight,iqr_weight,ks_weight,ks_sort_time_ms,ks_gap_time_ms,comparison_time_ms\n";

    for (const auto& row : rows) {
        const auto& anchor_record = std::get<0>(row);
        const auto& compare_record = std::get<1>(row);
        const auto& similarity = std::get<2>(row);

        stream << anchor_record.state_type << ','
               << anchor_record.category << ','
               << anchor_record.layer << ','
               << compare_record.layer << ','
               << anchor_record.tensor_key << ','
               << compare_record.tensor_key << ','
               << similarity.similarity_pct << ','
               << similarity.mean_similarity_pct << ','
               << similarity.std_similarity_pct << ','
               << similarity.skew_similarity_pct << ','
               << similarity.kurt_similarity_pct << ','
               << similarity.median_similarity_pct << ','
               << similarity.iqr_similarity_pct << ','
               << similarity.ks_similarity_pct << ','
               << similarity.weights.mean_weight << ','
               << similarity.weights.std_weight << ','
               << similarity.weights.skew_weight << ','
               << similarity.weights.kurt_weight << ','
               << similarity.weights.median_weight << ','
               << similarity.weights.iqr_weight << ','
               << similarity.weights.ks_weight << ','
               << similarity.ks_sort_runtime.time_ms << ','
               << similarity.ks_gap_runtime.time_ms << ','
               << similarity.comparison_time_ms << '\n';
    }
}

void write_quantile_benchmark_csv(
    const fs::path& output_path,
    const std::vector<tensor_analyzer_quantiles::QuantileBatchBenchmarkResult>& rows)
{
    std::ofstream stream(output_path);
    stream << "strategy_name,tensor_count,total_input_bytes,total_time_ms,effective_bandwidth_gbps\n";

    for (const auto& row : rows) {
        stream << row.strategy_name << ','
               << row.tensor_count << ','
               << row.total_input_bytes << ','
               << row.total_time_ms << ','
               << row.effective_bandwidth_gbps << '\n';
    }
}

void print_demo_report(const FingerprintRun& run)
{
    const auto& fp = run.fingerprint;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n=================== STATISTICAL FINGERPRINT ===================" << std::endl;
    std::cout << " Tensor Target Size   : " << fp.element_count << " elements" << std::endl;
    std::cout << " Mean                 : " << fp.mean << std::endl;
    std::cout << " StdDev               : " << fp.stddev << std::endl;
    std::cout << " Skewness             : " << fp.skewness << std::endl;
    std::cout << " Excess Kurtosis      : " << fp.excess_kurtosis << std::endl;
    std::cout << " Median               : " << fp.median << std::endl;
    std::cout << " IQR                  : " << fp.iqr << std::endl;
    std::cout << "=================== PERFORMANCE BREAKDOWN ====================" << std::endl;
    std::cout << " Device allocation    : " << run.device_allocation_runtime.time_ms << " ms" << std::endl;
    std::cout << " Pinned allocation    : " << run.pinned_allocation_runtime.time_ms << " ms" << std::endl;
    std::cout << " Host -> Device       : " << run.host_to_device_runtime.time_ms << " ms" << std::endl;
    std::cout << " Moments pass         : " << run.moments_runtime.time_ms << " ms" << std::endl;
    std::cout << " Mean finalize        : " << run.mean_runtime.time_ms << " ms" << std::endl;
    std::cout << " StdDev finalize      : " << run.std_runtime.time_ms << " ms" << std::endl;
    std::cout << " Skewness finalize    : " << run.skewness_runtime.time_ms << " ms" << std::endl;
    std::cout << " Kurtosis finalize    : " << run.kurtosis_runtime.time_ms << " ms" << std::endl;
    std::cout << " Selection pass       : " << run.sort_runtime.time_ms << " ms" << std::endl;
    std::cout << " Quantile / IQR       : " << run.quantile_runtime.time_ms << " ms" << std::endl;
    std::cout << " Total                : " << run.total_time_ms << " ms" << std::endl;
    std::cout << "=============================================================" << std::endl;
}

void run_synthetic_demo()
{
    const size_t element_count = 16 * 1024 * 1024;

    auto& rm = umpire::ResourceManager::getInstance();

#if defined(RAJA_ENABLE_CUDA)
    std::cout << ">> Orchestrating memory resources using Umpire (CUDA Backend)..." << std::endl;
    umpire::Allocator device_allocator =
        rm.makeAllocator<umpire::strategy::QuickPool>("VRAM_POOL", rm.getAllocator("DEVICE"));
#else
    std::cout << ">> Orchestrating memory resources using Umpire (CPU Fallback)..." << std::endl;
    umpire::Allocator device_allocator = rm.getAllocator("HOST");
#endif

    std::vector<float> host_tensor(element_count, 0.0f);
    std::mt19937 generator(42);
    std::normal_distribution<float> distribution(0.0f, 0.0005f);
    for (size_t i = 0; i < element_count; ++i) {
        host_tensor[i] = distribution(generator);
    }

    std::cout << ">> Launching synthetic tensor demo..." << std::endl;
    const FingerprintRun run = run_fingerprint_analysis(host_tensor, rm, device_allocator);
    print_demo_report(run);
}

void analyze_export_directory(const fs::path& export_dir)
{
    auto& rm = umpire::ResourceManager::getInstance();

#if defined(RAJA_ENABLE_CUDA)
    umpire::Allocator device_allocator =
        rm.makeAllocator<umpire::strategy::QuickPool>("CHECKPOINT_ANALYZER_POOL", rm.getAllocator("DEVICE"));
#else
    umpire::Allocator device_allocator = rm.getAllocator("HOST");
#endif

    const StateMap state_map = load_manifest(export_dir);

    std::vector<std::pair<TensorRecord, FingerprintRun>> fingerprint_rows;
    std::vector<PairwiseComparisonInput> pairwise_inputs;
    std::vector<std::tuple<TensorRecord, TensorRecord, SimilarityResult>> similarity_rows;
    std::unordered_map<std::string, FingerprintRun> fingerprint_cache;
    std::vector<std::vector<float>> quantile_benchmark_tensors;

    for (const auto& state_type : k_state_types) {
        const auto state_it = state_map.find(state_type);
        if (state_it == state_map.end()) {
            continue;
        }

        for (const auto& category : k_categories) {
            const auto category_it = state_it->second.find(category);
            if (category_it == state_it->second.end()) {
                continue;
            }

            const auto& layer_map = category_it->second;
            for (auto current_it = layer_map.begin(); current_it != layer_map.end(); ++current_it) {
                const auto& anchor_record = current_it->second;
                const std::string anchor_key = anchor_record.file_path.string();

                if (!fingerprint_cache.contains(anchor_key)) {
                    const std::vector<float> tensor = load_tensor_file(anchor_record.file_path, anchor_record.numel);
                    quantile_benchmark_tensors.push_back(tensor);
                    FingerprintRun run = run_fingerprint_analysis(tensor, rm, device_allocator);
                    fingerprint_cache[anchor_key] = run;
                    fingerprint_rows.emplace_back(anchor_record, run);
                }

                for (auto compare_it = std::next(current_it); compare_it != layer_map.end(); ++compare_it) {
                    const auto& compare_record = compare_it->second;
                    const std::string compare_key = compare_record.file_path.string();
                    if (!fingerprint_cache.contains(compare_key)) {
                        const std::vector<float> tensor = load_tensor_file(compare_record.file_path, compare_record.numel);
                        quantile_benchmark_tensors.push_back(tensor);
                        FingerprintRun run = run_fingerprint_analysis(tensor, rm, device_allocator);
                        fingerprint_cache[compare_key] = run;
                        fingerprint_rows.emplace_back(compare_record, run);
                    }

                    const std::vector<float> anchor_tensor =
                        load_tensor_file(anchor_record.file_path, anchor_record.numel);
                    const std::vector<float> compare_tensor =
                        load_tensor_file(compare_record.file_path, compare_record.numel);

                    pairwise_inputs.push_back(compute_pairwise_ks(
                        anchor_record,
                        compare_record,
                        anchor_tensor,
                        compare_tensor,
                        rm,
                        device_allocator));
                }
            }
        }
    }

    std::map<std::pair<std::string, std::string>, MetricWeights> weight_map;
    for (const auto& state_type : k_state_types) {
        for (const auto& category : k_categories) {
            weight_map[{state_type, category}] = compute_group_weights(
                fingerprint_rows,
                pairwise_inputs,
                state_type,
                category);
        }
    }

    for (const auto& pairwise_input : pairwise_inputs) {
        const std::string anchor_key = pairwise_input.anchor_record.file_path.string();
        const std::string compare_key = pairwise_input.compare_record.file_path.string();
        const MetricWeights& weights = weight_map.at({
            pairwise_input.anchor_record.state_type,
            pairwise_input.anchor_record.category,
        });

        const SimilarityResult similarity = compare_fingerprints(
            fingerprint_cache.at(anchor_key).fingerprint,
            fingerprint_cache.at(compare_key).fingerprint,
            weights,
            pairwise_input);
        similarity_rows.emplace_back(pairwise_input.anchor_record, pairwise_input.compare_record, similarity);
    }

    const fs::path analysis_dir = export_dir / "analysis";
    fs::create_directories(analysis_dir);

#if defined(RAJA_ENABLE_CUDA)
    umpire::Allocator benchmark_host_allocator = rm.getAllocator("PINNED");
#else
    umpire::Allocator benchmark_host_allocator = rm.getAllocator("HOST");
#endif
    std::vector<tensor_analyzer_quantiles::QuantileBatchBenchmarkResult> benchmark_rows;
    benchmark_rows.push_back(tensor_analyzer_quantiles::benchmark_sync_sort_batch(
        quantile_benchmark_tensors,
        rm,
        device_allocator,
        benchmark_host_allocator));
    benchmark_rows.push_back(tensor_analyzer_quantiles::benchmark_async_sort_batch(
        quantile_benchmark_tensors,
        rm,
        device_allocator,
        benchmark_host_allocator));

    write_fingerprint_csv(analysis_dir / "fingerprints.csv", fingerprint_rows);
    write_similarity_csv(analysis_dir / "pairwise_similarity.csv", similarity_rows);
    write_quantile_benchmark_csv(analysis_dir / "quantile_strategy_benchmarks.csv", benchmark_rows);

    std::cout << "Wrote fingerprint metrics to " << (analysis_dir / "fingerprints.csv") << std::endl;
    std::cout << "Wrote pairwise similarity to " << (analysis_dir / "pairwise_similarity.csv") << std::endl;
    std::cout << "Wrote quantile strategy benchmarks to "
              << (analysis_dir / "quantile_strategy_benchmarks.csv") << std::endl;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        if (argc == 1) {
            run_synthetic_demo();
            return 0;
        }

        analyze_export_directory(argv[1]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "TensorAnalyzer execution failed: " << error.what() << std::endl;
        return 1;
    }
}