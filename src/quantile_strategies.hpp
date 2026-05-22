#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <vector>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

#if defined(RAJA_ENABLE_CUDA)
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#endif

namespace tensor_analyzer_quantiles {

struct QuantileValues {
    double q1 {0.0};
    double median {0.0};
    double q3 {0.0};
};

struct QuantileExtractionResult {
    QuantileValues values {};
    double runtime_ms {0.0};
    size_t scratch_bytes {0};
    size_t result_bytes {0};
    double effective_bandwidth_gbps {0.0};
    double device_allocation_time_ms {0.0};
    size_t device_allocation_bytes {0};
    size_t peak_live_device_bytes {0};
};

inline double safe_divide(double numerator, double denominator)
{
    if (std::abs(denominator) <= 1.0e-12) {
        return 0.0;
    }

    return numerator / denominator;
}

inline double bytes_to_gib(size_t bytes)
{
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

inline size_t nearest_index(size_t count, double fraction)
{
    if (count == 0) {
        return 0;
    }

    return static_cast<size_t>(std::llround(static_cast<double>(count - 1) * fraction));
}

inline QuantileExtractionResult extract_quantiles_with_sort(
    const std::vector<float>& host_tensor,
    float* d_tensor,
    size_t input_bytes,
    umpire::ResourceManager& rm,
    umpire::Allocator& device_allocator,
    umpire::Allocator& host_allocator)
{
    QuantileExtractionResult result {};
    result.scratch_bytes = input_bytes;
    result.result_bytes = 3 * sizeof(float);

    auto allocation_start = std::chrono::high_resolution_clock::now();
    float* d_sorted = static_cast<float*>(device_allocator.allocate(input_bytes));
    auto allocation_end = std::chrono::high_resolution_clock::now();
    result.device_allocation_time_ms =
        std::chrono::duration<double, std::milli>(allocation_end - allocation_start).count();
    result.device_allocation_bytes = input_bytes;
    result.peak_live_device_bytes = device_allocator.getCurrentSize();

    const size_t q1_index = nearest_index(host_tensor.size(), 0.25);
    const size_t q2_index = nearest_index(host_tensor.size(), 0.50);
    const size_t q3_index = nearest_index(host_tensor.size(), 0.75);

    auto start = std::chrono::high_resolution_clock::now();
    rm.copy(d_sorted, d_tensor, input_bytes);
#if defined(RAJA_ENABLE_CUDA)
    thrust::device_ptr<float> sorted_begin(d_sorted);
    thrust::sort(thrust::device, sorted_begin, sorted_begin + host_tensor.size());
#else
    std::vector<float> host_sorted(host_tensor);
    std::nth_element(host_sorted.begin(), host_sorted.begin() + q1_index, host_sorted.end());
    std::nth_element(
        host_sorted.begin() + q1_index + 1,
        host_sorted.begin() + q2_index,
        host_sorted.end());
    std::nth_element(
        host_sorted.begin() + q2_index + 1,
        host_sorted.begin() + q3_index,
        host_sorted.end());
    float* h_sorted_staging = static_cast<float*>(host_allocator.allocate(input_bytes));
    std::copy(host_sorted.begin(), host_sorted.end(), h_sorted_staging);
    rm.copy(d_sorted, h_sorted_staging, input_bytes);
    host_allocator.deallocate(h_sorted_staging);
#endif

    float* h_quantiles = static_cast<float*>(host_allocator.allocate(3 * sizeof(float)));
    rm.copy(h_quantiles, d_sorted + q1_index, sizeof(float));
    rm.copy(h_quantiles + 1, d_sorted + q2_index, sizeof(float));
    rm.copy(h_quantiles + 2, d_sorted + q3_index, sizeof(float));
    auto end = std::chrono::high_resolution_clock::now();

    result.values.q1 = h_quantiles[0];
    result.values.median = h_quantiles[1];
    result.values.q3 = h_quantiles[2];
    result.runtime_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.effective_bandwidth_gbps =
        safe_divide(bytes_to_gib(input_bytes), result.runtime_ms / 1000.0);

    host_allocator.deallocate(h_quantiles);
    device_allocator.deallocate(d_sorted);
    return result;
}

}  // namespace tensor_analyzer_quantiles
