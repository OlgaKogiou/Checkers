#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

#if defined(RAJA_ENABLE_CUDA)
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#endif

namespace tensor_analyzer_quantiles {

struct QuantileBatchBenchmarkResult {
    std::string strategy_name;
    size_t tensor_count {0};
    size_t total_input_bytes {0};
    double total_time_ms {0.0};
    double effective_bandwidth_gbps {0.0};
};

inline double benchmark_safe_divide(double numerator, double denominator)
{
    if (std::abs(denominator) <= 1.0e-12) {
        return 0.0;
    }

    return numerator / denominator;
}

inline double benchmark_bytes_to_gib(size_t bytes)
{
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

inline QuantileBatchBenchmarkResult benchmark_sync_sort_batch(
    const std::vector<std::vector<float>>& tensors,
    umpire::ResourceManager& rm,
    umpire::Allocator& device_allocator,
    umpire::Allocator& host_allocator)
{
    QuantileBatchBenchmarkResult result {};
    result.strategy_name = "sync_sort";
    result.tensor_count = tensors.size();

    auto start = std::chrono::high_resolution_clock::now();
    for (const auto& tensor : tensors) {
        const size_t bytes = tensor.size() * sizeof(float);
        result.total_input_bytes += bytes;

        float* h_staging = static_cast<float*>(host_allocator.allocate(bytes));
        std::copy(tensor.begin(), tensor.end(), h_staging);
        float* d_tensor = static_cast<float*>(device_allocator.allocate(bytes));
        float* d_sorted = static_cast<float*>(device_allocator.allocate(bytes));

        rm.copy(d_tensor, h_staging, bytes);
        rm.copy(d_sorted, d_tensor, bytes);
#if defined(RAJA_ENABLE_CUDA)
        thrust::device_ptr<float> begin(d_sorted);
        thrust::sort(thrust::device, begin, begin + tensor.size());
#else
        std::sort(h_staging, h_staging + tensor.size());
#endif

        device_allocator.deallocate(d_sorted);
        device_allocator.deallocate(d_tensor);
        host_allocator.deallocate(h_staging);
    }
    auto end = std::chrono::high_resolution_clock::now();

    result.total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.effective_bandwidth_gbps = benchmark_safe_divide(
        benchmark_bytes_to_gib(result.total_input_bytes),
        result.total_time_ms / 1000.0);
    return result;
}

inline QuantileBatchBenchmarkResult benchmark_async_sort_batch(
    const std::vector<std::vector<float>>& tensors,
    umpire::ResourceManager& rm,
    umpire::Allocator& device_allocator,
    umpire::Allocator& host_allocator)
{
    QuantileBatchBenchmarkResult result {};
    result.strategy_name = "async_sort";
    result.tensor_count = tensors.size();

#if defined(RAJA_ENABLE_CUDA)
    if (tensors.empty()) {
        return result;
    }

    constexpr size_t k_pipeline_width = 2;
    std::array<cudaStream_t, k_pipeline_width> streams {};
    std::array<float*, k_pipeline_width> h_buffers {};
    std::array<float*, k_pipeline_width> d_input_buffers {};
    std::array<float*, k_pipeline_width> d_sorted_buffers {};
    std::array<size_t, k_pipeline_width> capacities {};

    for (size_t slot = 0; slot < k_pipeline_width; ++slot) {
        cudaStreamCreate(&streams[slot]);
        h_buffers[slot] = nullptr;
        d_input_buffers[slot] = nullptr;
        d_sorted_buffers[slot] = nullptr;
        capacities[slot] = 0;
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t tensor_index = 0; tensor_index < tensors.size(); ++tensor_index) {
        const size_t slot = tensor_index % k_pipeline_width;
        const auto& tensor = tensors[tensor_index];
        const size_t bytes = tensor.size() * sizeof(float);
        result.total_input_bytes += bytes;

        cudaStreamSynchronize(streams[slot]);
        if (capacities[slot] < bytes) {
            if (h_buffers[slot] != nullptr) {
                host_allocator.deallocate(h_buffers[slot]);
            }
            if (d_input_buffers[slot] != nullptr) {
                device_allocator.deallocate(d_input_buffers[slot]);
            }
            if (d_sorted_buffers[slot] != nullptr) {
                device_allocator.deallocate(d_sorted_buffers[slot]);
            }

            h_buffers[slot] = static_cast<float*>(host_allocator.allocate(bytes));
            d_input_buffers[slot] = static_cast<float*>(device_allocator.allocate(bytes));
            d_sorted_buffers[slot] = static_cast<float*>(device_allocator.allocate(bytes));
            capacities[slot] = bytes;
        }

        std::copy(tensor.begin(), tensor.end(), h_buffers[slot]);
        cudaMemcpyAsync(d_input_buffers[slot], h_buffers[slot], bytes, cudaMemcpyHostToDevice, streams[slot]);
        cudaMemcpyAsync(d_sorted_buffers[slot], d_input_buffers[slot], bytes, cudaMemcpyDeviceToDevice, streams[slot]);
        thrust::device_ptr<float> begin(d_sorted_buffers[slot]);
        thrust::sort(thrust::cuda::par.on(streams[slot]), begin, begin + tensor.size());
    }

    for (size_t slot = 0; slot < k_pipeline_width; ++slot) {
        cudaStreamSynchronize(streams[slot]);
    }
    auto end = std::chrono::high_resolution_clock::now();

    for (size_t slot = 0; slot < k_pipeline_width; ++slot) {
        if (h_buffers[slot] != nullptr) {
            host_allocator.deallocate(h_buffers[slot]);
        }
        if (d_input_buffers[slot] != nullptr) {
            device_allocator.deallocate(d_input_buffers[slot]);
        }
        if (d_sorted_buffers[slot] != nullptr) {
            device_allocator.deallocate(d_sorted_buffers[slot]);
        }
        cudaStreamDestroy(streams[slot]);
    }

    result.total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.effective_bandwidth_gbps = benchmark_safe_divide(
        benchmark_bytes_to_gib(result.total_input_bytes),
        result.total_time_ms / 1000.0);
#else
    result = benchmark_sync_sort_batch(tensors, rm, device_allocator, host_allocator);
    result.strategy_name = "async_sort_unavailable";
#endif
    return result;
}

}  // namespace tensor_analyzer_quantiles
