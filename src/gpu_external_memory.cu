// Strict CUDA single-file implementation of a GPU-resident paged external memory cache.
// Compile: nvcc -O3 -std=c++17 -arch=sm_80 gpu_external_memory.cu -o gpu_mem
// Run:     ./gpu_mem [num_items] [dim] [num_pages] [page_size]
// Example: ./gpu_mem 1048576 128 8192 16

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>

#define CUDA_CHECK(x) do { \
    cudaError_t err__ = (x); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        std::exit(1); \
    } \
} while (0)

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Fixed maximum vector dimension for demo simplicity. Real kernels usually specialize DIM at compile time.
constexpr int MAX_DIM = 256;
constexpr int EMPTY = -1;

struct GpuPagedMemory {
    int num_items;
    int dim;
    int page_size;        // items per physical page
    int num_pages;        // physical pages in memory pool
    int logical_pages;    // ceil(num_items / page_size)
    int slots;            // num_pages * page_size

    float* values;        // [num_pages, page_size, dim]
    int* page_table;      // logical_page -> physical_page, or -1
    int* reverse_table;   // physical_page -> logical_page, or -1
    unsigned int* age;    // physical_page -> last access epoch
    int* clock_hand;      // one int global allocator pointer
    unsigned int* epoch;  // global monotonically increasing access counter
};

__device__ __forceinline__ int ceil_div_i(int a, int b) { return (a + b - 1) / b; }

// A simple GPU-side victim picker. It approximates LRU with a bounded clock scan.
// For high-contention production systems, shard this per-SM or per-bucket.
__device__ int allocate_or_evict_page(GpuPagedMemory mem, int logical_page) {
    // Fast path: another thread already mapped it.
    int mapped = atomicAdd(&mem.page_table[logical_page], 0);
    if (mapped != EMPTY) return mapped;

    // Claim allocator work. Only one claimant wins the CAS for this logical page.
    int sentinel = -2;
    if (atomicCAS(&mem.page_table[logical_page], EMPTY, sentinel) != EMPTY) {
        while ((mapped = atomicAdd(&mem.page_table[logical_page], 0)) < 0) { /* spin */ }
        return mapped;
    }

    // Search physical pages using clock hand. Prefer empty, otherwise oldest sampled page.
    int best_page = -1;
    unsigned int best_age = 0xffffffffu;
    const int scans = min(mem.num_pages, 64);

    for (int s = 0; s < scans; ++s) {
        int p = atomicAdd(mem.clock_hand, 1) % mem.num_pages;
        int owner = atomicAdd(&mem.reverse_table[p], 0);
        unsigned int a = atomicAdd(&mem.age[p], 0);
        if (owner == EMPTY) { best_page = p; break; }
        if (a < best_age) { best_age = a; best_page = p; }
    }
    if (best_page < 0) best_page = atomicAdd(mem.clock_hand, 1) % mem.num_pages;

    // Evict old logical owner if present.
    int old_owner = atomicExch(&mem.reverse_table[best_page], logical_page);
    if (old_owner >= 0 && old_owner != logical_page) {
        atomicCAS(&mem.page_table[old_owner], best_page, EMPTY);
    }

    __threadfence();
    atomicExch(&mem.page_table[logical_page], best_page);
    atomicExch(&mem.age[best_page], atomicAdd(mem.epoch, 1));
    return best_page;
}

__device__ __forceinline__ int physical_offset(const GpuPagedMemory mem, int item_id, int phys_page) {
    int in_page = item_id % mem.page_size;
    return (phys_page * mem.page_size + in_page) * mem.dim;
}

// Store/update items into paged memory. One warp writes one item. Coalesced across dimensions.
__global__ void paged_store_kernel(GpuPagedMemory mem, const float* __restrict__ input, const int* __restrict__ ids, int n) {
    int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x & (WARP_SIZE - 1);
    if (warp_global >= n) return;

    int item_id = ids ? ids[warp_global] : warp_global;
    if (item_id < 0 || item_id >= mem.num_items) return;

    int logical_page = item_id / mem.page_size;
    int phys_page = allocate_or_evict_page(mem, logical_page);
    int base = physical_offset(mem, item_id, phys_page);
    int in_base = warp_global * mem.dim;

    for (int d = lane; d < mem.dim; d += WARP_SIZE) {
        mem.values[base + d] = input[in_base + d];
    }

    if (lane == 0) atomicExch(&mem.age[phys_page], atomicAdd(mem.epoch, 1));
}

// Read items from paged memory. Missing pages return 0. One warp reads one item.
__global__ void paged_load_kernel(GpuPagedMemory mem, float* __restrict__ output, const int* __restrict__ ids, int n) {
    int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x & (WARP_SIZE - 1);
    if (warp_global >= n) return;

    int item_id = ids ? ids[warp_global] : warp_global;
    int out_base = warp_global * mem.dim;
    if (item_id < 0 || item_id >= mem.num_items) {
        for (int d = lane; d < mem.dim; d += WARP_SIZE) output[out_base + d] = 0.0f;
        return;
    }

    int logical_page = item_id / mem.page_size;
    int phys_page = atomicAdd(&mem.page_table[logical_page], 0);
    if (phys_page < 0) {
        for (int d = lane; d < mem.dim; d += WARP_SIZE) output[out_base + d] = 0.0f;
        return;
    }

    int base = physical_offset(mem, item_id, phys_page);
    for (int d = lane; d < mem.dim; d += WARP_SIZE) {
        output[out_base + d] = mem.values[base + d];
    }
    if (lane == 0) atomicExch(&mem.age[phys_page], atomicAdd(mem.epoch, 1));
}

// Gather top-k-ish nearest stored vectors by approximate page sampling.
// This demonstrates external-memory retrieval, not a full ANN index.
__global__ void sampled_dot_retrieve_kernel(
    GpuPagedMemory mem,
    const float* __restrict__ queries, // [num_queries, dim]
    int num_queries,
    int samples_per_query,
    int* __restrict__ best_ids,
    float* __restrict__ best_scores
) {
    extern __shared__ float sh[];
    int q = blockIdx.x;
    int tid = threadIdx.x;
    if (q >= num_queries) return;

    // Load query to shared memory.
    for (int d = tid; d < mem.dim; d += blockDim.x) sh[d] = queries[q * mem.dim + d];
    __syncthreads();

    float local_best = -3.402823e38f;
    int local_id = -1;

    // Deterministic pseudo-random sample sequence.
    unsigned int x = 1664525u * (q + 1) + 1013904223u;
    for (int s = tid; s < samples_per_query; s += blockDim.x) {
        x = 1664525u * (x + s) + 1013904223u;
        int logical_page = (int)(x % (unsigned)mem.logical_pages);
        int phys_page = atomicAdd(&mem.page_table[logical_page], 0);
        if (phys_page < 0) continue;

        int item_in_page = (int)((x >> 8) % (unsigned)mem.page_size);
        int item_id = logical_page * mem.page_size + item_in_page;
        if (item_id >= mem.num_items) continue;

        int base = (phys_page * mem.page_size + item_in_page) * mem.dim;
        float dot = 0.0f;
        for (int d = 0; d < mem.dim; ++d) dot += sh[d] * mem.values[base + d];
        if (dot > local_best) { local_best = dot; local_id = item_id; }
    }

    // Block reduction in shared memory.
    float* score_buf = sh + mem.dim;
    int* id_buf = (int*)(score_buf + blockDim.x);
    score_buf[tid] = local_best;
    id_buf[tid] = local_id;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && score_buf[tid + stride] > score_buf[tid]) {
            score_buf[tid] = score_buf[tid + stride];
            id_buf[tid] = id_buf[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        best_ids[q] = id_buf[0];
        best_scores[q] = score_buf[0];
    }
}

GpuPagedMemory create_memory(int num_items, int dim, int num_pages, int page_size) {
    if (dim <= 0 || dim > MAX_DIM) {
        fprintf(stderr, "dim must be 1..%d\n", MAX_DIM);
        std::exit(1);
    }
    GpuPagedMemory mem{};
    mem.num_items = num_items;
    mem.dim = dim;
    mem.page_size = page_size;
    mem.num_pages = num_pages;
    mem.logical_pages = (num_items + page_size - 1) / page_size;
    mem.slots = num_pages * page_size;

    CUDA_CHECK(cudaMalloc(&mem.values, sizeof(float) * mem.slots * dim));
    CUDA_CHECK(cudaMalloc(&mem.page_table, sizeof(int) * mem.logical_pages));
    CUDA_CHECK(cudaMalloc(&mem.reverse_table, sizeof(int) * mem.num_pages));
    CUDA_CHECK(cudaMalloc(&mem.age, sizeof(unsigned int) * mem.num_pages));
    CUDA_CHECK(cudaMalloc(&mem.clock_hand, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&mem.epoch, sizeof(unsigned int)));

    CUDA_CHECK(cudaMemset(mem.values, 0, sizeof(float) * mem.slots * dim));
    CUDA_CHECK(cudaMemset(mem.page_table, 0xff, sizeof(int) * mem.logical_pages));
    CUDA_CHECK(cudaMemset(mem.reverse_table, 0xff, sizeof(int) * mem.num_pages));
    CUDA_CHECK(cudaMemset(mem.age, 0, sizeof(unsigned int) * mem.num_pages));
    CUDA_CHECK(cudaMemset(mem.clock_hand, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(mem.epoch, 1, sizeof(unsigned int)));
    return mem;
}

void destroy_memory(GpuPagedMemory& mem) {
    cudaFree(mem.values);
    cudaFree(mem.page_table);
    cudaFree(mem.reverse_table);
    cudaFree(mem.age);
    cudaFree(mem.clock_hand);
    cudaFree(mem.epoch);
    mem = {};
}

int main(int argc, char** argv) {
    int num_items = argc > 1 ? std::atoi(argv[1]) : 1 << 20;
    int dim       = argc > 2 ? std::atoi(argv[2]) : 128;
    int num_pages = argc > 3 ? std::atoi(argv[3]) : 8192;
    int page_size = argc > 4 ? std::atoi(argv[4]) : 16;
    int n_store = std::min(num_items, num_pages * page_size); // avoid intentional eviction in validation
    int n_query = 1024;

    printf("GpuPagedMemory: items=%d dim=%d physical_pages=%d page_size=%d resident_items=%d\n",
           num_items, dim, num_pages, page_size, num_pages * page_size);

    GpuPagedMemory mem = create_memory(num_items, dim, num_pages, page_size);

    std::vector<float> h_input((size_t)n_store * dim);
    std::vector<int> h_ids(n_store);
    for (int i = 0; i < n_store; ++i) h_ids[i] = i;
    for (size_t i = 0; i < h_input.size(); ++i) h_input[i] = std::sin((float)i * 0.001f);

    float *d_input, *d_output;
    int *d_ids;
    CUDA_CHECK(cudaMalloc(&d_input, sizeof(float) * h_input.size()));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float) * h_input.size()));
    CUDA_CHECK(cudaMalloc(&d_ids, sizeof(int) * h_ids.size()));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), sizeof(float) * h_input.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ids, h_ids.data(), sizeof(int) * h_ids.size(), cudaMemcpyHostToDevice));

    int threads = 256;
    int warps_per_block = threads / WARP_SIZE;
    int blocks = (n_store + warps_per_block - 1) / warps_per_block;

    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventRecord(a));
    paged_store_kernel<<<blocks, threads>>>(mem, d_input, d_ids, n_store);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float store_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&store_ms, a, b));

    CUDA_CHECK(cudaEventRecord(a));
    paged_load_kernel<<<blocks, threads>>>(mem, d_output, d_ids, n_store);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float load_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&load_ms, a, b));

    std::vector<float> h_output(h_input.size());
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, sizeof(float) * h_output.size(), cudaMemcpyDeviceToHost));
    double max_err = 0.0;
    for (size_t i = 0; i < h_input.size(); ++i) max_err = std::max(max_err, std::abs((double)h_input[i] - h_output[i]));

    std::vector<float> h_queries((size_t)n_query * dim);
    for (size_t i = 0; i < h_queries.size(); ++i) h_queries[i] = std::cos((float)i * 0.003f);
    float* d_queries;
    int* d_best_ids;
    float* d_best_scores;
    CUDA_CHECK(cudaMalloc(&d_queries, sizeof(float) * h_queries.size()));
    CUDA_CHECK(cudaMalloc(&d_best_ids, sizeof(int) * n_query));
    CUDA_CHECK(cudaMalloc(&d_best_scores, sizeof(float) * n_query));
    CUDA_CHECK(cudaMemcpy(d_queries, h_queries.data(), sizeof(float) * h_queries.size(), cudaMemcpyHostToDevice));

    int retrieve_threads = 128;
    size_t shmem = sizeof(float) * (dim + retrieve_threads) + sizeof(int) * retrieve_threads;
    sampled_dot_retrieve_kernel<<<n_query, retrieve_threads, shmem>>>(mem, d_queries, n_query, 4096, d_best_ids, d_best_scores);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> best_ids(n_query);
    std::vector<float> best_scores(n_query);
    CUDA_CHECK(cudaMemcpy(best_ids.data(), d_best_ids, sizeof(int) * n_query, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(best_scores.data(), d_best_scores, sizeof(float) * n_query, cudaMemcpyDeviceToHost));

    double bytes_store = (double)n_store * dim * sizeof(float);
    printf("store: %.3f ms, approx %.2f GB/s payload\n", store_ms, bytes_store / (store_ms * 1e6));
    printf("load : %.3f ms, approx %.2f GB/s payload\n", load_ms, bytes_store / (load_ms * 1e6));
    printf("validation max_abs_error=%g\n", max_err);
    printf("sample retrieval: q0 best_id=%d score=%f\n", best_ids[0], best_scores[0]);

    cudaFree(d_input); cudaFree(d_output); cudaFree(d_ids);
    cudaFree(d_queries); cudaFree(d_best_ids); cudaFree(d_best_scores);
    CUDA_CHECK(cudaEventDestroy(a)); CUDA_CHECK(cudaEventDestroy(b));
    destroy_memory(mem);
    return 0;
}
