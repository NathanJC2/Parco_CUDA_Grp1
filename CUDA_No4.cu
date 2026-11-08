%%writefile variant4.cu
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <algorithm>
#include <vector>
#include <chrono>
#include <iomanip>

#ifndef _WIN32
#include <unistd.h>
#else
#include <windows.h>
#endif

#define RUNS 30
#define THREADS_PER_BLOCK 256

// --------------------------------------------------
// CUDA ERROR CHECKING
// --------------------------------------------------

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA Error: %s\n",                        \
                cudaGetErrorString(err));                          \
        return 1;                                                  \
    }                                                              \
} while (0)

// --------------------------------------------------
// PAGE SIZE
// --------------------------------------------------

size_t get_page_size()
{
#ifdef _WIN32
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    return info.dwPageSize;
#else
    long page_size = sysconf(_SC_PAGESIZE);
    return (page_size > 0) ? (size_t)page_size : 4096;
#endif
}

// --------------------------------------------------
// PAGE CREATION
// --------------------------------------------------

void create_pages(double* A, double* B, size_t n)
{
    size_t page_size = get_page_size();
    size_t elements_per_page = page_size / sizeof(double);

    if (elements_per_page == 0)
        elements_per_page = 1;

    volatile double* page_A = A;
    volatile double* page_B = B;

    // Touch at least one element per system page.
    for (size_t i = 0; i < n; i += elements_per_page)
    {
        page_A[i] = 0.0;
        page_B[i] = 0.0;
    }
}

// --------------------------------------------------
// VECTOR INITIALIZATION
// --------------------------------------------------

void initialize_vectors(double* A, double* B, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        A[i] = sin((double)i * 0.001);
        B[i] = cos((double)i * 0.002);
    }
}

// --------------------------------------------------
// C REFERENCE IMPLEMENTATION
// --------------------------------------------------

double dot_c(const double* A, const double* B, size_t n)
{
    double sum = 0.0;

    for (size_t i = 0; i < n; i++)
    {
        sum += A[i] * B[i];
    }

    return sum;
}

// --------------------------------------------------
// CUDA GRID-STRIDE DOT PRODUCT KERNEL
// --------------------------------------------------

__global__ void dot_product_kernel(
    const double* A,
    const double* B,
    size_t n,
    double* partial_sums)
{
    __shared__ double shared_sum[THREADS_PER_BLOCK];

    unsigned int tid = threadIdx.x;

    size_t index =
        blockIdx.x * blockDim.x + threadIdx.x;

    size_t stride =
        blockDim.x * gridDim.x;

    double sum = 0.0;

    // Grid-stride loop
    for (size_t i = index; i < n; i += stride)
    {
        sum += A[i] * B[i];
    }

    shared_sum[tid] = sum;

    __syncthreads();

    // Parallel reduction within each block
    for (unsigned int offset = blockDim.x / 2;
         offset > 0;
         offset >>= 1)
    {
        if (tid < offset)
        {
            shared_sum[tid] += shared_sum[tid + offset];
        }

        __syncthreads();
    }

    // Each block produces one partial sum
    if (tid == 0)
    {
        partial_sums[blockIdx.x] = shared_sum[0];
    }
}

// --------------------------------------------------
// CORRECTNESS CHECK
// --------------------------------------------------

int check_result(double expected, double actual)
{
    double difference = fabs(expected - actual);

    double tolerance =
        1e-12 + 1e-10 * fabs(expected);

    return difference <= tolerance;
}

// --------------------------------------------------
// MAIN
// --------------------------------------------------

int main()
{
    const size_t sizes[] = {
        (size_t)1 << 20,
        (size_t)1 << 26,
        (size_t)1 << 28
    };

    int device;
    cudaDeviceProp prop;

    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("============================================\n");
    printf("VARIANT 4: CUDA DOT PRODUCT\n");
    printf("Grid-stride + Prefetch + Page Creation\n");
    printf("Without Memory Advice\n");
    printf("============================================\n");

    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n",
           prop.major, prop.minor);

    printf("Benchmark Runs: %d\n", RUNS);
    printf("Threads per Block: %d\n", THREADS_PER_BLOCK);

    printf("Initialization:\n");
    printf("A[i] = sin(i * 0.001)\n");
    printf("B[i] = cos(i * 0.002)\n");

    printf("Tolerance: 1e-12 + 1e-10 * |reference|\n");

    for (size_t size_index = 0;
         size_index < sizeof(sizes) / sizeof(sizes[0]);
         size_index++)
    {
        size_t n = sizes[size_index];

        size_t bytes = n * sizeof(double);

        printf("\n============================================\n");
        printf("Vector Size: %zu\n", n);
        printf("Memory per Vector: %.2f MiB\n",
               bytes / (1024.0 * 1024.0));
        printf("============================================\n");

        double* A = nullptr;
        double* B = nullptr;
        double* d_partial = nullptr;

        // Allocate Unified Memory
        CUDA_CHECK(cudaMallocManaged(&A, bytes));
        CUDA_CHECK(cudaMallocManaged(&B, bytes));

        // Determine grid size
        size_t required_blocks =
            (n + THREADS_PER_BLOCK - 1) /
            THREADS_PER_BLOCK;

        int max_blocks = prop.multiProcessorCount * 8;

        int blocks = (int)std::min(
            required_blocks,
            (size_t)max_blocks
        );

        CUDA_CHECK(cudaMalloc(
            &d_partial,
            blocks * sizeof(double)
        ));

        std::vector<double> h_partial(blocks);

        printf("Grid Blocks: %d\n", blocks);

        // ------------------------------------------
        // STEP 1: PAGE CREATION
        // ------------------------------------------

        auto page_start = std::chrono::high_resolution_clock::now();

        create_pages(A, B, n);

        auto page_end = std::chrono::high_resolution_clock::now();

        double page_time =
            std::chrono::duration<double>(
                page_end - page_start
            ).count();

        printf("Page Creation Time: %.6f seconds\n", page_time);

        // ------------------------------------------
        // STEP 2: INITIALIZE VECTORS
        // ------------------------------------------

        initialize_vectors(A, B, n);

        // Calculate C reference before prefetching
        double expected = dot_c(A, B, n);

        printf("C Reference Result: %.12f\n", expected);

        // ------------------------------------------
        // STEP 3: PREFETCH TO GPU
        // ------------------------------------------

        cudaEvent_t prefetch_start, prefetch_stop;

        CUDA_CHECK(cudaEventCreate(&prefetch_start));
        CUDA_CHECK(cudaEventCreate(&prefetch_stop));

        CUDA_CHECK(cudaEventRecord(prefetch_start));

        cudaMemLocation gpu_location = {};
        gpu_location.type = cudaMemLocationTypeDevice;
        gpu_location.id = device;

        CUDA_CHECK(cudaMemPrefetchAsync(
            A,
            bytes,
            gpu_location,
            0,
            0
        ));

        CUDA_CHECK(cudaMemPrefetchAsync(
            B,
            bytes,
            gpu_location,
            0,
            0
        ));

        CUDA_CHECK(cudaEventRecord(prefetch_stop));

        CUDA_CHECK(cudaEventSynchronize(prefetch_stop));

        float prefetch_ms = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &prefetch_ms,
            prefetch_start,
            prefetch_stop
        ));

        printf("Prefetch Time: %.6f milliseconds\n",
               prefetch_ms);

        CUDA_CHECK(cudaEventDestroy(prefetch_start));
        CUDA_CHECK(cudaEventDestroy(prefetch_stop));

        // ------------------------------------------
        // STEP 4: WARM-UP
        // ------------------------------------------

        dot_product_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            A, B, n, d_partial
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ------------------------------------------
        // STEP 5: BENCHMARK 30 RUNS
        // ------------------------------------------

        cudaEvent_t start, stop;

        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        double total_ms = 0.0;
        float min_ms = 1e30f;
        float max_ms = 0.0f;

        for (int run = 0; run < RUNS; run++)
        {
            CUDA_CHECK(cudaEventRecord(start));

            dot_product_kernel<<<blocks, THREADS_PER_BLOCK>>>(
                A, B, n, d_partial
            );

            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaEventRecord(stop));

            CUDA_CHECK(cudaEventSynchronize(stop));

            float elapsed_ms = 0.0f;

            CUDA_CHECK(cudaEventElapsedTime(
                &elapsed_ms, start, stop
            ));

            total_ms += elapsed_ms;

            if (elapsed_ms < min_ms)
                min_ms = elapsed_ms;

            if (elapsed_ms > max_ms)
                max_ms = elapsed_ms;
        }

        double average_ms = total_ms / RUNS;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        // ------------------------------------------
        // STEP 6: RETRIEVE PARTIAL SUMS
        // ------------------------------------------

        CUDA_CHECK(cudaMemcpy(
            h_partial.data(),
            d_partial,
            blocks * sizeof(double),
            cudaMemcpyDeviceToHost
        ));

        double actual = 0.0;

        for (int i = 0; i < blocks; i++)
        {
            actual += h_partial[i];
        }

        // ------------------------------------------
        // STEP 7: CORRECTNESS CHECK
        // ------------------------------------------

        int correct = check_result(expected, actual);

        printf("\n--- RESULTS ---\n");

        printf("CUDA Result: %.12f\n", actual);

        printf("Correctness: %s\n",
               correct ? "PASS" : "FAIL");

        // ------------------------------------------
        // STEP 8: EXECUTION TIME
        // ------------------------------------------

        printf("\n--- KERNEL EXECUTION TIME ---\n");

        printf("Runs: %d\n", RUNS);

        printf("Average: %.6f ms\n", average_ms);

        printf("Minimum: %.6f ms\n", min_ms);

        printf("Maximum: %.6f ms\n", max_ms);

        // ------------------------------------------
        // CLEANUP
        // ------------------------------------------

        CUDA_CHECK(cudaFree(A));
        CUDA_CHECK(cudaFree(B));
        CUDA_CHECK(cudaFree(d_partial));
    }

    printf("\n============================================\n");
    printf("VARIANT 4 COMPLETED\n");
    printf("============================================\n");

    return 0;
}
