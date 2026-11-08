%%writefile benchmark.cu
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define RUNS 30
#define MAX_BLOCKS 1024

__global__ void dot_product_kernel(
    const double* A,
    const double* B,
    size_t n,
    double* block_sums)
{
    extern __shared__ double shared_sums[];

    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    double sum = 0.0;

    for (size_t i = index; i < n; i += stride) {
        sum += A[i] * B[i];
    }

    shared_sums[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2;
         offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared_sums[threadIdx.x] +=
                shared_sums[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_sums[blockIdx.x] = shared_sums[0];
    }
}

__global__ void finish_reduction_kernel(
    const double* block_sums,
    size_t block_count,
    double* result)
{
    extern __shared__ double shared_sums[];

    double sum = 0.0;

    for (size_t i = threadIdx.x;
         i < block_count;
         i += blockDim.x) {
        sum += block_sums[i];
    }

    shared_sums[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2;
         offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared_sums[threadIdx.x] +=
                shared_sums[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        result[0] = shared_sums[0];
    }
}

static int cuda_check(cudaError_t error, const char* operation)
{
    if (error == cudaSuccess) {
        return 1;
    }

    fprintf(stderr, "%s failed: %s\n",
            operation, cudaGetErrorString(error));
    return 0;
}

static void initialize_vectors(double* A, double* B, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        A[i] = sin((double)i * 0.001);
        B[i] = cos((double)i * 0.002);
    }
}

static double sequential_dot(
    const double* A,
    const double* B,
    size_t n,
    double* magnitude_sum)
{
    double sum = 0.0;
    double absolute_sum = 0.0;

    for (size_t i = 0; i < n; i++) {
        double product = A[i] * B[i];
        sum += product;
        absolute_sum += fabs(product);
    }

    *magnitude_sum = absolute_sum;
    return sum;
}

static double correctness_tolerance(
    double expected,
    double magnitude_sum,
    size_t n)
{
    double levels = ceil(log2((double)n)) + 1.0;

    return 1e-12
        + 1e-10 * fabs(expected)
        + 16.0 * DBL_EPSILON * levels * magnitude_sum;
}

static int run_size(size_t n, int block_size, int device)
{
    size_t bytes = n * sizeof(double);
    size_t required_blocks = (n + block_size - 1) / block_size;
    size_t block_count = required_blocks < MAX_BLOCKS
        ? required_blocks
        : MAX_BLOCKS;

    double* A = NULL;
    double* B = NULL;
    double* block_sums = NULL;
    double* result = NULL;

    cudaEvent_t prefetch_start_event = NULL;
    cudaEvent_t prefetch_stop_event = NULL;
    cudaEvent_t start_event = NULL;
    cudaEvent_t stop_event = NULL;

    float times[RUNS];
    float prefetch_time = 0.0f;
    double expected;
    double magnitude_sum;
    double actual;
    double absolute_error;
    double tolerance;
    double average = 0.0;
    double minimum;
    double maximum;

    int run;
    int outcome = -1;
    cudaError_t status = cudaSuccess;

    printf("\n========================================\n");
    printf("Vector size: %zu\n", n);
    printf("Block size: %d\n", block_size);
    printf("Number of blocks: %zu\n", block_count);
    printf("========================================\n");

    status = cudaMallocManaged((void**)&A, bytes);
    if (status != cudaSuccess) goto allocation_failed;

    status = cudaMallocManaged((void**)&B, bytes);
    if (status != cudaSuccess) goto allocation_failed;

    status = cudaMallocManaged(
        (void**)&block_sums, block_count * sizeof(double));
    if (status != cudaSuccess) goto allocation_failed;

    status = cudaMallocManaged((void**)&result, sizeof(double));
    if (status != cudaSuccess) goto allocation_failed;

    initialize_vectors(A, B, n);
    expected = sequential_dot(A, B, n, &magnitude_sum);

    if (!cuda_check(cudaEventCreate(&prefetch_start_event),
                    "Create prefetch start event") ||
        !cuda_check(cudaEventCreate(&prefetch_stop_event),
                    "Create prefetch stop event") ||
        !cuda_check(cudaEventCreate(&start_event),
                    "Create start event") ||
        !cuda_check(cudaEventCreate(&stop_event),
                    "Create stop event")) {
        goto cleanup;
    }

    /*
     * Version 3: prefetch managed allocations to the GPU.
     * No cudaMemAdvise() and no explicit page-creation step.
     */
    {
        cudaMemLocation location;
        location.type = cudaMemLocationTypeDevice;
        location.id = device;

        if (!cuda_check(cudaEventRecord(prefetch_start_event),
                        "Record prefetch start")) {
            goto cleanup;
        }

        status = cudaMemPrefetchAsync(A, bytes, location, 0);
        if (!cuda_check(status, "Prefetch A")) goto cleanup;

        status = cudaMemPrefetchAsync(B, bytes, location, 0);
        if (!cuda_check(status, "Prefetch B")) goto cleanup;

        status = cudaMemPrefetchAsync(
            block_sums, block_count * sizeof(double), location, 0);
        if (!cuda_check(status, "Prefetch block sums")) goto cleanup;

        status = cudaMemPrefetchAsync(result, sizeof(double), location, 0);
        if (!cuda_check(status, "Prefetch result")) goto cleanup;

        if (!cuda_check(cudaEventRecord(prefetch_stop_event),
                        "Record prefetch stop") ||
            !cuda_check(cudaEventSynchronize(prefetch_stop_event),
                        "Synchronize prefetch")) {
            goto cleanup;
        }

        if (!cuda_check(cudaEventElapsedTime(
                &prefetch_time,
                prefetch_start_event,
                prefetch_stop_event),
                "Measure prefetch time")) {
            goto cleanup;
        }
    }

    /* Warm-up execution, excluded from the 30 timed runs. */
    dot_product_kernel<<<(unsigned int)block_count, block_size,
        block_size * sizeof(double)>>>(A, B, n, block_sums);

    if (!cuda_check(cudaGetLastError(), "Launch warm-up dot kernel")) {
        goto cleanup;
    }

    finish_reduction_kernel<<<1, block_size,
        block_size * sizeof(double)>>>(block_sums, block_count, result);

    if (!cuda_check(cudaGetLastError(), "Launch warm-up reduction")) {
        goto cleanup;
    }

    if (!cuda_check(cudaDeviceSynchronize(), "Warm-up synchronization")) {
        goto cleanup;
    }

    /* Time 30 executions of both kernels. */
    for (run = 0; run < RUNS; run++) {
        if (!cuda_check(cudaEventRecord(start_event),
                        "Record start event")) {
            goto cleanup;
        }

        dot_product_kernel<<<(unsigned int)block_count, block_size,
            block_size * sizeof(double)>>>(A, B, n, block_sums);

        if (!cuda_check(cudaGetLastError(), "Launch dot kernel")) {
            goto cleanup;
        }

        finish_reduction_kernel<<<1, block_size,
            block_size * sizeof(double)>>>(block_sums, block_count, result);

        if (!cuda_check(cudaGetLastError(), "Launch reduction kernel")) {
            goto cleanup;
        }

        if (!cuda_check(cudaEventRecord(stop_event), "Record stop event") ||
            !cuda_check(cudaEventSynchronize(stop_event),
                        "Synchronize stop event") ||
            !cuda_check(cudaEventElapsedTime(
                &times[run], start_event, stop_event),
                "Get execution time")) {
            goto cleanup;
        }
    }

    actual = result[0];
    absolute_error = fabs(expected - actual);
    tolerance = correctness_tolerance(expected, magnitude_sum, n);

    minimum = times[0];
    maximum = times[0];

    for (run = 0; run < RUNS; run++) {
        average += times[run];

        if (times[run] < minimum) minimum = times[run];
        if (times[run] > maximum) maximum = times[run];
    }

    average /= RUNS;

    printf("CPU reference result: %.17g\n", expected);
    printf("CUDA result:          %.17g\n", actual);
    printf("Absolute error:       %.17g\n", absolute_error);
    printf("Tolerance:            %.17g\n", tolerance);
    printf("Correctness:          %s\n",
           absolute_error <= tolerance ? "PASS" : "FAIL");
    printf("Prefetch time:        %.3f microseconds\n",
           prefetch_time * 1000.0f);
    printf("Average kernel time:  %.3f microseconds\n",
           average * 1000.0);
    printf("Minimum kernel time:  %.3f microseconds\n",
           minimum * 1000.0);
    printf("Maximum kernel time:  %.3f microseconds\n",
           maximum * 1000.0);
    printf("Timed executions:     %d\n", RUNS);

    outcome = absolute_error <= tolerance ? 1 : 0;
    goto cleanup;

allocation_failed:
    fprintf(stderr, "Unified Memory allocation failed: %s\n",
            cudaGetErrorString(status));

cleanup:
    if (prefetch_start_event != NULL)
        cudaEventDestroy(prefetch_start_event);
    if (prefetch_stop_event != NULL)
        cudaEventDestroy(prefetch_stop_event);
    if (start_event != NULL)
        cudaEventDestroy(start_event);
    if (stop_event != NULL)
        cudaEventDestroy(stop_event);

    if (result != NULL) cudaFree(result);
    if (block_sums != NULL) cudaFree(block_sums);
    if (B != NULL) cudaFree(B);
    if (A != NULL) cudaFree(A);

    return outcome;
}

int main(void)
{
    const size_t sizes[] = {
        (size_t)1 << 20,
        (size_t)1 << 26,
        (size_t)1 << 28
    };

    const int block_sizes[] = {256, 512, 1024};

    int device_count = 0;
    int device = 0;
    int result;
    int failures = 0;
    int block_index;
    size_t size_index;
    cudaDeviceProp properties;

    if (!cuda_check(cudaGetDeviceCount(&device_count),
                    "Get device count")) {
        return 1;
    }

    if (device_count == 0) {
        fprintf(stderr, "No CUDA-capable GPU is available.\n");
        return 1;
    }

    if (!cuda_check(cudaGetDevice(&device), "Get current device") ||
        !cuda_check(cudaGetDeviceProperties(&properties, device),
                    "Get device properties")) {
        return 1;
    }

    printf("CUDA device: %s\n", properties.name);
    printf("Version: Unified Memory with prefetching\n");
    printf("No separate page-creation step or memory advice\n");
    printf("Timed executions per test: %d\n", RUNS);

    for (block_index = 0; block_index < 3; block_index++) {
        for (size_index = 0;
             size_index < sizeof(sizes) / sizeof(sizes[0]);
             size_index++) {

            result = run_size(
                sizes[size_index], block_sizes[block_index], device);

            if (result <= 0) failures++;

            if (result < 0) {
                fprintf(stderr, "Benchmark stopped due to an error.\n");
                return 1;
            }
        }
    }

    printf("\nBenchmark completed.\n");
    printf("Failed tests: %d\n", failures);

    return failures == 0 ? 0 : 1;
}
