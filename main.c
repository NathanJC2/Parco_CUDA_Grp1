%% writefile main.c
#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <stddef.h>

#define RUNS 30

typedef double (*DotKernel)(const double*, const double*, size_t);

/* Vector sizes to test */
static const size_t VECTOR_SIZES[] = {
    (size_t)1 << 20,
    (size_t)1 << 26,
    (size_t)1 << 28
};

#define NUM_SIZES (sizeof(VECTOR_SIZES) / sizeof(VECTOR_SIZES[0]))

/* Sequential C dot product */
double dot_c(const double* A, const double* B, size_t n)
{
    double sum = 0.0;

    for (size_t i = 0; i < n; i++) {
        sum += A[i] * B[i];
    }

    return sum;
}

/* Initialize vectors */
void initialize_vectors(double* A, double* B, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        A[i] = sin((double)i * 0.001);
        B[i] = cos((double)i * 0.002);
    }
}

/* Calculate correctness tolerance */
double get_tolerance(double expected)
{
    return 1e-12 + 1e-10 * fabs(expected);
}

/* Check correctness */
int check_result(double expected, double actual)
{
    return fabs(expected - actual) <= get_tolerance(expected);
}

/* High-resolution timer */
double get_time(void)
{
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }

    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* Run and measure the CPU kernel */
int time_kernel(
    DotKernel kernel,
    const double* A,
    const double* B,
    size_t n,
    double expected)
{
    double total = 0.0;
    double minimum = 1e100;
    double maximum = 0.0;
    double result = 0.0;

    volatile double sink = 0.0;

    for (int i = 0; i < RUNS; i++) {
        double start = get_time();

        result = kernel(A, B, n);

        double end = get_time();
        double elapsed = end - start;

        sink = result;

        total += elapsed;

        if (elapsed < minimum) {
            minimum = elapsed;
        }

        if (elapsed > maximum) {
            maximum = elapsed;
        }
    }

    (void)sink;

    double error = fabs(expected - result);
    double tolerance = get_tolerance(expected);
    int passed = check_result(expected, result);

    printf("CPU reference result: %.15f\n", expected);
    printf("CPU result:           %.15f\n", result);
    printf("Absolute error:       %.15e\n", error);
    printf("Tolerance:            %.15e\n", tolerance);

    printf("Correctness:          %s\n",
        passed ? "PASS" : "FAIL");

    printf("Average execution time:  %.3f microseconds\n",
        (total / RUNS) * 1e6);

    printf("Minimum execution time:  %.3f microseconds\n",
        minimum * 1e6);

    printf("Maximum execution time:  %.3f microseconds\n",
        maximum * 1e6);

    printf("Timed executions:     %d\n", RUNS);

    return passed;
}

/* Run one vector-size test */
int run_test(size_t n)
{
    double* A = NULL;
    double* B = NULL;
    double expected;
    int passed;

    printf("\n========================================\n");
    printf("Vector size: %zu\n", n);
    printf("========================================\n");

    if (n > SIZE_MAX / sizeof(double)) {
        printf("Vector size is too large.\n");
        return 0;
    }

    A = (double*)malloc(n * sizeof(double));
    B = (double*)malloc(n * sizeof(double));

    if (A == NULL || B == NULL) {
        printf("Memory allocation failed.\n");
        free(A);
        free(B);
        return 0;
    }

    /* Initialization is outside the timed kernel */
    initialize_vectors(A, B, n);

    /* Calculate CPU reference result */
    expected = dot_c(A, B, n);

    /* Execute 30 timed runs */
    passed = time_kernel(dot_c, A, B, n, expected);

    free(A);
    free(B);

    return passed;
}

int main(void)
{
    int all_passed = 1;

    printf("Sequential C Baseline: Vector Dot Product\n");
    printf("Initialization: A[i] = sin(i * 0.001), "
        "B[i] = cos(i * 0.002)\n");
    printf("Timed executions per vector size: %d\n", RUNS);

    for (size_t i = 0; i < NUM_SIZES; i++) {
        if (!run_test(VECTOR_SIZES[i])) {
            all_passed = 0;
        }
    }

    printf("\n========================================\n");

    if (all_passed) {
        printf("All tests completed successfully.\n");
    }
    else {
        printf("One or more tests failed.\n");
    }

    printf("========================================\n");

    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}