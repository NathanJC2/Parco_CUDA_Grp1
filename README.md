# Dot Product Performance Analysis

## Project Information
### Group Members
1. Catapang, Nathan James 
2. Garzon, Harris Martin
3. Ruiz, Joseph Benjamin
4. Tamayo, Chelsea Nichole
   
### Project Specifications
- Programming Language: C/C++ and CUDA C/C++
- Development Environment: Google Colab
- GPU Used: NVIDIA Tesla T4
- Objective: To compare the performance of a C/C++ dot product program with four CUDA implementations using different optimization techniques.
- Vector Initialization:
  - A[i] = sin(i * 0.001)
  - B[i] = cos(i * 0.002)
- Vector Sizes: 2²⁴, 2²⁶, and 2²⁸ elements
- Block Sizes: 256, 512, and 1024 threads
- Number of Runs: 30 runs per configuration
- Correctness Check: The CUDA results are compared with the C/C++ reference results to check if they are correct.
- Profiling Tools: NVIDIA Nsight Systems and NVIDIA Nsight Compute
  
Program Variants:
- Variant 1: C/C++ program version
- Variant 2: CUDA program version using a grid-stride loop without prefetch and memory advise
- Variant 3: CUDA program version using a grid-stride loop with prefetch but without page creation and memory advise
- Variant 4: CUDA program version using a grid-stride loop with prefetch and page creation but without memory advise
- Variant 5: CUDA program version using a grid-stride loop with prefetch, page creation, and memory advise

### AI Usage Declaration
The group used ChatGPT mainly to help us understand CUDA concepts and how the different optimization techniques work. It also assisted us in debugging our code, checking program correctness, and organizing our results and documentation. All implementations were tested, and the results were verified by the group.

## Program Output and Correctness Checks
C/C++ Baseline

Figure 1. C/C++ baseline output showing execution times and correctness results for the tested vector sizes.

<img width="598" alt="C/C++ baseline output" src="https://github.com/user-attachments/assets/14a91ee1-7251-45b3-824b-6d55d48f4a82" />

CUDA Variant 2: Grid-Stride Loop

Figure 2. Correctness checks and execution results for CUDA Variant 2 using a grid-stride loop without prefetching or memory advice.

<p>
  <img width="627" alt="Variant 2 output 1" src="https://github.com/user-attachments/assets/131000ec-5662-4405-9456-8d25b6a00719" />
</p>

<p>
  <img width="478" alt="Variant 2 output 2" src="https://github.com/user-attachments/assets/5886ebb8-dffa-4b70-9b55-11c575d9a090" />
</p>

<p>
  <img width="492" alt="Variant 2 output 3" src="https://github.com/user-attachments/assets/5c8d96e5-7276-4f05-a30a-aa4dadcd0669" />
</p>

CUDA Variant 3: Grid-Stride Loop with Prefetching

Figure 3. Correctness checks and execution results for CUDA Variant 3 using a grid-stride loop with prefetching.

<!-- Insert Variant 3 screenshots here -->

CUDA Variant 4: Grid-Stride Loop with Page Creation and Prefetching

Figure 4. Correctness checks and execution results for CUDA Variant 4 using page creation and prefetching.

<!-- Insert Variant 4 screenshots here -->

CUDA Variant 5: Grid-Stride Loop with Page Creation, Prefetching, and Memory Advice

Figure 5. Correctness checks and execution results for CUDA Variant 5 using page creation, prefetching, and memory advice.

<!-- Insert Variant 5 screenshots here -->




  
## Nsight Systems Profiling for All CUDA Variants
// put screenshots here //

## Comparative Execution-Time Analysis
### A. Results for 2²⁴ Elements
### B. Results for 2²⁶ Elements
### C. Results for 2²⁸ Elements

## Analysis of Results
### A. Analysis of Kernel Execution Time
### B. Analysis of Speed Performance Across All Platforms

## Problems Encountered, Solutions, and Key Insights
### A. Problems Encountered
### B. Solutions Implemented
### C. Unique Methodology
### D. Key Insights and Aha Moments

## SIMD vs. SIMT in the Dot Product Use Case
### A. Differences Between SIMD and SIMT
### B. Advantages of SIMD
### C. Disadvantages of SIMD
### D. Advantages of SIMT
### E. Disadvantages of SIMT
### F. Comparison of SIMD and SIMT for This Project

## Conclusion

