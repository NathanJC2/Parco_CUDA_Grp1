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
**Figure 1.** C/C++ baseline output showing execution times and correctness results for the tested vector sizes.
<img width="598" height="1018" alt="image" src="https://github.com/user-attachments/assets/14a91ee1-7251-45b3-824b-6d55d48f4a82" />

**Figure 2.** Correctness checks for CUDA Variants 2–5 across different vector sizes and block sizes.
- Variant 2
  <img width="622" height="787" alt="image" src="https://github.com/user-attachments/assets/bf03f424-6e7e-41cf-888f-7c5ab2bf09d4" />
  <img width="485" height="1071" alt="image" src="https://github.com/user-attachments/assets/4389f7bc-5825-4cf7-ba7a-ff21ff4eaa19" />
  <img width="475" height="1036" alt="image" src="https://github.com/user-attachments/assets/b231a039-f17c-4396-9813-f1ae66515aaf" />
  <img width="477" height="403" alt="image" src="https://github.com/user-attachments/assets/2878d6a2-bb06-4646-b821-9c2732d3b963" />

  
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

