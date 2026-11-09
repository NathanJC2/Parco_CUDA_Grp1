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

<img width="678" height="1078" alt="image" src="https://github.com/user-attachments/assets/5a0c0fd0-f775-4b0c-95dc-03d428ce0b56" />

CUDA Variant 2: Grid-Stride Loop

Figure 2. Correctness checks and execution results for CUDA Variant 2 using a grid-stride loop without prefetching or memory advice.

<p>
  <img width="670" height="1078" alt="image" src="https://github.com/user-attachments/assets/480ca519-e880-419c-b7e0-1f747d6bfc95" />
</p>

<p>
  <img width="668" height="1045" alt="image" src="https://github.com/user-attachments/assets/c0c39db3-322d-4747-9cd6-a16102235a40" />
</p>

<p>
  <img width="671" height="1078" alt="image" src="https://github.com/user-attachments/assets/6490c6de-caf8-428e-bd6e-6b52c81d80da" />
</p>

CUDA Variant 3: Grid-Stride Loop with Prefetching

Figure 3. Correctness checks and execution results for CUDA Variant 3 using a grid-stride loop with prefetching.

<p>
  <img width="671" height="1078" alt="image" src="https://github.com/user-attachments/assets/d6ff9d30-4afd-43e7-8789-ff1ef069fc35" />
</p>

<p>
  <img width="683" height="1077" alt="image" src="https://github.com/user-attachments/assets/47fffbd4-a2d0-497a-af7f-fbe02af041c5" />
</p>

<p>
  <img width="680" height="1078" alt="image" src="https://github.com/user-attachments/assets/3e6c8e71-5f00-4d13-b600-9feb5f82288e" />
</p>

CUDA Variant 4: Grid-Stride Loop with Page Creation and Prefetching

Figure 4. Correctness checks and execution results for CUDA Variant 4 using page creation and prefetching.

<p>
  <img width="680" height="1070" alt="image" src="https://github.com/user-attachments/assets/0a9d56f4-0565-4cf2-91cc-784cdd644e93" />
</p>

<p>
  <img width="688" height="1078" alt="image" src="https://github.com/user-attachments/assets/3006299a-7c45-456f-b280-b0426aa58a56" />
</p>

<p>
  <img width="677" height="1040" alt="image" src="https://github.com/user-attachments/assets/81dcca92-0f8e-4a3b-a6d7-172607bfbed2" />
</p>

<p>
   <img width="680" height="1078" alt="image" src="https://github.com/user-attachments/assets/49327a88-895b-4a59-ae2d-7c92db4972dc" />
</p>

CUDA Variant 5: Grid-Stride Loop with Page Creation, Prefetching, and Memory Advice

Figure 5. Correctness checks and execution results for CUDA Variant 5 using page creation, prefetching, and memory advice.

<p>
   <img width="656" height="1078" alt="image" src="https://github.com/user-attachments/assets/6e3922b7-59fb-4a3f-be41-977e95b1875d" />
</p>

<p>
   <img width="655" height="1078" alt="image" src="https://github.com/user-attachments/assets/1163d997-0806-40ce-a61e-ff9f65e92eca" />
</p>

<p>
   <img width="652" height="1078" alt="image" src="https://github.com/user-attachments/assets/1d963d77-14cd-4489-ac25-88fcf76379f4" />
</p>

<p>
   <img width="655" height="1065" alt="image" src="https://github.com/user-attachments/assets/eb47c9bb-6f09-436e-98d1-790e52f25db4" />
</p>

<p>
   <img width="652" height="1078" alt="image" src="https://github.com/user-attachments/assets/103e3950-58b5-471c-a20d-3c7eca2df401" />
</p>

<p>
   <img width="652" height="1078" alt="image" src="https://github.com/user-attachments/assets/46ebfeed-e1c8-440b-a437-721a47467afe" />
</p>

<p>
   <img width="655" height="972" alt="image" src="https://github.com/user-attachments/assets/2240b90a-0fbd-43fb-bfeb-cc58ba4678f7" />
</p>

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

