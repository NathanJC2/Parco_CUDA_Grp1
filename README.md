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
### Variant 2

**Figure 6.** OS Runtime Summary for Variant 2.

<img width="1505" height="735" alt="image" src="https://github.com/user-attachments/assets/60d806af-14d9-4cab-9be8-1191a571d96e" />

**Figure 7.** CUDA API Summary for Variant 2.

<img width="1501" height="412" alt="image" src="https://github.com/user-attachments/assets/9c962c91-5294-41ed-bea3-b8510186a174" />

**Figure 8.** GPU Kernel and Memory-Transfer Statistics for Variant 2.

<img width="1782" height="581" alt="image" src="https://github.com/user-attachments/assets/1cb73d2e-4f9c-44dc-8efe-c504223007c7" />

### Variant 3

**Figure 9.** OS Runtime Summary for Variant 3.

<img width="1501" height="822" alt="image" src="https://github.com/user-attachments/assets/0b51593e-35e4-4f72-880c-bd87654727f7" />

**Figure 10.** CUDA API Summary for Variant 3.

<img width="1492" height="452" alt="image" src="https://github.com/user-attachments/assets/8cd26467-dfa3-4b63-8d0a-7571ef93ebf7" />

**Figure 11.** GPU Kernel and Memory-Transfer Statistics for Variant 3.

<img width="1771" height="582" alt="image" src="https://github.com/user-attachments/assets/dab2c3b9-9671-4e70-b44e-36be7aecc0b2" />

### Variant 4

**Figure 12.** OS Runtime Summary for Variant 4.

<img width="1501" height="800" alt="image" src="https://github.com/user-attachments/assets/e389bdce-027d-4899-b731-2ce5193df657" />

**Figure 13.** CUDA API Summary for Variant 4.

<img width="1492" height="471" alt="image" src="https://github.com/user-attachments/assets/e5f9b3cd-f394-4166-9168-fa3aac438533" />

**Figure 14.** GPU Kernel and Memory-Transfer Statistics for Variant 4.

<img width="1771" height="581" alt="image" src="https://github.com/user-attachments/assets/9d286a57-a4b1-4d92-8ec6-3d825d5499c8" />

## Comparative Execution-Time Analysis
### A. Results for 2²⁴ Elements
| Kernel | Block Size | Average Time (μs) | Speedup vs C |
|---|---:|---:|---:|
| C (Baseline) | - | 1407.891 | 1.00x |
| x86-64 Assembly | - | 893.823 | 1.58x |
| x86-64 SIMD XMM | - | 602.657 | 2.34x |
| x86-64 SIMD YMM | - | 385.100 | 3.66x |
| CUDA Unified | 256 | 348.353 | 4.04x |
| CUDA Unified | 512 | 202.532 | 6.95x |
| CUDA Unified | 1024 | 290.704 | 4.84x |
| CUDA Prefetch | 256 | 81.233 | 17.33x |
| CUDA Prefetch | 512 | 94.435 | 14.91x |
| CUDA Prefetch | 1024 | 171.421 | 8.21x |
| CUDA Prefetch + Page Creation | 256 | 103.938 | 13.55x |
| CUDA Prefetch + Page Creation | 512 | 172.268 | 8.17x |
| CUDA Prefetch + Page Creation | 1024 | 363.194 | 3.88x |
| CUDA Prefetch + Page Creation + Mem Advice | 256 | 106.269 | 13.25x |
| CUDA Prefetch + Page Creation + Mem Advice | 512 | 180.042 | 7.82x |
| CUDA Prefetch + Page Creation + Mem Advice | 1024 | 331.567 | 4.25x |

### B. Results for 2²⁶ Elements
| Kernel | Block Size | Average Time (μs) | Speedup vs C |
|---|---:|---:|---:|
| C (Baseline) | - | 113629.128 | 1.00x |
| x86-64 Assembly | - | 55515.197 | 2.05x |
| x86-64 SIMD XMM | - | 39002.567 | 2.91x |
| x86-64 SIMD YMM | - | 34083.980 | 3.33x |
| CUDA Unified | 256 | 12368.990 | 9.19x |
| CUDA Unified | 512 | 11199.752 | 10.15x |
| CUDA Unified | 1024 | 12232.219 | 9.29x |
| CUDA Prefetch | 256 | 4007.925 | 28.35x |
| CUDA Prefetch | 512 | 3978.243 | 28.56x |
| CUDA Prefetch | 1024 | 4070.829 | 27.91x |
| CUDA Prefetch + Page Creation | 256 | 3980.043 | 28.55x |
| CUDA Prefetch + Page Creation | 512 | 3945.934 | 28.80x |
| CUDA Prefetch + Page Creation | 1024 | 3956.501 | 28.72x |
| CUDA Prefetch + Page Creation + Mem Advice | 256 | 3979.824 | 28.55x |
| CUDA Prefetch + Page Creation + Mem Advice | 512 | 3950.001 | 28.77x |
| CUDA Prefetch + Page Creation + Mem Advice | 1024 | 3956.349 | 28.72x |

### C. Results for 2²⁸ Elements
| Kernel | Block Size | Average Time (μs) | Speedup vs C |
|---|---:|---:|---:|
| C (Baseline) | - | 444390.058 | 1.00x |
| x86-64 Assembly | - | 221395.230 | 2.01x |
| x86-64 SIMD XMM | - | 170132.387 | 2.61x |
| x86-64 SIMD YMM | - | 150384.397 | 2.96x |
| CUDA Unified | 256 | 65144.547 | 6.82x |
| CUDA Unified | 512 | 45842.321 | 9.69x |
| CUDA Unified | 1024 | 51228.078 | 8.67x |
| CUDA Prefetch | 256 | 15946.253 | 27.87x |
| CUDA Prefetch | 512 | 15851.203 | 28.04x |
| CUDA Prefetch | 1024 | 15821.575 | 28.09x |
| CUDA Prefetch + Page Creation | 256 | 15965.106 | 27.84x |
| CUDA Prefetch + Page Creation | 512 | 15868.160 | 28.01x |
| CUDA Prefetch + Page Creation | 1024 | 15776.598 | 28.17x |
| CUDA Prefetch + Page Creation + Mem Advice | 256 | 15961.716 | 27.84x |
| CUDA Prefetch + Page Creation + Mem Advice | 512 | 15866.989 | 28.01x |
| CUDA Prefetch + Page Creation + Mem Advice | 1024 | 15831.164 | 28.07x |

## Analysis of Results
### A. Analysis of Kernel Execution Time
## Performance Results – 2^20 Elements (1,048,576)

| Kernel | Block Size | Average Time (μs) | Minimum Time (μs) | Maximum Time (μs) | Speedup vs C |
|---|---:|---:|---:|---:|---:|
| C (Baseline) | - | 1407.891 | 1328.185 | 1575.761 | 1.00x |
| x86-64 Assembly | - | 893.823 | - | - | 1.58x |
| x86-64 SIMD XMM | - | 602.657 | - | - | 2.34x |
| x86-64 SIMD YMM | - | 385.100 | - | - | 3.66x |
| CUDA Unified | 256 | 348.353 | 94.240 | 7637.312 | 4.04x |
| CUDA Unified | 512 | 202.532 | 79.872 | 3712.992 | 6.95x |
| CUDA Unified | 1024 | 290.704 | 165.184 | 3889.888 | 4.84x |
| CUDA Prefetch | 256 | 81.233 | 78.656 | 100.864 | 17.33x |
| CUDA Prefetch | 512 | 94.435 | 92.160 | 99.328 | 14.91x |
| CUDA Prefetch | 1024 | 171.421 | 168.864 | 180.224 | 8.21x |
| CUDA Prefetch + Page Creation | 256 | 103.938 | 100.320 | 118.784 | 13.55x |
| CUDA Prefetch + Page Creation | 512 | 172.268 | 169.824 | 176.160 | 8.17x |
| CUDA Prefetch + Page Creation | 1024 | 363.194 | 361.152 | 367.360 | 3.88x |
| CUDA Prefetch + Page Creation + Mem Advice | 256 | 106.269 | 104.448 | 110.176 | 13.25x |
| CUDA Prefetch + Page Creation + Mem Advice | 512 | 180.042 | 176.160 | 182.752 | 7.82x |
| CUDA Prefetch + Page Creation + Mem Advice | 1024 | 331.567 | 319.488 | 395.424 | 4.25x |

## Performance Results – 2^26 Elements (67,108,864)

| Kernel | Block Size | Average Time (μs) | Minimum Time (μs) | Maximum Time (μs) | Speedup vs C |
|---|---:|---:|---:|---:|---:|
| C (Baseline) | - | 113629.128 | 105056.585 | 132172.179 | 1.00x |
| x86-64 Assembly | - | 55515.197 | - | - | 2.05x |
| x86-64 SIMD XMM | - | 39002.567 | - | - | 2.91x |
| x86-64 SIMD YMM | - | 34083.980 | - | - | 3.33x |
| CUDA Unified | 256 | 12368.990 | 3940.352 | 256582.855 | 9.19x |
| CUDA Unified | 512 | 11199.752 | 3933.856 | 221664.825 | 10.15x |
| CUDA Unified | 1024 | 12232.219 | 3951.616 | 252031.296 | 9.29x |
| CUDA Prefetch | 256 | 4007.925 | 3995.648 | 4020.032 | 28.35x |
| CUDA Prefetch | 512 | 3978.243 | 3960.640 | 4029.920 | 28.56x |
| CUDA Prefetch | 1024 | 4070.829 | 4057.152 | 4086.400 | 27.91x |
| CUDA Prefetch + Page Creation | 256 | 3980.043 | 3952.384 | 4012.992 | 28.55x |
| CUDA Prefetch + Page Creation | 512 | 3945.934 | 3940.000 | 3950.560 | 28.80x |
| CUDA Prefetch + Page Creation | 1024 | 3956.501 | 3930.464 | 4479.008 | 28.72x |
| CUDA Prefetch + Page Creation + Mem Advice | 256 | 3979.824 | 3952.096 | 4026.336 | 28.55x |
| CUDA Prefetch + Page Creation + Mem Advice | 512 | 3950.001 | 3945.408 | 3958.784 | 28.77x |
| CUDA Prefetch + Page Creation + Mem Advice | 1024 | 3956.349 | 3949.184 | 3985.408 | 28.72x |

## Performance Results – 2^28 Elements (268,435,456)

| Kernel | Block Size | Average Time (μs) | Minimum Time (μs) | Maximum Time (μs) | Speedup vs C |
|---|---:|---:|---:|---:|---:|
| C (Baseline) | - | 444390.058 | 418062.890 | 494333.567 | 1.00x |
| x86-64 Assembly | - | 221395.230 | - | - | 2.01x |
| x86-64 SIMD XMM | - | 170132.387 | - | - | 2.61x |
| x86-64 SIMD YMM | - | 150384.397 | - | - | 2.96x |
| CUDA Unified | 256 | 65144.547 | 15920.544 | 1492041.260 | 6.82x |
| CUDA Unified | 512 | 45842.321 | 15859.712 | 915004.883 | 9.69x |
| CUDA Unified | 1024 | 51228.078 | 15808.256 | 1077921.875 | 8.67x |
| CUDA Prefetch | 256 | 15946.253 | 15931.392 | 15964.832 | 27.87x |
| CUDA Prefetch | 512 | 15851.203 | 15835.136 | 15917.664 | 28.04x |
| CUDA Prefetch | 1024 | 15821.575 | 15804.416 | 15868.672 | 28.09x |
| CUDA Prefetch + Page Creation | 256 | 15965.106 | 15944.736 | 15987.200 | 27.84x |
| CUDA Prefetch + Page Creation | 512 | 15868.160 | 15856.384 | 15884.288 | 28.01x |
| CUDA Prefetch + Page Creation | 1024 | 15776.598 | 15763.424 | 15793.472 | 28.17x |
| CUDA Prefetch + Page Creation + Mem Advice | 256 | 15961.716 | 15929.568 | 15982.176 | 27.84x |
| CUDA Prefetch + Page Creation + Mem Advice | 512 | 15866.989 | 15855.552 | 15880.704 | 28.01x |
| CUDA Prefetch + Page Creation + Mem Advice | 1024 | 15831.164 | 15816.992 | 15843.520 | 28.07x |


### B. Analysis of Speed Performance Across All Platforms
| Kernel | Block Size | 2^20 Speedup | 2^26 Speedup | 2^28 Speedup |
|---|---:|---:|---:|---:|
| C (Baseline) | - | 1.00x | 1.00x | 1.00x |
| x86-64 Assembly | - | 1.58x | 2.05x | 2.01x |
| x86-64 SIMD XMM | - | 2.34x | 2.91x | 2.61x |
| x86-64 SIMD YMM | - | 3.66x | 3.33x | 2.96x |
| CUDA Unified | 256 | 4.04x | 9.19x | 6.82x |
| CUDA Unified | 512 | 6.95x | 10.15x | 9.69x |
| CUDA Unified | 1024 | 4.84x | 9.29x | 8.67x |
| CUDA Prefetch | 256 | 17.33x | 28.35x | 27.87x |
| CUDA Prefetch | 512 | 14.91x | 28.56x | 28.04x |
| CUDA Prefetch | 1024 | 8.21x | 27.91x | 28.09x |
| CUDA Prefetch + Page Creation | 256 | 13.55x | 28.55x | 27.84x |
| CUDA Prefetch + Page Creation | 512 | 8.17x | 28.80x | 28.01x |
| CUDA Prefetch + Page Creation | 1024 | 3.88x | 28.72x | 28.17x |
| CUDA Prefetch + Page Creation + Mem Advice | 256 | 13.25x | 28.55x | 27.84x |
| CUDA Prefetch + Page Creation + Mem Advice | 512 | 7.82x | 28.77x | 28.01x |
| CUDA Prefetch + Page Creation + Mem Advice | 1024 | 4.25x | 28.72x | 28.07x |

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

