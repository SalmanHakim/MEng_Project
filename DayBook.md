# DayBook

## 14/09/2021

Installed Armadillo library to handle matrix multiplication using C++. Further documentation can be found at <http://arma.sourceforge.net/docs.html>. To compile, use
>$g++ example.cpp -o example -std=c++11 -O2 -larmadillo

in the terminal. The example code was just a simple matrix multiplication. The plan for next time is to try solve a random Ax=b linear equation.

## 15/09/2021

Tried redoing the example given in the [documentation](http://arma.sourceforge.net/docs.html#spsolve) for sparse matrix solve. Then, I tried including circuit\_1 benchmark matrix from [SuiteSparse Collection](https://sparse.tamu.edu/). In order to implement that, I have to read the .mtx file into variables inside the code. The sparse matrix A has been successfully read with no error. The plan for next time is to try read vector b into the variable.

## 16/09/2021

Vector b has been successfully read with no error. Vector x also has been solved and checked with the reference vector x given. It is roughly the same except for the accuracy. I tried implementing the sparse solver as a function to do the analysis on the timing. Apparently the profiling tool hasn't detected any bottlenecks on the timing. The profiling tool used is gprof. The plan for next time is to try the code with other benchmark matrices.

## 20/09/2021

I can't seem to get the profiling tool the work well with my codes. When implementing the code with circuit\_4 benchmark matrices, the program run significantly slower. I then proceed with using the syntax
>$time ./\<name of program\>

in the terminal. It shows real time which is the clock time, and the user time which is the CPU time. These information is enough for now. When running circuit\_4, there are some inaccuracy present in the computed answer of x, compared to the reference vector of x given. Minor changes have been applied to the code to improve the accuracy. Currently doing some reading on the documentation of cuBLAS, cuSOLVER, cuSPARSE and cuSPARSELt before developing a code for GPU. All of that can be found [here](https://docs.nvidia.com/cuda-libraries/index.html). The cuSOLVER [documentation](https://docs.nvidia.com/cuda/cusolver/index.html#lu_examples) shows a sample code that uses LU factorisation. Copied the code and adjust the original C code to C++. The plan for next time is to run the code with benchmark matrices and observe execution time improvements.

## 22/09/2021

Run the code from before. It works just fine. I just noticed that the matrix used in the example is dense, hence the code revolves around dense matrix operation. Further reading of cuSPARSE is needed to ensure implementation of the solver to sparse matrix to work. Updated this document to markdown format instead of a text file format, as suggested by my supervisor. Fixed something on the previous codes as well. Will need to continue reading cuSPARSE and cuSPARSELt documentations to further understand the implementation on CUDA.

## 27/09/2021

I have been reading the documentations for cuSOLVER and cuSPARSE, along with some code examples. I then develop a code to read in the .mtx files. The .mtx file that contains the sparse matrix A can be easily stored in COO storage format. Vector b on the other hand, is stored in a row vector format instead of a column vector. This might change after I know how cuSOLVER deals with a sparse matrix and a vector operation. I found that `cusolverSpDcsrlsvlu()` function accepts sparse matrix with CSR storage format. So, I first need to convert matrix A from COO to CSR storage format. The function `cusparseXcoo2csr()` will do just that. But, it requires the input matrix to be sorted in row order. As of currently, the matrix A is sorted in column order. So, sorting is required before the conversion can take place. The function `cusparseXcoosortByRow()` will make that happen. I have to check whether all the functions stated requires the parameters to be in host or device memory.

## 28/09/2021

I finished the code from the sorting and conversion part, to the solving part. The code seems fine as there are no errors present in it. Upon compiling, a warning message appear that stated the function `cusparseDgthr()` is deprecated. I have to use `cusparseGather()` instead. Unfortunately, it is not as straightforward as it may seems. The plan for next time is to solve this issue and compile the code to see if it works as it should.

## 03/10/2021

The `cusparseGather()` is a fairly new function and there is not many guides as to how to use them. And it turns out that the code can still be compiled despite the warning message from using `cusparseDgthr()`. First try at running the executable file, an error came out.
>Segmentation fault (core dumped)

My suspicion is it has something to do with the function `cusolverSpDcsrlsvluHost()`. My approach in solving the linear equation consist of  a few major steps; sorting matrix A in row order, converting the sorted matrix A to CSR storage format, and finally the LU factorisation. I plan to use a simple 3x3 matrix as A, and 3x1 matrix as b.

```txt
    | 1 2 0 |               | 3 |
A = | 0 0 8 |           b = | 5 |
    | 0 5 0 |               | 1 |

This will give vector x,

    |  2.6  |
x = |  0.2  |
    | 0.625 |
```

I plan to print matrix A after sorting and after conversion to CSR to look for any errors that may be present there, and after that proceed with print vector x to see if the LU factorisation works fine or not.

## 04/10/2021

Set the code to print the row, column and value of matrix A after the sorting. It shows that matrix A is well sorted as it should, and that proves that the function `cusparseDgthr()` works just fine in this approach. Then, I included the `cusparseXcoo2csr()` function to convert the matrix storage format. It printed out a new row array in CSR format, which is expected. The value and column arrays remains unchanged. The last function to be tested is `cusolverSpDcsrlsvluHost()`. I do not have to print matrix A anymore. In this test, I just need to print out vector x. The function requires a matrix descriptor for matrix A, which is done easily. But, there is a conflict on whether to use the host or device version of some parameters. This affects the sequence of the code; the memory copy from device to host part either being before or after the function. I tried using the device parameters, and the same error came out. I shifted the copy memory from device to host part to be above the function, and changed all the parameters used in the function as host parameters. It finally showed some results. It printed the values of vector x, and I have crosschecked it with the reference vector x. The plan for next time is to set a test to compare the timing between GPU and CPU computation time of benchmark matrices and the accuracy of the computed values.

## 11/10/2021

I have been trying to use NVIDIA Nsight Systems to profile the code and see the timing and whether the functions actually running on the DEVICE and not the HOST. But, the same error keep coming up. The
>Segmentation fault (core dumped)

came up even with the code I previously did an analysis on successfully using the same tool. Have not figured out the reason yet.

## 12/10/2021

I decided to try the code with more benchmark matrices from the [SuiteSparse Collection](https://sparse.tamu.edu/) to see if it would work. My initial assumption was it will work no problem. But the `Segmentation fault` error came up when running scircuit and hcircuit benchmark matrices. I suspect it has something to do with the memory allocations in HOST. So, I changed the memory allocation of matrix A, vector b and vector x to dynamic using `malloc()` and `calloc()` functions. The error is no longer there, but the program took too long to execute. I have to terminate the program midway because it just won't show anything. The plan for next time is to print out a string that shows the code passes every steps.

## 13/10/2021

After printing out the status after each steps, I was able to identify the step that was preventing the code from finishing its execution. It was the final step that uses `cusolverSpDcsrlsvluHost()` function. Tried reading the [documentation](https://docs.nvidia.com/cuda/cusolver/index.html#cusolver-lt-t-gt-csrlsvlu) again and decided to try to use one of the reordering schemes, which can significantly affects the performance of the LU solver. After doing a bit of reading, I decided to use the `symamd` which implements Symmetric Approximate Minimum Degree Algorithm based on Quotient Graph. It finally works with bcircuit, hcircuit and scircuit benchmark matrices.

## 20/10/2021

I have run a few benchmark matrices on both CUDA (GPU) and Armadillo (CPU), and the results are quite surprising. The only speed up that I observed is when I run circuit_4 benchmark matrix (11x). Other than that, the CPU outperforms the GPU. I suspect that the `cusolverSpDcsrlsvluHost()` function actually run on HOST instead of DEVICE. Apparently NVIDIA have not released a DEVICE version of the function. My option now is to either use QR factorisation, which can be run on DEVICE, or write an LU factorisation function from scratch. I cannot confirm this as the `nsys profile` tool has not been working for me at all. I would not want to go to QR factorisation due to performance reduction compared to LU factorisation. But, it would be nice to observe any difference when running the code.

After changing and testing, linear solver using QR factorisation is sligtly slower than that using LU factorisation and therefore is not suitable in this project.

## 21/10/2021

I finally figured out the CLI profiling using Nsight Systems. The problem is there due to OpenGL tracing not working. So, the right command is
>$nsys profile --trace=cuda,osrt --sample=cpu -o \<filename\> --stats=true ./\<nameofprogram\>

This will create a qdrep file with the name \<filename\> in the working directory. With `--stats=true`, it will generate a summary of the analysis inside the terminal. The analysis gave a timing of each functions called by the code. I did not see anything from the analysis that says a cusolver function is used, indicating that the `cusolverSpDcsrlsvluHost()` function is done on HOST. Will reconfirm this later with more information from the analysis.

## 03/11/2021

I ran the benchmark matrices on two codes--sparse_solver.cpp and spsolver.cpp--that contains Armadillo library and CUDA library. The speedup is calculated by dividing the execution time on CUDA with the execution time on Armadillo. The Armadillo function that I use is the `spsolve()`, while the CUDA function is the `cusolverSpDcsrlsvluHost()`.

| Benchmark | Dimension, n | Non-zero elements, nnz | Execution time / s (Armadillo) | Execution time / s (CUDA) | Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: |
| add20 | 2395 | 17319 | 0.01059 | 0.60033 | 0.01764x |
| add32 | 4960 | 23884 | 0.00839 | 0.60959 | 0.01376x |
| circuit_1 | 2624 | 35823 | 0.13950 | 0.65754 | 0.21216x |
| circuit_4 | 80209 | 307604 | 56.29252 | 5.05437 | 11.13739x |
| bcircuit | 68902 | 375558 |0.20016 | 4.30923 | 0.04645x |
| hcircuit | 105676 | 513072 | 0.72707 | 8.54220 | 0.08511x |
| scircuit | 170998 | 958936 | 0.94049 | 22.33581 | 0.04211x |

## 09/11/2021

From the table above, it can be observed that only with `circuit_4` does GPU shows an improvement over CPU. I went and checked the sparsity pattern of matrix A of each benchmarks using MATLAB.

![add20](/sparsity/add20.png)
![add32](/sparsity/add32.png)
![circuit\_1](/sparsity/circuit_1.png)
![circuit\_4](/sparsity/circuit_4.png)
![hcircuit](/sparsity/hcircuit.png)
![scircuit](/sparsity/scircuit.png)

From the figures above, the only significant difference that `circuit_4` has compared to the others is its non-zero elements are concentrated on the diagonal, bottom, right-hand side of the matrix, while the others are a bit more spread out.

## 12/11/2021

I tried using the routines in cuSPARSE library. It has a sparse matrix-vector solver which will give a solution for x in Ax=b equation. The LU factorisation prior to that has to be done manually. The benefit of using this routine compared to `cusolverSpDcsrlsvluHost()` is it is done in the DEVICE instead of HOST. After writing the code, I tried running the circuit\_1 benchmark but I have to terminate the program before it finishes because it took a long time. It happens to circuit\_4, hcircuit and scircuit as well. The only benchmark matrices that managed to finish are add20, add32 and bcircuit. The results are shown below.

| Benchmark | Executuion time / s (Armadillo) | Execution time / s (cusolver) | Execution time / s (cusparse) |
| :--- | :---: | :---: | :---: |
| add20 | 0.01059 | 0.60033 | 0.21831 |
| add32 | 0.00839 | 0.60959 | 0.21339 |
| bcircuit | 0.20016 | 4.30923 | 0.23815 |

From the results, it can be observed that the cusparse routine showed significant improvement over cusolver. The reason for the cusparse routine not working with the other benchmark matrices is still unknown. They all have one thing in common, which is the function `cusparseXcsrilu02_zeroPivot()` during the analysis gave out an error. Might have to check out what does this function really do and how can I avoid this problem.

## 16/11/2021

After further read into incomplete LU factorisation (ILU), it is an iterative method and it is an approximation of LU factorisation so it does not return the exact solution for x in Ax=b. ILU is often use as a preconditioner for another iterative method such as Bi-Conjugate Gradient Stabilised (BiCGStab). So, even though it is fast, the ILU is far from accurate. The figures below show the solutions to `add32` benchmark matrix using ILU (cusparse) and LU (cusolver).

LU:
![LU](/sparsity/LU.png)

ILU:
![ILU](/sparsity/ILU.png)

Due to the solution being so small, the ILU assumes it to be 0. This happened to `add20` and `bcircuit` as well. This shows that the ILU iterative method is not suitable to be used to find the final solution of Ax=b, and is not acceptable in simulation.
