## 14/09/2021
Installed Armadillo library to handle matrix multiplication using C++. Further documentation can be found at <http://arma.sourceforge.net/docs.html>. To compile, use 
>$g++ example.cpp -o example -std=c++11 -O2 -larmadillo 

in the terminal. The example code was just a simple matrix multiplication. The plan for next time is to try solve a random Ax=b linear equation.

## 15/09/2021
Tried redoing the example given in the [documentation](http://arma.sourceforge.net/docs.html#spsolve) for sparse matrix solve. Then, I tried including circuit\_1 benchmark matrix from [SuiteSparse Collection](https://sparse.tamu.edu/). In order to implement that, I have to read the .mtx file into variables inside the code. The sparse matrix A has been successfully read with no error. The plan for next time is to try read vector b into the variable.

## 16/09/2021
Vector b has been successfully read with no error. Vector x also has been solved and checked with the reference vector of x given. It is roughly the same except for the accuracy. I tried implementing the sparse solver as a function to do the analysis on the timing. Apparently the profiling tool hasn't detected any bottlenecks on the timing. The profiling tool used is gprof. The plan for next time is to try the code with other benchmark matrices.

## 20/09/2021
I can't seem to get the profiling tool the work well with my codes. When implementing the code with circuit\_4 benchmark matrices, the program run significantly slower. I then proceed with using the syntax 
>$time ./\<name of program\> 

in the terminal. It shows real time which is the clock time, and the user time which is the CPU time. These information is enough for now. When running circuit\_4, there are some inaccuracy present in the computed answer of x, compared to the reference vector of x given. Minor changes have been applied to the code to improve the accuracy. Currently doing some reading on the documentation of cuBLAS, cuSOLVER, cuSPARSE and cuSPARSELt before developing a code for GPU. All of that can be found [here](https://docs.nvidia.com/cuda-libraries/index.html). The cuSOLVER [documentation](https://docs.nvidia.com/cuda/cusolver/index.html#lu_examples) shows a sample code that uses LU factorisation. Copied the code and adjust the original C code to C++. The plan for next time is to run the code with benchmark matrices and observe execution time improvements.

## 22/09/2021
Run the code from before. It works just fine. I just noticed that the matrix used in the example is dense, hence the code revolves around dense matrix operation. Further reading of cuSPARSE is needed to ensure implementation of the solver to sparse matrix to work. Updated this document to markdown format instead of a text file format, as suggested by my supervisor. Fixed something on the previous codes as well. Will need to continue reading cuSPARSE and cuSPARSELt documentations to further understand the implementation on CUDA.

## 27/09/2021
I have been reading the documentations for cuSOLVER and cuSPARSE, along with some code examples. I then develop a code to read in the .mtx files. The .mtx file that contains the sparse matrix A can be easily stored in COO storage format. Vector b on the other hand, is stored in a row vector format instead of a column vector. This might change after I know how cuSOLVER deals with a sparse matrix and a vector operation. I found that 
>cusolverSpDcsrlsvlu()

function accepts sparse matrix with CSR storage format. So, I first need to convert matrix A from COO to CSR storage format. The function
>cusparseXcoo2csr()

will do just that. But, it requires the input matrix to be sorted in row order. As of currently, the matrix A is sorted in column order. So, sorting is required before the conversion can take place. The function
>cusparseXcoosortByRow()

will make that happen. I have to check whether all the functions stated requires the parameters to be in host or device memory.

## 28/09/2021
I finished the code from the sorting and conversion part, to the solving part. The code seems fine as there are no errors present in it. Upon compiling, a warning message appear that stated the function 
>cusparseDgthr() 

is deprecated. I have to use 
>cusparseGather() 

instead. Unfortunately, it is not as straightforward as it may seems. The plan for next time is to solve this issue and compile the code to see if it works as it should.

## 03/10/2021
The `cusparseGather()` is a fairly new function and there is not many guides as to how to use them. And it turns out that the code can still be compiled despite the warning message from using `cusparseDgthr()`. First try at running the executable file, an error came out.
>Segmentation fault (core dumped)

My suspicion is it has something to do with the function `cusolverSpDcsrlsvluHost()`. My approach in solving the linear equation consist of  a few major steps; sorting matrix A in row order, converting the sorted matrix A to CSR storage format, and finally the LU factorisation. I plan to use a simple 3x3 matrix as A, and 3x1 matrix as b.
```
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
