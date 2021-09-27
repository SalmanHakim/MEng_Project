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