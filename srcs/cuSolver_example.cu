#include <iostream>
#include <math.h>
#include <cassert>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cusolverDn.h>

void printMatrix(int m, int n, const double *A, int lda, const char *name)
{
    for (int row=0; row<m; row++) {
        for (int col=0; col<n; col++) {
            double Areg = A[row + (col*lda)];
            std::cout << name << "(" << row+1 << "," << col+1 << ") = " << Areg << std::endl;
        }
    }
}

int main(int argc, char **argv)
{
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;

    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;

    const int m = 3;
    const int lda = m;
    const int ldb = m;

    double A[lda*m] = {1.0 , 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0};
    double B[m] = {1.0, 2.0, 3.0};
    double X[m];
    double LU[lda*m];

    int Ipiv[m];
    int info = 0;

    double *d_A = NULL;
    double *d_B = NULL;
    int *d_Ipiv = NULL;
    int *d_info = NULL;
    int lwork = 0;
    double *d_work = NULL;

    const int pivot_on = false;

    std::cout << "example of getrf" << std::endl;

    if (pivot_on) {
        std::cout << "pivot in on : compute P*A = L*U" << std::endl;
    }

    else {
        std::cout <<  "pivot is off : compute A = L*U" << std::endl;
    }

    std::cout << "A = (matlab base-1)" << std::endl;
    printMatrix(m, m, A, lda, "A");
    std::cout << "======================================================" << std::endl;

    std::cout << "B = (matlab base-1)" << std::endl;
    printMatrix(m, 1, B, ldb, "B");
    std::cout << "======================================================" << std::endl;

    //step 1 : create cuSolver handle, bind a stream
    status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaSuccess == cudaStat1);

    status = cusolverDnSetStream(cusolverH, stream);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    //step 2 : copy A to device
    cudaStat1 = cudaMalloc ((void**)&d_A, sizeof(double) * lda * m);
    cudaStat2 = cudaMalloc ((void**)&d_B, sizeof(double) * m);
    cudaStat3 = cudaMalloc ((void**)&d_Ipiv, sizeof(int) * m);
    cudaStat4 = cudaMalloc ((void**)&d_info, sizeof(int));
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);

    cudaStat1 = cudaMemcpy(d_A, A, sizeof(double)*lda*m, cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_B, B, sizeof(double)*m, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

    //step 3 : query working space of getrf
    status = cusolverDnDgetrf_bufferSize(cusolverH, m, m, d_A, lda, &lwork);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
    assert(cudaSuccess == cudaStat1);

    //step 4 : LU factorisation
    if (pivot_on) {
        status = cusolverDnDgetrf(cusolverH, m, m, d_A, lda, d_work, d_Ipiv, d_info);
    }

    else {
        status = cusolverDnDgetrf(cusolverH, m, m, d_A, lda, d_work, NULL, d_info);
    }
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == status);
    assert(cudaSuccess == cudaStat1);

    if (pivot_on) {
        cudaStat1 = cudaMemcpy(Ipiv, d_Ipiv, sizeof(int)*m, cudaMemcpyDeviceToHost);
    }
    cudaStat2 = cudaMemcpy(LU, d_A, sizeof(double)*lda*m, cudaMemcpyDeviceToHost);
    cudaStat3 = cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);

    if (0 > info) {
        std::cout << -info << "-th parameter is wrong" << std::endl;
        exit(1);
    }
    if (pivot_on) {
        std::cout << "pivoting sequence, matlab base-1" << std::endl;
        for (int j=0; j<m; j++) {
            std::cout << "Ipiv(" << j+1 << ") = " << Ipiv[j] << std::endl;
        }
    }
    std::cout << "L and U = (matlab base-1)" << std::endl;
    printMatrix(m, m, LU, lda, "LU");
    std::cout << "======================================================" << std::endl;

    //step 5 : solve A*X=B
    if (pivot_on) {
        status = cusolverDnDgetrs(
                            cusolverH, 
                            CUBLAS_OP_N, 
                            m,
                            1,
                            d_A,
                            lda,
                            d_Ipiv,
                            d_B,
                            ldb,
                            d_info);
    }else {
        status = cusolverDnDgetrs(
                            cusolverH,
                            CUBLAS_OP_N,
                            m,
                            1,
                            d_A,
                            lda,
                            NULL,
                            d_B,
                            ldb,
                            d_info);
    }
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == status);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(X, d_B, sizeof(double)*m, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    std::cout << "X = (matlab base-1)" << std::endl;
    printMatrix(m, 1, X, ldb, "X");
    std::cout << "======================================================" << std::endl;

    //free resources
    if (d_A     ) cudaFree(d_A);
    if (d_B     ) cudaFree(d_B);
    if (d_Ipiv  ) cudaFree(d_Ipiv);
    if (d_info  ) cudaFree(d_info);
    if (d_work  ) cudaFree(d_work);

    if (cusolverH   ) cusolverDnDestroy(cusolverH);
    if (stream      ) cudaStreamDestroy(stream);

    cudaDeviceReset();

    return 0;
}