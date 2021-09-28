#include <iostream>
#include <math.h>
#include <cassert>
#include <cstdlib>
#include <fstream>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusolverSp.h>
#include <cusparse.h>

void printMatrix(int m, int n, const double *x, int ldx, const char *name)
{
    for (int row=0; row<m; row++) {
        for (int col=0; col<n; col++) {
            double xreg = x[row + (col*ldx)];
            std::cout << name << "(" << row+1 << "," << col+1 << ") = " << xreg << std::endl;
        }
    }
}

int main(int argc, char** argv)
{
    std::ifstream finA("circuit_1/circuit_1.mtx");

    //Declare variable for matrix features
    int num_row, num_col, nnzA;

    //Ignore header
    while (finA.peek() == '%') {
        finA.ignore(2048, '\n');
    }
    
    //Read in the matrix features
    finA >> num_row >> num_col >> nnzA;

    //Create matrix in COO format
    double cooValA[nnzA];
    int cooRowIndA[nnzA];
    int cooColIndA[nnzA];

    //Read the data
    for (int i=0; i<nnzA; i++) {
        int a, b;
        double data;
        finA >> a >> b >> data;
        cooValA[i] = data;
        cooRowIndA[i] = a-1;
        cooColIndA[i] = b-1;
    }

    std::ifstream finb("circuit_1/circuit_1_b.mtx");

    //Ignore header
    while (finb.peek() == '%') {
        finb.ignore(2048, '\n');
    }

    //Read in the vector features
    finb >> num_row >> num_col;

    //Create vector b
    double Vecb[num_row];

    //Read the data
    for (int j=0; j<num_row; j++) {
        double data;
        finb >> data;
        Vecb[j] = data;
    }

    //Create vector x
    double Vecx[num_row];

    finA.close();
    finb.close();
    
    //////////////////////////////////////////////////////////////////////////////////////////
    
    cusolverSpHandle_t  cusolverH = NULL;
    cusparseHandle_t    cusparseH = NULL;
    cudaStream_t stream = NULL;

    cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    cudaError_t cudaStat5 = cudaSuccess;
    cudaError_t cudaStat6 = cudaSuccess;
    cudaError_t cudaStat7 = cudaSuccess;
    cudaError_t cudaStat8 = cudaSuccess;

    int P[nnzA];

    int *d_cooRowIndA           = NULL;
    int *d_cooColIndA           = NULL;
    int *d_csrRowPtrA           = NULL;
    int *d_P                    = NULL;
    double *d_cooValA           = NULL;
    double *d_cooValA_sorted    = NULL;
    double *d_Vecb              = NULL;
    double *d_Vecx              = NULL;
    size_t pBufferSizeInBytes   = 0;
    void *pBuffer               = NULL;

    //step 1 : create cuSolver and cuSparse handle
    cusolver_status = cusolverSpCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cusparse_status = cusparseCreate(&cusparseH);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);

    //step 2 : sort A in row order
    ////allocate buffer
    cusparse_status = cusparseXcoosort_bufferSizeExt(
                                                cusparseH,
                                                num_row,
                                                num_row,
                                                nnzA,
                                                d_cooRowIndA,
                                                d_cooColIndA,
                                                &pBufferSizeInBytes);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);

    ////copy data from host to device
    cudaStat1 = cudaMalloc(&d_cooRowIndA    , sizeof(int)*nnzA);
    cudaStat2 = cudaMalloc(&d_cooColIndA    , sizeof(int)*nnzA);
    cudaStat3 = cudaMalloc(&d_P             , sizeof(int)*nnzA);
    cudaStat4 = cudaMalloc(&d_cooValA       , sizeof(double)*nnzA);
    cudaStat5 = cudaMalloc(&d_cooValA_sorted, sizeof(double)*nnzA);
    cudaStat6 = cudaMalloc(&pBuffer         , sizeof(char)*pBufferSizeInBytes);
    cudaStat7 = cudaMalloc(&d_csrRowPtrA    , sizeof(int)*(num_row+1));
    cudaStat8 = cudaMalloc(&d_Vecb          , sizeof(double)*num_row);
    

    assert( cudaSuccess == cudaStat1 );
    assert( cudaSuccess == cudaStat2 );
    assert( cudaSuccess == cudaStat3 );
    assert( cudaSuccess == cudaStat4 );
    assert( cudaSuccess == cudaStat5 );
    assert( cudaSuccess == cudaStat6 );
    assert( cudaSuccess == cudaStat7 );
    assert( cudaSuccess == cudaStat8 );

    cudaStat1 = cudaMemcpy(d_cooRowIndA, cooRowIndA, sizeof(int)*nnzA   , cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_cooColIndA, cooColIndA, sizeof(int)*nnzA   , cudaMemcpyHostToDevice);
    cudaStat3 = cudaMemcpy(d_cooValA, cooValA, sizeof(double)*nnzA, cudaMemcpyHostToDevice);
    cudaStat4 = cudaMemcpy(d_Vecb, Vecb, sizeof(double)*num_row, cudaMemcpyHostToDevice);
    cudaStat5 = cudaDeviceSynchronize();
    assert( cudaSuccess == cudaStat1 );
    assert( cudaSuccess == cudaStat2 );
    assert( cudaSuccess == cudaStat3 );
    assert( cudaSuccess == cudaStat4 );
    assert( cudaSuccess == cudaStat5 );

    ////set value P
    cusparse_status = cusparseCreateIdentityPermutation(cusparseH, nnzA, d_P);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);
    
    ////sort A by row
    cusparse_status = cusparseXcoosortByRow(
                                        cusparseH,
                                        num_row,
                                        num_row,
                                        nnzA,
                                        d_cooRowIndA,
                                        d_cooColIndA,
                                        d_P,
                                        pBuffer);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);

    ////sort cooValA
    cusparse_status = cusparseDgthr(
                                        cusparseH,
                                        nnzA,
                                        d_cooValA,
                                        d_cooValA_sorted,
                                        d_P,
                                        CUSPARSE_INDEX_BASE_ZERO);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);
    assert(cudaSuccess == cudaStat1);

    //step 3 : change format of A from COO to CSR
    cusparse_status = cusparseXcoo2csr(
                                    cusparseH,
                                    d_cooRowIndA,
                                    nnzA,
                                    num_row,
                                    d_csrRowPtrA,
                                    CUSPARSE_INDEX_BASE_ZERO);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);

    //step 4 : LU factorisation and solving for x
    cusparseMatDescr_t MatADescr;
    cusparse_status = cusparseCreateMatDescr(&MatADescr);
    cudaStat1 = cudaMalloc(&d_Vecx, sizeof(double)*num_row);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);
    assert(cudaSuccess == cudaStat1);

    double tol  = 1.e-7;
    int *sing   = NULL;

    cusolver_status = cusolverSpDcsrlsvluHost(
                                        cusolverH,
                                        num_row,
                                        nnzA,
                                        MatADescr,
                                        d_cooValA_sorted,
                                        d_csrRowPtrA,
                                        d_cooColIndA,
                                        d_Vecb,
                                        tol,
                                        0,
                                        d_Vecx,
                                        sing);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);

    //step 4 : copy vector x from device to host
    cudaStat1 = cudaMemcpy(Vecx, d_Vecx, sizeof(double)*num_row, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    std::cout << "X = (matlab base-1)" << std::endl;
    printMatrix(num_row, 1, Vecx, num_row, "x");

    //Free resources
    if (d_cooColIndA)       cudaFree(d_cooColIndA);
    if (d_cooRowIndA)       cudaFree(d_cooRowIndA);
    if (d_cooValA)          cudaFree(d_cooValA);
    if (d_cooValA_sorted)   cudaFree(d_cooValA_sorted);
    if (d_csrRowPtrA)       cudaFree(d_csrRowPtrA);
    if (d_P)                cudaFree(d_P);
    if (pBuffer)            cudaFree(pBuffer);
    if (d_Vecb)             cudaFree(d_Vecb);
    if (d_Vecx)             cudaFree(d_Vecx);

    if (cusolverH)          cusolverSpDestroy(cusolverH);
    if (cusparseH)          cusparseDestroy(cusparseH);
    if (stream)             cudaStreamDestroy(stream);

    cudaDeviceReset();

    return 0;
}