#include <iostream>
#include <algorithm>
#include <math.h>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <chrono>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusolverSp.h>
#include <cusparse.h>

void printMatrix(int m, int n, double *x, int ldx, const char *name)
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
    //Check the number of parameters
    if (argc < 3) {
        //Show the correct syntax to user
        std::cerr << "The correct syntax: " << argv[0] << " MatrixA VectorB" << std::endl;
        return 1;
    }

    //Open file
    std::ifstream finA(argv[1]);      //file name should be written in the command line

    //Declare variable for matrix features
    int num_row, num_col, nnzA;

    //Ignore header
    while (finA.peek() == '%') {
        finA.ignore(2048, '\n');
    }
    
    //Read in the matrix features
    finA >> num_row >> num_col >> nnzA;

    //Create matrix in COO format
    double *cooValA = (double *)malloc(sizeof(double)*nnzA);
    int *cooRowIndA = (int *)malloc(sizeof(int)*nnzA);
    int *cooColIndA = (int *)malloc(sizeof(int)*nnzA);
    //double cooValA[nnzA];
    //int cooRowIndA[nnzA];
    //int cooColIndA[nnzA];

    int *csrRowPtrA = (int *)calloc(1, sizeof(int)*(num_row+1));
    //int csrRowPtrA[num_row+1];

    //Read the data
    for (int i=0; i<nnzA; i++) {
        int a, b;
        double data;
        finA >> a >> b >> data;
        cooValA[i] = data;
        cooRowIndA[i] = a-1;
        cooColIndA[i] = b-1;
    }

    //Open file
    std::ifstream finb(argv[2]);      //file name should be written in the command line

    //Ignore header
    while (finb.peek() == '%') {
        finb.ignore(2048, '\n');
    }

    //Read in the vector features
    finb >> num_row >> num_col;

    //Create vector b
    double *Vecb = (double *)malloc(sizeof(double)*num_row);
    //double Vecb[num_row];

    //Read the data
    for (int j=0; j<num_row; j++) {
        double data;
        finb >> data;
        Vecb[j] = data;
    }

    //Create vector x
    double *Vecx = (double *)calloc(1, sizeof(double)*num_row);
    //double Vecx[num_row];

    finA.close();
    finb.close();
    std::cout << "Data read successfully" << std::endl;

    /*int num_row = 3;
    int num_col = 3;
    int nnzA    = 4;

    int cooRowIndA[nnzA]    = {1,2,0,0};
    int cooColIndA[nnzA]    = {2,1,0,1};
    int csrRowPtrA[num_row+1];

    double cooValA[nnzA]    = {8.0,5.0,1.0,2.0};
    double Vecb[num_row]    = {3.0,5.0,1.0};
    //double *Vecx = (double *)malloc(sizeof(double)*num_row);       //{2.6, 0.2, 0.625}
    double Vecx[num_row];*/
    
    //////////////////////////////////////////////////////////////////////////////////////////
    //Start measuring time
    auto begin = std::chrono::high_resolution_clock::now();

    cusparseHandle_t    cusparseH = NULL;
    cudaStream_t stream = NULL;

    cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    cudaError_t cudaStat5 = cudaSuccess;
    cudaError_t cudaStat6 = cudaSuccess;
    cudaError_t cudaStat7 = cudaSuccess;
    cudaError_t cudaStat8 = cudaSuccess;
    cudaError_t cudaStat9 = cudaSuccess;

    //int P[nnzA];

    int *d_cooRowIndA           = NULL;
    int *d_cooColIndA           = NULL;
    int *d_csrRowPtrA           = NULL;
    int *d_P                    = NULL;
    double *d_cooValA           = NULL;
    double *d_cooValA_sorted    = NULL;
    double *d_Vecb              = NULL;
    double *d_Vecx              = NULL;
    double *d_Vecy              = NULL;
    size_t pBufferSizeInBytes   = 0;
    void *pBuffer               = NULL;

    cusparseMatDescr_t descr_A = 0;
    cusparseMatDescr_t descr_L = 0;
    cusparseMatDescr_t descr_U = 0;
    csrilu02Info_t info_A  = 0;
    csrsv2Info_t  info_L  = 0;
    csrsv2Info_t  info_U  = 0;
    int pBufferSize_A;
    int pBufferSize_L;
    int pBufferSize_U;
    int pBufferSize;
    void *pBuffer_ALU = 0;
    int structural_zero;
    int numerical_zero;
    const double alpha = 1.;
    const cusparseSolvePolicy_t policy_A = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L  = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseOperation_t trans_U  = CUSPARSE_OPERATION_NON_TRANSPOSE;

    ////copy data from host to device
    cudaStat1 = cudaMalloc(&d_cooRowIndA    , sizeof(int)*nnzA);
    cudaStat2 = cudaMalloc(&d_cooColIndA    , sizeof(int)*nnzA);
    cudaStat3 = cudaMalloc(&d_P             , sizeof(int)*nnzA);
    cudaStat4 = cudaMalloc(&d_cooValA       , sizeof(double)*nnzA);
    cudaStat5 = cudaMalloc(&d_cooValA_sorted, sizeof(double)*nnzA);
    cudaStat6 = cudaMalloc(&d_Vecy          , sizeof(double)*num_row);
    cudaStat7 = cudaMalloc(&d_csrRowPtrA    , sizeof(int)*(num_row+1));
    cudaStat8 = cudaMalloc(&d_Vecb          , sizeof(double)*num_row);
    cudaStat9 = cudaMalloc(&d_Vecx          , sizeof(double)*num_row);

    assert( cudaSuccess == cudaStat1 );
    assert( cudaSuccess == cudaStat2 );
    assert( cudaSuccess == cudaStat3 );
    assert( cudaSuccess == cudaStat4 );
    assert( cudaSuccess == cudaStat5 );
    assert( cudaSuccess == cudaStat6 );
    assert( cudaSuccess == cudaStat7 );
    assert( cudaSuccess == cudaStat8 );
    assert( cudaSuccess == cudaStat9 );

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

    //step 1 : create cuSparse handle, stream
    cusparse_status = cusparseCreate(&cusparseH);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);

    cudaStat1 = cudaStreamCreate(&stream);
    assert(cudaSuccess == cudaStat1);

    cusparse_status = cusparseSetStream(cusparseH, stream);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);

    //step 2 : create matrix descr for a A, L and U
    cusparse_status = cusparseCreateMatDescr(&descr_A);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);
    cusparse_status = cusparseSetMatIndexBase(descr_A, CUSPARSE_INDEX_BASE_ZERO);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);
    cusparse_status = cusparseSetMatType(descr_A, CUSPARSE_MATRIX_TYPE_GENERAL);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);

    cusparse_status = cusparseCreateMatDescr(&descr_L);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);
    cusparse_status = cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);
    cusparse_status = cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);
    cusparse_status = cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);
    cusparse_status = cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);

    cusparse_status = cusparseCreateMatDescr(&descr_U);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);
    cusparse_status = cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);
    cusparse_status = cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);
    cusparse_status = cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);
    cusparse_status = cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);

    //step 3 : create info
    cusparse_status = cusparseCreateCsrilu02Info(&info_A);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);
    cusparse_status = cusparseCreateCsrsv2Info(&info_L);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);
    cusparse_status = cusparseCreateCsrsv2Info(&info_U);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);

    //step 4 : sorting
    cusparse_status = cusparseXcoosort_bufferSizeExt(
                                                cusparseH,
                                                num_row,
                                                num_row,
                                                nnzA,
                                                d_cooRowIndA,
                                                d_cooColIndA,
                                                &pBufferSizeInBytes);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);

    cudaStat1 = cudaMalloc(&pBuffer         , sizeof(char)*pBufferSizeInBytes);
    assert( cudaSuccess == cudaStat1 );

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

    //step 5 : change format of A from COO to CSR
    cusparse_status = cusparseXcoo2csr(
                                    cusparseH,
                                    d_cooRowIndA,
                                    nnzA,
                                    num_row,
                                    d_csrRowPtrA,
                                    CUSPARSE_INDEX_BASE_ZERO);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);

    std::cout << "Conversion to CSR is successful" << std::endl;
    

    //step 6 : allocate buffer for LU
    cusparse_status = cusparseDcsrilu02_bufferSize(
                                                cusparseH, 
                                                num_row, 
                                                nnzA,
                                                descr_A, 
                                                d_cooValA_sorted, 
                                                d_csrRowPtrA, 
                                                d_cooColIndA, 
                                                info_A, 
                                                &pBufferSize_A);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);

    cusparse_status = cusparseDcsrsv2_bufferSize(
                                                cusparseH, 
                                                trans_L, 
                                                num_row, 
                                                nnzA,
                                                descr_L, 
                                                d_cooValA_sorted, 
                                                d_csrRowPtrA, 
                                                d_cooColIndA, 
                                                info_L, 
                                                &pBufferSize_L);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);

    cusparse_status = cusparseDcsrsv2_bufferSize(
                                                cusparseH, 
                                                trans_U, 
                                                num_row, 
                                                nnzA,
                                                descr_U, 
                                                d_cooValA_sorted, 
                                                d_csrRowPtrA, 
                                                d_cooColIndA, 
                                                info_U, 
                                                &pBufferSize_U);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);

    pBufferSize = std::max(pBufferSize_A, std::max(pBufferSize_L, pBufferSize_U));

    cudaStat1 = cudaMalloc((void**)&pBuffer_ALU, pBufferSize);
    assert( cudaSuccess == cudaStat1 );

    std::cout << "Allocation of buffer is successful" << std::endl;

    //step 7 : perform analysis on A, L and U
    cusparse_status = cusparseDcsrilu02_analysis(
                                                cusparseH, 
                                                num_row, 
                                                nnzA, 
                                                descr_A,
                                                d_cooValA_sorted, 
                                                d_csrRowPtrA, 
                                                d_cooColIndA, 
                                                info_A,
                                                policy_A, 
                                                pBuffer_ALU);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);

    cusparse_status = cusparseXcsrilu02_zeroPivot(cusparseH, info_A, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == cusparse_status){
        printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
    }

    cusparse_status = cusparseDcsrsv2_analysis(
                                                cusparseH, 
                                                trans_L, 
                                                num_row, 
                                                nnzA, 
                                                descr_L,
                                                d_cooValA_sorted, 
                                                d_csrRowPtrA, 
                                                d_cooColIndA,
                                                info_L, 
                                                policy_L, 
                                                pBuffer_ALU);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);

    cusparse_status = cusparseDcsrsv2_analysis(
                                                cusparseH, 
                                                trans_U, 
                                                num_row, 
                                                nnzA, 
                                                descr_U,
                                                d_cooValA_sorted, 
                                                d_csrRowPtrA, 
                                                d_cooColIndA,
                                                info_U, 
                                                policy_U, 
                                                pBuffer_ALU);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);

    std::cout << "Analysis on A, L and U are successful" << std::endl;

    //step 8 : A = L * U
    cusparse_status = cusparseDcsrilu02(
                                                cusparseH, 
                                                num_row, 
                                                nnzA, 
                                                descr_A,
                                                d_cooValA_sorted, 
                                                d_csrRowPtrA, 
                                                d_cooColIndA, 
                                                info_A, 
                                                policy_A, 
                                                pBuffer_ALU);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);
    
    cusparse_status = cusparseXcsrilu02_zeroPivot(cusparseH, info_A, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == cusparse_status){
        printf("U(%d,%d) is zero\n", numerical_zero, numerical_zero);
    }

    //step 9 : L * y = b
    cusparse_status = cusparseDcsrsv2_solve(
                                                cusparseH, 
                                                trans_L, 
                                                num_row, 
                                                nnzA, 
                                                &alpha, 
                                                descr_L,
                                                d_cooValA_sorted, 
                                                d_csrRowPtrA, 
                                                d_cooColIndA, 
                                                info_L,
                                                d_Vecb, 
                                                d_Vecy, 
                                                policy_L, 
                                                pBuffer_ALU);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);

    //step 10 : U * x = y
    cusparse_status = cusparseDcsrsv2_solve(
                                                cusparseH, 
                                                trans_U, 
                                                num_row, 
                                                nnzA, 
                                                &alpha, 
                                                descr_U,
                                                d_cooValA_sorted, 
                                                d_csrRowPtrA, 
                                                d_cooColIndA, 
                                                info_U,
                                                d_Vecy, 
                                                d_Vecx, 
                                                policy_U, 
                                                pBuffer_ALU);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);

    //Stop measuring time and calculate elapsed time
    auto end        = std::chrono::high_resolution_clock::now();
    auto elapsed    = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    std::cout << "Vector X is solved successfully" << std::endl;
    std::cout << std::endl;

    std::cout << "Time measured (seconds) : " << elapsed.count() * 1e-9 << std::endl;

    //Free resources
    if (d_cooColIndA)       cudaFree(d_cooColIndA);
    if (d_cooRowIndA)       cudaFree(d_cooRowIndA);
    if (d_cooValA)          cudaFree(d_cooValA);
    if (d_cooValA_sorted)   cudaFree(d_cooValA_sorted);
    if (d_csrRowPtrA)       cudaFree(d_csrRowPtrA);
    if (d_P)                cudaFree(d_P);
    if (pBuffer)            cudaFree(pBuffer);
    if (pBuffer_ALU)        cudaFree(pBuffer_ALU);
    if (d_Vecb)             cudaFree(d_Vecb);
    if (d_Vecx)             cudaFree(d_Vecx);
    if (d_Vecy)             cudaFree(d_Vecy);

    if (cusparseH)          cusparseDestroy(cusparseH);
    if (stream)             cudaStreamDestroy(stream);
    if (descr_A)            cusparseDestroyMatDescr(descr_A);
    if (descr_L)            cusparseDestroyMatDescr(descr_L);
    if (descr_U)            cusparseDestroyMatDescr(descr_U);

    if (info_A)             cusparseDestroyCsrilu02Info(info_A);
    if (info_L)             cusparseDestroyCsrsv2Info(info_L);
    if (info_U)             cusparseDestroyCsrsv2Info(info_U);

    cudaDeviceReset();

    free(cooValA);
    free(cooColIndA);
    free(cooRowIndA);
    free(csrRowPtrA);
    free(Vecb);
    free(Vecx);

    return 0;
}