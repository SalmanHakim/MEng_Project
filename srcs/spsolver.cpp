#include <iostream>
#include <math.h>
#include <cassert>
#include <cstdlib>
#include <fstream>

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

    //int P[nnzA];

    int *d_cooRowIndA           = NULL;
    int *d_cooColIndA           = NULL;
    int *d_csrRowPtrA           = NULL;
    int *d_P                    = NULL;
    double *d_cooValA           = NULL;
    double *d_cooValA_sorted    = NULL;
    double *d_Vecb              = NULL;
    //double *d_Vecx              = NULL;
    size_t pBufferSizeInBytes   = 0;
    void *pBuffer               = NULL;

    //step 1 : create cuSolver and cuSparse handle, stream
    cusolver_status = cusolverSpCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cusparse_status = cusparseCreate(&cusparseH);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);

    cudaStat1 = cudaStreamCreate(&stream);
    assert(cudaSuccess == cudaStat1);

    cusparse_status = cusparseSetStream(cusparseH, stream);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);

    cusolver_status = cusolverSpSetStream(cusolverH, stream);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    std::cout << "Handles and stream created successfully" << std::endl;

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

    std::cout << "Data copied from HOST to DEVICE successfully" << std::endl;

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

    std::cout << "Matrix A sorted successfully" << std::endl;

    //step 3 : change format of A from COO to CSR
    cusparse_status = cusparseXcoo2csr(
                                    cusparseH,
                                    d_cooRowIndA,
                                    nnzA,
                                    num_row,
                                    d_csrRowPtrA,
                                    CUSPARSE_INDEX_BASE_ZERO);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);

    std::cout << "Storage format converted successfully" << std::endl;

    //step 4 : copy parameters from device to host
    cusparseMatDescr_t MatADescr;
    cusparse_status = cusparseCreateMatDescr(&MatADescr);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);
    cusparse_status = cusparseSetMatDiagType(MatADescr, CUSPARSE_DIAG_TYPE_NON_UNIT);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);
    
    //cudaStat1 = cudaMalloc(&d_Vecx, sizeof(double)*num_row);
    //assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaDeviceSynchronize();  //wait until the computation is done
    //cudaStat2 = cudaMemcpy(Vecx, d_Vecx, sizeof(double)*num_row, cudaMemcpyDeviceToHost);
    cudaStat3 = cudaMemcpy(cooRowIndA, d_cooRowIndA, sizeof(int)*nnzA   , cudaMemcpyDeviceToHost);
    cudaStat4 = cudaMemcpy(cooColIndA, d_cooColIndA, sizeof(int)*nnzA   , cudaMemcpyDeviceToHost);
    cudaStat5 = cudaMemcpy(csrRowPtrA, d_csrRowPtrA, sizeof(int)*(num_row+1)   , cudaMemcpyDeviceToHost);
    //cudaStat6 = cudaMemcpy(P,       d_P      , sizeof(int)*nnzA   , cudaMemcpyDeviceToHost);
    cudaStat7 = cudaMemcpy(cooValA, d_cooValA_sorted, sizeof(double)*nnzA, cudaMemcpyDeviceToHost);
    cudaStat8 = cudaDeviceSynchronize();
    assert( cudaSuccess == cudaStat1 );
    assert( cudaSuccess == cudaStat2 );
    assert( cudaSuccess == cudaStat3 );
    assert( cudaSuccess == cudaStat4 );
    assert( cudaSuccess == cudaStat5 );
    assert( cudaSuccess == cudaStat6 );
    assert( cudaSuccess == cudaStat7 );
    assert( cudaSuccess == cudaStat8 );

    std::cout << "Data copied from DEVICE to HOST successfully" << std::endl;

    //step 5 : LU factorisation and solving for x
    double tol  = 1.e-7;
    int sing   = 0;

    cusolver_status = cusolverSpDcsrlsvluHost(
                                        cusolverH,
                                        num_row,
                                        nnzA,
                                        MatADescr,
                                        cooValA,
                                        csrRowPtrA,
                                        cooColIndA,
                                        Vecb,
                                        tol,
                                        2,
                                        Vecx,
                                        &sing);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);

    std::cout << "Vector X is solved successfully" << std::endl;


    /*printf("sorted csr: \n");
    printf("Row = {");
    for(int j = 0 ; j < num_row+1; j++){
        printf("%d, ", csrRowPtrA[j]);
    };
    printf("}\n");

    printf("Col = {");
    for(int j = 0 ; j < nnzA; j++){
        printf("%d, ", cooColIndA[j]);
    };
    printf("}\n");

    printf("Val = {");
    for(int j = 0 ; j < nnzA; j++){
        printf("%f, ", cooValA[j]);
    };
    printf("}\n");

    for(int j = 0 ; j < nnzA; j++){
        printf("P[%d] = %d \n", j, P[j] );
    };*/

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
    //if (d_Vecx)             cudaFree(d_Vecx);

    if (cusolverH)          cusolverSpDestroy(cusolverH);
    if (cusparseH)          cusparseDestroy(cusparseH);
    if (stream)             cudaStreamDestroy(stream);
    if (MatADescr)          cusparseDestroyMatDescr(MatADescr);

    cudaDeviceReset();

    free(cooValA);
    free(cooColIndA);
    free(cooRowIndA);
    free(csrRowPtrA);
    free(Vecb);
    free(Vecx);

    return 0;
}