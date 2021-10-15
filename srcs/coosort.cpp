#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <library_types.h>

int main(int argc, char*argv[])
{
    cusparseHandle_t handle = NULL;
    cudaStream_t stream = NULL;

    cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    cudaError_t cudaStat5 = cudaSuccess;
    cudaError_t cudaStat6 = cudaSuccess;

/*
 * A is a 3x3 sparse matrix 
 *     | 1 2 0 | 
 * A = | 0 5 0 | 
 *     | 0 8 0 | 
 */
    const int m = 3;
    const int n = 3;
    const int nnz = 4;


/* index starts at 0 */
    int h_cooRows[nnz] = {2, 1, 0, 0 };
    int h_cooCols[nnz] = {1, 1, 0, 1 }; 

    double h_cooVals[nnz] = {8.0, 5.0, 1.0, 2.0 };
    int h_P[nnz];

    int *d_cooRows = NULL;
    int *d_cooCols = NULL;
    int *d_P       = NULL;
    double *d_cooVals = NULL;
    double *d_cooVals_sorted = NULL;
    size_t pBufferSizeInBytes = 0;
    void *pBuffer = NULL;

    printf("m = %d, n = %d, nnz=%d \n", m, n, nnz );

/* step 1: create cusparse handle, bind a stream */
    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaSuccess == cudaStat1);

    status = cusparseCreate(&handle);
    assert(CUSPARSE_STATUS_SUCCESS == status);

    status = cusparseSetStream(handle, stream);
    assert(CUSPARSE_STATUS_SUCCESS == status);

/* step 2: allocate buffer */ 
    status = cusparseXcoosort_bufferSizeExt(
        handle,
        m,
        n,
        nnz,
        d_cooRows,
        d_cooCols,
        &pBufferSizeInBytes
    );
    assert( CUSPARSE_STATUS_SUCCESS == status);

    printf("pBufferSizeInBytes = %lld bytes \n", (long long)pBufferSizeInBytes);

    cudaStat1 = cudaMalloc( &d_cooRows, sizeof(int)*nnz);
    cudaStat2 = cudaMalloc( &d_cooCols, sizeof(int)*nnz);
    cudaStat3 = cudaMalloc( &d_P      , sizeof(int)*nnz);
    cudaStat4 = cudaMalloc( &d_cooVals, sizeof(double)*nnz);
    cudaStat5 = cudaMalloc( &d_cooVals_sorted, sizeof(double)*nnz);
    cudaStat6 = cudaMalloc( &pBuffer, sizeof(char)* pBufferSizeInBytes);

    assert( cudaSuccess == cudaStat1 );
    assert( cudaSuccess == cudaStat2 );
    assert( cudaSuccess == cudaStat3 );
    assert( cudaSuccess == cudaStat4 );
    assert( cudaSuccess == cudaStat5 );
    assert( cudaSuccess == cudaStat6 );

    cudaStat1 = cudaMemcpy(d_cooRows, h_cooRows, sizeof(int)*nnz   , cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_cooCols, h_cooCols, sizeof(int)*nnz   , cudaMemcpyHostToDevice);
    cudaStat3 = cudaMemcpy(d_cooVals, h_cooVals, sizeof(double)*nnz, cudaMemcpyHostToDevice);
    cudaStat4 = cudaDeviceSynchronize();
    assert( cudaSuccess == cudaStat1 );
    assert( cudaSuccess == cudaStat2 );
    assert( cudaSuccess == cudaStat3 );
    assert( cudaSuccess == cudaStat4 );

/* step 3: setup permutation vector P to identity */
    status = cusparseCreateIdentityPermutation(
        handle,
        nnz,
        d_P);
    assert( CUSPARSE_STATUS_SUCCESS == status);

/* step 4: sort COO format by Row */
    status = cusparseXcoosortByRow(
        handle, 
        m, 
        n, 
        nnz, 
        d_cooRows, 
        d_cooCols, 
        d_P, 
        pBuffer
    ); 
    assert( CUSPARSE_STATUS_SUCCESS == status);

/* step 5: gather sorted cooVals */
    /*cusparseDnVecDescr_t unsorted;
    status = cusparseCreateDnVec(&unsorted, nnz, d_cooVals, CUDA_R_64F);
    assert(CUSPARSE_STATUS_SUCCESS == status);

    cusparseSpVecDescr_t sorted;
    status = cusparseCreateSpVec(
                                &sorted, 
                                nnz, 
                                nnz, 
                                d_P, 
                                d_cooVals_sorted, 
                                CUSPARSE_INDEX_64I, 
                                CUSPARSE_INDEX_BASE_ZERO, 
                                CUDA_R_64F);
    assert(CUSPARSE_STATUS_SUCCESS == status);*/

    status = cusparseDgthr(
        handle, 
        nnz, 
        d_cooVals, 
        d_cooVals_sorted, 
        d_P, 
        CUSPARSE_INDEX_BASE_ZERO
    );

    //status = cusparseGather(handle, unsorted, sorted);
    assert( CUSPARSE_STATUS_SUCCESS == status);

    //status = cusparseSpVecGetValues(sorted, &d_cooVals_sorted);
    //assert( CUSPARSE_STATUS_SUCCESS == status);

    cudaStat1 = cudaDeviceSynchronize(); /* wait until the computation is done */
    cudaStat2 = cudaMemcpy(h_cooRows, d_cooRows, sizeof(int)*nnz   , cudaMemcpyDeviceToHost);
    cudaStat3 = cudaMemcpy(h_cooCols, d_cooCols, sizeof(int)*nnz   , cudaMemcpyDeviceToHost);
    cudaStat4 = cudaMemcpy(h_P,       d_P      , sizeof(int)*nnz   , cudaMemcpyDeviceToHost);
    cudaStat5 = cudaMemcpy(h_cooVals, d_cooVals_sorted, sizeof(double)*nnz, cudaMemcpyDeviceToHost);
    cudaStat6 = cudaDeviceSynchronize();
    assert( cudaSuccess == cudaStat1 );
    assert( cudaSuccess == cudaStat2 );
    assert( cudaSuccess == cudaStat3 );
    assert( cudaSuccess == cudaStat4 );
    assert( cudaSuccess == cudaStat5 );
    assert( cudaSuccess == cudaStat6 );

    printf("sorted coo: \n");
    for(int j = 0 ; j < nnz; j++){
        printf("(%d, %d, %f) \n", h_cooRows[j], h_cooCols[j], h_cooVals[j] );
    }

    for(int j = 0 ; j < nnz; j++){
        printf("P[%d] = %d \n", j, h_P[j] );
    }

/* free resources */
    if (d_cooRows     ) cudaFree(d_cooRows);
    if (d_cooCols     ) cudaFree(d_cooCols);
    if (d_P           ) cudaFree(d_P);
    if (d_cooVals     ) cudaFree(d_cooVals);
    if (d_cooVals_sorted ) cudaFree(d_cooVals_sorted);
    if (pBuffer       ) cudaFree(pBuffer);
    if (handle        ) cusparseDestroy(handle);
    if (stream        ) cudaStreamDestroy(stream);
    //if (sorted          ) cusparseDestroySpVec(sorted);
    //if (unsorted        ) cusparseDestroyDnVec(unsorted);
    cudaDeviceReset();
    return 0;
}