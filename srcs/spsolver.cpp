#include <iostream>
#include <math.h>
#include <cassert>
#include <cstdlib>
#include <fstream>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusolverSp.h>

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

    //Create vector
    double Valb[num_row];

    //Read the data
    for (int j=0; j<num_row; j++) {
        double data;
        finb >> data;
        Valb[j] = data;
    }

    finA.close();
    finb.close();
    
    //////////////////////////////////////////////////////////////////////////////////////////
    
    cusolverSpHandle_t  cusolverH = NULL;
    cusparseHandle_t    cusparseH = NULL;
    //cudaStream_t stream = NULL;

    cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    //cudaError_t cudaStat1 = cudaSuccess;
    //cudaError_t cudaStat2 = cudaSuccess;
    //cudaError_t cudaStat3 = cudaSuccess;
    //cudaError_t cudaStat4 = cudaSuccess;
    //cudaError_t cudaStat5 = cudaSuccess;

    //step 1 : create cuSolver and cuSparse handle
    cusolver_status = cusolverSpCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cusparse_status = cusparseCreate(&cusparseH);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);

    //step 2 : change format of A from COO to CSR
    int csrRowPtrA[num_row+1];
    cusparse_status = cusparseXcoo2csr(
                                    cusparseH,
                                    cooRowIndA,
                                    nnzA,
                                    num_row,
                                    csrRowPtrA,
                                    CUSPARSE_INDEX_BASE_ZERO);
    assert(CUSPARSE_STATUS_SUCCESS == cusparse_status);

    //step 3 : copy necessary data from host to device

}