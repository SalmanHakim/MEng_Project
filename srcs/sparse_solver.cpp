#include <iostream>
#include <armadillo>
#include <chrono>

using namespace arma;

void printMatrix(int m, int n, vec x, int ldx, const char *name)
{
    for (int row=0; row<m; row++) {
        for (int col=0; col<n; col++) {
            double xreg = x(row + (col*ldx));
            std::cout << name << "(" << row+1 << "," << col+1 << ") = " << xreg << std::endl;
        }
    }
}

//Function to solve the linear equation
vec solve(sp_mat A, vec b, int num_row) {

    superlu_opts opts;

    //opts.allow_ugly     = true;
    //opts.equilibrate    = true;
    opts.refine         = superlu_opts::REF_DOUBLE;

    vec x = spsolve(A, b, "superlu");

    bool status = spsolve(x, A, b);
    if (status == false) {
        std::cout << "no solution" << std::endl;
    }

    return x;
}

int main(int argc, char **argv)
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
    int num_row, num_col, num_lines;

    //Ignore header
    while (finA.peek() == '%') {
        finA.ignore(2048, '\n');
    }
    
    //Read in the matrix features
    finA >> num_row >> num_col >> num_lines;

    //Create matrix, fill with zeros
    sp_mat A(num_row, num_col);
    A.zeros();

    //Read the data
    for (int i=0; i<num_lines; i++) {
        int m, n;
        double data;
        finA >> m >> n >> data;
        A(m-1, n-1) = data;
    }

    //Open file
    std::ifstream finb(argv[2]);      //file name should be written in the command line

    //Ignore header
    while (finb.peek() == '%') {
        finb.ignore(2048, '\n');
    }

    //Read in vector features
    finb >> num_row >> num_col;

    //create vector
    vec b(num_row);

    //Read data
    for (int j=0; j<num_row; j++) {
        double data;
        finb >> data;
        b(j) = data;
    }

    finA.close();
    finb.close();

    std::cout << "START" << std::endl;
    auto begin = std::chrono::high_resolution_clock::now();
    vec x = solve(A, b, num_row);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    std::cout << "Time measured (seconds) : " << elapsed.count() * 1e-9 << std::endl;

    /*std::cout << "X = (matlab base-1)" << std::endl;
    printMatrix(num_row, 1, x , num_row, "x");*/

//    sp_mat A = sprandu<sp_mat>(100, 100, 0.1);

//    vec b(100, fill::randu);

//    vec x = spsolve(A, b);

//    bool status = spsolve(x, A, b);
//    if( status == false) {
//        cout << "no solution" << endl;
//    }

//    spsolve(x, A, b, "lapack");
//    spsolve(x, A, b, "superlu");

//    superlu_opts opts;

//    opts.allow_ugly     = true;
//    opts.equilibrate    = true;

//    spsolve(x, A, b, "superlu", opts);

//    cout << "x = " << x;
}