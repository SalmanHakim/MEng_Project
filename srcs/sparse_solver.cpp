#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

int main(int argc, char **argv)
{
    //Open file
    ifstream finA(argv[1]);      //file name should be written in the command line

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
    ifstream finb(argv[2]);      //file name should be written in the command line

    //Ignore header
    while (finb.peek() == '%') {
        finb.ignore(2048, '\n');
    }

    //Read in vector features
    finb >> num_row >> num_col;

    //create vector


    finA.close();

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