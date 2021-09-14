#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

int main()
{
    mat A(4, 3, fill::randu);
    mat B(3, 4, fill::randu);


    cout << "A=\t" << A << endl;
    cout << "B=\t" << B << endl;
    cout << "A*B=\t" << A*B << endl;

    return 0;
}