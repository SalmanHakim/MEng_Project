#include <iostream>
#include <armadillo>
#include <fstream>
#include <chrono>

using namespace arma;

int main(int argc, char **argv)
{
    if (argc < 3) {
        //Show the correct syntax to user
        std::cerr << "The correct syntax: " << argv[0] << " nnz nameof_VectorB" << std::endl;
        return 1;
    }

    std::ofstream vecb;

    int element = atoi(argv[1]);

    vec v(element, fill::randn);

    vecb.open(argv[2]);
    vecb << element << " " << 1 << std::endl;
    for (int i=0; i<element; i++){
        vecb << v(i) <<std::endl;
    }

    //std::cout << v;
}