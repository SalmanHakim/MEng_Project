# Guide on how to compile the files

## sparse_solver.cpp (Armadillo)

>$g++ sparse_solver.cpp -o \<desire name of exe file\> -std=c++11 -O2 -larmadillo

## spsolver.cpp (CUDA)

>$nvcc -c -I/usr/local/cuda/include spsolver.cpp

and then,
>$g++ -o \<desired name of exe file\> spsolver.o -L/usr/local/cuda/lib64 -lcusparse -lcudart -lcusolver
