# Guide on how to compile the files

## sparse_solver.cpp (Armadillo)

>$g++ sparse_solver.cpp -o \<desire name of exe file\> -std=c++11 -O2 -larmadillo

## spsolver.cpp (CUDA)

>$nvcc -c -I/usr/local/cuda/include spsolver.cpp

and then,
>$g++ -o \<desired name of exe file\> spsolver.o -L/usr/local/cuda/lib64 -lcusparse -lcudart -lcusolver

## vecb_gen.cpp (Vector generator)

>$g++ vecb_gen.cpp -o \<desire name of exe file\> -std=c++11 -O2 -larmadillo

When running the file, it will ask for `nnz` and `name of vector b mtx file`. For example, the command

>$./vecb_gen 321821 ASIC_320k_b.mtx

will make a vector b mtx file for `ASIC_320k` benchmark.
