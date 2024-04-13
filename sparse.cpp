#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <iostream>
#include <fstream>

// using tuples of length 3 to represent sparse matrices
std::vector<Tuple> generate_sparse(int s, int N, int p, int rank) {
    std::vector<Tuple> gen_mat;
    for (int r = rank * (N / p); r < (rank + 1) * (N / p); r++) {
        for (int c = 0; c < N; c++) {
            int rand_num = rand() % N;
            if (rand_num < s * N) {
                Tuple tup = new Tuple(t.row, t.col, rand() % 10); //row, col, value
                gen_mat.push_back(tup);
            }
        }
    }
    return gen_mat;
}


