#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <iostream>
#include <fstream>
#include <vector>

struct Point {
    int row;
    int col;
    int value;
};

// using tuples of length 3 to represent sparse matrices
#include <vector>

std::vector<Point> generate_sparse(int s, int N, int p, int rank) {
    std::vector<Point> gen_mat;
    int start_row = rank * (N / p);
    int end_row = (rank + 1) * (N / p);
    for (int c = 0; c < N; c++) {
        for (int r = start_row; r < end_row; r++) {
            if (rand() % N < s * N) {
                Point point = {r, c, rand() % 10}; //row, col, value
                gen_mat.push_back(point);
            }
        }
    }
    return gen_mat;
}

void transpose_matrix(std::vector<Point>& matrix, int N, int p) {
    std::vector<int> sendcounts(p, 0); //sending number per proc
    std::vector<int> sdispls(p, 0); //displacement

    for (Point& p : matrix) {
        sendcounts[p.col / (N / p)]++;
    }

    sdispls[0] = 0;
    for (int i = 1; i < p; i++) {
        sdispls[i] = sdispls[i - 1] + sendcounts[i - 1]; //running total for displacement per proc
    }

    std::vector<Point> transposed_matrix(matrix.size()); //storage for transposed matrix

    // i think??
    MPI_Alltoallv(matrix.data(), sendcounts.data(), sdispls.data(), MPI_INT, transposed_matrix.data(), sendcounts.data(), sdispls.data(), MPI_INT, MPI_COMM_WORLD);

    for (Point& p: transposed_matrix) {
        int temp = p.row;
        p.row = p.col;
        p.col = temp; //cuz we have to transpose
    }

    matrix = transposed_matrix;
}

int main(int argc, char** argv) {
    #include <mpi.h> // Include the MPI header file

    MPI_Init(&argc, &argv);

    int rank, p;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);

    int N = std::stoi(argv[1]);  // size
    double s = std::stod(argv[2]);  // sparsoty
    int pf = std::stoi(argv[3]);  // printing flag
    std::string out_file = argv[4];  // Ofile name

    std::vector<Point> A = generate_sparse(s, N, p, rank);
    std::vector<Point> B = generate_sparse(s, N, p, rank);

    transpose_matrix(B, N, p);

    int C_size = N * N / p;
    int* C = new int[C_size];
    for (int i = 0; i < C_size; i++) {
        C[i] = 0;
    }
} 


