#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <iostream>
#include <fstream>
#include <vector>

struct Point {
    int r; //row
    int c; //col
    int v; //val
};

using namespace std;
vector<Point> generate_sparse(int s, int N, int p, int rank) {
    vector<Point> gen_mat;
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

void transpose_matrix(vector<Point>& matrix, int N, int p) {
    vector<int> sendcounts(p, 0); //sending number per proc
    vector<int> sdispls(p, 0); //displacement

    for (Point& point : matrix) {
        sendcounts[point.c / (N / p)]++;
    }

    sdispls[0] = 0;
    for (int i = 1; i < p; i++) {
        sdispls[i] = sdispls[i - 1] + sendcounts[i - 1]; //running total for displacement per proc
    }

    #include <mpi.h> // Include the MPI header file

    vector<Point> transposed_matrix(matrix.size()); //storage for transposed matrix

    MPI_Alltoallv(matrix.data(), sendcounts.data(), sdispls.data(), MPI_INT, transposed_matrix.data(), sendcounts.data(), sdispls.data(), MPI_INT, MPI_COMM_WORLD);

    for (Point& point: transposed_matrix) {
        int temp = point.r;
        point.r = point.c;
        point.c = temp; //cuz we have to transpose
    }

    matrix = transposed_matrix;
}

void mat_mul(vector<Point>& a, vector<Point>& b, int* c, int N, int p) {
    for (Point& pb : b) {
        for (Point& pa : a) {
            if (pa.c == pb.r) {
                int new_val = pa.v * pb.v;
                int idx = pa.r * N + pb.c;
                c[idx] += new_val;
            }
        }
    }
}

int main(int argc, char** argv) {
    #include <mpi.h> // Include the MPI header file

    MPI_Init(&argc, &argv);

    int rank, p;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    MPI_Comm comm;
    int dim[1] = {p};
    int period[1] = {1};
    int reorder = 0;
    MPI_Cart_create(MPI_COMM_WORLD, 1, dim, period, reorder, &comm);

    int N = stoi(argv[1]);  // size
    double s = stod(argv[2]);  // sparsoty
    int pf = stoi(argv[3]);  // printing flag
    string out_file = argv[4];  // Ofile name

    vector<Point> A = generate_sparse(s, N, p, rank);
    vector<Point> B = generate_sparse(s, N, p, rank);

    transpose_matrix(B, N, p);

    int C_size = N * N / p;
    int* C = new int[C_size];
    for (int i = 0; i < C_size; i++) {
        C[i] = 0;
    }

    double start_time, end_time;
    if (rank == 0) {
        start_time = MPI_Wtime();
    }


    int src, dst;
    MPI_Cart_shift(comm, 0, 1, &src, &dst);

    for (int iter = 0; iter < p; iter++) {
        mat_mul(A, B, C, N, p);
        
        vector<Point> rec_buffer(B.size());
        MPI_Sendrecv(B.data(), B.size(), MPI_INT, dst, 0, rec_buffer.data(), rec_buffer.size(), MPI_INT, src, 0, comm, MPI_STATUS_IGNORE);

        B = rec_buffer;
    }

    if (rank == 0) {
        end_time = MPI_Wtime();
        cout << "Time: " << end_time - start_time << endl;
    }

    if (pf == 1) {
        // ADD PRINTING CODE HERE
    }

    delete[] C;

    MPI_Finalize();
    return 0;
} 


