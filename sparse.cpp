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
vector<Point> generate_sparse(float s, int N, int p, int rank) {
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
    vector<int> sendcounts(p, 0);
    vector<int> sdispls(p, 0);
    vector<int> recvcounts(p, 0);
    vector<int> rdispls(p, 0);
    for (Point& point : matrix) {
        sendcounts[point.c / (N / p)]++;
    }

    sdispls[0] = 0;
    for (int i = 1; i < p; i++) {
        sdispls[i] = sdispls[i - 1] + sendcounts[i - 1];
    }
    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    
    rdispls[0] = 0;
    int recv_buffer_size = recvcounts[0];
    for (int i = 1; i < p; i++) {
        rdispls[i] = rdispls[i - 1] + recvcounts[i - 1];
        recv_buffer_size += recvcounts[i];
    }
    vector<Point> transposed_matrix(recv_buffer_size);
    MPI_Alltoallv(matrix.data(), sendcounts.data(), sdispls.data(), MPI_INT,
                  transposed_matrix.data(), recvcounts.data(), rdispls.data(), MPI_INT,
                  MPI_COMM_WORLD);
    for (Point& point : transposed_matrix) {
        int temp = point.r;
        point.r = point.c;
        point.c = temp;
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

void print_matrix_to_file(int* matrix, int N, ofstream& outfile) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            outfile << matrix[i*N + j] << " ";
        }
        outfile << endl;
    }
    outfile << endl;
}

void print_matrices_to_file(int* A, int* B, int* C, int N, int p, const char* out_file) {

    ofstream outfile(out_file);
    print_matrix_to_file(A, N, outfile);
    print_matrix_to_file(B, N, outfile);
    print_matrix_to_file(C, N, outfile);
    outfile.close();
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    int rank, p;
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
    char* out_file = argv[4];  // Ofile name
    vector<Point> A = generate_sparse(s, N, p, rank);
    vector<Point> B = generate_sparse(s, N, p, rank);

    transpose_matrix(B, N, p);

    int C_size = N * N / p;
    int* C = new int[C_size];
    for (int i = 0; i < C_size; i++) {
        C[i] = 0;
    }

    double start_time;
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

    double end_time;
    if (rank == 0) {
        end_time = MPI_Wtime();
        double time_taken = end_time - start_time;
        cout << "Time: " << time_taken << endl;
    }

    cout << "wtf";
    MPI_Barrier(MPI_COMM_WORLD);

    int* dense_A = new int[N * N / p];
    int* dense_B = new int[N * N / p];
    for (const Point& point : A) {
        dense_A[point.r * N + point.c] = point.v;
    }
    for (const Point& point : B) {
        dense_B[point.r * N + point.c] = point.v;
    }

    printf("FNISIHING THIS");
    // int* global_dense_A = nullptr;
    // int* global_dense_B = nullptr;
    // int* global_dense_C = nullptr;

    // if (rank == 0) {
    int* global_dense_A = new int[N * N];
    int* global_dense_B = new int[N * N];
    int* global_dense_C = new int[N * N];
    // }

    

    printf("STARTING GATHERS");
    MPI_Gather(dense_A, N * N / p, MPI_INT, global_dense_A, N * N / p, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(dense_B, N * N / p, MPI_INT, global_dense_B, N * N / p, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(C, N * N / p, MPI_INT, global_dense_C, N * N / p, MPI_INT, 0, comm);
    printf("ENDING GARTHERS");
    if (pf == 1) {
        printf("ENTEREDF HERE");
        int* global_C = nullptr;
        
        if (rank == 0) {
            ofstream outfile(out_file);
            print_matrices_to_file(global_dense_A, global_dense_B, global_dense_C, N, p, out_file);
            
        }
        delete[] global_C;
    }

    delete[] dense_A;
    delete[] dense_B;
    delete[] C;
    
    delete[] global_dense_A;
    delete[] global_dense_B;
    delete[] global_dense_C;

    MPI_Finalize();
    return 0;
} 