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

void print_matrix(int* matrix, char* outfile, int dim1, int dim2){
    FILE * fp = fopen(outfile, "w");
    for(int i = 0; i < dim1; i++){
        for(int j = 0; j < dim2; j++){
            fprintf(fp, "%d ", matrix[i*dim2 + j]);
        }
        fprintf(fp, "\n");
    }
}

void print_mat(vector<Point> matrix){
    int all_vals = matrix.size();
    char *filename = "example.txt";
    FILE * fp = fopen(filename, "w");
    for (int i = 0; i < all_vals; i++) {
        fprintf(fp, "(%d, %d, %d)\n", matrix[i].r, matrix[i].c, matrix[i].v);
    }
}

void print_mat_int(vector<int> matrix){
    int all_vals = matrix.size();
    char *filename = "example.txt";
    FILE * fp = fopen(filename, "w");
    for (int i = 0; i < all_vals; i++) {
        fprintf(fp, "%d ", matrix[i]);
    }
}

int* vec_mat(vector<Point> matrix, int N, int p){
    int* mat = new int[N*N/p];

    for (int i = 0; i < matrix.size(); i++) {
        int row = matrix[i].r;
        int col = matrix[i].c;
        mat[row*N + col] += matrix[i].v;
    }    
    return mat;
}

// void transpose_matrix(vector<Point>& matrix, int N, int p, int rank, vector<int> sendcounts, vector<int> sdispls, vector<int> recvcounts(p, 0), vector<int> rdispls(p, 0)) {
//     vector<int> sendcounts(p, 0); //sending number per proc
//     vector<int> sdispls(p, 0); //displacement

//     vector<int> recvcounts(p, 0); //sending number per proc
//     vector<int> rdispls(p, 0); //displacement

//     //how much does this current processor send to each other processor?
//     //how much is sent to each processor i.e. how much does rank 0 receive?
//     for (Point& point : matrix) {
//         if((point.c / (N / p)) != rank){ // just stays in place
//             sendcounts[point.c / (N / p)]++;
//         }
//     }


//     // MPI_Allreduce(&sendcounts, recvcounts.data(), sendcounts.size(), MPI_INT, MPI_SUM, )
//     //for every processor, iterate through the matrix and store an array of 
//     //all processors that it is sending to and increment that joint array

//     // for (Point& point : matrix) {
//     //     if((point.c / (N / p)) != rank){ // just stays in place
//     //         recvcounts[point.c / (N / p)]++;
//     //     }
//     // }
//     //how much does each processor send to this current processor?
    
//     // print_mat_int(sendcounts);
//     // //how much is being sent out from each processor ie. how much does rank 0 send out?
//     // for (Point& point : matrix) {
//     //     recvcounts[point.c / (N / p)]++;
//     // }

//     sdispls[0] = 0;
//     for (int i = 1; i < p; i++) {
//         sdispls[i] = sdispls[i - 1] + sendcounts[i - 1]; //running total for displacement per proc
//     }

//     vector<Point> transposed_matrix(matrix.size()); //storage for transposed matrix

//     MPI_Alltoallv(matrix.data(), sendcounts.data(), sdispls.data(), MPI_INT, transposed_matrix.data(), sendcounts.data(), sdispls.data(), MPI_INT, MPI_COMM_WORLD);

//     for (Point& point: transposed_matrix) {
//         int temp = point.r;
//         point.r = point.c;
//         point.c = temp; //cuz we have to transpose
//     }

//     matrix = transposed_matrix;
// }

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

    if(rank == 0){
        int* curr_mat = vec_mat(B, N, p);
        print_matrix(curr_mat, "original.txt", N/p, N);
    }

    vector<Point> matrix = B;
    vector<int> sendcounts(p, 0); //sending number per proc
    vector<int> sdispls(p, 0); //displacement

    vector<int> recvcounts(p, 0); //sending number per proc
    vector<int> rdispls(p, 0); //displacement


    for (Point& point : matrix) {
        sendcounts[point.c / (N / p)]++;
    }

    sdispls[0] = 0;
    for (int i = 1; i < p; i++) {
        sdispls[i] = sdispls[i - 1] + sendcounts[i - 1]; //running total for displacement per proc
    }

    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    rdispls[0] = 0;
    for (int i = 1; i < p; i++) {
        rdispls[i] = rdispls[i - 1] + recvcounts[i - 1]; //running total for displacement per proc
    }

    vector<Point> transposed_matrix(matrix.size()); //storage for transposed matrix

    MPI_Alltoallv(matrix.data(), sendcounts.data(), sdispls.data(), MPI_INT, transposed_matrix.data(), recvcounts.data(), rdispls.data(), MPI_INT, MPI_COMM_WORLD);

    for (Point& point: transposed_matrix) {
        int temp = point.r;
        point.r = point.c;
        point.c = temp; //cuz we have to transpose
    }

    B = transposed_matrix;
    if(rank == 0){
        int* new_mat = vec_mat(transposed_matrix, N, p);
        print_matrix(new_mat, "transpose.txt", N, N/p);
    }

    // print_mat(B, "transpose");

    // transpose_matrix(B, N, p, rank);

    // if(rank == 0){
    //     print_mat(B, N, p, 0);
    // }


    // int C_size = N * N / p;
    // int* C = new int[C_size];
    // for (int i = 0; i < C_size; i++) {
    //     C[i] = 0;
    // }

    // double start_time;
    // if (rank == 0) {
    //     start_time = MPI_Wtime();
    // }


    // int src, dst;
    // MPI_Cart_shift(comm, 0, 1, &src, &dst);

    // for (int iter = 0; iter < p; iter++) {
    //     mat_mul(A, B, C, N, p);
        
    //     vector<Point> rec_buffer(B.size());
    //     MPI_Sendrecv(B.data(), B.size(), MPI_INT, dst, 0, rec_buffer.data(), rec_buffer.size(), MPI_INT, src, 0, comm, MPI_STATUS_IGNORE);

    //     B = rec_buffer;
    // }

    // double end_time;
    // if (rank == 0) {
    //     end_time = MPI_Wtime();
    //     double time_taken = end_time - start_time;
    //     cout << "Time: " <<time_taken << endl;
    // }


    // if (pf == 1) {
    //     // printf("Running");
    //     int* final_mat;
    //     MPI_Gather(&C, N*N/p, MPI_INT, &final_mat, N*N, MPI_INT, 0, comm);
    //     print_matrix(final_mat, out_file, N, p);
    // }

    // delete[] C;

    MPI_Finalize();
    return 0;
} 


