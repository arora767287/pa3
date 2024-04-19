#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace std;

struct Point {
    int r;
    int c;
    int v;
};

struct CSR {
    vector<int> rows;
    vector<int> cols;
    vector<int> vals;
};

bool sort_by_row(const Point &a, const Point &b) {
    return a.r < b.r;
}

bool sort_by_col(const Point &a, const Point &b) {
    if (a.c != b.c) return a.c < b.c;
    return a.r < b.r;
}

void printVector(const std::vector<int>& vec) {
    std::cout << "[";
    for (size_t i = 0; i < vec.size(); i++) {
        std::cout << vec[i];
        if (i < vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

void printCSRForm(const CSR& matrix){
    vector<int> rows = matrix.rows;
    vector<int> cols = matrix.cols;
    vector<int> vals = matrix.vals;
    
    printf("Rows: \n");
    printVector(rows);
    printf("Columns: \n");
    printVector(cols);
    printf("Values: \n");
    printVector(vals);
}

void printCSRMatrix(const CSR& matrix, int numRows, int numCols) {
    int currentRow = 0;
    for (int r = 0; r < numRows; r++) {
        int rowStart = matrix.rows[r];
        int rowEnd = (r + 1 < numRows) ? matrix.rows[r + 1] : matrix.cols.size();

        int currentColIndex = rowStart;

        for (int c = 0; c < numCols; c++) {
            if (currentColIndex < rowEnd && matrix.cols[currentColIndex] == c) {
                std::cout << matrix.vals[currentColIndex] << " ";
                currentColIndex++;
            } else {
                std::cout << "0 ";
            }
        }
        std::cout << std::endl;
    }
}

void serializeCSR(const CSR& matrix, std::vector<int>& buffer) {
    buffer.clear();
    //allocate memory using reserve for buffer
    buffer.push_back(matrix.rows.size());  // Number of row pointers
    buffer.push_back(matrix.cols.size());  // Number of column indices (and number of values)
    buffer.insert(buffer.end(), matrix.rows.begin(), matrix.rows.end());
    buffer.insert(buffer.end(), matrix.cols.begin(), matrix.cols.end());
    buffer.insert(buffer.end(), matrix.vals.begin(), matrix.vals.end());
}

void deserializeCSR(const std::vector<int>& buffer, CSR& matrix) {
    size_t idx = 0;
    size_t rows_size = buffer[idx++];
    size_t cols_size = buffer[idx++];
    matrix.rows.assign(buffer.begin() + idx, buffer.begin() + idx + rows_size);
    idx += rows_size;
    matrix.cols.assign(buffer.begin() + idx, buffer.begin() + idx + cols_size);
    idx += cols_size;
    matrix.vals.assign(buffer.begin() + idx, buffer.begin() + idx + cols_size);
}

int* gather_and_assemble_CSR(int N, int p, int rank, const CSR& local_csr) {
    std::vector<int> serialized_data;
    serializeCSR(local_csr, serialized_data);
    int local_size = serialized_data.size();

    std::vector<int> all_sizes(p);
    MPI_Gather(&local_size, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displacements;
    int total_size = 0;
    if (rank == 0) {
        displacements.resize(p);
        for (int i = 0; i < p; ++i) {
            displacements[i] = total_size;
            total_size += all_sizes[i];
        }
    }

    std::vector<int> all_data(total_size);
    MPI_Gatherv(serialized_data.data(), local_size, MPI_INT,
                all_data.data(), all_sizes.data(), displacements.data(), MPI_INT, 0, MPI_COMM_WORLD);

    int* full_matrix = nullptr;
    if (rank == 0) {
        full_matrix = new int[N * N]();  // Initialize full matrix
        int offset = 0;
        for (int proc = 0; proc < p; ++proc) {
            CSR temp_csr;
            std::vector<int> temp_buffer(all_data.begin() + displacements[proc], all_data.begin() + displacements[proc] + all_sizes[proc]);
            deserializeCSR(temp_buffer, temp_csr);
            
            int start_row = proc * (N/p);
            for (size_t r = 0; r < temp_csr.rows.size() - 1; ++r) {
                for (int idx = temp_csr.rows[r]; idx < temp_csr.rows[r + 1]; ++idx) {
                    int col = temp_csr.cols[idx];
                    int val = temp_csr.vals[idx];
                    full_matrix[((start_row + r) % N) * N + col] = val;
                }
            }
        }
    }
    return full_matrix;
}

int* gather_and_assemble_CSR_T(int N, int p, int rank, const CSR& local_csr) {
    std::vector<int> serialized_data;
    serializeCSR(local_csr, serialized_data);
    int local_size = serialized_data.size();

    std::vector<int> all_sizes(p);
    MPI_Gather(&local_size, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displacements;
    int total_size = 0;
    if (rank == 0) {
        displacements.resize(p);
        for (int i = 0; i < p; ++i) {
            displacements[i] = total_size;
            total_size += all_sizes[i];
        }
    }


    std::vector<int> all_data(total_size);
    MPI_Gatherv(serialized_data.data(), local_size, MPI_INT,
                all_data.data(), all_sizes.data(), displacements.data(), MPI_INT, 0, MPI_COMM_WORLD);

    int* full_matrix = nullptr;
    if (rank == 0) {
        full_matrix = new int[N * N](); 
        int offset = 0;
        for (int proc = 0; proc < p; ++proc) {
            CSR temp_csr;
            std::vector<int> temp_buffer(all_data.begin() + displacements[proc], all_data.begin() + displacements[proc] + all_sizes[proc]);
            deserializeCSR(temp_buffer, temp_csr);
            
            int start_row = proc * (N/p);
            for (size_t r = 0; r < temp_csr.rows.size() - 1; ++r) {
                for (int idx = temp_csr.rows[r]; idx < temp_csr.rows[r + 1]; ++idx) {
                    int col = temp_csr.cols[idx];
                    int val = temp_csr.vals[idx];
                    full_matrix[((start_row + r - (N/p)*proc) % N) * N + col] = val;
                }
            }
        }
    }
    return full_matrix;
}

void generate_sparse_bonus_T(float s, int N, int p, int rank, int seed, CSR& transpose, CSR& m) {
    std::vector<int> rows(1, 0);
    std::vector<int> cols;
    std::vector<int> vals;

    std::vector<int> transpose_cols;
    std::vector<int> transpose_temp_rows;
    std::vector<int> transpose_vals;

    srand(time(NULL) + rank + seed + 2); 

    int val_count = 0;

    for (int r = rank * (N / p); r < (rank + 1) * (N / p); r++) {
        for (int c = 0; c < N; c++) {
            if (rand() % N < s * N) {
                int rand_value = (rand() % 9) + 1;
                val_count++;
                cols.push_back(c);
                vals.push_back(rand_value);

                // Store transpose matrix values, swapping row and column indices
                transpose_temp_rows.push_back(c); // This acts as the "row" in the transpose
                transpose_cols.push_back(r); // This acts as the "column" in the transpose
                transpose_vals.push_back(rand_value);
            }
        }
        rows.push_back(val_count);
    }

    // set up transpose
    std::vector<int> transpose_rows(N + 1, 0);
    std::vector<int> count(N, 0);
    for (int i = 0; i < transpose_temp_rows.size(); ++i) {
        count[transpose_temp_rows[i]]++;
    }
    for (int i = 1; i <= N; ++i) {
        transpose_rows[i] = transpose_rows[i-1] + count[i-1];
    }

    std::vector<int> temp_position = transpose_rows;
    std::vector<int> sorted_transpose_cols(transpose_cols.size());
    std::vector<int> sorted_transpose_vals(transpose_vals.size());

    for (size_t i = 0; i < transpose_temp_rows.size(); ++i) {
        int pos = temp_position[transpose_temp_rows[i]]++;
        sorted_transpose_cols[pos] = transpose_cols[i];
        sorted_transpose_vals[pos] = transpose_vals[i];
    }

    transpose.rows = transpose_rows;
    transpose.cols = sorted_transpose_cols;
    transpose.vals = sorted_transpose_vals;

    m.rows = rows;
    m.cols = cols;
    m.vals = vals;
}

MPI_Datatype create_mpi_csr_type() {
    MPI_Datatype csr_type;
    MPI_Type_contiguous(2, MPI_INT, &csr_type);
    MPI_Type_commit(&csr_type);
    return csr_type;
}

CSR transpose_csr_matrix(CSR& curr_matrix, int N, int p, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Prepare counts and displacements for MPI communication
    std::vector<int> sendcounts(p, 0), sdispls(p), recvcounts(p), rdispls(p);

    // Prepare data to send
    std::vector<int> senddata; // Flatten data for sending
    for (int i = 0; i < p; ++i) {
        int start_row = i * (N / p);
        int end_row = (i + 1) * (N / p);
        for (int row = start_row; row < end_row; ++row) {
            for (int idx = curr_matrix.rows[row]; idx < curr_matrix.rows[row + 1]; ++idx) {
                senddata.push_back(curr_matrix.cols[idx]);
                senddata.push_back(curr_matrix.vals[idx]);
            }
        }
        sendcounts[i] = (curr_matrix.rows[end_row] - curr_matrix.rows[start_row]) * 2; // Each element consists of two ints
        sdispls[i] = (i == 0) ? 0 : sdispls[i - 1] + sendcounts[i - 1];
    }

    // Communicate the amount of data to be received
    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);

    // Calculate displacements for received data
    rdispls[0] = 0;
    std::vector<int> recvdata;
    int totalRecv = 0;
    for (int i = 0; i < p; ++i) {
        rdispls[i] = (i == 0) ? 0 : rdispls[i - 1] + recvcounts[i - 1];
        totalRecv += recvcounts[i];
    }
    recvdata.resize(totalRecv);

    // Perform all-to-all communication of CSR data
    MPI_Alltoallv(senddata.data(), sendcounts.data(), sdispls.data(), MPI_INT,
                  recvdata.data(), recvcounts.data(), rdispls.data(), MPI_INT, comm);

    // Rebuild the local transposed matrix from recvdata

    // Rebuild the local transposed matrix from recvdata
    CSR transposed;
    transposed.rows.resize(N + 1, 0);
    for (int i = 0; i < totalRecv; i += 2) {
        int col = recvdata[i];
        int val = recvdata[i + 1];
        transposed.cols.push_back(col);
        transposed.vals.push_back(val);
        transposed.rows[col / (N / p) + 1]++;
    }
    // Compute prefix sums to finalize rows
    for (int i = 1; i <= N / p; ++i) {
        transposed.rows[i] += transposed.rows[i - 1];
    }

    return transposed;
}



void print_matrix_all(int* matrix, int* matrix2, int* matrix3, char* outfile, int dim1, int dim2){
    FILE * fp = fopen(outfile, "w");
    for(int i = 0; i < dim1; i++){
        for(int j = 0; j < dim2; j++){
            fprintf(fp, "%d ", matrix[i*dim2 + j]);
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "\n");
    for(int i = 0; i < dim1; i++){
        for(int j = 0; j < dim2; j++){
            fprintf(fp, "%d ", matrix2[i*dim2 + j]);
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "\n");
    for(int i = 0; i < dim1; i++){
        for(int j = 0; j < dim2; j++){
            fprintf(fp, "%d ", matrix3[i*dim2 + j]);
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "\n");
}

void mat_mul_bonus(CSR a, CSR b, int* c, int N, int p) {
    // Initialize the result matrix c to zero
    memset(c, 0, (N * N / p) * sizeof(int));

    // Perform matrix multiplication
    for (int i = 0; i < N; ++i) {  // Loop over rows of A
        for (int j = a.rows[i]; j < a.rows[i + 1]; ++j) {  // Loop over non-zero entries in row i of A
            int a_col = a.cols[j]; // Column index in A, corresponds to row index in B^T
            int a_val = a.vals[j]; // Value at A[i, a_col]
            for (int k = b.rows[a_col]; k < b.rows[a_col + 1]; ++k) {
                int b_col = b.cols[k];  // Column index in B^T, row index in original B
                int b_val = b.vals[k]; // Value at B^T[a_col, b_col], which is B[b_col, a_col]
                c[i * p + b_col] += a_val * b_val; // Accumulate product in C[i, b_col]
            }
        }
    }
}

void mat_mul_csr(const CSR& A, const CSR& BT, int* C, int rowsA, int colsBT) {
    std::memset(C, 0, rowsA * colsBT * sizeof(int));
    for (int i = 0; i < rowsA; ++i) {
        for (int j = A.rows[i]; j < A.rows[i + 1]; ++j) {
            int colA = A.cols[j];
            int valA = A.vals[j];
            for (int k = BT.rows[colA]; k < BT.rows[colA + 1]; ++k) {
                int rowBT = BT.cols[k];
                int valBT = BT.vals[k];
                C[i * colsBT + rowBT] += valA * valBT;
            }
        }
    }
}

void printMatrix(int* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int* mat_mul_serial(int* first, int* second, int N){
    int* global_C = new int[N*N];
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            global_C[i * N + j] = 0;
            for (int k = 0; k < N; k++) {
                global_C[i * N + j] += first[i * N + k] * second[k * N + j];
            }
        }
    }
    return global_C;
}

int main(int argc, char** argv) {
    cout << "STARTING" << std::flush;
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

    CSR tranA;
    CSR A;
    CSR tranB;
    CSR B;
    
    printf("HERE0");
    
    generate_sparse_bonus_T(s, N, p, rank, 0, tranA, A);
    generate_sparse_bonus_T(s, N, p, rank, 1, tranB, B);
    if (rank == 0){
        printf("HERE1");
    }
    
    int C_size = N * N / p;
    int* C = new int[C_size];
    for (int i = 0; i < C_size; i++) {
        C[i] = 0;
    }
    int* mat_A = gather_and_assemble_CSR(N, p, rank, B);

    int* mat_B = gather_and_assemble_CSR_T(N, p, rank, tranB);
    if (rank == 0){
        printf("HERE2");
    }
    int src, dst;
    MPI_Cart_shift(comm, 0, 1, &src, &dst);

   double start_time;
    if (rank == 0) {
        start_time = MPI_Wtime();
    }
    CSR new_tranB;
    

    new_tranB = transpose_csr_matrix(tranB, N, p, comm);
    if (rank == 0) {
        printf("HERE3");

    }
    printf("");
    if (rank == 2) {
        print_matrix_all(mat_A, mat_B, mat_B, out_file, N, N);
        printCSRForm(new_tranB);

    }
    // for (int iter = 0; iter < p; iter++) {
    //     // mat_mul_csr(A, tranB, C, N/p, N);
    //     // if (rank == 0) {
    //     //     printf("HERE4");

    //     // }   
    //     // vector<int> send_buffer;
    //     // serializeCSR(tranB, send_buffer);
    //     // int send_size = send_buffer.size();

    //     // int recv_size;
    //     // MPI_Sendrecv(&send_size, 1, MPI_INT, dst, 0, &recv_size, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
    //     // vector<int> recv_buffer(recv_size);
    //     // MPI_Sendrecv(send_buffer.data(), send_size, MPI_INT, dst, 0, recv_buffer.data(), recv_size, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    //     // deserializeCSR(recv_buffer, new_tranB);
    //     // tranB = new_tranB;
    // }

//     double end_time;
//     if (rank == 0) {
//         end_time = MPI_Wtime();
//         double time_taken = end_time - start_time;
//         cout << "Time: " << time_taken << endl;
//     }

//     int* local_C = new int[N * N / p];
//     for (int i = 0; i < N * N / p; ++i) {
//         local_C[i] = C[i];
//     }

//     int* global_C = nullptr;
//     if (rank == 0) {
//         global_C = new int[N * N];
//     }
//     MPI_Gather(local_C, N * N / p, MPI_INT, global_C, N * N / p, MPI_INT, 0, MPI_COMM_WORLD);

//     delete[] local_C;
//     if (rank == 0) {
//         delete[] global_C;
//     }

    MPI_Finalize();
    return 0;
} 