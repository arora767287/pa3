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
            // if(proc == 1){
            //     printf("processor 1: \n");       
            //     printCSRForm(temp_csr);
            //     printf("\n");
            // }
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
            // if(proc == 1){
            //     printf("processor 1: \n");       
            //     printCSRForm(temp_csr);
            //     printf("\n");
            // }

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
    vector<int> rows(1, 0);
    vector<int> cols;
    vector<int> vals;

    vector<int> transpose_cols;
    vector<int> transpose_rows;
    vector<int> transpose_vals;

    vector<int> count_all(N, 0);

    int start_row = rank * (N / p);
    int end_row = (rank + 1) * (N / p);
    srand(time(NULL) + rank + seed + 2); 

    int val_count = 0;

    for (int r = start_row; r < end_row; r++) {
        for (int c = 0; c < N; c++) {
            int randd = rand() % N;
            if (randd < s * N) {
                int rand_value = (rand() % 9) + 1;
                val_count++;
                cols.push_back(c);
                vals.push_back(rand_value);

                transpose_cols.push_back(r);
                transpose_rows.push_back(c);
                transpose_vals.push_back(rand_value);
                count_all[c]++;
            }
        }
        rows.push_back(val_count);
    }

    struct Comparator {
        const vector<int>& ref;
        Comparator(const vector<int>& v) : ref(v) {}
        bool operator()(int i, int j) const {
            return ref[i] < ref[j];
        }
    };

    vector<int> index(transpose_rows.size(), 0);
    iota(index.begin(), index.end(), 0);
    sort(index.begin(), index.end(), Comparator(transpose_rows));

    vector<int> csc_cols(N + 1, 0), csc_rows, csc_vals;
    for (int i : index) {
        csc_rows.push_back(transpose_cols[i]);
        csc_vals.push_back(transpose_vals[i]);
    }

    for (int i = 0, col_offset = 0; i < transpose_rows.size(); ++i) {
        while (transpose_rows[index[i]] > col_offset) {
            csc_cols[++col_offset] = i;
        }
    }
    csc_cols[N] = transpose_rows.size();

    transpose.rows = csc_cols;
    transpose.cols = csc_rows;
    transpose.vals = csc_vals;

    m.rows = rows;
    m.cols = cols;
    m.vals = vals;
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

int* gather_and_return_CSR_matrix(const CSR& curr_matrix, int N, int p, int rank) {
    std::vector<int> curr_buffer;
    serializeCSR(curr_matrix, curr_buffer);
    int curr_size = curr_buffer.size();

    std::vector<int> sizes(p);
    MPI_Gather(&curr_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displs(p, 0);
    if (rank == 0) {
        for (int i = 1; i < p; ++i) {
            displs[i] = displs[i - 1] + sizes[i - 1];
        }
    }

    std::vector<int> all_data;
    if (rank == 0) {
        all_data.resize(displs[p - 1] + sizes[p - 1]);
    }

    MPI_Gatherv(curr_buffer.data(), curr_size, MPI_INT,
                all_data.data(), sizes.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int* matrix = new int[N * N](); 

        int start_idx = 0;
        for (int i = 0; i < p; ++i) {
            vector<int> sub_buffer(all_data.begin() + start_idx, all_data.begin() + start_idx + sizes[i]);
            CSR csr;
            deserializeCSR(sub_buffer, csr);

            for (size_t row = 0; row + 1 < csr.rows.size(); ++row) {
                for (int j = csr.rows[row]; j < csr.rows[row + 1]; ++j) {
                    int col = csr.cols[j];
                    int val = csr.vals[j];
                    matrix[row * N + col] = val;
                }
            }
            start_idx += sizes[i];
        }
        return matrix;
    }
    return NULL;
}


//transpose matrix first, so b's rows are original b matrix columns


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

void mat_mul_csr(const CSR& A, const CSR& BT, int* C, int rowsA, int colsB) {
    std::memset(C, 0, rowsA * colsB * sizeof(int));
    for (int i = 0; i < rowsA; ++i) { 
        for (int j = A.rows[i]; j < A.rows[i + 1]; ++j) { 
            int colA = A.cols[j];
            int valA = A.vals[j]; 
            for (int k = BT.rows[colA]; k < BT.rows[colA + 1]; ++k) {
                int colB = BT.cols[k];
                int valBT = BT.vals[k];
                C[i * colsB + colB] += valA * valBT;
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

    generate_sparse_bonus_T(s, N, p, rank, 0, tranA, A);
    generate_sparse_bonus_T(s, N, p, rank, 1, tranB, B);

    // if(rank == 1){
    //     printCSRForm(B);
    //     printf("\n");
    //     printCSRForm(tranB);
    // }

    int C_size = N * N / p;
    int* C = new int[C_size];
    for (int i = 0; i < C_size; i++) {
        C[i] = 0;
    }

    // printCSRForm(tranB);
    // printf("\n");
    if(rank == 0){
        printCSRMatrix(tranB, N/p, N);
        printf("\n");
        printCSRForm(tranB);
    }

    // int* mat_A = gather_and_return_CSR_matrix(B, N, p, rank);

    // int* mat_B = gather_and_return_CSR_matrix(tranB, N, p, rank);

    int* mat_A = gather_and_assemble_CSR(N, p, rank, B);

    int* mat_B = gather_and_assemble_CSR_T(N, p, rank, tranB);


    int src, dst;
    MPI_Cart_shift(comm, 0, 1, &src, &dst);

    double start_time;
    if (rank == 0) {
        start_time = MPI_Wtime();
    }
    CSR new_tranB;
    // for (int iter = 0; iter < p; iter++) {
    //     mat_mul_csr(A, tranB, C, N/p, N);
        

    //     vector<int> send_buffer;
    //     serializeCSR(tranB, send_buffer);
    //     int send_size = send_buffer.size();
    //     if(rank == 0){
    //         printCSRMatrix(tranB, N/p, N);
    //         printf("\n");
    //     }

    //     int recv_size;
    //     MPI_Sendrecv(&send_size, 1, MPI_INT, dst, 0, &recv_size, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    //     vector<int> recv_buffer(recv_size);
    //     MPI_Sendrecv(send_buffer.data(), send_size, MPI_INT, dst, 0, recv_buffer.data(), recv_size, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    //     deserializeCSR(recv_buffer, new_tranB);


    //     tranB = new_tranB; 

    // }

    double end_time;
    if (rank == 0) {
        end_time = MPI_Wtime();
        double time_taken = end_time - start_time;
        cout << "Time: " <<time_taken << endl;
    }

    int* global_C = new int[N*N];
    for (int i = 0; i < N*N; i++) {
        global_C[i] = 0;
    }
    MPI_Gather(C, N*N/p , MPI_INT, global_C , N*N/p , MPI_INT, 0, MPI_COMM_WORLD);

    if (pf == 1) {
        if(rank == 0){
            printMatrix(mat_mul_serial(mat_A, mat_B, N), N, N);
            print_matrix_all(mat_A, mat_B, global_C, out_file, N, N);
        }
    }
    MPI_Finalize();
    return 0;
} 