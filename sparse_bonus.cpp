#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <ctime>


using namespace std;

struct Point {
   uint64_t r;
   uint64_t c;
   uint64_t v;
};

struct CSR {
    vector<uint64_t> rows;
    vector<uint64_t> cols;
    vector<uint64_t> vals;
};

bool pointComparator(const Point& a, const Point& b) {
    if (a.c == b.c)
        return a.r < b.r;
    return a.c < b.c;
}

CSR convertToCSC(const std::vector<Point>& points,uint64_t cols) {
    CSR csc;
    if (points.empty()) {
        return csc;
    }

    uint64_t maxCol = 0;
    for (const auto& point : points) {
        if (point.c > maxCol) {
            maxCol = point.c;
        }
    }

    std::vector<Point> sortedPoints = points;
    std::sort(sortedPoints.begin(), sortedPoints.end(), pointComparator);

    csc.cols.resize(cols + 1, 0);

    for (const auto& point : sortedPoints) {
        if (point.c >= csc.cols.size()) {
            csc.cols.resize(point.c + 1, 0);
        }
        csc.rows.push_back(point.r);
        csc.vals.push_back(point.v);
    }

    uint64_t currentColumn = -1;
    for (size_t i = 0; i < sortedPoints.size(); ++i) {
        while (currentColumn < sortedPoints[i].c) {
            currentColumn++;
            csc.cols[currentColumn] = i;
        }
    }
    csc.cols[currentColumn + 1] = sortedPoints.size(); 

    for (int i = currentColumn + 2; i <= cols; i++) {
        csc.cols[i] = sortedPoints.size();
    }
    vector<uint64_t> temp = csc.rows;
    csc.rows = csc.cols;
    csc.cols = temp;
    return csc;
}

CSR convertToCSR(const vector<Point>& points,uint64_t num_rows) {
    CSR csr;
    if (points.empty()) {
        return csr;
    }
    csr.rows.resize(num_rows + 1, 0);
    csr.cols.reserve(points.size());
    csr.vals.reserve(points.size());

    for (auto& point : points) {
        csr.cols.push_back(point.c);
        csr.vals.push_back(point.v);
    }

    for (auto& point : points) {
        csr.rows[(point.r%num_rows + 1)]++;
    }

    for (int i = 1; i <= num_rows; ++i) {
        csr.rows[i] += csr.rows[i - 1];
    }

    return csr;
}

bool sort_by_row(const Point &a, const Point &b) {
    return a.r < b.r;
}

bool sort_by_col(const Point &a, const Point &b) {
    if (a.c != b.c) return a.c < b.c;
    return a.r < b.r;
}

void printVector(const std::vector<uint64_t>& vec) {
    std::cout << "[";
    for (size_t i = 0; i < vec.size(); i++) {
        std::cout << vec[i];
        if (i < vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

void serializeCSR(const CSR& matrix, std::vector<uint64_t>& buffer) {
    buffer.clear();
    buffer.push_back(matrix.rows.size());  // Number of row pointers
    buffer.push_back(matrix.cols.size());
    buffer.insert(buffer.end(), matrix.rows.begin(), matrix.rows.end());
    buffer.insert(buffer.end(), matrix.cols.begin(), matrix.cols.end());
    buffer.insert(buffer.end(), matrix.vals.begin(), matrix.vals.end());
}

void deserializeCSR(const std::vector<uint64_t>& buffer, CSR& matrix) {
    size_t idx = 0;
    size_t rows_size = buffer[idx++];
    size_t cols_size = buffer[idx++];
    matrix.rows.assign(buffer.begin() + idx, buffer.begin() + idx + rows_size);
    idx += rows_size;
    matrix.cols.assign(buffer.begin() + idx, buffer.begin() + idx + cols_size);
    idx += cols_size;
    matrix.vals.assign(buffer.begin() + idx, buffer.begin() + idx + cols_size);
}

//used to gather CSR for normal matrix and print/return asuint64_tarray
uint64_t* gather_and_assemble_CSR(int N,uint64_t p,uint64_t rank, const CSR& local_csr) {
    std::vector<uint64_t> serialized_data;
    serializeCSR(local_csr, serialized_data);
    int local_size = serialized_data.size();

    std::vector<int> all_sizes(p);
    MPI_Gather(&local_size, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displacements;
    uint64_t total_size = 0;
        if (rank == 0) {
            displacements.resize(p);
            for (int i = 0; i < p; ++i) {
                displacements[i] = total_size;
                total_size += all_sizes[i];
            }
        }

    std::vector<uint64_t> all_data(total_size);
    MPI_Gatherv(serialized_data.data(), local_size, MPI_UINT64_T,
                all_data.data(), all_sizes.data(), displacements.data(), MPI_UINT64_T, 0, MPI_COMM_WORLD);

    uint64_t* full_matrix = nullptr;
    if (rank == 0) {
        full_matrix = new uint64_t[N * N]();  // Initialize full matrix
       uint64_t offset = 0;
        for (int proc = 0; proc < p; ++proc) {
            CSR temp_csr;
            std::vector<uint64_t> temp_buffer(all_data.begin() + displacements[proc], all_data.begin() + displacements[proc] + all_sizes[proc]);
            deserializeCSR(temp_buffer, temp_csr);
            uint64_t start_row = proc * (N/p);
            for (size_t r = 0; r < temp_csr.rows.size() - 1; ++r) {
                for (int idx = temp_csr.rows[r]; idx < temp_csr.rows[r + 1]; ++idx) {
                   uint64_t col = temp_csr.cols[idx];
                   uint64_t val = temp_csr.vals[idx];
                   full_matrix[((start_row + r) % N) * N + col] = val;
                }
            }
        }
    }
    return full_matrix;
}

void generate_sparse_bonus_T(float s,uint64_t N,uint64_t p,uint64_t rank,uint64_t seed, CSR& transpose, CSR& m) {
    std::vector<uint64_t> rows(1, 0);
    std::vector<uint64_t> cols;
    std::vector<uint64_t> vals;

    std::vector<uint64_t> transpose_cols;
    std::vector<uint64_t> transpose_temp_rows;
    std::vector<uint64_t> transpose_vals;

    srand(time(NULL) + rank + seed + 2); 

   uint64_t val_count = 0;

    for (int r = rank * (N / p); r < (rank + 1) * (N / p); r++) {
        for (int c = 0; c < N; c++) {
            if (rand() % N < s * N) {
               uint64_t rand_value = (rand() % 9) + 1;
                val_count++;
                cols.push_back(c);
                vals.push_back(rand_value);

                transpose_temp_rows.push_back(c); 
                transpose_cols.push_back(r); 
                transpose_vals.push_back(rand_value);
            }
        }
        rows.push_back(val_count);
    }

    // set up transpose
    std::vector<uint64_t> transpose_rows(N + 1, 0);
    std::vector<uint64_t> count(N, 0);
    for (int i = 0; i < transpose_temp_rows.size(); ++i) {
        count[transpose_temp_rows[i]]++;
    }
    for (int i = 1; i <= N; ++i) {
        transpose_rows[i] = transpose_rows[i-1] + count[i-1];
    }

    std::vector<uint64_t> temp_position = transpose_rows;
    std::vector<uint64_t> sorted_transpose_cols(transpose_cols.size());
    std::vector<uint64_t> sorted_transpose_vals(transpose_vals.size());

    for (size_t i = 0; i < transpose_temp_rows.size(); ++i) {
       uint64_t pos = temp_position[transpose_temp_rows[i]]++;
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
    MPI_Type_contiguous(2, MPI_UINT64_T, &csr_type);
    MPI_Type_commit(&csr_type);
    return csr_type;
}

MPI_Datatype create_point_type() {
    MPI_Datatype point;
    int parts[3] = {1, 1, 1};

    MPI_Aint disp[3];
    disp[0] = offsetof(Point, r);
    disp[1] = offsetof(Point, c);
    disp[2] = offsetof(Point, v);

    MPI_Datatype part_types[3] = {MPI_UINT64_T, MPI_UINT64_T, MPI_UINT64_T};

    MPI_Type_create_struct(3, parts, disp, part_types, &point);
    MPI_Type_commit(&point);
    return point;
}

CSR transpose_csr_matrix(CSR& curr_matrix,uint64_t N,uint64_t p, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    CSR transposed;
    MPI_Datatype point_type = create_point_type();

    std::vector<int> sendcounts(p, 0), sdispls(p), recvcounts(p), rdispls(p);
    std::vector<uint64_t> senddata;
    std::vector<Point> temp_store;

   uint64_t currentRow = 0;
   uint64_t numRows = N/p;
   uint64_t numCols = N;
    for (int r = 0; r < numRows; r++) {
       uint64_t rowStart = curr_matrix.rows[r];
       uint64_t rowEnd = (r + 1 < numRows) ? curr_matrix.rows[r + 1] : curr_matrix.cols.size();

       uint64_t currentColIndex = rowStart;

        for (int c = 0; c < numCols; c++) {
            if (currentColIndex < rowEnd && curr_matrix.cols[currentColIndex] == c) {
               uint64_t row = (rank*N/p) + r;
               uint64_t col = curr_matrix.cols[currentColIndex];
               uint64_t val = curr_matrix.vals[currentColIndex];
                sendcounts[col / (N/p)]++;
                Point curr_point = {row, col, val};
                temp_store.push_back(curr_point);
                currentColIndex++;
            }
        }
    }
    std::sort(temp_store.begin(), temp_store.end(), sort_by_col);

    sdispls[0] = 0;
    for (int i = 1; i < p; i++) {
        sdispls[i] = sdispls[i - 1] + sendcounts[i - 1];
    }

    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    rdispls[0] = 0;
   uint64_t recv_buffer_size = recvcounts[0];
    for (int i = 1; i < p; i++) {
        rdispls[i] = rdispls[i - 1] + recvcounts[i - 1];
        recv_buffer_size += recvcounts[i];
    }
    std::vector<Point> points(recv_buffer_size);

    MPI_Alltoallv(temp_store.data(), sendcounts.data(), sdispls.data(), point_type, points.data(), recvcounts.data(), rdispls.data(), point_type, MPI_COMM_WORLD);

    for(auto& point : points){
       uint64_t temp = point.r;
        point.r = point.c;
        point.c = temp;
    }

    std::sort(points.begin(), points.end(), sort_by_row);

    CSR result = convertToCSR(points, N/p);
    CSR cscresult = convertToCSC(points, N);

    MPI_Type_free(&point_type);
    return cscresult;
}


void print_matrix_all(uint64_t* matrix, uint64_t* matrix2, uint64_t* matrix3, char* outfile,uint64_t dim1,uint64_t dim2){
    FILE * fp = fopen(outfile, "w");
    for(int i = 0; i < dim1; i++){
        for(int j = 0; j < dim2; j++){
            fprintf(fp, "%llu ", matrix[i*dim2 + j]);
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "\n");
    for(int i = 0; i < dim1; i++){
        for(int j = 0; j < dim2; j++){
            fprintf(fp, "%llu ", matrix2[i*dim2 + j]);
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "\n");
    for(int i = 0; i < dim1; i++){
        for(int j = 0; j < dim2; j++){
            fprintf(fp, "%llu ", matrix3[i*dim2 + j]);
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "\n");
}

void printMatrix(uint64_t* matrix,uint64_t rows,uint64_t cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

void mat_mul_csr(const CSR& A, const CSR& B, uint64_t* C, uint64_t rowsA, uint64_t colsB, uint64_t N, uint64_t iter, uint64_t rank) {
    for (uint64_t i = 0; i < rowsA; ++i) {
        for (uint64_t j = 0; j < colsB; ++j) {
            uint64_t sum = 0;
            for (uint64_t k = A.rows[i]; k < A.rows[i + 1]; ++k) {
                uint64_t a_col = A.cols[k];
                uint64_t a_val = A.vals[k];
                for (uint64_t l = B.rows[j]; l < B.rows[j + 1]; ++l) {
                    uint64_t b_row = B.cols[l];
                    if (b_row == a_col) {
                        uint64_t b_val = B.vals[l];
                        sum += a_val * b_val;
                        break;
                    }
                }
            }
            C[i * N + j] += sum;
        }
    }
}


uint64_t* mat_mul_serial(uint64_t* first, uint64_t* second,uint64_t N){
    uint64_t* global_C = new uint64_t[N*N];
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

   uint64_t N = stoi(argv[1]);  // size
    double s = stod(argv[2]);  // sparsoty
   uint64_t pf = stoi(argv[3]);  // printing flag
    char* out_file = argv[4];  // Ofile name

    CSR tranA;
    CSR A;
    CSR tranB;
    CSR B;

    generate_sparse_bonus_T(s, N, p, rank, 0, tranA, A);
    generate_sparse_bonus_T(s, N, p, rank, 1, tranB, B);

   uint64_t C_size = N * N / p;
    uint64_t* C = new uint64_t[C_size];
    for (int i = 0; i < C_size; i++) {
        C[i] = 0;
    }

    uint64_t* mat_A = gather_and_assemble_CSR(N, p, rank, A);
    uint64_t* mat_B = gather_and_assemble_CSR(N, p, rank, B);

    int src, dst;
    MPI_Cart_shift(comm, 0, 1, &src, &dst);
    
    double start_time;
    if (rank == 0) {
        start_time = MPI_Wtime();
    }
    tranB = transpose_csr_matrix(B, N, p, comm);

    CSR new_tranB;

    for (int iter = 0; iter < p; iter++) {
        mat_mul_csr(A, tranB, C, N/p, N/p, N, iter, rank);

        vector<uint64_t> send_buffer;
        serializeCSR(tranB, send_buffer);
        int send_size = send_buffer.size();

        int recv_size;
        MPI_Sendrecv(&send_size, 1, MPI_INT, dst, 0, &recv_size, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        vector<uint64_t> recv_buffer(recv_size);
        MPI_Sendrecv(send_buffer.data(), send_size, MPI_UINT64_T, dst, 0, recv_buffer.data(), recv_size, MPI_UINT64_T, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        deserializeCSR(recv_buffer, new_tranB);

        tranB = new_tranB; 

        MPI_Barrier(comm);
    }

    double end_time;

    if (rank == 0) {
        end_time = MPI_Wtime();
        double time_taken = end_time - start_time;
        cout << "Time: " <<time_taken << endl;
    }

    uint64_t* global_C = new uint64_t[N*N];
    for (int i = 0; i < N*N; i++) {
        global_C[i] = 0;
    }

    MPI_Gather(C, N*N/p , MPI_UINT64_T, global_C , N*N/p , MPI_UINT64_T, 0, MPI_COMM_WORLD);

    if (pf == 1) {
        if(rank == 0){
            printMatrix(mat_mul_serial(mat_A, mat_B, N), N, N);
            print_matrix_all(mat_A, mat_B, global_C, out_file, N, N);
        }
    }
    MPI_Finalize();
    return 0;
} 
