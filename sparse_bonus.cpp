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
    int r;
    int c;
    int v;
};

struct CSR {
    vector<int> rows;
    vector<int> cols;
    vector<int> vals;
};

CSR convertToCSC(const CSR& csr) {

    int numRows = *std::max_element(csr.rows.begin(), csr.rows.end());
    int numCols = csr.rows.size() > 1 ? csr.rows[1] : 0;

    // Initialize CSC matrix data structures
    CSR csc;
    csc.rows.resize(numCols + 1, 0);
    std::vector<std::vector<std::pair<int, int>>> temp(numRows + 1);

    // Sort and store non-zero elements per column
    for (int i = 0; i < csr.cols.size(); ++i) {
        temp[csr.cols[i]].emplace_back(csr.rows[i], csr.vals[i]);
    }

    // Compute row offsets and sort column indices within each row
    int offset = 0;
    for (int i = 0; i < numCols; ++i) {
        std::sort(temp[i].begin(), temp[i].end());
        csc.rows[i] = offset;
        offset += temp[i].size();
    }
    csc.rows[numCols] = offset;

    // Flatten sorted column indices and values into cols and vals arrays
    int numNonZeros = offset;
    csc.cols.resize(numNonZeros);
    csc.vals.resize(numNonZeros);

    int idx = 0;
    for (int i = 0; i < numCols; ++i) {
        for (const auto& pair : temp[i]) {
            csc.cols[idx] = pair.first;
            csc.vals[idx] = pair.second;
            ++idx;
        }
    }

    return csc;
}

CSR convertToCSC(const std::vector<Point>& points, int num_cols, int r) {
        CSR csc;
    if (points.empty() || num_cols <= 0) {
        return csc;
    }

    // Initialize the rows vector with size num_cols + 1 for column pointers
    csc.rows.resize(num_cols + 1, 0);

    // Count non-zero elements per column
    for (const auto& point : points) {
        csc.rows[point.c]++;
    }

    // Convert counts to actual indices in rows
    for (int i = 1; i <= num_cols; ++i) {
        csc.rows[i] += csc.rows[i - 1];
    }

    // Resize cols and vals arrays
    int numNonZeros = points.size();
    csc.cols.resize(numNonZeros);
    csc.vals.resize(numNonZeros);

    // Fill cols and vals
    std::vector<int> colIndices(num_cols, 0);
    for (const auto& point : points) {
        int col = point.c;
        int idx = csc.rows[col] + colIndices[col];
        csc.cols[idx] = point.r;
        csc.vals[idx] = point.v;
        colIndices[col]++;
    }

    // Rearrange rows and vals to ensure row indices are in ascending order within each column
    for (int i = 0; i < num_cols; ++i) {
        int start = csc.rows[i];
        int end = csc.rows[i + 1];
        if (start < end) {
            std::vector<int> tempRowIndices(end - start);
            std::vector<int> tempValues(end - start);
            for (int j = start; j < end; ++j) {
                tempRowIndices[j - start] = csc.cols[j];
                tempValues[j - start] = csc.vals[j];
            }
            std::sort(tempRowIndices.begin(), tempRowIndices.end());
            for (int j = start; j < end; ++j) {
                csc.cols[j] = tempRowIndices[j - start];
                csc.vals[j] = tempValues[j - start];
            }
        }
    }

    return csc;
}

bool pointComparator(const Point& a, const Point& b) {
    if (a.c == b.c)
        return a.r < b.r;
    return a.c < b.c;
}

CSR convertToCSC(const std::vector<Point>& points, int cols) {
    CSR csc;
    if (points.empty()) {
        return csc;
    }

    // Determine the number of columns
    int maxCol = 0;
    for (const auto& point : points) {
        if (point.c > maxCol) {
            maxCol = point.c;
        }
    }

    // Sort points by column index, and then row index
    std::vector<Point> sortedPoints = points;
    std::sort(sortedPoints.begin(), sortedPoints.end(), pointComparator);

    // Initialize column pointers
    csc.cols.resize(cols + 1, 0); // +2 to include the past-the-end index for the last column

    // Process each point to fill rows and vals
    for (const auto& point : sortedPoints) {
        if (point.c >= csc.cols.size()) {
            csc.cols.resize(point.c + 1, 0);
        }
        csc.rows.push_back(point.r);
        csc.vals.push_back(point.v);
    }

    // Create column pointers in 'cols'
    int currentColumn = -1;
    for (size_t i = 0; i < sortedPoints.size(); ++i) {
        while (currentColumn < sortedPoints[i].c) {
            currentColumn++;
            csc.cols[currentColumn] = i;
        }
    }
    csc.cols[currentColumn + 1] = sortedPoints.size(); // set the past-the-end index for the last column

    // Fill the remaining columns' pointers if any
    for (int i = currentColumn + 2; i <= cols; i++) {
        csc.cols[i] = sortedPoints.size();
    }
    vector<int> temp = csc.rows;
    csc.rows = csc.cols;
    csc.cols = temp;
    return csc;
}

// CSR convertToCSC(const vector<Point>& points, int num_cols) {
//     CSR csr;
//     if (points.empty()) {
//         return csr;
//     }

//     // Initialize the rows vector with size num_rows + 1 for row pointers
//     csr.rows.resize(num_rows + 1, 0);

//     // Reserve space assuming a dense distribution of non-zeros (optional, for efficiency)
//     csr.cols.reserve(points.size());
//     csr.vals.reserve(points.size());

//     // Fill cols and vals
//     for (auto& point : points) {
//         csr.cols.push_back(point.c);
//         csr.vals.push_back(point.v);
//     }

//     // Compute the row pointers
//     for (auto& point : points) {
//         csr.rows[(point.r%num_rows + 1)]++;
//     }

//     // Convert counts to actual indices
//     for (int i = 1; i <= num_rows; ++i) {
//         csr.rows[i] += csr.rows[i - 1];
//     }

//     return csr;
// }

CSR convertToCSCFinal(const vector<Point>& points, int num_cols) {
    CSR csr;
    if (points.empty()) {
        return csr;
    }

    // Initialize the rows vector with size num_rows + 1 for row pointers
    csr.rows.resize(num_cols + 1, 0);

    // Reserve space assuming a dense distribution of non-zeros (optional, for efficiency)
    csr.cols.reserve(points.size());
    csr.vals.reserve(points.size());

    // Fill cols and vals
    for (auto& point : points) {
        csr.cols.push_back(point.r);
        csr.vals.push_back(point.v);
    }

    // Compute the row pointers
    for (auto& point : points) {
        csr.rows[(point.c%num_cols + 1)]++;
    }

    // Convert counts to actual indices
    for (int i = 1; i <= num_cols; ++i) {
        csr.rows[i] += csr.rows[i - 1];
    }

    return csr;
}

CSR convertToCSR(const vector<Point>& points, int num_rows) {
    CSR csr;
    if (points.empty()) {
        return csr;
    }

    // Initialize the rows vector with size num_rows + 1 for row pointers
    csr.rows.resize(num_rows + 1, 0);

    // Reserve space assuming a dense distribution of non-zeros (optional, for efficiency)
    csr.cols.reserve(points.size());
    csr.vals.reserve(points.size());

    // Fill cols and vals
    for (auto& point : points) {
        csr.cols.push_back(point.c);
        csr.vals.push_back(point.v);
    }

    // Compute the row pointers
    for (auto& point : points) {
        csr.rows[(point.r%num_rows + 1)]++;
    }

    // Convert counts to actual indices
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

//column numbers can be out of order, so not prining properly, e.g. 7 and 6 as col then skips 6 and puts a 0 in it.
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

//used to gather CSR for normal matrix and print/return as int array
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

//used to gather CSR for transpose and print/return as int array
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
            printf("CSR Form: \n");
            printCSRForm(temp_csr);
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
//generate matrix as CSR with transpose
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

MPI_Datatype create_point_type() {
    MPI_Datatype point;
    int parts[3] = {1, 1, 1};

    MPI_Aint disp[3];
    disp[0] = offsetof(Point, r);
    disp[1] = offsetof(Point, c);
    disp[2] = offsetof(Point, v);

    MPI_Datatype part_types[3] = {MPI_INT, MPI_INT, MPI_INT};

    MPI_Type_create_struct(3, parts, disp, part_types, &point);
    MPI_Type_commit(&point);
    return point;
}

void print_points(const vector<Point>& points) {
    cout << "Points in COO format:" << endl;
    for (const Point& point : points) {
        cout << "Row: " << point.r << ", Column: " << point.c << ", Value: " << point.v << endl;
    }
}

CSR transpose_csr_matrix(CSR& curr_matrix, int N, int p, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    CSR transposed;
    MPI_Datatype point_type = create_point_type();
    // Prepare counts and displacements for MPI communication
    std::vector<int> sendcounts(p, 0), sdispls(p), recvcounts(p), rdispls(p);

    // Prepare data to send
    std::vector<int> senddata; // Flatten data for sending
    // for(int i = 0; i < N/p; i++){
    //     int start_row = i;
    //     int end_row = i+1;
    //     for (int row = start_row; row < end_row; ++row) {
    //         for (int idx = curr_matrix.rows[row]; idx < curr_matrix.rows[row + 1]; ++idx) {
    //             senddata.push_back(curr_matrix.cols[idx]);
    //             senddata.push_back(curr_matrix.vals[idx]);
    //         }
    //     }
    // }
    std::vector<Point> temp_store;

    int currentRow = 0;
    int numRows = N/p;
    int numCols = N;
    for (int r = 0; r < numRows; r++) {
        int rowStart = curr_matrix.rows[r];
        int rowEnd = (r + 1 < numRows) ? curr_matrix.rows[r + 1] : curr_matrix.cols.size();

        int currentColIndex = rowStart;

        for (int c = 0; c < numCols; c++) {
            if (currentColIndex < rowEnd && curr_matrix.cols[currentColIndex] == c) {
                int row = (rank*N/p) + r;
                int col = curr_matrix.cols[currentColIndex];
                int val = curr_matrix.vals[currentColIndex];
                sendcounts[col / (N/p)]++;
                Point curr_point = {row, col, val};
                temp_store.push_back(curr_point);
                currentColIndex++;
            }
        }
    }
    std::sort(temp_store.begin(), temp_store.end(), sort_by_col);
    // if(rank == 0){
    //     print_points(temp_store);
    // }
    //count how many columns need to be sent and only send them
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
    std::vector<Point> points(recv_buffer_size);

    MPI_Alltoallv(temp_store.data(), sendcounts.data(), sdispls.data(), point_type, points.data(), recvcounts.data(), rdispls.data(), point_type, MPI_COMM_WORLD);

    // sort(points.begin(), points.end(), [](const Point& a, const Point& b) {
    //     if (a.r == b.r) return a.c < b.c;
    //     return a.r < b.r;
    // });

    for(auto& point : points){
        int temp = point.r;
        point.r = point.c;
        point.c = temp;
    }

    std::sort(points.begin(), points.end(), sort_by_row);
    // if(rank == 1){
    //     printf("POINTS: \n");
    //     print_points(points);
    // }

    CSR result = convertToCSR(points, N/p);
    CSR cscresult = convertToCSC(points, N);
    // if(rank == 0){
    //     printf("\n");
    //     printCSRForm(result);
    //     printf("\n");
    //     printCSRForm(cscresult);
    // }
    // Initialize row pointers and fill column indices and values
    // if(rank == 0){
    //     printVector(senddata);
    //     printf("\n");
    //     printCSRForm(curr_matrix);
    // }

    // Communicate the amount of data to be received
    // MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);

    // Calculate displacements for received data
    // rdispls[0] = 0;
    // std::vector<int> recvdata;
    // int totalRecv = 0;
    // for (int i = 0; i < p; ++i) {
    //     rdispls[i] = (i == 0) ? 0 : rdispls[i - 1] + recvcounts[i - 1];
    //     totalRecv += recvcounts[i];
    // }
    // recvdata.resize(totalRecv);

    // Perform all-to-all communication of CSR data
    // MPI_Alltoallv(senddata.data(), sendcounts.data(), sdispls.data(), MPI_INT,
    //               recvdata.data(), recvcounts.data(), rdispls.data(), MPI_INT, comm);

    // if(rank == 0){
    //     printf("Received: \n");
    //     printVector(recvdata);
    // }

    // // Rebuild the local transposed matrix from recvdata
    // CSR transposed;
    // transposed.rows.resize(N / p + 1, 0);
    // int currentVal = 0;
    // for (int i = 0; i < totalRecv; i += 2) {
    //     int col = recvdata[i]; // Column index in the transposed matrix is the row index
    //     int val = recvdata[i + 1];
    //     transposed.cols.push_back(col);
    //     transposed.vals.push_back(val);
    //     transposed.rows[col % (N / p) + 1]++;
    // }
    // // Compute prefix sums to finalize rows
    // std::partial_sum(transposed.rows.begin(), transposed.rows.end(), transposed.rows.begin());
    // return transposed;
    MPI_Type_free(&point_type);
    return cscresult;
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
void printMatrix(int* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

// void mat_mul_csr(const CSR& A, const CSR& B, int* C, int rowsA, int rowsB, int N, int iter, int rank) {
//     std::memset(C, 0, rowsA * rowsB * sizeof(int));

//     for (int i = 0; i < rowsA; ++i) {
//         for (int k = 0; k < rowsB; ++k) {
//             int sum = 0;
//             for (int j = A.rows[i]; j < A.rows[i + 1]; ++j) {
//                 int colA = A.cols[j];
//                 int valA = A.vals[j];

//                 for (int m = B.rows[k]; m < B.rows[k + 1]; ++m) {
//                     if (B.cols[m] == colA) {
//                         int valB = B.vals[m];
//                         sum += valA * valB;
//                     }
//                 }
//             }
//             C[(i % rowsB)*(N) + ((k + (rank + 1)*rowsB)%N)] = sum;
//         }
//     }
//     // if(rank == 1){
//     //     // printf("Print Rows: \n");
//     //     // printMatrix(C, rowsA, sizeof(C));
//     // }
// }

// void mat_mul_csr(const CSR& A, const CSR& B, int* C, int rowsA, int rowsB, int N, int iter, int rank) {
//     std::memset(C, 0, rowsA * rowsB * sizeof(int));

//     for (int i = 0; i < rowsA; ++i) {
//         for (int k = 0; k < rowsB; ++k) {
//             int sum = 0;
//             for (int j = A.rows[i]; j < A.rows[i + 1]; ++j) {
//                 int colA = A.cols[j];
//                 int valA = A.vals[j];

//                 for (int m = B.rows[k]; m < B.rows[k + 1]; ++m) {
//                     if (B.cols[m] == colA) {
//                         int valB = B.vals[m];
//                         sum += valA * valB;
//                         break;  // Break once a matching element is found in B
//                     } else if (B.cols[m] > colA) {
//                         break;  // No need to search further if B.cols[m] exceeds colA
//                     }
//                 }
//             }
//             C[i * N + k] = sum;
//         }
//     }
//     // if(rank == 1){
//     //     printf("Print Rows: \n");
//     //     printMatrix(C, rowsA, sizeof(C));
//     // }
// }

void mat_mul_dot_product_csr(const vector<int>& a_row, const vector<int>& a_col, const vector<int>& a_val,
                             const vector<int>& b_row, const vector<int>& b_col, const vector<int>& b_val,
                             int* c, int N, int p) {
    // Convert input matrices to CSR format if not already in CSR format
    // Sort A and B by row indices if not already sorted (CSR matrices are typically sorted)
    
    // Iterate through each row of matrix A
    for (int i = 0; i < N; ++i) {
        // Iterate through the non-zero elements of the i-th row of matrix A
        for (int j = a_row[i]; j < a_row[i + 1]; ++j) {
            int a_col_idx = a_col[j]; // Column index of non-zero element in matrix A
            int a_val_ij = a_val[j];  // Value of non-zero element in matrix A

            // Iterate through the non-zero elements of the a_col_idx-th row of matrix B
            for (int k = b_row[a_col_idx]; k < b_row[a_col_idx + 1]; ++k) {
                int b_col_idx = b_col[k]; // Column index of non-zero element in matrix B
                int b_val_ik = b_val[k];  // Value of non-zero element in matrix B

                // Update the corresponding element of the result matrix c
                int idx = ((i % (N/p)) * N) + b_col_idx;
                c[idx] += a_val_ij * b_val_ik;
            }
        }
    }
    // if(rank == 1){
    //     printf("Print Rows: \n");
    //     printMatrix(C, rowsA, sizeof(C));
    // }
}

void mat_mul_csr(const CSR& A, const CSR& B, int* C, int rowsA, int rowsB, int N, int iter, int rank) {
    // std::memset(C, 0, rowsA * N * sizeof(int));  // Clearing result matrix C
    // if(rank == 3){
    //     printCSRForm(A);
    //     printf("\n");
    //     printCSRForm(B);
    //     printf("\n");
    // }

    for (int i = 0; i < rowsA; ++i) {  // Iterate over rows of A
        for (int j = A.rows[i]; j < A.rows[i + 1]; ++j) {  // Non-zeros in row i of A
            int a_col = A.cols[j];
            int a_val = A.vals[j];

            // if (rank == 3) {  // Debugging output for first process
            //     printf("Processing row %d, col %d with value %d\n", i, a_col, a_val);
            // }

            for (int k = B.rows[a_col]; k < B.rows[a_col + 1]; ++k) {
                int b_col = B.cols[k];
                int b_val = B.vals[k];
                C[i * N + b_col] += a_val * b_val;

                // if (rank == 3) {  // More debugging output
                //     printf("Multiplying A[%d][%d]=%d by B[%d][%d]=%d, adding to C[%d][%d]\n",
                //            i, a_col, a_val, a_col, b_col, b_val, i, b_col);
                // }
            }
        }
    }
}


// void mat_mul_csr(const CSR& A, const CSR& B, int* C, int rowsA, int rowsB, int N, int iter, int rank) {
//     std::memset(C, 0, rowsA * rowsB * sizeof(int));

//     for (int i = 0; i < rowsA; ++i) {
//         for (int k = 0; k < rowsB; ++k) {
//             int sum = 0;
//             int ptrA = A.rows[i];
//             int ptrB = B.rows[k];

//             // Iterate through the row of A and the column of B simultaneously
//             while (ptrA < A.rows[i + 1] && ptrB < B.rows[k + 1]) {
//                 int colA = A.cols[ptrA];
//                 int colB = B.cols[ptrB];
                
//                 if (colA == colB) {
//                     sum += A.vals[ptrA] * B.vals[ptrB];
//                     ptrA++;
//                     ptrB++;
//                 } else if (colA < colB) {
//                     ptrA++;
//                 } else {
//                     ptrB++;
//                 }
//             }
//             C[(i % rowsB)*(N) + ((k + (iter + rank + 1)*rowsB)%N)] = sum;
//         }
//     }
//     // if(rank == 1){
//     //     printf("Print Rows: \n");
//     //     printMatrix(C, rowsA, sizeof(C));
//     // }
// }

// void mat_mul_csr(const CSR& A, const CSR& B, int* C, int rowsA, int rowsB, int N, int iter, int rank) {
//     std::memset(C, 0, rowsA * rowsB * sizeof(int));

//     for (int i = 0; i < rowsA; ++i) {
//         for (int k = 0; k < rowsB; ++k) {
//             int sum = 0;
//             int ptrA = A.rows[i];
//             int ptrB = B.rows[k];

//             // Binary search for matching columns in matrix B
//             while (ptrA < A.rows[i + 1] && ptrB < B.rows[k + 1]) {
//                 int colA = A.cols[ptrA];
//                 int colB = B.cols[ptrB];

//                 if (colA == colB) {
//                     sum += A.vals[ptrA] * B.vals[ptrB];
//                     ptrA++;
//                     ptrB++;
//                 } else if (colA < colB) {
//                     ptrA++;
//                 } else {
//                     ptrB = std::lower_bound(B.cols.begin() + ptrB, B.cols.begin() + B.rows[k + 1], colA) - B.cols.begin();
//                 }
//             }
//             C[(i % rowsB)*(N) + ((k + (rank + 1)*rowsB)%N)] = sum;
//         }
//     }
//     // if(rank == 1){
//     //     printf("Print Rows: \n");
//     //     printMatrix(C, rowsA, sizeof(C));
//     // }
// }

// void mat_mul_csr(const CSR& A, const CSR& B, int* C, int rowsA, int rowsB, int N, int iter, int rank) {
//     std::memset(C, 0, rowsA * rowsB * sizeof(int));

//     for (int i = 0; i < rowsA; ++i) {
//         for (int k = 0; k < rowsB; ++k) {
//             int sum = 0;
//             std::vector<int> colB_vals;

//             // Preload values of matrix B column k
//             for (int m = B.rows[k]; m < B.rows[k + 1]; ++m) {
//                 colB_vals.push_back(B.vals[m]);
//             }

//             // Perform multiplication
//             for (int j = A.rows[i]; j < A.rows[i + 1]; ++j) {
//                 int colA = A.cols[j];
//                 int valA = A.vals[j];

//                 // Search for colA in matrix B's column k
//                 auto it = std::lower_bound(B.cols.begin() + B.rows[k], B.cols.begin() + B.rows[k + 1], colA);
//                 if (it != B.cols.begin() + B.rows[k + 1] && *it == colA) {
//                     int idx = it - B.cols.begin();
//                     sum += valA * colB_vals[idx - B.rows[k]];
//                 }
//             }

//             // Update index into C using the original formula
//             C[(i % (rowsB)) * N + (k + (rank + 1) * rowsB) % N] = sum;
//         }
//     }
// }

// void mat_mul_csr(const CSR& A, const CSR& B, int* C, int rowsA, int rowsB, int N, int iter, int rank) {
//     std::memset(C, 0, rowsA * rowsB * sizeof(int));

//     // OpenMP parallelization for outer loop
//     #pragma omp parallel for
//     for (int i = 0; i < rowsA; ++i) {
//         for (int k = 0; k < rowsB; ++k) {
//             int sum = 0;
//             for (int j = A.rows[i]; j < A.rows[i + 1]; ++j) {
//                 int colA = A.cols[j];
//                 int valA = A.vals[j];
//                 for (int m = B.rows[k]; m < B.rows[k + 1]; ++m) {
//                     if (B.cols[m] == colA) {
//                         sum += valA * B.vals[m];
//                         break;
//                     } else if (B.cols[m] > colA) {
//                         break; // No need to search further if B.cols[m] exceeds colA
//                     }
//                 }
//             }
//             // Update index into C using the original formula
//             C[(i % rowsB) * N + (k + (rank + 1) * rowsB) % N] = sum;
//         }
//     }
// }


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

    int C_size = N * N / p;
    int* C = new int[C_size];
    for (int i = 0; i < C_size; i++) {
        C[i] = 0;
    }

    // if(rank == 0){
    //     printCSRMatrix(tranB, N/p, N);
    //     printf("\n");
    //     printCSRForm(tranB);
    //     printf("\n");
    //     printCSRMatrix(B, N/p, N);
    //     printf("\n");
    //     printCSRForm(B);
    // }

    // int* mat_A = gather_and_return_CSR_matrix(B, N, p, rank);

    // int* mat_B = gather_and_return_CSR_matrix(tranB, N, p, rank);

    int* mat_A = gather_and_assemble_CSR(N, p, rank, A);
    int* mat_B = gather_and_assemble_CSR(N, p, rank, B);
    // int* mat_tranB = gather_and_assemble_CSR(N, p, rank, tranB);

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
        // time_t start = time(NULL);
        // while (time(NULL) - start < 2 * rank);
        // printf("%d\n", rank);
        // MPI_Barrier(comm);
        // if(rank == 1){
        //     printf("Mult \n");
        //     printMatrix(C, N/p, N);
        //     printf("\n: ");
        // }

        vector<int> send_buffer;
        serializeCSR(tranB, send_buffer);
        int send_size = send_buffer.size();

        int recv_size;
        MPI_Sendrecv(&send_size, 1, MPI_INT, dst, 0, &recv_size, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        vector<int> recv_buffer(recv_size);
        MPI_Sendrecv(send_buffer.data(), send_size, MPI_INT, dst, 0, recv_buffer.data(), recv_size, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

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
