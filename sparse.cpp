#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

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

vector<Point> generate_sparse(float s, int N, int p, int rank, int seed) {
    vector<Point> m;
    int count = 0;
    int start_row = rank * (N / p);
    int end_row = (rank + 1) * (N / p);
    srand(time(NULL) + rank + seed); 
    for (int c = 0; c < N; c++) {
        for (int r = start_row; r < end_row; r++) {
            int randd = rand() % N;
            if (randd < s * N) {
                int rand_value = (rand() % 10);
                Point point = {r, c, rand_value};
                m.push_back(point);
                count += 1;
            }
        }
    }
    return m;
}

//generates matrix in CSR
CSR generate_sparse_bonus(float s, int N, int p, int rank, int seed) {
    CSR m;
    vector<int> rows;
    vector<int> cols;
    vector<int> vals;
    rows.push_back(0);

    int start_row = rank * (N / p);
    int end_row = (rank + 1) * (N / p);
    srand(time(NULL) + rank + seed); 
    int val_count = 0;
    for (int r = start_row; r < end_row; r++) {
        for (int c = 0; c < N; c++){
            int randd = rand() % N;
            if (randd < s * N) {
                int rand_value = (rand() % 10);
                val_count += 1;
                cols.push_back(c);
                vals.push_back(rand_value);
            }
        }
        rows.push_back(val_count);
    }

    return m;
}

//generates matrix with the transpose as well (both in CSR)
CSR generate_sparse_bonus_T(float s, int N, int p, int rank, int seed, CSR& transpose) {
    CSR m;
    vector<int> rows;
    vector<int> cols;
    vector<int> vals;

    vector<int> csc_rows;
    vector<int> csc_cols;
    vector<int> csc_vals;

    vector<int> count_all(N, 0);
    rows.push_back(0);
    csc_cols.push_back(0);

    int start_row = rank * (N / p);
    int end_row = (rank + 1) * (N / p);
    srand(time(NULL) + rank + seed); 
    int val_count = 0;
    for (int r = start_row; r < end_row; r++) {
        for (int c = 0; c < N; c++){
            int randd = rand() % N;
            if (randd < s * N) {
                int rand_value = (rand() % 10);
                val_count += 1;
                cols.push_back(c);
                vals.push_back(rand_value);
                csc_vals.push_back(rand_value);
                csc_rows.push_back(r);
                count_all[c] += 1;
            }
        }
        rows.push_back(val_count);
    }

    int total_count = 0;
    for(int i = 0; i < N; i++){
        csc_cols[i] = total_count;
        total_count += count_all[i];
    }
    csc_cols.push_back(total_count);

    transpose.rows = csc_cols;
    transpose.cols = csc_rows;
    transpose.vals = csc_vals;

    m.rows = rows;
    m.cols = cols;
    m.vals = vals;

    return m;
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




vector<Point> transpose_matrix(std::vector<Point>& matrix, int N, int p) {
    MPI_Datatype point_type = create_point_type();
    std::vector<int> sendcounts(p, 0), sdispls(p, 0), recvcounts(p, 0), rdispls(p, 0);

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
    std::vector<Point> transposed_matrix(recv_buffer_size);

    MPI_Alltoallv(matrix.data(), sendcounts.data(), sdispls.data(), point_type, transposed_matrix.data(), recvcounts.data(), rdispls.data(), point_type, MPI_COMM_WORLD);

    MPI_Type_free(&point_type);
    return transposed_matrix;
}

int* gather_and_return_matrix(const std::vector<Point>& curr_matrix, int N, int p, int rank) {
    MPI_Datatype point_type = create_point_type();

    int curr_size = curr_matrix.size();
    std::vector<int> sizes(p);
    MPI_Gather(&curr_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displs(p, 0);
    if (rank == 0) {
        for (int i = 1; i < p; ++i) {
            displs[i] = displs[i - 1] + sizes[i - 1];
        }
    }

    vector<Point> all_points(0);

    if(rank == 0){
        all_points.resize(displs[p - 1] + sizes[p - 1]);
    }

    MPI_Gatherv(curr_matrix.data(), curr_size, point_type, all_points.data(), sizes.data(), displs.data(), point_type, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        int* matrix = new int[N*N];
        for(int i = 0; i < N*N; i++){
            matrix[i] = 0;
        }
        for (Point& point : all_points) {
            matrix[(point.r)*N + point.c] = point.v;
        }
        return matrix;
    }
    MPI_Type_free(&point_type);
    return NULL;
}


// for testing, delete later
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

// for testing, delete later
void printMatrix(int* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}


void mat_mul_naive(vector<Point>& a, vector<Point>& b, int* c, int N, int p) {
    for (Point& pb : b) {
        for (Point& pa : a) {
            if (pa.c == pb.c) {
                int new_val = pa.v * pb.v;
                int idx = ((pa.r % (N/p))* N) + pb.r;
                c[idx] += new_val;
            }
        }
    }
}


void mat_mul_dot_product(vector<Point>& a, vector<Point>& b, int* c, int N, int p) {
    // sort A by row
    // sort B by col
    // for all A, multiply with rows in B where B_r = A_c, result goes into ((pa.r % (N/p))* N) + pb.r;
    std::sort(a.begin(), a.end(), sort_by_col);
    std::sort(b.begin(), b.end(), sort_by_row);

    int bpointer = 0;
    int bstart = 0;
    int old_a_col = -1;

    for (Point &pa: a) {
        if (old_a_col == pa.c) bpointer = bstart;
        while (bpointer < b.size() && b[bpointer].r < pa.c) {
            bpointer++;
        }
        bstart = bpointer;
        while (bpointer < b.size() && pa.c == b[bpointer].r) {
            int new_val = pa.v * b[bpointer].v;
            int idx = ((pa.r % (N/p))* N) + b[bpointer].c;
            c[idx] += new_val;
            bpointer++;
        }
        old_a_col = pa.c;
    }
}

//transpose matrix first, so b's rows are original b matrix columns
void mat_mul_bonus(CSR a, CSR b, int* c, int N, int p){
    vector<int> a_rows = a.rows;
    vector<int> b_rows = b.rows;    

    vector<int> a_cols = a.cols;
    vector<int> b_cols = b.cols;

    vector<int> a_vals = a.vals;
    vector<int> b_vals = b.vals;

    int a_col = 0;
    int b_col = 0;
    for(int i = 1; i < a_rows.size(); i++){
        for(int j = 1; j < b_rows.size(); j++){
            int counter_a = 0;
            int counter_b = 0;
            while(counter_a < a_rows[i] - a_rows[i-1] && counter_b < b_rows[i] - b_rows[i-1]){
                int a_row = i;
                int a_col = a_cols[a_rows[i] + counter_a];
                int b_col = b_cols[b_rows[i] + counter_b];
                if(a_col < b_col){
                    counter_a += 1;
                } else if (b_col < a_col){
                    counter_b += 1;
                } else {
                    c[i*N + j] += a_vals[a_rows[i] + counter_a]*b_vals[b_rows[i] + counter_a];    
                    counter_a += 1;   
                    counter_b += 1;      
                }

            }
        }
    }
}

void mat_mul_outer_product(vector<Point>& a, vector<Point>& b, int* c, int N, int p) {
    // sort A by row
    // sort B by col
    // for all A, multiply with rows in B where B_r = A_c, result goes into ((pa.r % (N/p))* N) + pb.r;
    std::sort(a.begin(), a.end(), sort_by_col);
    std::sort(b.begin(), b.end(), sort_by_row);

    int bpointer = 0;
    int bstart = 0;
    int old_a_col = -1;

    for (Point &pa: a) {
        if (old_a_col == pa.c) bpointer = bstart;
        while (bpointer < b.size() && b[bpointer].r < pa.c) {
            bpointer++;
        }
        bstart = bpointer;
        while (bpointer < b.size() && pa.c == b[bpointer].r) {
            int new_val = pa.v * b[bpointer].v;
            int idx = ((pa.r % (N/p))* N) + b[bpointer].c;
            c[idx] += new_val;
            bpointer++;
        }
        old_a_col = pa.c;
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
    
    vector<Point> A = generate_sparse(s, N, p, rank, 0);
    vector<Point> B = generate_sparse(s, N, p, rank, 1);

    // CSR tranB;
    // CSR A = generate_sparse_bonus(s, N, p, rank, 0);
    // CSR B = generate_sparse_bonus_T(s, N, p, rank, 1, &tranB);

    int C_size = N * N / p;
    int* C = new int[C_size];
    for (int i = 0; i < C_size; i++) {
        C[i] = 0;
    }

    vector<Point> oldB = B;

    int src, dst;
    MPI_Cart_shift(comm, 0, 1, &src, &dst);

    int* mat_A = gather_and_return_matrix(A, N, p, rank);
    int* mat_B = gather_and_return_matrix(oldB, N, p, rank);

    double start_time;
    if (rank == 0) {
        start_time = MPI_Wtime();
    }

    //normal implemnetation
    vector<Point> tranB = transpose_matrix(B, N, p);

    // if(rank == 0){ // for testing
    //     if(mat_A != NULL || mat_B != NULL){
    //         printMatrix(mat_mul_serial(mat_A, mat_B, N), N, N);
    //     }
        
    // }

    MPI_Datatype point_type = create_point_type();
    for (int iter = 0; iter < p; iter++) {

        // mat_mul_bonus(A, tranB, C, N, p);
        mat_mul_dot_product(A, tranB, C, N, p);

        int send = tranB.size();
        int recv;
        MPI_Sendrecv(&send, 1, MPI_INT, dst, 0, &recv, 1, MPI_INT, src, 0, comm, MPI_STATUS_IGNORE);
        vector<Point> rec_buffer(recv);
        MPI_Sendrecv(tranB.data(), send, point_type, dst, 0, rec_buffer.data(), recv, point_type, src, 0, comm, MPI_STATUS_IGNORE);
        tranB.resize(recv);
        tranB = rec_buffer;
    }
    MPI_Type_free(&point_type);

    double end_time;
    if (rank == 0) {
        end_time = MPI_Wtime();
        double time_taken = end_time - start_time;
        cout << "Time: " <<time_taken << endl;
    }

    // int* global_C = new int[N*N];
    // for (int i = 0; i < N*N; i++) {
    //     global_C[i] = 0;
    // }
    // MPI_Gather(C, N*N/p , MPI_INT, global_C , N*N/p , MPI_INT, 0, MPI_COMM_WORLD);

    // if (pf == 1) {
    //     if(rank == 0){
    //         print_matrix_all(mat_A, mat_B, global_C, out_file, N, N);
    //     }
    // }
    MPI_Finalize();
    return 0;
} 