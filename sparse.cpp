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


// void mat_mul_naive(vector<Point>& a, vector<Point>& b, int* c, int N, int p) {
//     for (Point& pb : b) {
//         for (Point& pa : a) {
//             if (pa.c == pb.c) {
//                 int new_val = pa.v * pb.v;
//                 int idx = ((pa.r % (N/p))* N) + pb.r;
//                 c[idx] += new_val;
//             }
//         }
//     }
// }


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

    vector<Point> tranB = transpose_matrix(B, N, p);

    MPI_Datatype point_type = create_point_type();
    for (int iter = 0; iter < p; iter++) {
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