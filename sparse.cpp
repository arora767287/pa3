#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstdint>

using namespace std;

struct Point {
   uint64_t r;
   uint64_t c;
   uint64_t v;
};

bool sort_by_row(const Point &a, const Point &b) {
    return a.r < b.r;
}

bool sort_by_col(const Point &a, const Point &b) {
    if (a.c != b.c) return a.c < b.c;
    return a.r < b.r;
}

vector<Point> generate_sparse(float s,uint64_t N,uint64_t p,uint64_t rank,uint64_t seed) {
    vector<Point> m;
   uint64_t count = 0;
   uint64_t start_row = rank * (N / p);
   uint64_t end_row = (rank + 1) * (N / p);
    srand(time(NULL) + rank + seed); 
    for (uint64_t c = 0; c < N; c++) {
        for (uint64_t r = start_row; r < end_row; r++) {
           uint64_t randd = rand() % N;
           
            if (randd < s * N) {
                uint64_t rand_value = static_cast<uint64_t>(rand() + 1);
                Point point = {r, c, rand_value};
                m.push_back(point);
                count += 1;
            }
        }
    }
    return m;
}

void print_matrix_all(uint64_t* matrix, uint64_t* matrix2, uint64_t* matrix3, char* outfile,uint64_t dim1,uint64_t dim2){
    FILE * fp = fopen(outfile, "w");
    for(int i = 0; i < dim1; i++){
        for(int j = 0; j < dim2; j++){
            fprintf(fp, "%llu", matrix[i*dim2 + j]);
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

vector<Point> transpose_matrix(std::vector<Point>& matrix,uint64_t N,uint64_t p) {
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
   uint64_t recv_buffer_size = recvcounts[0];
    for (int i = 1; i < p; i++) {
        rdispls[i] = rdispls[i - 1] + recvcounts[i - 1];
        recv_buffer_size += recvcounts[i];
    }
    std::vector<Point> transposed_matrix(recv_buffer_size);

    MPI_Alltoallv(matrix.data(), sendcounts.data(), sdispls.data(), point_type, transposed_matrix.data(), recvcounts.data(), rdispls.data(), point_type, MPI_COMM_WORLD);

    MPI_Type_free(&point_type);
    return transposed_matrix;
}

uint64_t* gather_and_return_matrix(const std::vector<Point>& curr_matrix,uint64_t N,uint64_t p,uint64_t rank) {
    MPI_Datatype point_type = create_point_type();

   uint64_t curr_size = curr_matrix.size();
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
        uint64_t* matrix = new uint64_t[N*N];
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

void printMatrix(uint64_t* matrix,uint64_t rows,uint64_t cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

void mat_mul_dot_product(vector<Point>& a, vector<Point>& b, uint64_t* c,uint64_t N,uint64_t p) {
    std::sort(a.begin(), a.end(), sort_by_col); // sN/p
    std::sort(b.begin(), b.end(), sort_by_row); // sN
   uint64_t sum = 0;
   uint64_t bpointer = 0;
   uint64_t bstart = 0;
   uint64_t old_a_col = -1;

    for (Point &pa: a) {
        if (old_a_col == pa.c) bpointer = bstart;
        while (bpointer < b.size() && b[bpointer].r < pa.c) {
            bpointer++;
        }
        bstart = bpointer;
        while (bpointer < b.size() && pa.c == b[bpointer].r) {
           uint64_t new_val = pa.v * b[bpointer].v;
           uint64_t idx = ((pa.r % (N/p))* N) + b[bpointer].c;
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

    uint64_t N = stoi(argv[1]);  // size
    double s = stod(argv[2]);  // sparsoty
    uint64_t pf = stoi(argv[3]);  // printing flag
    char* out_file = argv[4];  // Ofile name
    
    vector<Point> A = generate_sparse(s, N, p, rank, 0);
    vector<Point> B = generate_sparse(s, N, p, rank, 1);

    uint64_t C_size = N * N / p;
    uint64_t* C = new uint64_t[C_size];
    for (int i = 0; i < C_size; i++) {
        C[i] = 0;
    }

    vector<Point> oldB = B;

    int src, dst;
    MPI_Cart_shift(comm, 0, 1, &src, &dst);

    uint64_t* mat_A = gather_and_return_matrix(A, N, p, rank);
    uint64_t* mat_B = gather_and_return_matrix(oldB, N, p, rank);

    double start_time;
    if (rank == 0) {
        start_time = MPI_Wtime();
    }

    vector<Point> tranB = transpose_matrix(B, N, p);

    MPI_Datatype point_type = create_point_type();
    for (int iter = 0; iter < p; iter++) {
        mat_mul_dot_product(A, tranB, C, N, p);

       uint64_t send = tranB.size();
       uint64_t recv;
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

    uint64_t* global_C = new uint64_t[N*N];
    for (int i = 0; i < N*N; i++) {
        global_C[i] = 0;
    }
    MPI_Gather(C, N*N/p , MPI_INT, global_C , N*N/p , MPI_INT, 0, MPI_COMM_WORLD);

    if (pf == 1) {
        if(rank == 0){
            print_matrix_all(mat_A, mat_B, global_C, out_file, N, N);
        }
    }
    MPI_Finalize();
    return 0;
} 
