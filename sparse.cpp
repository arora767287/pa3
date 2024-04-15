#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <iostream>
#include <fstream>
#include <vector>

struct Point {
    int r;
    int c;
    int v;
};

using namespace std;
vector<Point> generate_sparse(float s, int N, int p, int rank, int seed) {
    vector<Point> gen_mat;
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
                gen_mat.push_back(point);
                count += 1;
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

void print_mat(vector<Point> matrix, char* outfile){
    int all_vals = matrix.size();
    char *filename = outfile;
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

void print_dense_matrix(const vector<Point>& matrix, int N, int p, int rank) {
    vector< vector<int> > dense_matrix(N, vector<int>(N, 0));
    for (const Point& point : matrix) {
        dense_matrix[point.r][point.c] = point.v;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << dense_matrix[i][j] << " ";
        }
        cout << endl;
    }
}

int* vec_mat(vector<Point> matrix, int dim1, int dim2){
    int* mat = new int[dim1*dim2];

    for (int i = 0; i < matrix.size(); i++) {
        int row = matrix[i].r;
        int col = matrix[i].c;
        mat[row*dim2 + col] += matrix[i].v;
    }    
    return mat;
}

MPI_Datatype create_point_type() {
    MPI_Datatype point_type;
    int blocklengths[3] = {1, 1, 1};
    MPI_Aint offsets[3];
    offsets[0] = offsetof(Point, r);
    offsets[1] = offsetof(Point, c);
    offsets[2] = offsetof(Point, v);
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};

    MPI_Type_create_struct(3, blocklengths, offsets, types, &point_type);
    MPI_Type_commit(&point_type);

    return point_type;
}

void transpose_matrix(std::vector<Point>& matrix, int N, int p) {
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

    MPI_Alltoallv(matrix.data(), sendcounts.data(), sdispls.data(), point_type,
                  transposed_matrix.data(), recvcounts.data(), rdispls.data(), point_type,
                  MPI_COMM_WORLD);

    for (Point& point : transposed_matrix) {
        std::swap(point.r, point.c);
    }

    matrix = std::move(transposed_matrix);

    MPI_Type_free(&point_type);
}

void printMatrix(int* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << std::setw(4) << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

void gather_and_print_matrix(const std::vector<Point>& local_matrix, int N, int p, int rank) {
    MPI_Datatype point_type = create_point_type();

    int local_size = local_matrix.size();
    
    std::vector<int> sizes(p);

    MPI_Gather(&local_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displs(p, 0);
    if (rank == 0) {
        for (int i = 1; i < p; ++i) {
            displs[i] = displs[i - 1] + sizes[i - 1];
        }
    }

    std::vector<Point> all_points;
    if(rank == 0){
        all_points.resize(displs[p - 1] + sizes[p - 1]);
    }
    
    MPI_Gatherv(local_matrix.data(), local_size, point_type, all_points.data(), sizes.data(), displs.data(), point_type, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::vector< std::vector<int> > matrix(N, std::vector<int>(N, 0));
        for (const auto& point : all_points) {
            int row = point.r;
            int col = point.v;
            matrix[point.r][point.c] = point.v;
        }

        std::cout << "Complete Matrix:" << std::endl;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                std::cout << matrix[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    MPI_Type_free(&point_type);
}

int* gather_and_return_matrix(const std::vector<Point>& local_matrix, int N, int p, int rank) {
    MPI_Datatype point_type = create_point_type();

    int local_size = local_matrix.size();
    std::vector<int> sizes(p);
    MPI_Gather(&local_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displs(p, 0);
    if (rank == 0) {
        for (int i = 1; i < p; ++i) {
            displs[i] = displs[i - 1] + sizes[i - 1];
        }
    }

    std::vector<Point> all_points(rank == 0 ? (displs[p - 1] + sizes[p - 1]) : 0);
    MPI_Gatherv(local_matrix.data(), local_size, point_type, 
                all_points.data(), sizes.data(), displs.data(), point_type, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int* matrix = new int[N*N];
        for (const auto& point : all_points) {
            matrix[(point.r)*N + point.c] = point.v;
        }
        return matrix;
    }

    MPI_Type_free(&point_type);
}

int* mat_convert(vector<Point>& all_points, int N){
    int* global_C = new int[N*N];
    for (const auto& point : all_points) {
        global_C[point.r * N + point.c] = point.v;
    }
    return global_C;
}

int* mat_mul_real(int* first, int* second, int N){
    int* global_C = new int[N*N];
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            global_C[i * N + j] = 0;
            for (int k = 0; k < N; k++) {
                global_C[i * N + j] += first[i * N + k] * second[k * N + j];
                // printf("%d", global_C[i * N + j]);
            }
        }
    }
    return global_C;
}

void mat_mul(vector<Point>& a, vector<Point>& b, int* c, int N, int p) {
    for (Point& pb : b) {
        for (Point& pa : a) {
            if (pa.c == pb.r) {
                int new_val = pa.v * pb.v;
                int idx = ((pa.r % (N/p))* N) + pb.c;
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
    vector<Point> A = generate_sparse(s, N, p, rank, 0);
    vector<Point> B = generate_sparse(s, N, p, rank, 1);
    transpose_matrix(B, N, p);

    printf("\n");

    printf("\n");

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

    int local_B_size = B.size();
    std::vector<int> sizes(p);
    MPI_Allgather(&local_B_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int* mat_A = gather_and_return_matrix(A, N, p, rank);
    int* mat_B = gather_and_return_matrix(B, N, p, rank);


    for (int i = 0; i < sizes.size(); i++) {
        cout << sizes.at(i) << " ";
    }

    MPI_Datatype point_type = create_point_type();
    for (int iter = 0; iter < p; iter++) {
        mat_mul(A, B, C, N, p);

        vector<Point> rec_buffer(sizes[src]);
        MPI_Sendrecv(B.data(), B.size(), point_type, dst, 0, rec_buffer.data(), sizes[src], point_type, src, 0, comm, MPI_STATUS_IGNORE);

        B.resize(sizes[src]);
        B = rec_buffer;
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

    if(rank == 0){
        printMatrix(global_C, N, N);
        printf("\n");
        printMatrix(mat_mul_real(mat_A, mat_B, N), N, N);
    }

    if (pf == 1) {

    }


    MPI_Finalize();
    return 0;
} 