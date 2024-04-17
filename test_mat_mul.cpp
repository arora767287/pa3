#include <vector>
#include <iostream>
#include <cstring>

struct CSR {
    std::vector<int> row_ptr;
    std::vector<int> cols;
    std::vector<int> vals;
};

void mat_mul_csr(const CSR& A, const CSR& BT, int* C, int rowsA, int colsB) {
    // Initialize the result matrix C to zero
    std::memset(C, 0, rowsA * colsB * sizeof(int));

    // Perform matrix multiplication
    for (int i = 0; i < rowsA; ++i) {  // Loop over rows of A
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {  // Loop over non-zero entries in row i of A
            int colA = A.cols[j]; // Column index in A, corresponds to row index in BT
            int valA = A.vals[j]; // Value at A[i, colA]

            // Multiply with corresponding row in BT (which is col in original B)
            for (int k = BT.row_ptr[colA]; k < BT.row_ptr[colA + 1]; ++k) {
                int colB = BT.cols[k]; // Column index in BT, corresponds to column index in B
                int valBT = BT.vals[k]; // Value at BT[colA, colB]

                C[i * colsB + colB] += valA * valBT; // Accumulate product in C[i, colB]
            }
        }
    }
}

int main() {
    CSR A, BT;

    // Manually push back elements to vectors for CSR A
    A.row_ptr.push_back(0);
    A.row_ptr.push_back(2);
    A.row_ptr.push_back(4);
    A.cols.push_back(0);
    A.cols.push_back(1);
    A.cols.push_back(0);
    A.cols.push_back(1);
    A.vals.push_back(1);
    A.vals.push_back(2);
    A.vals.push_back(3);
    A.vals.push_back(4);

    // Manually push back elements to vectors for CSR BT
    BT.row_ptr.push_back(0);
    BT.row_ptr.push_back(2);
    BT.row_ptr.push_back(4);
    BT.cols.push_back(0);
    BT.cols.push_back(1);
    BT.cols.push_back(0);
    BT.cols.push_back(1);
    BT.vals.push_back(5);
    BT.vals.push_back(6);
    BT.vals.push_back(7);
    BT.vals.push_back(8);

    int rowsA = 2;
    int colsB = 2;
    int* C = new int[rowsA * colsB];

    mat_mul_csr(A, BT, C, rowsA, colsB);

    // Print the resulting matrix C
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            std::cout << C[i * colsB + j] << " ";
        }
        std::cout << "\n";  // Use \n instead of std::endl for potentially better performance
    }

    delete[] C;
    return 0;
}
