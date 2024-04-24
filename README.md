Sriharsha Kocherla, Sachitt Arora, Nitya Arora

Program Description:

This program multiplies two sparse matrices generated and filled with random numbers using inputs of the number of processors to run the program on, an input sparsity value to generate a non-zero value in a matrix with probability s, and an input N corresponding to the size of the square matrix to generate (N by N). First, we generate $\frac{N}{p}$ rows of each matrix on every one of the processors allocated for the program. This block distributes the rows of $A$ and $B$ among the processors. Then, we transpose the matrix B using an MPI All-to-All so that every processor holds $\frac{N}{p}$ columns of the original matrix B instead of $\frac{N}{p}$ rows of it. Every processor then completes a dot product between the rows of A and the rows of the now transposed matrix B (so the columns of the original matrix B). We utilize the ring topology in MPI to shift the local matrices B of size $\frac{N}{p} by N$ $p$ times so that every processor is able to run a local multiplication between the $\frac{N}{p}$ rows of A that it holds and the $\frac{N}{p}$ columns of B across all of B. The results of the multiplication are stored in dense matrix format in a separate matrix C and printed to the file specified in the command line input upon running the program. If the print flag is enabled in the command line input, A, B, and C are all printed to the output file.

Machine Used:

We used the first available CPU queued by the ICE system (ours was an Intel CPU) with 1 node and 24 cores per node (since we only ran our program on 1 core through up to 16 cores).

Running program:

To run our program, you can use the following commands:

- To make: make
- To run the program on matrices of some size NxN with sparsity s: make run np=(number of processors) n=(matrix dimension) s=(sparsity) pf=(whether to print) out=(output file name)
