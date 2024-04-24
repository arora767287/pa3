Program Description:

This program multiplies two sparse matrices generated and filled withe random numbers using inputs of the number of processors to run the program on, an input sparsity value to generate a non-zero value in a matrix with probability s, and an input N corresponding to the size of the square matrix to generate (N by N). First, we generate \frac{N}{p} rows of each matrix on every one of the processors allocated for the program. This block distributes the rows of $A$ and $B$ among the processors. Then, we transpose the matrix B using an MPI All-to-All so that every processor holds \frac{N}{p} columns of the original matrix B instead of \frac{N}{p} rows of it. Every processor then completes a dot product between the rows of A and the rows of the now transposed matrix B (so the columns of the original matrix B). We utilize the ring topology in MPI to shift the local matrices B of size $\frac{N}{p} by N$. 

The program runs as below:
\begin{enumerate}
    \item Each processor is in charge of $\frac{N}{p}$ rows, so each one generates $\frac{sN^2}{p}$ elements for matrix $A$ and $\frac{sN^2}{p}$ for matrix $B$. This block distributes the rows of $A$ and $B$ among the processors.
    \item $B$ undergoes an MPI All-to-All to transpose the matrix.
    \item Each processor completes a dot product between their rows of $A$ and the rows of transposed $B$.
    \item Each processor sends their transposed $B$ to the next processor in the ring.
    \item Steps 3 and 4 are repeated $p$ times until all their rows of $A$ computed a dot product with all the rows of $B$.
\end{enumerate}
After this, the matrix can be considered as multiplied, as timing starts after step 1 and ends after step 5, but it can be gathered into one processor for I/O. \\


Sriharsha Kocherla, Sachitt Arora, Nitya Arora

Program description: 

This program transposes an input matrix using an input number of processors where the matrix is square and has dimensions divisible by the number of the processors used. To transpose the input matrix, we first scatter the matrix among the $p$ processors such that each has an $\frac{N}{p}$ by $N$ matrix stored. Then, we locally transpose each $\frac{N}{p}$ by $N$ sub-matrix that each processor has. We then treat each of the $p$ $\frac{N}{p}$ by $\frac{N}{p}$ matrices as sub-matrices that need to be re-positioned in their transposed locations among the larger $N$ by $N$ matrix. For instance, the second $\frac{N}{p}$ by $\frac{N}{p}$ matrix in the first processor needs to swap positions with the first $\frac{N}{p}$ by $\frac{N}{p}$ in the second processor. For this, we use an all-to-all communication (implemented one of three ways out of using hypercubic permutations, arbitrary permutations or the MPI implementation). Finally, we gather the output across all of the processors in the processor with rank 0. During our implementation, we record the time for this computation and exchange to occur, starting our timer after block distibuting the rows and ending it before gathering the final matrix. 

Machine Used:

We used the first available CPU queued by the ICE system (ours was an Intel CPU) with 1 node and 24 cores per node (since we only ran our program on 1 core through up to 16 cores).

Running program:

To run our program, you can use the following commands:

- To make: make
- To run the program on the hypercubic permuation all to all implementation: make runh np=(number of processors) n=(matrix dimenson)
- To run the program on the arbitrary permuation all to all implementation: make runa np=(number of processors) n=(matrix dimenson)
- To run the program on the MPI all to all implementation: make rund np=(number of processors) n=(matrix dimenson)
