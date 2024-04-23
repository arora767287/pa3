Theoretical Analysis \\
The program runs as below:
\begin{enumerate}
    \item Each processor is in charge of $\frac{N}{p}$ rows, so each one generates $\frac{sN^2}{p}$ elements for matrix $A$ and $\frac{sN^2}{p}$ for matrix $B$. This block distributes the rows of $A$ and $B$ among the processors.
    \item $B$ undergoes an MPI All-to-All to transpose the matrix.
    \item Each processor completes a dot product between their rows of $A$ and the rows of transposed $B$.
    \item Each processor sends their transposed $B$ to the next processor in the ring.
    \item Steps 3 and 4 are repeated $p$ times until all their rows of $A$ computed a dot product with all the rows of $B$.
\end{enumerate}
After this, the matrix can be considered as multiplied, as timing starts after step 1 and ends after step 5, but it can be gathered into one processor for I/O. \\
