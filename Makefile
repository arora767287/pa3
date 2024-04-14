all:
	mpicxx -o sparse sparse.cpp

run:
	mpirun -np $(np) ./sparse $(n) $(s) $(pf) $(out)

clean:
	rm -f sparse

# sample:
# 	mpirun -np 8 ./sparse 10000 0.001 1 sparse_out
sample:
	mpirun -np 4 ./sparse 6 1 1 sparse_out