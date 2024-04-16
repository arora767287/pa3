all:
	mpicxx -o spmat sparse.cpp

run:
	mpirun -np $(np) ./spmat $(n) $(s) $(pf) $(out)

clean:
	rm -f spmat

sample:
	mpirun -np 4 ./spmat 8 0.1 1 sparse_out
