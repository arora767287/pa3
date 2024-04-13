all:
	mpicxx -o sparse sparse.cpp

run:
	mpirun -np $(np) ./sparse $(n) $(s) $(pf) $(out)

clean:
	rm -f sparse

sample:
	mpirun -np 8 ./sparse 10000 0.001 0 sparse_out