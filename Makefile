all:
	mpicxx -o spmat sparse.cpp

run:
	mpirun -np $(np) ./spmat $(n) $(s) $(pf) $(out)

clean:
	rm -f spmat

sample:
	mpirun -np 8 ./spmat 200 0.001 1 sparse_out
# sample:
# 	mpirun -np 4 ./sparse 8 0.7 1 sparse_out