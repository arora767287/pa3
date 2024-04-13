all:
	mpicxx -o spmat spmat.cpp

run:
	mpirun -np $(np) ./spmat $(n) $(s) $(pf) $(out)

clean:
	rm -f spmat

sample:
	mpirun -np 8 ./spmat 10000 0.001 0 spmat_out