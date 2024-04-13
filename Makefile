all:
	mpicxx -o spmat spmat.cpp

clean:
	rm -f spmat

run:
	mpirun -np $(np) ./spmat $(n) $(s) $(pf) $(out)

sample:
	mpirun -np 8 ./spmat 10000 0.001 0 spmat_out