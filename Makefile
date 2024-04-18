all:
	mpicxx -std=c++17 -O3 -o spmat sparse.cpp

bonus:
	mpicxx -std=c++17 -O3 -o spmat sparse_bonus.cpp

run:
	mpirun -np $(np) ./spmat $(n) $(s) $(pf) $(out)

clean:
	rm -f spmat

sample:
	mpirun -np 4 ./spmat 8 0.5 1 sparse_out

pacesample:
	srun -n 4 ./spmat 8 1 1 sparse_out

pacerun:
	srun -n $(np) ./spmat $(n) $(s) $(pf) $(out)

runhpcfirst:
	srun -n 16 ./spmat 1000 0.01 0 sparse_out ;
	&> runhpcfirst.txt
	srun -n 16 ./spmat 2000 0.01 0 sparse_out ;
	&> runhpcfirst.txt
	srun -n 16 ./spmat 3000 0.01 0 sparse_out ;
	&> runhpcfirst.txt

runhpcsecond16:
	srun -n 16 ./spmat 10000 0.1 0 sparse_out ;
	srun -n 16 ./spmat 10000 0.01 0 sparse_out ;
	srun -n 16 ./spmat 10000 0.001 0 sparse_out ;
	&> runhpcsecond16.txt

runhpcsecond8:
	srun -n 8 ./spmat 10000 0.1 0 sparse_out ;
	srun -n 8 ./spmat 10000 0.01 0 sparse_out ;
	srun -n 8 ./spmat 10000 0.001 0 sparse_out ;
	&> runhpcsecond8.txt

runhpcsecond4:
	srun -n 4 ./spmat 10000 0.1 0 sparse_out ;
	srun -n 4 ./spmat 10000 0.01 0 sparse_out ;
	srun -n 4 ./spmat 10000 0.001 0 sparse_out ;
	&> runhpcsecond4.txt

runhpcsecond2:
	srun -n 2 ./spmat 10000 0.1 0 sparse_out ;
	srun -n 2 ./spmat 10000 0.01 0 sparse_out ;
	srun -n 2 ./spmat 10000 0.001 0 sparse_out ;
	&> runhpcsecond2.txt

runhpc:
	make runhpcfirst ; \
	make runhpcsecond2 ; \
	make runhpcsecond4 ; \
	make runhpcsecond8 ; \
	make runhpcsecond16 ; \