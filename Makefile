all:
	mpicxx -o spmat sparse.cpp

run:
	mpirun -np $(np) ./spmat $(n) $(s) $(pf) $(out)

clean:
	rm -f spmat

sample:
	mpirun -np 4 ./spmat 8 0.1 1 sparse_out

pacesample:
	srun -n 4 ./spmat 8 0.1 1 sparse_out

pacerun:
	srun -n $(np) ./spmat $(n) $(s) $(pf) $(out)

runhpcfirst:
	@/bin/bash -c 'number=16 ; while [[ $$number -le 512 ]] ; do \
		echo ; \
		echo $$number ; \
               	echo ; \
		srun -n 8 ./transpose matrix.txt transpose.txt h $$number; \
		((number = number + 16)) ; \
	done' &> log_8h.txt

runhpcsecond:
	@/bin/bash -c 'number=16 ; while [[ $$number -le 512 ]] ; do \
		echo ; \
		echo $$number ; \
		echo ; \
		srun -n 16 ./transpose matrix.txt transpose.txt m $$number; \
		((number = number + 16)) ; \
	done' &> log_16d.txt

runhpc:
	make runhpc8a ; \
	make runhpc8h ; \
	make runhpc8d ; \
	make runhpc16a ; \
	make runhpc16h ; \
	make runhpc16d ; \