#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <iostream>
#include <fstream>

//tuples are in groups of three, so just need pointer
int * generate_sparse(int s, int N, int p){
    int * gen_mat = new int[N*N/p];
    int counter = 0;
    for(int r = 0; r < N/p; r++){
        for(int c = 0; c < N; c++){
            int rand_num = rand() % N;
            if(rand_num < (1-s)*N){
                gen_mat[counter] = r;
                gen_mat[counter + 1] = c;
                gen_mat[counter + 2] = rand() % 10;
            }   
        }
    }
    return gen_mat;
}

