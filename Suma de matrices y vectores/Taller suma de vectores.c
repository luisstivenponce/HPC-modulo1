#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define SIZE 2000
#define BLOCKSIZE 1024

__global__ void vecAdd(int *A, int *B, int *C, int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//int i = threadIdx.x;

			//blockIdx.x;
  if (i < n){
		C[i] = A[i] + B[i];
	    // printf("%d. %d + %d = %d\n",i, A[i], B[i], C[i]);
  }
}
int vectorAddGPU( int *A, int *B, int *C, int n){
	int size = n*sizeof(int);
	int *d_A, *d_B, *d_C;
	//Reservo Memoria en el dispositivo
	cudaMalloc((void **)&d_A, size);
	cudaMalloc((void **)&d_B, size);
	cudaMalloc((void **)&d_C, size);
	//Copio los datos al device
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
	// Ejecuto el Kernel (del dispositivo)
	int dimGrid = ceil(SIZE/BLOCKSIZE);
	printf(" DimGrid %d\n", dimGrid);
	vecAdd<<< dimGrid, BLOCKSIZE >>>(d_A, d_B, d_C, n);
	// vecAdd<<< DIMGRID, HILOS >>>
	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	return 0;
}


int vectorAddCPU( int *A, int *B, int *C, int n){
	int i;
	for(i=0;i< n; i++){
		C[i]=A[i]+B[i];
		//printf("%d. %d+", i, A[i]);
    //printf("%d=",B[i]);
		//printf("%d\n",C[i]);
	}
	return 0;
}

int main(){
	int j;
	int SIZES[] = {512, 1024, 3000, 5000, 1000000, 5000000, 10000000, 20000000,30000000};
	for (j = 0; j < sizeof(SIZES)/sizeof(SIZES[0]); ++j)
	{
		int *A=(int *) malloc(SIZES[j]*sizeof(int));
		int *B=(int *) malloc(SIZES[j]*sizeof(int));
		int *C=(int *) malloc(SIZES[j]*sizeof(int));
		clock_t inicioCPU, inicioGPU,finCPU, finGPU;
		int i;
		for(i=0;i< SIZES[j]; i++){
			A[i]=rand()%21;
			B[i]=rand()%21;
			// A[i]=srand(time(NULL));
			// B[i]=srand(time(NULL));

		}
		// Ejecuto por GPU
		inicioGPU=clock();
		vectorAddGPU(A, B, C, SIZES[j]);
		finGPU = clock();
		// Ejecuto por CPU
		inicioCPU=clock();
		vectorAddCPU(A, B, C, SIZES[j]);
		finCPU=clock();
		printf("Size %d\n", SIZES[j]);
		printf("El tiempo GPU es: %f\n",(double)(finGPU - inicioGPU) / CLOCKS_PER_SEC);
		printf("El tiempo CPU es: %f\n",(double)(finCPU - inicioCPU) / CLOCKS_PER_SEC);
		free(A);
		free(B);
	}
	return 0;
}
