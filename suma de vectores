#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cstdlib>
#include <time.h>
#define TAM 16

void inicializar(int *M){
int i=0;
int j=0;
  
for(i=0; i<TAM; i++){
  int j=i;
    M[i] = 0;
    printf("%i ",M[i]);
  if(i==sqrt(TAM)){
  	printf("\n");
    j=0;
  
  }
     j++;
}

}
int main(){
	int n; //longitud del vector
	int * A;
	int * B;
	int * C;
  	n=TAM;

	A = (int*)malloc( n*sizeof(int) );
	B = (int*)malloc( n*sizeof(int) );
	C = (int*)malloc( n*sizeof(int) );
	inicializar(A);
	inicializar(B);
  return 0;
	}
