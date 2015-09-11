#include <stdlib.h> 
#include <stdio.h>
#define filas 2
#define columnas 2

int main(void) 
{ 

	int **M1;	
	int **M2;
  int **M3;
  
	int i;	 // Recorre filas 
	int j;	 // Recorre columnas 

	// Reserva de Memoria 
	M1 = (int **)malloc(filas*sizeof(int*)); 
	M2 = (int **)malloc(filas*sizeof(int*));
  M3 = (int **)malloc(filas*sizeof(int*));
  
  for (i=0;i<filas;i++){
		M1[i] = (int*)malloc(columnas*sizeof(int));
		M2[i] = (int*)malloc(columnas*sizeof(int));
  	M3[i] = (int*)malloc(columnas*sizeof(int));
  }
	// Damos Valores a l

	// Dibujamos la Matriz en pantalla 
  for (i=0; i<filas; i++) 
	{ 
    for (j=0; j<columnas; j++){ 
      M1[i][j] =  rand() % 4; 
    	M2[i][j] =  rand() % 5;
    	M3[i][j] =  0; 
    }
	} 
  
	for (i=0; i<filas; i++) 
	{ 
		 
		for (j=0; j<columnas; j++) 
			printf("%i\t", M1[i][j] ); 
    	printf("\n");
	} 
  
  for (i=0; i<filas; i++) 
	{ 
		printf("\n"); 
		for (j=0; j<columnas; j++) 
			printf("%i\t", M2[i][j] ); 
	}
  printf("\n");
  /*for (i=0; i<filas; i++) 
	{ 
		printf("\n"); 
		for (j=0; j<columnas; j++) 
			printf("%i\t", M3[i][j] ); 
	} */
  printf("\n");
 for (i=0; i<filas; i++) 
	{ 
   for (j=0; j<columnas; j++){
				M3[i][j]=M1[i][j]+M2[i][j];
     		printf("%i\t",M3[i][j]);
   	}
   printf("\n");
	} 	
   
	return 0; 
} 
