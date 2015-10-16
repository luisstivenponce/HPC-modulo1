#include<stdio.h>
#include<stdlib.h>
#define tama 3
 
int num=0;
void inimatriz(int M[tama][tama]){
int i = 0, j = 0;
  num = num +1;
  srand((unsigned) time(NULL));
  for(i = 0;i < tama; i++){
    for(j = 0; j < tama; j++){
    M[i][j] = rand() % (3 + num); 
      printf("%i\t", M[i][j]);
    }
    printf("\n");
  }
}

int main(){
  int M1[tama][tama],M2[tama][tama],M3[tama][tama];
  int i=0, j=0, c=0, numero = 0;
  inimatriz(M1);
  printf("\n");
  inimatriz(M2);
  
  printf("\n");
  for(i=0;i < tama; i++){
  	for(c=0;c < tama; c++){
      numero = 0;
  		for(j=0;j < tama; j++){
        		numero = numero + M1[i][j] * M2[j][c];  	
  		}
      M3[i][c] = numero;
      printf("%i\t",M3[i][c]);
  	}
    printf("\n");
  }
  
return 0;
}
