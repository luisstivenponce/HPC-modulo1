#include<iostream>
#include<cstdlib>
#include<cstdlib>
#include<cuda.h>
#include<highgui.h>
#include<cv.h>

#define Mask_size  3
#define TILE_SIZE  32
#define BLOCK_SIZE 32

__constant__ char M[Mask_size*Mask_size];
__constant__ char M1[Mask_size*Mask_size];

using namespace std;
using namespace cv;


// Es necesario utilizar __device__ debido a que es llamada desde el kernel
__device__ unsigned char clamp(int value) 
{
    if(value < 0)
        value = 0;
    else
        if(value > 255)
            value = 255;
    return  value;
}


// Memoria Global
__global__ void convolution2DGlobalMemKernel(unsigned char *In,char *MaskX,char *MaskY, unsigned char *Out,int Mask_Width,int Rowimg,int Colimg)
{

   unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
   unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;

   int PvalueX = 0;
   int PvalueY = 0;
   double SUM = 0;

   int N_start_point_row = row - (Mask_Width/2);
   int N_start_point_col = col - (Mask_Width/2);

   for(int i = 0; i < Mask_Width; i++)
   {
       for(int j = 0; j < Mask_Width; j++ )
       {
        if((N_start_point_col + j >=0 && N_start_point_col + j < Rowimg)
        &&(N_start_point_row + i >=0 && N_start_point_row + i < Colimg))
        {
          PvalueX += In[(N_start_point_row + i)*Rowimg+(N_start_point_col + j)] * MaskX[i*Mask_Width+j];
		  		PvalueY += In[(N_start_point_row + i)*Rowimg+(N_start_point_col + j)] * MaskY[i*Mask_Width+j];
        }
       }
   }
   
   SUM = sqrt(pow((double) PvalueX, 2) + pow((double) PvalueY, 2));

   Out[row*Rowimg+col] = clamp(SUM);

}

// Memoria Constante
__global__ void convolution2DConstantMemKernel(unsigned char *In,unsigned char *Out,int Mask_Width,int Rowimg,int Colimg)
 {
   unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
   unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;

   int Pvalue = 0;
   int PvalueY = 0; 		
   double SUM = 0;	
	
   int N_start_point_row = row - (Mask_Width/2);
   int N_start_point_col = col - (Mask_Width/2);

   for(int i = 0; i < Mask_Width; i++)
   {
       for(int j = 0; j < Mask_Width; j++ )
       {
         if((N_start_point_col + j >=0 && N_start_point_col + j < Rowimg)
         &&(N_start_point_row + i >=0 && N_start_point_row + i < Colimg))
         {
           Pvalue += In[(N_start_point_row + i)*Rowimg+(N_start_point_col + j)] * M[i*Mask_Width+j];
		   		 PvalueY += In[(N_start_point_row + i)*Rowimg+(N_start_point_col + j)] * M1[i*Mask_Width+j];
         }
       }
    }
	 SUM = sqrt(pow((double) Pvalue, 2) + pow((double) PvalueY, 2));

   Out[row*Rowimg+col] = clamp(SUM);
}

// Memoria Compartida - Tiles
__global__ void convolution2DSharedMemKernel(unsigned char *imageInput,unsigned char *imageOutput,
 int maskWidth, int width, int height)
{
    __shared__ float N_ds[TILE_SIZE + Mask_size - 1][TILE_SIZE + Mask_size - 1];
    int n = maskWidth/2;
    int dest = threadIdx.y*TILE_SIZE+threadIdx.x, destY = dest / (TILE_SIZE+Mask_size-1), destX = dest % (TILE_SIZE+Mask_size-1),
        srcY = blockIdx.y * TILE_SIZE + destY - n, srcX = blockIdx.x * TILE_SIZE + destX - n,
        src = (srcY * width + srcX);
    if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
        N_ds[destY][destX] = imageInput[src];
    else
        N_ds[destY][destX] = 0;

    // Second batch loading
    dest = threadIdx.y * TILE_SIZE + threadIdx.x + TILE_SIZE * TILE_SIZE;
    destY = dest /(TILE_SIZE + Mask_size - 1), destX = dest % (TILE_SIZE + Mask_size - 1);
    srcY = blockIdx.y * TILE_SIZE + destY - n;
    srcX = blockIdx.x * TILE_SIZE + destX - n;
    src = (srcY * width + srcX);
    if (destY < TILE_SIZE + Mask_size - 1)
    {
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
            N_ds[destY][destX] = imageInput[src];
        else
            N_ds[destY][destX] = 0;
    }
    __syncthreads();

    int Pvalue = 0;
		int PvalueY = 0;
    int y, x;
	double SUM;
  for (y = 0; y < maskWidth; y++){
        for (x = 0; x < maskWidth; x++){
            Pvalue += N_ds[threadIdx.y + y][threadIdx.x + x] * M[y * maskWidth + x];
						PvalueY += N_ds[threadIdx.y + y][threadIdx.x + x] *M1[y * maskWidth + x];
        }
		}
    y = blockIdx.y * TILE_SIZE + threadIdx.y;
    x = blockIdx.x * TILE_SIZE + threadIdx.x;
    SUM = sqrt(pow((double) Pvalue, 2) + pow((double) PvalueY, 2));
  
    if (y < height && x < width)		
        imageOutput[(y * width + x)] = clamp(SUM);
	    __syncthreads();
}

// llamado a los diferentes kernels
void KernelCalls(Mat image,unsigned char *In,unsigned char *Out,char *h_Mask,char *v_Mask,
  int Mask_Width,int Row,int Col, int op)
{
  // Variables
  int Size_of_bytes =  sizeof(unsigned char)*Row*Col*image.channels();
  int Mask_size_bytes =  sizeof(char)*(Mask_size*Mask_size);
  unsigned char *d_In, *d_Out;
  char *d_Mask, *d2_Mask;
  float Blocksize=BLOCK_SIZE;

  // Reserva de memoria para el dispositivo
  cudaMalloc((void**)&d_In,Size_of_bytes);
  cudaMalloc((void**)&d_Out,Size_of_bytes);
  cudaMalloc((void**)&d_Mask,Mask_size_bytes);
  cudaMalloc((void**)&d2_Mask,Mask_size_bytes);

  cudaMemcpy(d_In,In,Size_of_bytes,cudaMemcpyHostToDevice);
  cudaMemcpy(d_Mask,h_Mask,Mask_size_bytes,cudaMemcpyHostToDevice); //Gx
  cudaMemcpy(d2_Mask,v_Mask,Mask_size_bytes,cudaMemcpyHostToDevice); //Gy
  
  cudaMemcpyToSymbol(M,h_Mask,Mask_size_bytes);// Usando memoria constante

  dim3 dimGrid(ceil(Row/Blocksize),ceil(Col/Blocksize),1);
  dim3 dimBlock(Blocksize,Blocksize,1);
  // La opcion me indica que Kernel voy a escoger
  switch(op)
  {
    case 1:
      cout<<"Convolucion usando Memoria Global"<<endl;
      convolution2DGlobalMemKernel<<<dimGrid,dimBlock>>>(d_In,d_Mask, d2_Mask, d_Out,Mask_Width,Row,Col);
      break;
    case 2:
      cout<<"Convolucion usando Memoria Constante"<<endl;
      convolution2DConstantMemKernel<<<dimGrid,dimBlock>>>(d_In,d_Out,Mask_Width,Row,Col);
      break;
    case 3:
      cout<<"Convolucion usando Memoria Compartida-Tiles"<<endl;
      convolution2DSharedMemKernel<<<dimGrid,dimBlock>>>(d_In,d_Out,Mask_Width,Row,Col);
      break;
    default:
      cout<<"La opcion es incorrecta, digite un numero del 1 al 3\n";
      break;
  }

  cudaDeviceSynchronize();
  
  cudaMemcpy (Out,d_Out,Size_of_bytes,cudaMemcpyDeviceToHost);
  
  // Liberamos la memoria del dispositivo
  cudaFree(d_In);
  cudaFree(d_Out);
  cudaFree(d_Mask);
}



int main()
{

  clock_t start, finish; //Clock variables
  double tiempoParalelo;
  //double tiempoSecuencial;
  int Mask_Width =  Mask_size;
  Mat image;
  image = imread("inputs/img6.jpg",0);   // Con cero indico que la cargue en escala de grises
  // Opcion para el llamado a paralelo
  //int op = 1;

  Size s = image.size();
  int Row = s.width;
  int Col = s.height;
  char h_Mask[] = {-1,0,1,-2,0,2,-1,0,1}; // Gx
  char v_Mask[] = {-1,-2,-1,0,0,0,1,2,1}; // Gy Mascara para el eje Y
  
  unsigned char *img = (unsigned char*)malloc(sizeof(unsigned char)*Row*Col*image.channels());
  unsigned char *imgOut = (unsigned char*)malloc(sizeof(unsigned char)*Row*Col*image.channels());

  if( !image.data )
  {
    cout<<"Problemas al cargar la imagen..."<<endl;
    return -1;
  }

  img = image.data;
/*
  Mat grad_x;
  start = clock();
  Sobel(image,grad_x,CV_8UC1,1,0,3,1,0,BORDER_DEFAULT);
  finish = clock();
  //imwrite("./outputs/1088273734.png",grad_x);
  tiempoSecuencial = (((double) (finish - start)) / CLOCKS_PER_SEC );
  cout<< "El tiempo secuencial fue de: " << tiempoSecuencial << " segundos "<< endl;
*/
	start = clock();
	KernelCalls(image,img,imgOut,h_Mask,v_Mask,Mask_Width,Row,Col,1);
	finish = clock();
	tiempoParalelo = (((double) (finish - start)) / CLOCKS_PER_SEC );
	cout<< "El tiempo paralelo fue de: " << tiempoParalelo << " segundos "<< endl;
 
  Mat gray_image;
  gray_image.create(Col,Row,CV_8UC1);
  gray_image.data = imgOut;
  imwrite("./outputs/1059448819.png",gray_image); 

  return 0;
}
