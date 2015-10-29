#include <cv.h>
#include <highgui.h>
#include <time.h>
#include <cuda.h>

#define RED 2
#define GREEN 1
#define BLUE 0

using namespace cv;

__device__ unsigned char clamp(int value)
{
    if(value < 0)
        value = 0;
    else
        if(value > 255)
            value = 255;
    return (unsigned char)value;
}


__global__ void sobelFilter(unsigned char *imageInput, int width, int height, unsigned int maskWidth,\
        char *M,unsigned char *imageOutput)
{
    unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;

    int Pvalue = 0;

    int N_start_point_row = row - (maskWidth/2);
    int N_start_point_col = col - (maskWidth/2);

    for(int i = 0; i < maskWidth; i++)
    {
        for(int j = 0; j < maskWidth; j++ )
        {
            if((N_start_point_col + j >=0 && N_start_point_col + j < width) \
                    &&(N_start_point_row + i >=0 && N_start_point_row + i < height)){
                Pvalue += imageInput[(N_start_point_row + i) * width + (N_start_point_col + j)] * M[i*maskWidth+j];
            }
        }
    }
    imageOutput[row*width+col] = clamp(Pvalue);
}

__global__ void img2gray(unsigned char *imageInput, int width, int height, unsigned char *imageOutput)
{
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if((row < height) && (col < width))
    {
        imageOutput[row*width+col] = imageInput[(row*width+col)*3+RED]*0.299 + imageInput[(row*width+col)*3+GREEN]*0.587 \
                                     + imageInput[(row*width+col)*3+BLUE]*0.114;
    }
}


int main()
{
    cudaError_t error = cudaSuccess;
    clock_t start, end, startGPU, endGPU;
    double cpu_time_used, gpu_time_used,TpromedioCPU = 0,TpromedioGPU = 0;

    char h_M[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};    // Gx
    char v_M[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};    // Gy

    char *d_M;

    unsigned char *dataRawImage, *d_dataRawImage, *d_imageOutput, *h_imageOutput, *d_sobelOutput;
    Mat image;
    image = imread("./inputs/img1.jpg", 1);

    if(!image.data)
    {
        printf("No image Data \n");
        return -1;
    }

    Size s = image.size();

    int width = s.width;
    int height = s.height;
    int size = sizeof(unsigned char)*width*height*image.channels();
    int sizeGray = sizeof(unsigned char)*width*height;


    dataRawImage = (unsigned char*)malloc(size);
    error = cudaMalloc((void**)&d_dataRawImage,size);
    if(error != cudaSuccess)
    {
        printf("Error reservando memoria para d_dataRawImage\n");
        exit(-1);
    }

    h_imageOutput = (unsigned char *)malloc(sizeGray);
    error = cudaMalloc((void**)&d_imageOutput,sizeGray);
    if(error != cudaSuccess)
    {
        printf("Error reservando memoria para d_imageOutput\n");
        exit(-1);
    }

    error = cudaMalloc((void**)&d_M,sizeof(char)*9);
    if(error != cudaSuccess)
    {
        printf("Error reservando memoria para d_M\n");
        exit(-1);
    }

    error = cudaMalloc((void**)&d_sobelOutput, sizeGray);
    if(error != cudaSuccess)
    {
        printf("Error reservando memoria para d_sobelOutput\n");
        exit(-1);
    }

    dataRawImage = image.data;

  for(int i = 0; i < 20; i++){
    	printf("--------------------- Ejecucion %i -------------------\n\n",i+1);
    	startGPU = clock();
    	error = cudaMemcpy(d_dataRawImage, dataRawImage, size, cudaMemcpyHostToDevice);
    	if(error != cudaSuccess)
    	{
        printf("Error copiando los datos de dataRawImage a d_dataRawImage \n");
        exit(-1);
    	}

    error = cudaMemcpy(d_M, h_M, sizeof(char)*9, cudaMemcpyHostToDevice);
    if(error != cudaSuccess)
    {
        printf("Error copiando los datos de h_M a d_M \n");
        exit(-1);
    }

    int blockSize = 32;
    dim3 dimBlock(blockSize, blockSize, 1);
    dim3 dimGrid(ceil(width/float(blockSize)), ceil(height/float(blockSize)), 1);

    img2gray<<<dimGrid, dimBlock>>>(d_dataRawImage, width, height, d_imageOutput);

    cudaDeviceSynchronize();
    sobelFilter<<<dimGrid,dimBlock>>>(d_imageOutput, width, height, 3, d_M, d_sobelOutput);
    cudaMemcpy(h_imageOutput, d_sobelOutput, sizeGray, cudaMemcpyDeviceToHost);
    
  	endGPU = clock();
   	
    
  
    Mat gray_image;
    gray_image.create(height, width, CV_8UC1);
    gray_image.data = h_imageOutput;

    start = clock();
    Mat gray_image_opencv, grad_x, abs_grad_x;
    cvtColor(image, gray_image_opencv, CV_BGR2GRAY);
    Sobel(gray_image_opencv, grad_x, CV_8UC1, 1, 0, 3, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
  
    end = clock();
		cpu_time_used = (((double) (end - start)) / CLOCKS_PER_SEC );
  	

    imwrite("./outputs/1059448819.png",gray_image);

    gpu_time_used = ((double) (endGPU - startGPU)) / CLOCKS_PER_SEC;
    cpu_time_used = ((double) (end - start)) /CLOCKS_PER_SEC;
    printf("Tiempo de ejecucion de la GPU: %f\n", gpu_time_used);
    printf("Tiempo de ejecucion de la CPU: %f\n\n", cpu_time_used);
    TpromedioGPU = TpromedioGPU + gpu_time_used;
    TpromedioCPU = TpromedioCPU + cpu_time_used;  	
  }
  	TpromedioGPU = TpromedioGPU / 20;
  	TpromedioCPU = TpromedioCPU / 20;
  	printf("\n-------------------------------------------------------------------------\n\n");
  	printf("Tiempo promedio de ejecucion de la GPU: %f\n", TpromedioGPU);
  	printf("Tiempo promedio de ejecucion de la CPU: %f\n", TpromedioCPU);
    cudaFree(d_dataRawImage);
    cudaFree(d_imageOutput);
    cudaFree(d_M);
    cudaFree(d_sobelOutput);
    return 0;
}
