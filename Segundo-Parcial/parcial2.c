#include<iostream>
#include<cstdlib>
#include<cstdlib>
#include<cuda.h>
#include<highgui.h>
#include<cv.h>

using namespace std;
using namespace cv;


int main()
{
  clock_t start, finish; 
  double elapsedSequential;
  
  Mat image;
  image = imread("inputs/img1.jpg",0);   // Se utiliza el parametro en cero para imagen en escla de grises y el 1 para imagen a color
  Size s = image.size();
  int Row = s.width;
  int Col = s.height;

  unsigned char *img = (unsigned char*)malloc(sizeof(unsigned char)*Row*Col*image.channels());

  if( !image.data )
  {
    cout<<"Problemas cargando la Imagen"<<endl;
    return -1;
  }

  img = image.data;

  cout<<"... Secuencial ...\n"<<endl;
  Mat imagen_salida;
  
  start = clock();
  
  Sobel(image, imagen_salida, CV_8UC1, 1, 0, 3, 1, 0, BORDER_DEFAULT);
  
  finish = clock();

  imwrite("./outputs/1059448819.png",imagen_salida);

  elapsedSequential = (((double) (finish - start)) / CLOCKS_PER_SEC );
  cout<< "El proceso secuencial tomo: " << elapsedSequential << " en ejecutar\n "<< endl;

  return 0;
}
