#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui.hpp>

#include <stdio.h>

#include <iostream>
#include <sstream>

#include <cuda.h>
#include <cutil.h>

#include "bg_sub_kernel.h"


using namespace cv;
using namespace std;


Mat frame[6];
Mat fgMaskMOG2;
Mat result;

Matrix frame_d[6];

int output = 0;
char outputPath[30];

void processImages();
Matrix AllocateMatrix(int, int , Mat); 


int main(int argc, char* argv[]) {
  if(argc < 7 || argc >= 9) {
    cerr << "USAGE: ./bgsg backgroundfile1...5 substractionfile\n" << endl;
    return EXIT_FAILURE;
  }
  else if(argc == 8) {
    output = 1;
    strcpy( outputPath, argv[7]);
  }

  namedWindow("Background");
  namedWindow("Original");
  namedWindow("Result");

  for(int i=0; i<6; i++) {
    frame[i] = imread(argv[i+1], IMREAD_GRAYSCALE);
    if(frame[i].empty()) {
      cerr << "Unable to open image: " << i << "\n" << endl;
      exit(EXIT_FAILURE);
    }
  }
  
  processImages();

//  destroyAllWindows();
  return EXIT_SUCCESS;
}

void processImages() {
  Matrix input[6];

  for(int i=0; i<6; i++) {
    input[i] = AllocateMatrix(frame[i].rows, frame[i].cols,frame[i]);
    frame_d[i] = AllocateDeviceMatrix(input[i]);
    CopyToDeviceMatrix(frame_d[i], input[i]);
  }

  result = frame[0].clone();
  Matrix tmp_r = AllocateMatrix(result.rows, result.cols, result);

  //r_d need to set to 0.
  Matrix r_d = AllocateDeviceMatrix(tmp_r);
  CopyToDeviceMatrix(r_d, tmp_r);

  bg_caller(frame_d, frame_d[5], r_d);

  CopyFromDeviceMatrix(tmp_r, r_d);

  for(int y=0; y<tmp_r.height; y++) {
    for(int x=0; x<tmp_r.width; x++) {
      result.at<uchar>(y,x) = tmp_r.elements[y * tmp_r.width + x];
    }
  }
  if(output == 1) {
    imwrite(outputPath, result);
  }
  cout << "tmp_r.height: " << tmp_r.height << "---";
  cout << "tmp_r.width: " << tmp_r.width << endl;
  cout << "Result.rows: " << result.rows << "---";
  cout << "Result.cols: " << result.cols << endl;
  imshow("Background", frame[0]);
  imshow("Original", frame[5]);
  imshow("Result", result);


  waitKey(0);
}

Matrix AllocateMatrix(int height, int width, Mat frame) {
  Matrix M;
  M.width = M.pitch = width;
  M.height = height;
  int size = M.width * M.height;
  M.elements = NULL;

  M.elements = (unsigned char*) malloc(size*sizeof(unsigned char));

  for(unsigned int i=0; i<M.height * M.width; i++ ) {
    M.elements[i] = frame.data[i];
  }

  return M;
}
  
