

typedef struct {
  unsigned int width;
  unsigned int height;
  unsigned int pitch;
  unsigned char* elements;
} Matrix;

void bg_caller(const Matrix*, const Matrix, Matrix);

Matrix AllocateDeviceMatrix(const Matrix);

void CopyToDeviceMatrix(Matrix, const Matrix);

void CopyFromDeviceMatrix(Matrix, const Matrix);

void FreeDeviceMatrix(Matrix*);

void FreeMatrix(Matrix*);
