#define BLOCK_SIZE 16

typedef struct {
  unsigned int width;
  unsigned int height;
  unsigned int pitch;
  unsigned char* elements;
} Matrix;

