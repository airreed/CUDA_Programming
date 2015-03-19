#include <stdio.h>
#include <time.h>
#include <sys/time.h>

#define BLOCK_SIZE 16

typedef struct {
  unsigned int width;
  unsigned int height;
  unsigned int pitch;
  unsigned char* elements;
} Matrix;

#define TIME(_ROUTINE_NAME, _LOOPS, _ACTION) \
{\
  printf("Timing '%s' started\n", _ROUTINE_NAME);\
  struct timeval tv;\
  struct timezone tz;\
  const clock_t startTime = clock();\
  gettimeofday(&tv, &tz); long GTODStartTime = tv.tv_sec * 1000 + tv.tv_usec / 1000;\
  for(int loop = 0; loop<(_LOOPS); ++loops)\
  {\
    _ACTION;\
  }\
  gettimeofday(&tv, &tz); long GTODEndTime = tv.tv_sec * 1000 + tv.tv_usec / 1000;\
  const clock_t endTime = clock();\
  const clock_t elapsedTime = endTime - startTime;\
  const double timeInSeconds = (elapsedTime/(double)CLOCKS_PER_SEC);\
  printf("GetTimeOfDay Time (for %d iterations) = %g\n", _LOOPS, (double)(GTODEndTime - GTODStartTime/ 1000.)); \
  printf("Clock Time (for %d iterations) = %g\n", _LOOPS, timeInSeconds );\
  printf("Timing '%s' ended\n", _ROUTINE_NAME);\
}

#endif
