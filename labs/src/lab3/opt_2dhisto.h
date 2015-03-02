#ifndef OPT_KERNEL
#define OPT_KERNEL

void opt_2dhisto(uint32_t*, unsigned int*  );

/* Include below the function headers of any other functions that you implement */
void* AllocateDeviceMemory(size_t);

void CopyToDeviceMemory(void*, void*, size_t);

void CopyFromDeviceMemory(void*,  void*, size_t);

void FreeDeviceMemory(void*);



#endif
