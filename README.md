# CUDA_Programming

##Lab 1

Lab 1 is about matrix multiplication in CUDA.

##Lab 2

This lab is about tiled matrix multiplication in CUDA. I have broken the input and output matrices into tiles. Each thread block should compute one output tile at a time. And I also used shared memory to speed up and added padding to avoid shared memory bank conflict in memory.

##Lab 3

Lab 3 is about histogram calculation in CUDA.

Histograms are a commonly used analysis tool in many application domains, including image processing and data mining. They show the frequency of occurrence of data elements over discrete intervals, also known as bins.
A simple example for the use of histograms is determining the distribution of a set of grades. Say you have a record of grades: 0, 1, 1, 4, 0, 2, 5, 5, 5. The above grades ranging from 0 to 5 result in the following 6-bin histogram, with each bin representing one grade: 2, 2, 1, 0, 1, 3.
