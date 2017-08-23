# GPU-PCC
This repository contains the source code of GPU-PCC algorithm

## Compilation:

Use the following command for compiling:

nvcc -arch=sm_35 -Xptxas="-dlcm=ca” GPU_PCC.cu -o out

## Running:

./out M N

M: number of elements

N: length of time series


## note:

Make sure that you add the path to your data in the source code

The data should be stored in row major format (first N elements corresponds to time series of first element, second N elements corresponds to time series of first element and …)
