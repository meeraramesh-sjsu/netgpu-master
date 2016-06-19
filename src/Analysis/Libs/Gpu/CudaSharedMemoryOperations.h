/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef CudaSharedMemoryOperations_h
#define CudaSharedMemoryOperations_h

#include "/usr/local/cuda/include/cuda.h"
#include <inttypes.h>

#ifdef __CUDACC__

//TODO: BIG ENDIAN

__device__ __inline__ int8_t cmpZero(uint8_t *x, size_t s)
{
	for(int i= s-1; i >=0; i--) {
		if(x[i] > 0) return 1;
	}
	return 0;
}

template<typename T>
__device__ __inline__ int8_t cudaSharedIsNotNull(T *x){
	return cmpZero((uint8_t*)x,sizeof(T));
}
__device__ __inline__ int8_t cmp(uint8_t *x, uint8_t *y, size_t s)
{
	for(int i= s-1; i >=0; i--) {
		if(x[i] > y[i]) return 1;
		else if(x[i] < y[i]) return -1;
	}
	return 0;
}

template<typename T>
__device__ __inline__ int8_t cudaSharedMemcmp(T *x, T *y){
	return cmp((uint8_t*)x,(uint8_t*)y,sizeof(T));
}


__device__ __inline__ void setZero(uint8_t *x, int8_t value, size_t s)
{
	for(int i= 0;i <s; i++) {
		x[i] = value;
	}
}

template<typename T>
__device__ __inline__ void cudaSharedMemset(T *x,int8_t value){

	setZero((uint8_t*) x,value,sizeof(T));
	
}

#endif //CUDACC

#endif //cudaSharedMemoryOperations



