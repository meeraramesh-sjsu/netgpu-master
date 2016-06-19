/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef SimpleSorter_h
#define SimpleSorter_h

#include <stdlib.h>
#include <inttypes.h>
#include <iostream>
#include <cuda.h>

#include "../../../../../Util.h"
#include "../CudaSharedMemoryOperations.h"

#define SIMPLESORTER_TPB 128

using namespace std;

#ifdef __CUDACC__

//kernels

template<typename T, typename S,typename R> __global__ void sSort(T* GPU_keys, S* GPU_data1,R* GPU_data2,T* GPU_keys_tmp, S* GPU_data1_tmp,R* GPU_data2_tmp,int start,int end)
{
	int i,blockIndex,blockElements,pos,isWorker;
	__shared__ T blockItems[SIMPLESORTER_TPB];
	__shared__ T myValues[SIMPLESORTER_TPB];
	T myValue;

	//Set start 
	pos = start;
	blockIndex = blockIdx.x;
	isWorker = (start+threadIdx.x+blockDim.x*blockIdx.x)<=end;
	if(isWorker)
		myValue = GPU_keys[threadIdx.x+blockIdx.x*blockDim.x];	
	else	
		myValue = 0;

	//Mine first block
	blockItems[threadIdx.x] = myValue;
	myValues[threadIdx.x] = myValue;
	__syncthreads();		

	do{	
		//Mine next block
		if(blockIdx.x != blockIndex && (start+threadIdx.x+blockDim.x*blockIndex)<=end) 
			blockItems[threadIdx.x] = GPU_keys[threadIdx.x+blockIndex*blockDim.x];
		__syncthreads();


		//Count actual blockElements 
		if((start+(blockIndex+1)*blockDim.x-1)>end)
			blockElements = blockDim.x -((start+(blockIndex+1)*blockDim.x -1) - end); 
		else
			blockElements = blockDim.x;

		//Count position elements below 
		for(i=0;i<blockElements;i++){
			/*if(blockItems[i]<myValue ||
			 (blockItems[i] == myValue && (threadIdx.x+blockIdx.x*blockDim.x > i+blockIndex*blockDim.x)))
				pos++;
			*/
			int cmpVal = cudaSharedMemcmp(&blockItems[i],&myValues[threadIdx.x]);
			if(cmpVal<0 || (cmpVal==0 && (threadIdx.x+blockIdx.x*blockDim.x > i+blockIndex*blockDim.x)))
				pos++;


		}	
		__syncthreads();

		//Next block index
		blockIndex++;
		if(blockIndex>=gridDim.x)
			blockIndex -= gridDim.x;
		
	}while(blockIndex != blockIdx.x);

/*	GPU_keys_tmp[threadIdx.x+blockIdx.x*blockDim.x] =-1;
	__syncthreads();
*/
	//Fill array
	if(isWorker == 1){
		GPU_keys_tmp[pos] = myValue;
//		GPU_keys_tmp[threadIdx.x+blockIdx.x*blockDim.x] =j;
	//	GPU_keys_tmp[threadIdx.x+blockIdx.x*blockDim.x] =pos;
		if(GPU_data1!=NULL)
	
		GPU_data1_tmp[pos] = GPU_data1[threadIdx.x+blockIdx.x*blockDim.x];
		if(GPU_data2!=NULL)
			GPU_data2_tmp[pos] = GPU_data2[threadIdx.x+blockIdx.x*blockDim.x];
	
	}
}

//HOST functions
/***************************************************************************************************************/
/* Note wrapper has to be called: wrapper_SimpleSort<type A,type B, type C>(a,start,end,[b],[c])               */
/*   - types B and C, although are optional, MUST be specified, in order to compile                            */
/*                                                                                                             */
/***************************************************************************************************************/
template<typename T, typename S, typename R> void wrapper_SimpleSort(T* GPU_keys, int start, int end, S* GPU_data1=NULL, R* GPU_data2=NULL)
{
	T* GPU_keys_tmp=NULL;
	S* GPU_data1_tmp=NULL;
	R* GPU_data2_tmp=NULL;
	
//	cerr<<"Entro a simple sorter; start: "<<start<<"end: "<<end<<endl;
	//Keys tmp	
        cudaAssert(cudaMalloc((void **)&GPU_keys_tmp,sizeof(T)*(end-start+1)));

//	fprintf(stderr,"GPU_keys_tmp:%p, GPU_keys:%p\n",GPU_keys_tmp,GPU_keys);
	//Data 1 tmp
	if(GPU_data1 != NULL)
	        cudaAssert(cudaMalloc((void **)&GPU_data1_tmp,sizeof(S)*(end-start+1)));

	//Data 2 tmp
	if(GPU_data2 != NULL)
	        cudaAssert(cudaMalloc((void **)&GPU_data2_tmp,sizeof(R)*(end-start+1)));

	//Check cudaMallocs
	if(GPU_keys_tmp == NULL  || (GPU_data1_tmp==NULL&&GPU_data1 != NULL) || (GPU_data2_tmp==NULL&&GPU_data2 != NULL)){
		fprintf(stderr,"Punters: %p->%d, %p->%d, %p->%d\n",GPU_keys,sizeof(T),GPU_data1,sizeof(S),GPU_data2,sizeof(R));
		ABORT("Malloc/CudaMalloc failed at SimpleSorter\n");
	}
	
	//memset tmp keys
        cudaAssert(cudaMemset(GPU_keys_tmp,0,sizeof(T)*(end-start+1)));
	cudaAssert(cudaThreadSynchronize());	
	
	//Start sorting
	sSort<T,S,R><<<((end-start)/SIMPLESORTER_TPB) + 1,SIMPLESORTER_TPB>>>(GPU_keys,GPU_data1,GPU_data2,GPU_keys_tmp,GPU_data1_tmp,GPU_data2_tmp,start,end);
	cudaAssert(cudaThreadSynchronize());

	//Copy results to right array	
	cudaAssert(cudaMemcpy(GPU_keys,GPU_keys_tmp,sizeof(T)*((end-start)+1), cudaMemcpyDeviceToDevice));
	cudaAssert(cudaThreadSynchronize());

	if(GPU_data1!=NULL){
		cudaAssert(cudaMemcpy(GPU_data1,GPU_data1_tmp,sizeof(S)*((end-start)+1), cudaMemcpyDeviceToDevice));
		cudaAssert(cudaThreadSynchronize());
	}
	if(GPU_data2!=NULL){
		cudaAssert(cudaMemcpy(GPU_data2,GPU_data2_tmp,sizeof(R)*((end-start)+1), cudaMemcpyDeviceToDevice));
		cudaAssert(cudaThreadSynchronize());
	}

	//Free tmp arrays
	cudaAssert(cudaFree(GPU_keys_tmp));
	cudaAssert(cudaFree(GPU_data1_tmp));
	cudaAssert(cudaFree(GPU_data2_tmp));
	//cerr<<"surto de simple sorter"<<endl;
}

#endif //ifdef CUDACC


#endif // SimpleSorter_h
