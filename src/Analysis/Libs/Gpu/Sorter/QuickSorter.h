/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef QuickSorter_h
#define QuickSorter_h

#include <stdlib.h>
#include <inttypes.h>
#include <iostream>
#include <cuda.h>

#include "../../../../../Util.h"
#include "../../../../../Common/PacketBuffer.h"
#include "../CudaSharedMemoryOperations.h"

#define QUICKSORTER_TPB 128
//#define QUICKSORTER_TPB 64

typedef struct{
	uint32_t pivotPosition;
	uint32_t below;
	uint32_t above;
}elementCounters_t;


#ifdef __CUDACC__

//kernels

__device__ inline int compare(const uint8_t* s1, const uint8_t* s2,size_t n)
{
    const uint8_t *p1 = (const uint8_t*)s1, *p2 = (const uint8_t*)s2;
    while(n--)
        if( *p1 != *p2 )
            return *p1 - *p2;
        else
            *p1++,*p2++;
    return 0;
}


template<typename T> __global__ void setPivot(T* GPU_keys,int start,int end,elementCounters_t* GPU_counter)
{
	int i;
	/*int offset, counter,i;
	__shared__ T blockKeys[QUICKSORTER_TPB];		
	__shared__ int blockDiff[QUICKSORTER_TPB];		
*/
       // if((end-start) < blockDim.x){
		if(threadIdx.x == 0){
			for(i=0;i<blockDim.x;i++){
				GPU_counter[i].pivotPosition =(((end-start)/2))+start;
			}
		}
		return;
//	}
        //blockDim.x offset 
       /* offset = (end-start)/blockDim.x;

	//Mining
	blockKeys[threadIdx.x] = GPU_keys[start+offset*threadIdx.x];
	__syncthreads();

	for(i=0,counter=0;i<end-start;i++)
		counter+=blockKeys[i*offset];

	for(i=0,counter=0;i<blockDim.x;i++){
		counter = abs(cudaMemcmp((uint8_t*)&blockKeys[threadIdx.x],(uint8_t*)&blockKeys[i],sizeof(T)));
	}
	
	blockDiff[threadIdx.x] = counter; 
	__syncthreads();

	for(i=0,counter=0;i<blockDim.x;i++){
		if(counter>blockDiff[i])
			return; 
	}
	for(i=0;i<blockDim.x;i++)
		GPU_counter[i].pivotPosition = start+threadIdx.x*offset;*/
}

template<typename T> __global__ void countAboveAndBelowBlockElements(T* GPU_keys, elementCounters_t* GPU_counters,int start,int end)
{	
	int i,pos,elements;
	elementCounters_t regCounters;
	bool isWorker;
	__shared__ T pivotValue;
	__shared__ T blockKeys[QUICKSORTER_TPB];
	
	pos = start+threadIdx.x+blockIdx.x*blockDim.x;
	isWorker = pos<=end;

	if((start+(blockIdx.x+1)*blockDim.x-1)>end)
		elements = blockDim.x -((start+(blockIdx.x+1)*blockDim.x -1) - end); //TODO: delete register
	else
		elements = blockDim.x;
		
		
	//Move pivot to reg (previously on Global Memory, before calling this kernel)
	regCounters.pivotPosition = GPU_counters[blockIdx.x].pivotPosition;
	if(threadIdx.x == 0)
		pivotValue = GPU_keys[regCounters.pivotPosition]; 
	
	//Data Mining
	if(isWorker){		
		blockKeys[threadIdx.x] = GPU_keys[pos];
	}	
	regCounters.above = 0;
	regCounters.below = 0;
	__syncthreads();	

	if(threadIdx.x == 0){
	//count items above and below
		for(i=0;i<elements;i++){
			if((start+blockIdx.x*blockDim.x+i) != regCounters.pivotPosition){
				if(cudaSharedMemcmp(&blockKeys[i],&pivotValue)>0){
				//if(blockKeys[i]>pivotValue){
					regCounters.above++;
				}else{
					regCounters.below++;
				}
			}
		}
		//Fill value
		GPU_counters[blockIdx.x].above = regCounters.above;
		GPU_counters[blockIdx.x].below = regCounters.below;
		
	}
} 

template<typename T, typename S> __global__ void sortBlockElements(T* GPU_keys, S* GPU_data1,T* GPU_keys_result, S* GPU_data1_result, elementCounters_t* GPU_counters,int start,int end,int numOfBlocks,int newPivotPos)
{
	int i, pos,isWorker,left,intraBlockOffset; 
	__shared__ T pivotValue;
	__shared__ elementCounters_t blockCounter;
	__shared__ T blockKeys[QUICKSORTER_TPB];	

	pos = threadIdx.x + blockDim.x*blockIdx.x + start;

	//Data mining
	blockKeys[threadIdx.x] = GPU_keys[pos];

	if(threadIdx.x == 0){
		blockCounter = GPU_counters[blockIdx.x];
	}
	__syncthreads();
	
	//Is worker?
	
	isWorker = (pos<=end && pos != blockCounter.pivotPosition);		
	
	if(threadIdx.x == 0)
		pivotValue = GPU_keys[blockCounter.pivotPosition]; 
	__syncthreads();
	
	left = (blockKeys[threadIdx.x]<=pivotValue);

	//--------------------------------

	if(pos == blockCounter.pivotPosition){	
		GPU_keys_result[newPivotPos] = blockKeys[threadIdx.x];
		if(GPU_data1 != NULL)
			GPU_data1_result[newPivotPos] = GPU_data1[blockCounter.pivotPosition];
	}
	__syncthreads();

	if(isWorker){
		for(i=0,intraBlockOffset=0;start+blockIdx.x*blockDim.x+i<=end&&i<blockDim.x;i++){
			
			if(i == threadIdx.x || start+blockIdx.x*blockDim.x+i == blockCounter.pivotPosition)
				continue;
					
			if(left==1){
				if(blockKeys[i]<blockKeys[threadIdx.x]|| (blockKeys[i]==blockKeys[threadIdx.x] && i<threadIdx.x)) //TODO:cudaShared
					intraBlockOffset++;	
						
			}else{
				if((blockKeys[i]<blockKeys[threadIdx.x] && blockKeys[i]>pivotValue) || (blockKeys[i]==blockKeys[threadIdx.x] && i<threadIdx.x)) //TODO: cudaShared
					intraBlockOffset++;				
		
			}	
		}
		if(left == 1){
			GPU_keys_result[start+blockCounter.below+intraBlockOffset] = blockKeys[threadIdx.x];
			//GPU_keys_result[start+blockCounter.below+intraBlockOffset] = pos;
//			GPU_keys_result[pos] = start+blockCounter.below+intraBlockOffset;
			//GPU_data1_result[pos] = intraBlockOffset;
			if(GPU_data1 != NULL)
				GPU_data1_result[start+blockCounter.below+intraBlockOffset] = GPU_data1[pos];
		}else{
			GPU_keys_result[(newPivotPos+1)+blockCounter.above+intraBlockOffset] = blockKeys[threadIdx.x];
//			GPU_keys_result[(newPivotPos+1)+blockCounter.above+intraBlockOffset] =pos;
			//GPU_keys_result[pos] = (newPivotPos+1)+blockCounter.above+intraBlockOffset;
			//GPU_data1_result[pos] = (newPivotPos+1)+blockCounter.above+intraBlockOffset;
			if(GPU_data1 != NULL)
				GPU_data1_result[(newPivotPos+1)+blockCounter.above+intraBlockOffset] = GPU_data1[pos];
		}	
	}
	
}

template<typename T, typename S> __host__ void sort(T* GPU_keys, S* GPU_data1,T* GPU_keys_tmp, S* GPU_data1_tmp,int start,int end)
{
	int i,numOfBlocks,accumulativeBelow,accumulativeAbove,newPivotPos;
	elementCounters_t *GPU_elementCounters, *elementCounters;
	
	uint32_t *result = (uint32_t*)malloc(sizeof(uint32_t)*MAX_BUFFER_PACKETS);
	
	if(start>=end){ 
		return;
	}	
	
	//Calculate #Blocks
	numOfBlocks = (end-start)/QUICKSORTER_TPB;

	if((end-start)%QUICKSORTER_TPB>0)
		numOfBlocks++;

	//fprintf(stderr,"PASSO start: %d, end: %d,  BLOCKS: %d\n",start,end,numOfBlocks);

	//Allocate memory & create Stream
	cudaMalloc((void **)&GPU_elementCounters,sizeof(elementCounters_t)*numOfBlocks);
	elementCounters = (elementCounters_t*)malloc(sizeof(elementCounters_t)*numOfBlocks);

	cudaThreadSynchronize();
	
	if(GPU_elementCounters == NULL  || elementCounters == NULL){
		ABORT("Malloc/CudaMalloc failed at QuickSorter\n");
	}

	//Set Pivot
	setPivot<T><<<1,numOfBlocks>>>(GPU_keys,start,end,GPU_elementCounters); //elementCounters	
	cudaThreadSynchronize();

	//Count elements
	countAboveAndBelowBlockElements<T><<<numOfBlocks,QUICKSORTER_TPB>>>(GPU_keys,GPU_elementCounters,start,end);
	cudaThreadSynchronize();

	//Accumulate (CPU)
	cudaMemcpy(elementCounters,GPU_elementCounters,sizeof(elementCounters_t)*numOfBlocks, cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();

	for(i=0,accumulativeBelow=0,accumulativeAbove=0;i<numOfBlocks;i++){
		accumulativeBelow += elementCounters[i].below; 
		accumulativeAbove += elementCounters[i].above;
	}
	newPivotPos = start+accumulativeBelow;


	for(i=(numOfBlocks-1);i>=0;i--){
		accumulativeBelow -= elementCounters[i].below;
		elementCounters[i].below = accumulativeBelow; 
		accumulativeAbove -= elementCounters[i].above;
		elementCounters[i].above = accumulativeAbove;
	}	

	cudaMemcpy(GPU_elementCounters,elementCounters,sizeof(elementCounters_t)*numOfBlocks, cudaMemcpyHostToDevice);
	cudaThreadSynchronize();
		

	//Sort
	sortBlockElements<T,S><<<numOfBlocks,QUICKSORTER_TPB>>>(GPU_keys,GPU_data1,GPU_keys_tmp,GPU_data1_tmp,GPU_elementCounters,start,end,numOfBlocks,newPivotPos);
	cudaThreadSynchronize();

	cudaMemcpy(GPU_keys+start,GPU_keys_tmp+start,sizeof(T)*((end-start)+1), cudaMemcpyDeviceToDevice);
	cudaThreadSynchronize();

	if(GPU_data1!=NULL){
		cudaMemcpy(GPU_data1+start,GPU_data1_tmp+start,sizeof(S)*((end-start)+1), cudaMemcpyDeviceToDevice);
		cudaThreadSynchronize();
	}
	
	//Free memory on GPU	
	cudaFree(GPU_elementCounters);
	free(elementCounters);
	
	//Recursive call for 2 parts
	if((end-start)> 2){
		if(((newPivotPos-1)-start)>0)
			sort(GPU_keys,GPU_data1,GPU_keys_tmp,GPU_data1_tmp,start,newPivotPos-1);
		
		if((end-(newPivotPos+1))>0)
			sort(GPU_keys,GPU_data1,GPU_keys_tmp,GPU_data1_tmp,newPivotPos+1,end);
	}
}

//HOST functions
template<typename T, typename S> void wrapper_QuickSort(T* GPU_keys, S* GPU_data1,int start,int end)
{
	T *GPU_keys_tmp;
	S *GPU_data1_tmp;
	
        cudaMalloc((void **)&GPU_keys_tmp,sizeof(T)*(end-start+1));
	if(GPU_data1 != NULL)
	        cudaMalloc((void **)&GPU_data1_tmp,sizeof(S)*(end-start+1));
	else
		GPU_data1_tmp = NULL;

        cudaMemset(GPU_keys_tmp,0,sizeof(T)*(end-start+1));

	cudaThreadSynchronize();	

	//Start sorting
	sort<T,S>(GPU_keys,GPU_data1,GPU_keys_tmp,GPU_data1_tmp,start,end);
}

#endif //ifdef CUDACC


#endif // QuickSorter_h
