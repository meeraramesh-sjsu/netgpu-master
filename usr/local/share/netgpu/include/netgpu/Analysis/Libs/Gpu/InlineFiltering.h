/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef InlineFiltering_h
#define InlineFiltering_h

#include <inttypes.h>
#include <iostream>
#include <arpa/inet.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "CudaSafeAccess.h"
#include "CudaSharedMemoryOperations.h"

enum FilterOps{
	Equal,			// ==
      	NotEqual,		// !=
	LessThan,		// <
	GreaterThan,		// >
	LessOrEqualThan,	// <=
	GreaterOrEqualThan,	// >=
      	InRangeStrict,		// ()
      	NotInRangeStrict,	// !()
      	InRange,		// []
      	NotInRange		// ![]
};

//Prototypes

#ifdef __CUDACC__

template<FilterOps op, int analysis_tpb, typename T, typename R> 
__device__ __inline__ void inlineFilter(T* GPU_toCompare, R* GPU_data, T op1){
	inlineFilter<op,analysis_tpb,T>(GPU_toCompare,GPU_data, op1, op1);
}


template<FilterOps op, int analysis_tpb, typename T, typename R>
__device__ __inline__ void inlineFilter(T* GPU_toCompare,R* GPU_data, T op1, T op2){

	bool comp;
	__shared__ T elements[analysis_tpb]; 
	__shared__ T sOp1;
	__shared__ T sOp2;
	__shared__ R nullElement;

	if(threadIdx.x == 0){
		sOp1 = op1;
		sOp2 = op2;
		cudaSharedMemset(&nullElement,0);
	}
	
	elements[threadIdx.x] = cudaSafeGet(GPU_toCompare);
	__syncthreads();

	//Perform comparison 
	switch(op){

		case Equal: comp = (cudaSharedMemcmp(&elements[threadIdx.x],&sOp1)==0); 
			break;
		case NotEqual:comp =(cudaSharedMemcmp(&elements[threadIdx.x],&sOp1)!=0);
			break;
		case LessThan:comp = (cudaSharedMemcmp(&elements[threadIdx.x],&sOp1)<0);
			break;
		case LessOrEqualThan:comp = (cudaSharedMemcmp(&elements[threadIdx.x],&sOp1)<=0);
			break;
		case GreaterThan:comp = (cudaSharedMemcmp(&elements[threadIdx.x],&sOp1)>0);
			break;
		case GreaterOrEqualThan:comp = (cudaSharedMemcmp(&elements[threadIdx.x],&sOp1)>=0);
			break;
		case InRangeStrict: comp = ((cudaSharedMemcmp(&elements[threadIdx.x],&sOp1)>0)&&(cudaSharedMemcmp(&elements[threadIdx.x],&sOp2)<0));
			break;
		case NotInRangeStrict: comp = !((cudaSharedMemcmp(&elements[threadIdx.x],&sOp1)>0)&&(cudaSharedMemcmp(&elements[threadIdx.x],&sOp2)<0));
			break;
		case InRange: comp = ((cudaSharedMemcmp(&elements[threadIdx.x],&sOp1)>=0)&&(cudaSharedMemcmp(&elements[threadIdx.x],&sOp2)<=0));
			break;
		case NotInRange: comp = !((cudaSharedMemcmp(&elements[threadIdx.x],&sOp1)>=0)&&(cudaSharedMemcmp(&elements[threadIdx.x],&sOp2)<=0));
			break;
	}

	//move Data 	
	if(comp)
		*GPU_data = nullElement; 
		
}

#endif //__CUDACC__


#endif //InlineFiltering_h
