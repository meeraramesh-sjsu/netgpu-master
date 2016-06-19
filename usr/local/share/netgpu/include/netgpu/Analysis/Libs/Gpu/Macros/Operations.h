/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef Operations_h
#define Operations_h

#include "../../../AnalysisState.h"

#if HAS_WINDOW == 1

	#define IF_WINDOW_LIMIT_REACHED_EXECUTE_CODE_BELOW()\
		SYNCTHREADS();\
		if(threadIdx.x == 0)\
			state.GPU_codeRequiresWLR[blockIdx.x] = 1;\
		SYNCTHREADS()
		

	#define ENDIF_WINDOW_LIMIT_REACHED_EXECUTE_CODE_BELOW()\
		SYNCTHREADS();\
		if(threadIdx.x == 0)\
			state.GPU_codeRequiresWLR[blockIdx.x] = 0;\
		SYNCTHREADS()


	#ifdef __CUDACC__

	bool __device__ checkOperationsCodeExecutionFlag(analysisState_t state){
		__shared__ uint32_t value;
		if(threadIdx.x == 0)
			value = state.GPU_codeRequiresWLR[blockIdx.x];
		SYNCTHREADS();
		return ( (value == 1 && state.windowState.hasReachedWindowLimit) || value == 0 ); 
	}	


	#define IF_OPERATION_HAS_PERMISSION()\
		if(checkOperationsCodeExecutionFlag(state)) 


	#endif //CUDACC
#else
	#ifdef __CUDACC__
		#define IF_OPERATION_HAS_PERMISSION()\
		if(1)

	#endif //CUDACC

#endif //HAS_WINDOW

#endif //Operations_h
