/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef Filtering_h
#define Filtering_h

/* Filters data/results by its own fields */

#define PRE_FILTER(field_to_compare,operation,op1,...)\
		inlineFilter<operation,ANALYSIS_TPB>(&field_to_compare,&(GPU_data[POS]),op1,##__VA_ARGS__)

#define POST_FILTER(field_to_compare,operation,op1,...)\
		do{\
			state.blockIterator += blockIdx.x;\
			while( state.blockIterator < state.windowState.totalNumberOfBlocks ){\
				inlineFilter<operation,ANALYSIS_TPB>(&(field_to_compare),&(GPU_results[POS]),op1,##__VA_ARGS__);\
				state.blockIterator += gridDim.x;\
			}\
		}while(0)


/* Filters by a field contained in GPU_buffer */
/* TODO: FILTER_BY_EXTERN_FIELD*/

#define PRE_FILTER_BY_EXTERN_FIELD(cond)\
	if(cond)\
		cudaSharedMemset(&DATA_ELEMENT,0);

#define POST_FILTER_BY_EXTERN_FIELD(cond)\
	if(cond)\
		cudaSharedMemset(&DATA_ELEMENT,0);




/*End of file */
#endif //Filtering_h

