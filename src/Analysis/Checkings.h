/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

//Checks preprocessor VARS & other stuff
#ifdef __CUDACC__
	/* Checking if the PrePreProcessor has been executed */
	#ifndef __PPP__
		#error "The PrePreProcessor (PPP) has not been executed or files cannot be found."
	#endif
/* Check user kernels and precoded kernels */
#if  __SYNCBLOCKS_COUNTER>0  && __SYNCBLOCKS_PRECODED_COUNTER>0 
	#error "SYNCBLOCKS cannot be called when using predefined Analysis routines."
#endif

/* CHECK number of threads >= 8 */

#if ANALYSIS_TPB < 8
	#error ANALYSIS_TPB must be greater or equal to 8
#endif

/* Check MAX_BUFFER_PACKETS%ANALYSIS_TPB == 0 */
#define COMPILATIONASSERT_THREADS(expn) typedef char analysis_threads_is_divisor_of_max_buffer_packets[(expn)?1:-1]
COMPILATIONASSERT_THREADS(MAX_BUFFER_PACKETS % ANALYSIS_TPB == 0);

/* CHECK ANALYSIS_INPUT_TYPE ALIGNMENT */
#define COMPILATIONASSERT_INPUT(expn) typedef char analysis_input_type_is_not_aligned_to_4_bytes[(expn)?1:-1]
COMPILATIONASSERT_INPUT(sizeof(ANALYSIS_INPUT_TYPE) % 4 ==0);


/* CHECK ANALYSIS_OUTPUT_TYPE ALIGNMENT */
#ifdef ANALYSIS_OUTPUT_TYPE
	#define COMPILATIONASSERT_OUTPUT(expn) typedef char analysis_output_type_is_not_aligned_to_4_bytes[(expn)?1:-1]
	COMPILATIONASSERT_OUTPUT(sizeof(ANALYSIS_INPUT_TYPE) % 4 ==0);
#endif

#endif
