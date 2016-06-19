/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef AnalysisState_h
#define AnalysisState_h

#include "Windows.h"

#define MAX_INPUT_EXTENDED_PARAMETERS 5

/**** INPUTS ****/

	//Inputs struct
	typedef struct{
		//int64_t[MAX_INPUT_EXTENDED_PARAMETERS] Used to pass parameters from kernel to kernel(Extra Kernels)
		int64_t* GPU_extendedParameters; 
	}analysisInputs_t;

/**** OUTPUTS ****/	
	//Statistical outputs
	typedef struct{
		float min;
		float max;
		float mean;
		float std;	
	}analysisStatisticalOutputs_t;
	
	//Outputs struct
	typedef struct{
		analysisStatisticalOutputs_t* GPU_statics;			
	}analysisOutputs_t;


/**** STATE STRUCT ****/	
typedef struct{
	/* Analysis state*/
	int32_t blockIterator; //Iterator (windowed Analysis)
	uint32_t lastPacket; //Last packet NOT NULL
	
	/* Operations Code Execution flags */
	uint32_t* GPU_codeRequiresWLR;

	/* Auxiliar Arrays (pointers) */
	void* GPU_aux;		
	int64_t* GPU_auxBlocks;
	
	/* Window State */
	windowState_t windowState;

	/* INPUTS */
	analysisInputs_t inputs;

	/* OUTPUTS */
	analysisOutputs_t outputs;
}analysisState_t;


#endif //AnalysisState_h
