/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

/*Include skeleton */
#include "../Scheduler/Scheduler.h"
#include "AnalysisSkeleton.h"
#include "Libs/Host/Database/QueryManager.h"
#include "Libs/Host/Database/DatabaseManager.h"


//If output type is not defined, use the same as input type
#ifndef ANALYSIS_OUTPUT_TYPE
	#define ANALYSIS_OUTPUT_TYPE ANALYSIS_INPUT_TYPE
#endif

class ANALYSIS_NAME:public AnalysisSkeleton{

public:
	static void launchAnalysis(PacketBuffer* packetBuffer, packet_t* GPU_buffer);
	static QueryManager queryManager;
	static int numOfPatterns;
private:
};


#ifdef __CUDACC__ /* Don't erase this */

QueryManager ANALYSIS_NAME::queryManager; //Scheduler::dbManager->getManager() );

/* Launch analysis method */
void ANALYSIS_NAME::launchAnalysis(PacketBuffer* packetBuffer, packet_t* GPU_buffer){

	//Launch Analysis (wrapper from C++ to C)
	COMPOUND_NAME(ANALYSIS_NAME,launchAnalysis_wrapper)<ANALYSIS_INPUT_TYPE,ANALYSIS_OUTPUT_TYPE>(packetBuffer, GPU_buffer,numOfPatterns);
	
}
#endif //ifdef CUDACC
