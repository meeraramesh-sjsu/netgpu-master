/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef Scheduler_h
#define Scheduler_h

#include <pcap.h>
#include <inttypes.h>
#include <iostream>
#include <arpa/inet.h>
#include "/usr/local/cuda/include/cuda.h"
#include "/usr/local/cuda/include/cuda_runtime.h"

#include "../Util.h"
#include "../Common/PacketBuffer.h"
#include "../Common/PacketFeeder.h"
#include "../Analysis/Libs/Host/GpuMemoryManagement/BMMS.h"
#include "../Analysis/Libs/Host/Database/DatabaseManager.h"
#include "../Analysis/Libs/Host/Database/ODBCDatabaseManager.h"

#define SCHEDULER_MAX_ANALYSIS_POOL_SIZE 256
#define SCHEDULER_MAX_FEEDERS_POOL_SIZE 1 //DO NOT MODIFY. Still not able to handle more than 1 feeder at the time

typedef struct{
	PacketFeeder* feeder;
	pthread_t* thread;
}feeders_t;

using namespace std;

class Scheduler{
public:

	static void start(int noOfPatterns);
	static void term(void);
	static DatabaseManager* dbManager;

	//Add to analysis Pool
	static void addAnalysisToPool(void (*func)(PacketBuffer* packetBuffer, packet_t* GPU_buffer,int noOfPatterns));

	//Add to feeders pool
	static void addFeederToPool(PacketFeeder* feeder,int limit=-1);
	//Program handler routine

private:	
	static void init(void);
	static void programHandler(void);
	static void analyzeBuffer(PacketBuffer* buffer,int noOfPatterns);

	static packet_t* loadBufferToGPU(PacketBuffer* packetBuffer);
	static void unloadBufferFromGPU(packet_t* GPU_buffer);

	//Analysis Pointers Pool
	static	void (*analysisFunctions[SCHEDULER_MAX_ANALYSIS_POOL_SIZE])(PacketBuffer* packetBuffer, packet_t* GPU_buffer, int noOfPatterns);

	//Feeders Pool
	static feeders_t feedersPool[SCHEDULER_MAX_FEEDERS_POOL_SIZE]; 	     
};

#endif // Scheduler_h
