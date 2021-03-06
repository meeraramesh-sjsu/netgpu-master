/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef OfflinePacketFeeder_h
#define OfflinePacketFeeder_h

#include <pcap.h>
#include <cstring>
#include <cstdlib>
#include <inttypes.h> 
#include <pthread.h> 
#include <semaphore.h> 


#include "../Util.h"
#include "../Common/PacketFeeder.h"
#include "SizeDissector.h"

#define CAPTURING_TIMEms 1000
#define SNIFFER_BUFFER_SIZE 8192
#define SNIFFER_NUM_OF_BUFFERS 2


#define OFFLINE_SNIFFER_GO_STATE 0
#define OFFLINE_SNIFFER_LASTBUFFER_STATE 1
#define OFFLINE_SNIFFER_END_STATE 2

using namespace std;

class OfflinePacketFeeder:public PacketFeeder {

public:

	OfflinePacketFeeder(const char* file);
	~OfflinePacketFeeder(void);
	pthread_t* start(int limit);	
	
	static void packetCallback(u_char *useless,const struct pcap_pkthdr* pkthdr,const u_char* packet);

	PacketBuffer* getSniffedPacketBuffer(void);
	
	void flushAndExit(void);
	static int noOfPatterns;

private:
	//PCAP descriptor
        pcap_t* descr;
	
	//Counter and limit
	int packetCounter;
	int maxPackets;

	//Array of 2 packetBuffers and actualindex
	PacketBuffer* packetBufferArray;
	short int bufferIndex;	

	//Device name
	const char* file;

	//State
	int state;

	//Mutex
	pthread_mutex_t mutex;

	//Synchronization semaphore
	sem_t* waitForSwap;
	sem_t* waitForOfflinePacketFeederToEnd;

	void _start(void);
	static void* startThreadWrapper(void* object);	
	inline void setDeviceDataLinkInfoToBuffers(int deviceDataLink);
};



#endif // OfflinePacketFeeder_h
