#include "LivePacketFeeder.h"

LivePacketFeeder::LivePacketFeeder(const char* device){
	
	packetBufferArray = new PacketBuffer[SNIFFER_NUM_OF_BUFFERS]();
	bufferIndex = 0;	
	dev = device;
	packetCounter = 0;
	state= SNIFFER_GO_STATE;

	//INIT MUTEX
	pthread_mutex_init(&mutex,NULL);

	//INIT SEMAPHORES
	waitForSwap = new sem_t;
       	sem_init(waitForSwap,0,0);
	waitForLivePacketFeederToEnd = new sem_t;
       	sem_init(waitForLivePacketFeederToEnd,0,0);
	
	//TODO: set PCAP buffer size 
}

LivePacketFeeder::~LivePacketFeeder(void){
	
	//DESTROY ALL
	pthread_mutex_destroy(&mutex);
	delete waitForLivePacketFeederToEnd;
	delete waitForSwap;

	//close descriptor
	pcap_close(descr);
	
	//Erase buffers
	delete [] packetBufferArray;	
}

PacketBuffer* LivePacketFeeder::getSniffedPacketBuffer(void){
	
	PacketBuffer* to_return;
	
	if(state == SNIFFER_END_STATE){
		return NULL;
	}


	//Waits for buffer to be filled
	sem_wait(waitForLivePacketFeederToEnd);

	//LOCK
	pthread_mutex_lock(&mutex);

	//Set to return actual buffer
	to_return = &packetBufferArray[bufferIndex];
	
	//Swaps  buffers and clean new one
	bufferIndex = (bufferIndex+1)%SNIFFER_NUM_OF_BUFFERS;

	DEBUG2("Swapped to buffer: %d",bufferIndex);
	
	packetBufferArray[bufferIndex].clearContent();	

	//Green light to sniffer thread
	sem_post(waitForSwap);
	
	if(state == SNIFFER_LASTBUFFER_STATE){
		state = SNIFFER_END_STATE; 
	}

	//UNLOCK
	pthread_mutex_unlock(&mutex);
	
	return to_return;		
}

void LivePacketFeeder::packetCallback(u_char* sniffer,const struct pcap_pkthdr* pkthdr,const u_char* packet){
	
	int noOfPatterns = 0;
	//LOCK
	pthread_mutex_lock(&((LivePacketFeeder*)sniffer)->mutex);

	//If pushPacket fails (no space) or packetCounter is in the limit -> Swap buffers

	if(((LivePacketFeeder*)sniffer)->packetBufferArray[((LivePacketFeeder*)sniffer)->bufferIndex].pushPacket((uint8_t*)packet,pkthdr,noOfPatterns)<0
		|| ++((LivePacketFeeder*)sniffer)->packetCounter == ((LivePacketFeeder*)sniffer)->maxPackets){

		
		//Green light for thread waiting to get PacketBuffer (getSniffedPacketBuffer) 
		sem_post(((LivePacketFeeder*)sniffer)->waitForLivePacketFeederToEnd);

		DEBUG2("Waiting for swap, collected : %d",((LivePacketFeeder*)sniffer)->packetCounter);
		
		//UNLOCK		
		pthread_mutex_unlock(&((LivePacketFeeder*)sniffer)->mutex);


		//Waits for processor thread to catch buffer
		sem_wait(((LivePacketFeeder*)sniffer)->waitForSwap);
		
		//LOCK		
		pthread_mutex_lock(&((LivePacketFeeder*)sniffer)->mutex);

		//Retry push packet
		((LivePacketFeeder*)sniffer)->packetBufferArray[((LivePacketFeeder*)sniffer)->bufferIndex].pushPacket((uint8_t*)packet,pkthdr);
	}
	//UNLOCK		
	pthread_mutex_unlock(&((LivePacketFeeder*)sniffer)->mutex);


	
}

void LivePacketFeeder::setDeviceDataLinkInfoToBuffers(int deviceDataLink){

	int i;

	for(i=0;i<SNIFFER_NUM_OF_BUFFERS;i++)
		packetBufferArray[i].setDeviceDataLinkInfo(deviceDataLink);
}

void LivePacketFeeder::flushAndExit(void){
	
	//LOCK
	pthread_mutex_lock(&mutex);

	//Set current buffer to be flushed
	packetBufferArray[bufferIndex].setFlushFlag(true);		

	//break pcap_loop
	pcap_breakloop(descr);		

	//Green light for thread waiting to get PacketBuffer (getSniffedPacketBuffer) 
	sem_post(waitForLivePacketFeederToEnd);

	//UNLOCK
	pthread_mutex_unlock(&mutex);
	
}


void LivePacketFeeder::_start(void){

        char errbuf[PCAP_ERRBUF_SIZE];

	//Open pcap device
        descr = pcap_open_live(dev,SNIFFER_BUFFER_SIZE,1,CAPTURING_TIMEms,errbuf);	

	//TODO: Set Buffer size & timeout
	// pcap_set_buffer_size()
	// pcap_set_timeout() 
            
	//Check if opened correctly	
        if (descr == NULL){
                printf("ERROR:  %s\n",errbuf);
		exit(1);
        }

	//Set appropiate datalink info
	setDeviceDataLinkInfoToBuffers(pcap_datalink(descr));

	//PCAP_LOOP
	pcap_loop(descr, -1, LivePacketFeeder::packetCallback,(u_char*)this);

	DEBUG2("Exiting pcap loop");

	//Setting state to LASTBUFFER for consumers	
	state = SNIFFER_LASTBUFFER_STATE; 
	sem_post(waitForLivePacketFeederToEnd);
	
	pthread_exit(NULL);
}

void* LivePacketFeeder::startThreadWrapper(void* object){

	((LivePacketFeeder*)object)->_start();	
}

pthread_t* LivePacketFeeder::start(int limit){
	
	static pthread_t thread;
	static int sLimit = limit;

	//Setting limit
	maxPackets = limit;
	
	//Creating thread and calling _start routine through wrapper	
	pthread_create(&thread,NULL,LivePacketFeeder::startThreadWrapper,(void*)this);

	return &thread;
}
