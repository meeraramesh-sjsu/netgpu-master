#include "OfflinePacketFeeder.h"

OfflinePacketFeeder::OfflinePacketFeeder(const char* tcpdumpFile){
	//cout<<"OfflinePacketFeeder";		
	packetBufferArray = new PacketBuffer[SNIFFER_NUM_OF_BUFFERS]();
	bufferIndex = 0;	
	file = tcpdumpFile;
	packetCounter = 0;
	state= OFFLINE_SNIFFER_GO_STATE;

	//INIT MUTEX
	pthread_mutex_init(&mutex,NULL);

	//INIT SEMAPHORE
	waitForSwap = new sem_t;
       	sem_init(waitForSwap,0,0);
	waitForOfflinePacketFeederToEnd = new sem_t;
       	sem_init(waitForOfflinePacketFeederToEnd,0,0);
	
	//TODO: set PCAP buffer size 
}

OfflinePacketFeeder::~OfflinePacketFeeder(void){
	
	//DESTROY ALL
	pthread_mutex_destroy(&mutex);
	delete waitForOfflinePacketFeederToEnd;
	delete waitForSwap;

	//close descriptor
	pcap_close(descr);
	
	//Erase buffers 
	delete [] packetBufferArray;	
}

PacketBuffer* OfflinePacketFeeder::getSniffedPacketBuffer(void){
	
	PacketBuffer* to_return;
	
	if(state == OFFLINE_SNIFFER_END_STATE){
		return NULL; 
	}

	//Waits for buffer to be filled
	sem_wait(waitForOfflinePacketFeederToEnd);

	//LOCK
	pthread_mutex_lock(&mutex);

	//Set to return actual buffer
	to_return = &packetBufferArray[bufferIndex];
	
	//Swaps  buffers and clean new one
	bufferIndex = (bufferIndex+1)%SNIFFER_NUM_OF_BUFFERS;

	DEBUG2("Swapped to buffer: %d",bufferIndex);
	
	packetBufferArray[bufferIndex].clearContent();	

	//Signals sniffer thread
	sem_post(waitForSwap);
	
	if(state == OFFLINE_SNIFFER_LASTBUFFER_STATE){
		state = OFFLINE_SNIFFER_END_STATE; 
	}

	//UNLOCK
	pthread_mutex_unlock(&mutex);
	
	return to_return;		
}

void OfflinePacketFeeder::packetCallback(u_char* sniffer,const struct pcap_pkthdr* pkthdr,const u_char* packet){
	
	cout<<"In packet callback"<<endl;
	//LOCK
	pthread_mutex_lock(&((OfflinePacketFeeder*)sniffer)->mutex);

	//If pushPacket fails (no space) or packetCounter is in the limit-> swap buffers
         //cout<<"call1:In packet Callback";
	if((((OfflinePacketFeeder*)sniffer)->packetBufferArray[((OfflinePacketFeeder*)sniffer)->bufferIndex].pushPacket((uint8_t*)packet,pkthdr)<0) 
		|| (++((OfflinePacketFeeder*)sniffer)->packetCounter == ((OfflinePacketFeeder*)sniffer)->maxPackets)){

		
		//Signal thread waiting to getBuffer (getSniffedPacketBuffer) 
		sem_post(((OfflinePacketFeeder*)sniffer)->waitForOfflinePacketFeederToEnd);

		DEBUG2("Waiting for swap, collected : %d",((OfflinePacketFeeder*)sniffer)->packetCounter);
			
		//UNLOCK		
		pthread_mutex_unlock(&((OfflinePacketFeeder*)sniffer)->mutex);


		//Waits for processor thread to catch buffer 		
		sem_wait(((OfflinePacketFeeder*)sniffer)->waitForSwap);
		//LOCK		
		pthread_mutex_lock(&((OfflinePacketFeeder*)sniffer)->mutex);
		
		//Retry push packet
		((OfflinePacketFeeder*)sniffer)->packetBufferArray[((OfflinePacketFeeder*)sniffer)->bufferIndex].pushPacket((uint8_t*)packet,pkthdr,noOfPatterns);

	}
	//UNLOCK		
	pthread_mutex_unlock(&((OfflinePacketFeeder*)sniffer)->mutex);


	
}

void OfflinePacketFeeder::setDeviceDataLinkInfoToBuffers(int deviceDataLink){

	int i;

	for(i=0;i<SNIFFER_NUM_OF_BUFFERS;i++)
		packetBufferArray[i].setDeviceDataLinkInfo(deviceDataLink);
}

void OfflinePacketFeeder::flushAndExit(void){
	
	//LOCK
	pthread_mutex_lock(&mutex);

	//Set current buffer to be flushed
	packetBufferArray[bufferIndex].setFlushFlag(true);		

	//break pcap_loop
	pcap_breakloop(descr);		

	//Green light for thread waiting to get PacketBuffer (getSniffedPacketBuffer) 
	sem_post(waitForOfflinePacketFeederToEnd);

	//UNLOCK
	pthread_mutex_unlock(&mutex);
	
}
void OfflinePacketFeeder::_start(void){

	cout<<"In start , offline packet feeder"<<endl;
        char errbuf[PCAP_ERRBUF_SIZE];

	/*TODO: FILTERS TO PCAP */

	//Open pcap device
        descr = pcap_open_offline(file,errbuf);	

	//Check if opened correctly	
        if (descr == NULL){
                printf("%s",errbuf);        }

	//Set appropiate datalink info
	setDeviceDataLinkInfoToBuffers(pcap_datalink(descr));

	//PCAP_LOOP
	pcap_loop(descr, -1, OfflinePacketFeeder::packetCallback,(u_char*)this);
	DEBUG2("Exiting pcap loop");

	//Setting state to LASTBUFFER for consumers	
	state = OFFLINE_SNIFFER_LASTBUFFER_STATE; 
	
	sem_post(waitForOfflinePacketFeederToEnd);
	
	pthread_exit(NULL);
}

void* OfflinePacketFeeder::startThreadWrapper(void* object){
	
	((OfflinePacketFeeder*)object)->_start();	
}

pthread_t* OfflinePacketFeeder::start(int limit){
	
	static pthread_t thread;
	static int sLimit=limit;
	
	//Setting limit
	//maxPackets = limit;
	noOfPatterns = limit;
	
	//Creating thread and calling _start routine through wrapper	
	pthread_create(&thread,NULL,OfflinePacketFeeder::startThreadWrapper,(void*)this);
	return &thread;
}
