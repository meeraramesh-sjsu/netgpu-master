#include "PacketBuffer.h"

PacketBuffer::PacketBuffer(void){
	
	//Allocate memory for buffer
	buffer = new packet_t[MAX_BUFFER_PACKETS];
	memset(buffer,0,sizeof(packet_t)*MAX_BUFFER_PACKETS);
	lastPacketIndex = 0;
	lostPackets = 0;
	flushWindows = false;
}

PacketBuffer::~PacketBuffer(void){
	
//	delete [] buffer;
	
}

void PacketBuffer::clearContent(void){
	
	lastPacketIndex = 0;
	lostPackets = 0;
	
}

void PacketBuffer::setDeviceDataLinkInfo(int deviceDataLinkInfo){

	deviceDataLink = deviceDataLinkInfo;
}

int PacketBuffer::getDeviceDataLinkInfo(void){
	return deviceDataLink;
}
unsigned int PacketBuffer::getNumOfPackets(void){
	return lastPacketIndex;
}

unsigned int PacketBuffer::getNumOfLostPackets(void){
	return lostPackets;
}

packet_t* PacketBuffer::getBuffer(void){
	return buffer;
}
/*Push packet into stack */
int PacketBuffer::pushPacket(uint8_t* packetPointer, const struct pcap_pkthdr* hdr,int noOfPatterns){

	cout<<"call2:in push packet";
//	SizeDissector sizeDissector;
	PreAnalyzerDissector preAnalyzerDissector;
	int totalLength;
	headers_t headers;	

	//Check for space in buffer

	if((lastPacketIndex+1)>=MAX_BUFFER_PACKETS){
		DEBUG2("No positions left");
		return -1;
	}

	//0 headers

	memset(&headers,0,sizeof(headers_t));
	
	//get size from packet(headers) & fill headers
	totalLength = preAnalyzerDissector.dissect(packetPointer,hdr,deviceDataLink,&headers,noOfPatterns);
	//totalLength = sizeDissector.dissect(packetPointer,hdr,deviceDataLink,&headers);
	
	if(totalLength<0){
		//Not supported
		lostPackets++;
		return 0;
	}
	
	DEBUG2("Trying to push packet");
	if(totalLength > 94) totalLength = 94;
	//Check Size of packet
/*	if(totalLength+TIMESTAMP_OFFSET>MAX_BUFFER_PACKET_SIZE){

		lostPackets++;
		DEBUG2("Packet discarted: >LIMIT:%d",MAX_BUFFER_PACKET_SIZE);
		return 0;
	}*/

	
	//Copy timestamp 
	//buffer[lastPacketIndex].timestamp = hdr->ts;
	
	//Copy headers
	
	//Insert capt into Vector & increment counter
	lastPacketIndex++;

	return 1;
}

packet_t* PacketBuffer::getPacket(int index){
	
	return &buffer[index];
}

