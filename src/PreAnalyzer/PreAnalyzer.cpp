#include "PreAnalyzer.h"

void PreAnalyzer::dumpBufferStats(PacketBuffer* buffer){
	
	unsigned int packets, lost;
	float lp;
	
	packets = buffer->getNumOfPackets();
	lost = buffer->getNumOfLostPackets();
	lp = lost/(lost+packets);

	cout<<"In buffer: "<<packets<<", Lost: "<<lost<<", LP: "<<lp<<endl;

}


void PreAnalyzer::preAnalyze(PacketBuffer* buffer){
	static int totalPackets = 0;	
	static int lostPackets = 0;	
	int unsigned i, num_of_packets;
	int deviceDataLink;
	

	//initializing constants
	deviceDataLink = buffer->getDeviceDataLinkInfo();	
	num_of_packets = buffer->getNumOfPackets();	
	
	//PACKETS	
	for(i=0;i<num_of_packets;i++){
#if DEBUG_LEVEL > 0
		cout<<endl<<"Packet nº: "<<i<<endl<<"-------------"<<endl;
#endif
		preAnalyzerDissector.dissect(buffer->getPacket(i)->packet,NULL,deviceDataLink,NULL);
//TODO: eliminar		dissect_wrapper_prova(buffer->getPacket(i)+TIMESTAMP_OFFSET,NULL,deviceDataLink,NULL);
	}
	
#if DEBUG_LEVEL > 0
	totalPackets += buffer->getNumOfPackets();	
	lostPackets += buffer->getNumOfLostPackets();	
	cerr<<"TOTAL packets:"<<totalPackets<<endl;
	cerr<<"TOTAL lost packets:"<<lostPackets<<endl;
#endif
}

