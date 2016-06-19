#include "PreAnalyzerDissector.h"

void PreAnalyzerDissector::EthernetVirtualAction(const uint8_t* packetPointer,unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,Ethernet2Header* header,void* user){

#if DEBUG_LEVEL > 0
	header->dump();
#endif
}

void PreAnalyzerDissector::Ip4VirtualAction(const uint8_t* packetPointer,unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,Ip4Header* header,void* user){

#if DEBUG_LEVEL > 0
	header->dump();
#endif
		


}

void PreAnalyzerDissector::TcpVirtualAction(const uint8_t* packetPointer,unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,TcpHeader* header,void* user){

#if DEBUG_LEVEL > 0
	header->dump();
	#endif
	
}

void PreAnalyzerDissector::UdpVirtualAction(const uint8_t* packetPointer,unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,UdpHeader* header,void* user){
#if DEBUG_LEVEL > 0
	header->dump();
	#endif
	}

void PreAnalyzerDissector::IcmpVirtualAction(const uint8_t* packetPointer,unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,IcmpHeader* header,void* user){

#if DEBUG_LEVEL > 0
	header->dump();
	#endif
	}

void PreAnalyzerDissector::EndOfDissectionVirtualAction(unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,void* user){

	
}

