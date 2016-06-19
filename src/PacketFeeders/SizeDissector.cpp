#include "SizeDissector.h"

void SizeDissector::EthernetVirtualAction(const uint8_t* packetPointer,unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,Ethernet2Header* header,void* user)
{//      cout<<"In Ethernet Virtual Action"; 

	INSERT_HEADER_ETHERNET((headers_t*)user,2, *totalHeaderLength);
}

void SizeDissector::Ip4VirtualAction(const uint8_t* packetPointer,unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,Ip4Header* header,void* user)
{

	INSERT_HEADER_IP4((headers_t*)user,3, *totalHeaderLength);

}

void SizeDissector::TcpVirtualAction(const uint8_t* packetPointer,unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,TcpHeader* header,void* user)
{
	INSERT_HEADER_TCP((headers_t*)user,4, *totalHeaderLength);

}

void SizeDissector::UdpVirtualAction(const uint8_t* packetPointer,unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,UdpHeader* header,void* user)
{

	INSERT_HEADER_UDP((headers_t*)user,4, *totalHeaderLength);
}

void SizeDissector::IcmpVirtualAction(const uint8_t* packetPointer,unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,IcmpHeader* header,void* user)
{

	INSERT_HEADER_ICMP((headers_t*)user,4, *totalHeaderLength);
}
void SizeDissector::EndOfDissectionVirtualAction(unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,void* user)
{

}


