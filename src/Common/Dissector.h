/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef Dissector_h
#define Dissector_h

#include <pcap.h>
#include <inttypes.h>
#include <iostream>
#include <arpa/inet.h>
#include <algorithm>
#include <fstream>
#include "../Util.h"

//Protocols
#include "Protocols/Ethernet2Header.h"
#include "Protocols/Ip4Header.h"
#include "Protocols/TcpHeader.h"
#include "Protocols/UdpHeader.h"
#include "Protocols/IcmpHeader.h"

using namespace std;

class Dissector {


public:
	unsigned int dissect(const uint8_t* packetPointer,const struct pcap_pkthdr* hdr,const int deviceDataLinkInfo,void* user);
private:
	void dissectEthernet(const uint8_t* packetPointer,unsigned int * totalHeaderLength,const struct pcap_pkthdr* hdr,void* user);

	void dissectIp4(const uint8_t* packetPointer,unsigned int * totalHeaderLength,const struct pcap_pkthdr* hdr,void* user);
	void dissectTcp(const uint8_t* packetPointer,unsigned int * totalHeaderLength,const struct pcap_pkthdr* hdr,void* user);
	void dissectUdp(const uint8_t* packetPointer,unsigned int * totalHeaderLength,const struct pcap_pkthdr* hdr,void* user);
	void dissectIcmp(const uint8_t* packetPointer,unsigned int * totalHeaderLength,const struct pcap_pkthdr* hdr,void* user);
	void payLoadRabinKarp(char* packetPointer);
//Virtual Actions:

	 virtual void EthernetVirtualAction(const uint8_t* packetPointer,unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,Ethernet2Header* header,void* user)=0;
	 virtual void Ip4VirtualAction(const uint8_t* packetPointer,unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,Ip4Header* header,void* user)=0;
	 virtual void Ip4VirtualActionnew(const uint8_t* packetPointer,unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,Ip4Header16* header,void* user)=0;
	 virtual void TcpVirtualAction(const uint8_t* packetPointer,unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,TcpHeader* header,void* user)=0;virtual void UdpVirtualAction(const uint8_t* packetPointer,unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,UdpHeader* header,void* user)=0;
	 virtual void IcmpVirtualAction(const uint8_t* packetPointer,unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,IcmpHeader* header,void* user)=0;
	
	 virtual void EndOfDissectionVirtualAction(unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,void* user)=0;

};
#endif // Dissector_h
