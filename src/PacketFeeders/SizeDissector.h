/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef SizeDissector_h
#define SizeDissector_h

#include <pcap.h>
#include <inttypes.h>
#include <iostream>
#include <arpa/inet.h>

#include "../Util.h"
#include "../Common/Dissector.h"
#include "../Common/PacketBuffer.h"

using namespace std;

class SizeDissector: public Dissector {


public:

private:
	
	//Virtual Actions:
	  void EthernetVirtualAction(const uint8_t* packetPointer,unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,Ethernet2Header* header,void* user);
	  void Ip4VirtualAction(const uint8_t* packetPointer,unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,Ip4Header* header,void* user);
	  void TcpVirtualAction(const uint8_t* packetPointer,unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,TcpHeader* header,void* user);
	  void UdpVirtualAction(const uint8_t* packetPointer,unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,UdpHeader* header,void* user);
	  void IcmpVirtualAction(const uint8_t* packetPointer,unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,IcmpHeader* header,void* user);

	  void EndOfDissectionVirtualAction(unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,void* user);
};
#endif // SizeDissector_h
