#include "Ip4Header.h"

void Ip4Header::dump(void){

	cout <<"IP4 Header ; ";

	fprintf(stdout,"Src address: %s ",inet_ntoa(*(struct in_addr*)&ip4->ip_src));
	fprintf(stdout,"Dst address: %s \n\n",inet_ntoa(*(struct in_addr*)&ip4->ip_dst));
}

uint8_t Ip4Header::getHeaderLength(void){
	return ip4->headerLength;
}
uint32_t Ip4Header::getHeaderLengthInBytes(void){
	return (uint32_t)ip4->headerLength*4;
}
uint8_t Ip4Header::getProtocol(void){

	return ip4->protocol; 
}
uint32_t Ip4Header::calcHeaderLengthInBytes(const uint8_t * ipPointer){
	return (uint32_t)((struct ip4_header *)ipPointer)->headerLength*4;
}
uint32_t Ip4Header::totalPacketLength(const uint8_t * ipPointer)
		{
	return (uint32_t) ntohs(((struct ip4_header *)ipPointer)->totalLength) + 14;
		}
