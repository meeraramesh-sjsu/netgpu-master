#include "TcpHeader.h"

void TcpHeader::dump(void){

	cout <<"TCP Header ; ";
	cout<< "Port src: "<< ntohs(tcp->sport)  <<" Port dst: "<< ntohs(tcp->dport) <<endl<<endl;
}

uint8_t TcpHeader::getHeaderLength(void){
	return tcp->offset; 
}
uint32_t TcpHeader::getHeaderLengthInBytes(void){
	return tcp->offset*4; 
}
uint32_t TcpHeader::calcHeaderLengthInBytes(const uint8_t * tcpPointer){
	return (uint32_t)(((struct tcp_header*)tcpPointer)->offset*4); 
}
