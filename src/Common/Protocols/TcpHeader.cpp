#include "TcpHeader.h"

void TcpHeader::dump(void){

	cout <<"TCP Header ; ";
	cout<< "Port src: "<< ntohs(tcp->sport)  <<" Port dst: "<< ntohs(tcp->dport) <<endl<<endl;

	//Checking if the TCP flags are malicious
	if(tcp->flags == 3 ||
		tcp->flags == 11 || tcp->flags == 7 ||
		tcp->flags == 15 || tcp->flags == 1 ||
		tcp->flags == 0 || tcp->unused != 0)
		{
		if(tcp->unused == 0) cout<<"Malicious flag bit combinations found"<<endl;
		else cout<<"The reserved bits are set"<<endl;
		}

		/*Checking if the source or Destination port is 0*/
		if(tcp->sport == 0 || tcp->dport == 0)
		{
		cout<<"Error! Source or destination port is zero"<<endl;
		}

		/*Checking if the ackowledgement bit is set, than the ACK number should not be zero*/
		if((tcp->flags & 16) ==1 && tcp->ack==0)
			cout<<"The ack bit is set, but the ack number is zero"<<endl;

		//Printing the payload
	/*	char* payload=tcp->data;
		int i=0;
		while(payload!='\0') {
			cout<<payload[i++];
		   payload++;
		}
		cout<<endl;*/
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
