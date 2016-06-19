#include "Ethernet2Header.h"

#include <arpa/inet.h>

void Ethernet2Header::dump(void){
	
	cout <<"Ethernet 2 Header ; ";
	cout <<"Ethernet type (onboard): "<<getType()<<endl<<endl;
}
uint16_t Ethernet2Header::getType(void){
	
	return ntohs(ether->type); 
	
}

