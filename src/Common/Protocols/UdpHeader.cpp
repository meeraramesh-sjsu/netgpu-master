#include "UdpHeader.h"

void UdpHeader::dump(void){

	cout <<"UDP Header ; ";

	cout<< "Port src: "<< ntohs(udp->sport)  <<" Port dst: "<<ntohs(udp->dport) <<endl<<endl;
}
