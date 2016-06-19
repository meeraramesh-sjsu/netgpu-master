#include "InlineFiltering.h"

#if 0
int main(int args, char* argv[]){
	uint16_t* GPU_items,*items;
	
	items = (uint16_t*)malloc(sizeof(uint16_t)*MAX_BUFFER_PACKETS);

	//wrapper_TcpDestPortsMiner(GPU_buffer,0, MAX_BUFFER_PACKETS, &GPU_items);
	
	//Volcat	
	cudaAssert(cudaMemcpy(items,GPU_items,sizeof(uint16_t)*MAX_BUFFER_PACKETS,cudaMemcpyDeviceToHost));	

	//Dump
	fprintf(stderr,"Volcant ports abans filtre:\n [");
	for(int i=0;i<MAX_BUFFER_PACKETS;i++)
		fprintf(stderr,"%d:%d,",i,items[i]);
	fprintf(stderr,"]\n\n");

	//Filter	
	wrapper_filter<uint16_t,GreaterThan>(GPU_items,MAX_BUFFER_PACKETS, 80,80);
//	GPU_items = wrapper_mineAndFilter<uint16_t,GreaterThan>(GPU_buffer,MAX_BUFFER_PACKETS,12, 80,80);

	//Volcat	
	cudaAssert(cudaMemcpy(items,GPU_items,sizeof(uint16_t)*MAX_BUFFER_PACKETS,cudaMemcpyDeviceToHost));	

	//Dump
	fprintf(stderr,"Volcant ports despres filtre:\n [");
	for(int i=0;i<MAX_BUFFER_PACKETS;i++)
		fprintf(stderr,"%d:%d,",i,items[i]);
	fprintf(stderr,"]\n\n");
	
	BMMS::freeBMMS(GPU_items);
	return 0;
}

#endif
