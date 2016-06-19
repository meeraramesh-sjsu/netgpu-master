#include "QuickSorter.h"

//KERNELS



//Host functions


#if 0
int main(int args, char *argv[]){
        int i;
	int counter;	
	uint32_t *a,*b,*result,*result2;
	cudaMalloc((void **)&a,sizeof(uint32_t)*MAX_BUFFER_PACKETS);
        cudaMalloc((void **)&b,sizeof(uint32_t)*MAX_BUFFER_PACKETS);
	result = (uint32_t*)malloc(sizeof(uint32_t)*MAX_BUFFER_PACKETS);
	result2 = (uint32_t*)malloc(sizeof(uint32_t)*MAX_BUFFER_PACKETS);
#define MAM MAX_BUFFER_PACKETS
	for(i=0,counter=0;i<MAM;i++){
		result[i] = MAX_BUFFER_PACKETS -i;
//		result[i] = i&0x7326432;
		/*result[i] = 1;
		if(i>63)
		result[i]++;
		if(i>127)
		result[i]++;
		if(i>159)
		result[i]++;
		*/	
		fprintf(stderr,"[%d:%d] ",i,result[i]);
		if(i%(16-1) == 0 && i!=0)
			fprintf(stderr,"\n");
		counter +=result[i];
	}
	fprintf(stderr,"\nCompto: %d\n",counter);
 
	cudaMemcpy(a,result,sizeof(uint32_t)*MAX_BUFFER_PACKETS, cudaMemcpyHostToDevice);
	cudaThreadSynchronize();       

	cudaMemcpy(b,result2,sizeof(uint32_t)*MAX_BUFFER_PACKETS, cudaMemcpyHostToDevice);
	cudaThreadSynchronize();  
	unsigned int t1,t2;	
	t1=clock();
	wrapper_QuickSort(a,b,0,MAM-1);
	cudaThreadSynchronize();

	t2=clock();
	fprintf(stderr,"Temps aproximat: %d\n",t2-t1);
	cudaMemcpy(result,a,sizeof(uint32_t)*MAX_BUFFER_PACKETS, cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
	 
	fprintf(stderr,"I el resultat és:\n");

	for(i=0,counter=0;i<MAM;i++){
		//fprintf(stderr,"[%d] ",result[i]);
		fprintf(stderr,"[%d,%d] ",i,result[i]);
		if(i%(16-1) == 0 && i!=0)
			fprintf(stderr,"\n");
		
		counter +=result[i];
	
	}

	fprintf(stderr,"\nHe comptat: %d\n",counter);
	exit(1);
}
#endif
