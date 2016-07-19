#include "IpScan.h" //Include your modified(from Template) .h
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
/*
	Using IPSCAN_DETECTOR module to try to detect ipscans
*/

template<typename T,typename R>
__device__ void COMPOUND_NAME(ANALYSIS_NAME,mining)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results,analysisState_t state){

/*	printf("Mining");	
 CUDA CODE: Implement MINING code here 

 printf("%d \n",GPU_buffer->headers.proto);
  printf("%d \n",GPU_buffer->headers.offset);

  for(int i=0;i<94;i++)
  {
   printf("%d ",GPU_buffer->packet[i]);
  }
*/
	$IPSCAN_DETECTOR$AUTO_MINE();
}

template<typename T,typename R>
__device__ void COMPOUND_NAME(ANALYSIS_NAME,filtering)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results,analysisState_t state){
/*	printf("Filtering");
	CUDA CODE: Implement PreAnalysis Routine Filtering here. Use predefined Filtering tool or Implement code here IF NECESSARY 
  printf("%d \n",GPU_buffer->headers.proto);
  printf("%d \n",GPU_buffer->headers.offset);

  for(int i=0;i<94;i++)
  {
   printf("%d ",GPU_buffer->packet[i]);
  }

*/

}

template<typename T,typename R>
__device__ void COMPOUND_NAME(ANALYSIS_NAME,analysis)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results,analysisState_t state){
//	printf("Analysis");
//static int i = 0;
//  --added on June 1st 
/*  printf("ThreadIDSSSSSSSS");
  printf("threadId %d",threadIdx.x);
  printf("%d \n",GPU_buffer->headers.proto);
  printf("%d \n",GPU_buffer->headers.offset);

  for(int i=0;i<94;i++)
  {
   printf("%d ",GPU_buffer->packet[i]);
  }
*/
	int *counter = 0;	
   /* CUDA CODE: Implement Analysis Routine here. Use a predefined Analysis or Implement code here */
	$IPSCAN_DETECTOR$ANALYSIS(15);

}

template<typename T,typename R>
__device__ void COMPOUND_NAME(ANALYSIS_NAME,operations)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results,analysisState_t state){
//	printf("operations");

/*  --added on June 1st 
  printf("%d \n",GPU_buffer->headers.proto);
  printf("%d \n",GPU_buffer->headers.offset);

  for(int i=0;i<94;i++)
  {
   printf("%d \n",GPU_buffer->packet[i]);
  }
*/ 

//Only Global Thread Index should modify

	/* CUDA CODE: Implement post Analysis Operations/Filtering here. Use predefined Operations/Filtering tools or Implement code here IF NECESSARY */
	$IPSCAN_DETECTOR$MULTI_THRESHOLDS$EXCLUDE(RESULT_ELEMENT.ipSrc==IP4(82,70,103,152)); //Exclude 82.70.103.152 from results
}

template<typename R>
void COMPOUND_NAME(ANALYSIS_NAME,hooks)(PacketBuffer* packetBuffer,R* results, analysisState_t state, int64_t* auxBlocks){
	printf("hooks");
        /* HOST CODE: Implement HOOKS code here. Use predefined hooks or define new ones. */
	
	//Dump to stderr
	//$IPSCAN_DETECTOR$DUMPER$DUMP_ALARMS_TO_STDERR(1,1);	

//	printf("Dumping to file");
//	printf("TCP = %d, UDP = %d, ICMP = %d",(*results).indexTcp,(*results).indexUdp,(*results).indexIcmp );
	//Dump to file
	$IPSCAN_DETECTOR$DUMPER$DUMP_ALARMS_TO_FILE("IPSCAN.alarms",7,10);	
	
	//Dump to DB
	// $IPSCAN_DETECTOR$DB_DUMPER$DUMP_ALARMS();



	//Example how to use PROGRAM_LAUNCHER
#if 0 
	printf("In program Launcher");	
	$PROGRAM_LAUNCHER$PREPARE();
	$PROGRAM_LAUNCHER$ADD_TEXT_ARG("Test1 ipscan");
	$PROGRAM_LAUNCHER$ADD_TEXT_ARG("Test2 ipscan");
	$PROGRAM_LAUNCHER$ADD_TEXT_ARG("Test3 ipscan 1");
	$PROGRAM_LAUNCHER$ADD_ARG_AS_IP(ipSrc);
	$PROGRAM_LAUNCHER$EXECUTE("./prova.sh");
#endif
}



