#include "Anomalies.h" //Include your modified(from Template) .h

/*
	Looking for simple anomalies in network traffic (using DETECTOR module) 
*/

template<typename T,typename R>
__device__ void COMPOUND_NAME(ANALYSIS_NAME,mining)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results, analysisState_t state){
	
	/* CUDA CODE: Implement MINING code here */

	if(IS_ETHERNET() && IS_IP4() && IS_TCP() && (GET_FIELD(TCP_HEADER.dport)==56166 ))//|| GET_FIELD(TCP_HEADER.sport)==22) )
		DATA_ELEMENT = 1;


}

template<typename T,typename R>
__device__ void COMPOUND_NAME(ANALYSIS_NAME,filtering)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results, analysisState_t state){
	
	/* CUDA CODE: Implement PreAnalysis Routine Filtering here. Use predefined Filtering tool or Implement code here IF NECESSARY */

}

template<typename T,typename R>
__device__ void COMPOUND_NAME(ANALYSIS_NAME,analysis)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results, analysisState_t state){

	/* CUDA CODE: Implement Analysis Routine here. Use a predefined Analysis or Implement code here */
	$DETECTORS$ANALYSIS();
}

template<typename T,typename R>
__device__ void COMPOUND_NAME(ANALYSIS_NAME,operations)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results, analysisState_t state){
	
	/* CUDA CODE: Implement post Analysis Operations/Filtering here. Use predefined Operations/Filtering tools or Implement code here IF NECESSARY */

}

template<typename R>
void COMPOUND_NAME(ANALYSIS_NAME,hooks)(PacketBuffer* packetBuffer,R* results, analysisState_t state, int64_t* auxBlocks){
	
	/* HOST CODE: Implement HOOKS code here. Use predefined hooks or define new ones. */
	
	$DETECTOR$DUMPER$PREPARE_TO_DUMP_ALARMS_TO_STDERR();
	$DETECTOR$DUMPER$ADD_FIELD();		
	$DETECTOR$DUMPER$DUMP_ALARMS_TO_STDERR();
	
}

