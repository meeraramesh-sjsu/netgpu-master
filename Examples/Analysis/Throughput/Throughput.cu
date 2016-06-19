#include "Throughput.h" //Include your modified(from Template) .

/*
	Using THROUGHPUTS module to detect throughput abuse
*/

template<typename T,typename R>
__device__ void COMPOUND_NAME(ANALYSIS_NAME,mining)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results, analysisState_t state){
	
	/* CUDA CODE: Implement MINING code here */

	if(IS_ETHERNET() && IS_IP4() && (IP4_NETID(GET_FIELD(IP4_HEADER.ip_src),8) ==IP4(225,0,0,0))){
			$THROUGHPUTS$MINE_QUANTITY(GET_FIELD(IP4_HEADER.totalLength));
			DATA_ELEMENT.ipsrc = GET_FIELD(IP4_HEADER.ip_src);
			DATA_ELEMENT.ipdst = GET_FIELD(IP4_HEADER.ip_dst); 
	}
}

template<typename T,typename R>
__device__ void COMPOUND_NAME(ANALYSIS_NAME,filtering)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results, analysisState_t state){
	
	/* CUDA CODE: Implement PreAnalysis Routine Filtering here. Use predefined Filtering tool or Implement code here IF NECESSARY */
	
}

template<typename T,typename R>
__device__ void COMPOUND_NAME(ANALYSIS_NAME,analysis)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results, analysisState_t state){

	/* CUDA CODE: Implement Analysis Routine here. Use a predefined Analysis or Implement code here */
	$THROUGHPUTS$ANALYSIS(400*KBPS);
}

template<typename T,typename R>
__device__ void COMPOUND_NAME(ANALYSIS_NAME,operations)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results, analysisState_t state){
	
	/* CUDA CODE: Implement post Analysis Operations/Filtering here. Use predefined Operations/Filtering tools or Implement code here IF NECESSARY */
	
	//Modify throughput threshold for ipsrc 225.86.179.79
	$THROUGHPUTS$MULTI_THRESHOLDS$MODIFY_THRESHOLD(RESULT_ELEMENT.ipsrc == IP4(225,86,179,79),500*KBPS);		
}

template<typename R>
void COMPOUND_NAME(ANALYSIS_NAME,hooks)(PacketBuffer* packetBuffer,R* results, analysisState_t  state, int64_t* auxBlocks){
	
	/* HOST CODE: Implement HOOKS code here. Use predefined hooks or define new ones. */

	//Dump results to stderr
	$THROUGHPUTS$DUMPER$PREPARE_TO_DUMP_ALARMS_TO_STDERR();
	$THROUGHPUTS$DUMPER$ADD_FIELD_AS_IP_COMPLEX(ipsrc);		
	$THROUGHPUTS$DUMPER$ADD_FIELD_AS_IP_COMPLEX(ipdst);		
	$THROUGHPUTS$DUMPER$DUMP_ALARMS_TO_STDERR();


	//Dump results to DB
	$THROUGHPUTS$DB_DUMPER$PREPARE_TO_DUMP_ALARMS_TO_DBTABLE("flows");
	$THROUGHPUTS$DB_DUMPER$ADD_FIELD_AS_IP_COMPLEX(ipsrc,ipsrc);		
	$THROUGHPUTS$DB_DUMPER$ADD_FIELD_AS_IP_COMPLEX(ipdst,ipdst);		
	$THROUGHPUTS$DB_DUMPER$DUMP_ALARMS();		

}

