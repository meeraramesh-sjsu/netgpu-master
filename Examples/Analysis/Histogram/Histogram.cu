#include "Histogram.h" //Include your modified(from Template) .h

/*
	Creating an histogram of the compund key [ipdest,port], using HISTOGRAMS module
*/

template<typename T,typename R>
__device__ void COMPOUND_NAME(ANALYSIS_NAME,mining)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results,analysisState_t state){
	
	/* CUDA CODE: Implement MINING code here */

	if(IS_ETHERNET() && IS_IP4() && IS_TCP()){
		DATA_ELEMENT.ip_dst = GET_FIELD(IP4_HEADER.ip_dst);
		DATA_ELEMENT.port_dst = GET_FIELD(TCP_HEADER.dport);
	}

}

template<typename T,typename R>
__device__ void COMPOUND_NAME(ANALYSIS_NAME,filtering)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results,analysisState_t state){
	
	/* CUDA CODE: Implement PreAnalysis Routine Filtering here. Use predefined Filtering tool or Implement code here IF NECESSARY */

}

template<typename T,typename R>
__device__ void COMPOUND_NAME(ANALYSIS_NAME,analysis)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results,analysisState_t state){

	/* CUDA CODE: Implement Analysis Routine here. Use a predefined Analysis or Implement code here */
	
	$HISTOGRAMS$ANALYSIS();

}

template<typename T,typename R>
__device__ void COMPOUND_NAME(ANALYSIS_NAME,operations)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results,analysisState_t state){
	
	/* CUDA CODE: Implement post Analysis Operations/Filtering here. Use predefined Operations/Filtering tools or Implement code here IF NECESSARY */

		//$HISTOGRAMS$NORMALIZER$NORMALIZE(); //used to normalize
		$HISTOGRAMS$NOISE$FILTER(50); //Filter noise 50% (noise submodule)

}

template<typename R>
void COMPOUND_NAME(ANALYSIS_NAME,hooks)(PacketBuffer* packetBuffer,R* results, analysisState_t state, int64_t* auxBlocks){
	
	/* HOST CODE: Implement HOOKS code here. Use predefined hooks or define new ones. */

	//Dumping to stderr using submodule DUMPER
	$HISTOGRAMS$DUMPER$PREPARE_TO_DUMP_TO_STDERR();
	$HISTOGRAMS$DUMPER$ADD_FIELD_AS_IP_COMPLEX(ip_dst);
	$HISTOGRAMS$DUMPER$ADD_FIELD_COMPLEX(port_dst);
	$HISTOGRAMS$DUMPER$DUMP_TO_STDERR();
	
	//Dumping to database table using submodule DB_DUMPER 
	$HISTOGRAMS$DB_DUMPER$PREPARE_TO_DUMP_TO_DBTABLE("hist");
	$HISTOGRAMS$DB_DUMPER$ADD_FIELD_AS_IP_COMPLEX(ip_dst,ipdst);
	$HISTOGRAMS$DB_DUMPER$ADD_FIELD_COMPLEX(port_dst,portdst);
	$HISTOGRAMS$DB_DUMPER$DUMP();


}

