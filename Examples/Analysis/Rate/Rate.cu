#include "Rate.h" //Include your modified(from Template) .h

/*
	Using RATES module to try to detect SYN flood abuse over a host
*/

template<typename T,typename R>
__device__ void COMPOUND_NAME(ANALYSIS_NAME,mining)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results, analysisState_t state){
	
	/* CUDA CODE: Implement MINING code here */
	if(IS_ETHERNET() && IS_IP4() && IS_TCP()&& (GET_FIELD(TCP_HEADER.flags)&0x12)==0x02 ){
			DATA_ELEMENT = 1; 
	}
}

template<typename T,typename R>
__device__ void COMPOUND_NAME(ANALYSIS_NAME,filtering)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results, analysisState_t state){
	
	/* CUDA CODE: Implement PreAnalysis Routine Filtering here. Use predefined Filtering tool or Implement code here IF NECESSARY */

}

template<typename T,typename R>
__device__ void COMPOUND_NAME(ANALYSIS_NAME,analysis)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results, analysisState_t state){

	/* CUDA CODE: Implement Analysis Routine here. Use a predefined Analysis or Implement code here */
	$RATES$ANALYSIS(30);

}

template<typename T,typename R>
__device__ void COMPOUND_NAME(ANALYSIS_NAME,operations)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results, analysisState_t state){
	
	/* CUDA CODE: Implement post Analysis Operations/Filtering here. Use predefined Operations/Filtering tools or Implement code here IF NECESSARY */

}

template<typename R>
void COMPOUND_NAME(ANALYSIS_NAME,hooks)(PacketBuffer* packetBuffer,R* results, analysisState_t state, int64_t* auxBlocks){
	
	/* HOST CODE: Implement HOOKS code here. Use predefined hooks or define new ones. */
/*	//Dump to file
	$RATES$DUMPER$PREPARE_TO_DUMP_ALARMS_TO_FILE("prova");
	$RATES$DUMPER$ADD_FIELD();		
	$RATES$DUMPER$DUMP_ALARMS_TO_FILE();
*/	
	//Dump to stderr
	$RATES$DUMPER$PREPARE_TO_DUMP_ALARMS_TO_STDERR();
	$RATES$DUMPER$ADD_FIELD();		
	$RATES$DUMPER$DUMP_ALARMS_TO_STDERR();

	//Dump to database
	$RATES$DB_DUMPER$PREPARE_TO_DUMP_ALARMS_TO_DBTABLE("level4");
	$RATES$DB_DUMPER$ADD_FIELD(dummy);
	$RATES$DB_DUMPER$DUMP_ALARMS();

	//example how to use program execution
#if 0
	//Program execution
	$PROGRAM_LAUNCHER$PREPARE();

	$PROGRAM_LAUNCHER$ADD_TEXT_ARG("prova1");	
	$PROGRAM_LAUNCHER$ADD_TEXT_ARG("prova2");	

	$PROGRAM_LAUNCHER$ADD_ARG(user);

	$PROGRAM_LAUNCHER$EXECUTE("./prova.sh");
#endif
}

