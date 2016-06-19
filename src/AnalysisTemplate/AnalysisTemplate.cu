#include "AnalysisTemplate.h" //Include your modified(from Template) .h

/*
	This is a blank Analysis TEMPLATE. Read documentation for more information. 
	
	You are encouraged to use PREDEFINED FUNCTIONS(API), MACROS and OBJECTS for each section(function) -> READ DOCUMENTATION.
	Fill the code inside the different functions, leaving arguments and names as they are.

*/

#ifdef __CUDACC__ /* Don't erase this */

template<typename T,typename R>
__device__  void COMPOUND_NAME(ANALYSIS_NAME,mining)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results, analysisState_t state){
	
	/* CUDA CODE: Implement MINING code here */

}


template<typename T,typename R>
__device__  void COMPOUND_NAME(ANALYSIS_NAME,filtering)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results, analysisState_t state){
	
	/* CUDA CODE: Implement PreAnalysis Routine Filtering here. Use predefined Filtering tool or Implement code here IF NECESSARY */

}


template<typename T,typename R>
__device__  void COMPOUND_NAME(ANALYSIS_NAME,analysis)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results,analysisState_t state){
	/* CUDA CODE: Implement Analysis Routine here. Implement code here IF NECESSARY (LEAVE IT BLANK if using a PREDEFINED ANALYSIS)*/

	
}


template<typename T,typename R>
__device__  void COMPOUND_NAME(ANALYSIS_NAME,operations)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results,analysisState_t state){
	/* CUDA CODE: Implement post Analysis Operations/Filtering here. Use predefined Operations/Filtering tools or Implement code here IF NECESSARY */

}

template<typename R>
void COMPOUND_NAME(ANALYSIS_NAME,hooks)(PacketBuffer *packetBuffer, R* results,analysisState_t state, int64_t* auxBlocks){
	
	/* HOST CODE: Implement HOOKS code here. Use predefined hooks or define new ones. */

}

#endif //ifdef CUDACC
