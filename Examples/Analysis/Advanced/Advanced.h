// Do not delete/edit this line
#include <netgpu/Analysis/Initializer.h>

/* 
	ANALYSIS_NAME MACRO: NAME OF THE ANALYSIS.
	
	MUST BE HERE, BEFORE INCLUDE OF PROTOTYPE(SKELETON) OR ANYTHING. 
	FILL WITH APPROPIATE (UNIQUE) ANALYSIS NAME

	Note that the value of ANALYSIS_NAME will be the Class name
	of the analysis (to be used in Analyzer.cpp)
		
*/


/*********   Edit this section   **********/

// [[GENERAL PARAMETERS]]

		//--> Analysis Name: unique name here for all the program
	#define ANALYSIS_NAME AdvancedExample

		//--> int,uint,floats, double(*) intXX_t, uintXX_t, structs etc.. or typedefs (define new types below) 
	#define ANALYSIS_INPUT_TYPE  int 

		//--> Threads Per Block (unidimensional): [8-512], default 128 
	#define ANALYSIS_TPB 128 

	/*** DEFINE COMPLEX TYPES HERE ***/
	//typedef struct{
	//	int x,y,z;
	//}mytype;

	/*** DEFINE HERE WINDOW PARAMS ***/
		//--> HAS_WINDOW
	#define HAS_WINDOW 0 
		//--> WINDOW_TYPE
	#define WINDOW_TYPE TYPE
		//--> WINDOW_LIMIT
	#define WINDOW_LIMIT 10
	
	/***  OUTPUT DATA TYPE  ***/
		//--> If you are NOT USING PREDEFINED ANALYSIS OR if INPUT TYPE IS DIFFERENT THAN OUTPUT TYPE, uncomment and modify
		//--> this line 
	//#define ANALYSIS_OUTPUT_TYPE  type 		

// [[--------------]]

/********* End of editable section **********/

/*DO NOT EDIT REST OF THE FILE */
#include <netgpu/Analysis/AnalysisPrototype.h>
