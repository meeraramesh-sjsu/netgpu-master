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

/*** General Parameters ***/
#define ANALYSIS_NAME Anomalies
#define ANALYSIS_INPUT_TYPE  int //int,uint,floats, double(*) intXX_t, uintXX_t, structs etc.. or typedefs (define new types below) 
#define ANALYSIS_TPB 128 //Threads Per Block (unidimensional): [8-512], default 128 

/*** Define complex types here (i.e. structs)  ***/
//typedef struct{
//	int x,y,z;
//}mytype;

/***  Output type ***/

/* If you are NOT USING PREDEFINED ANALYSIS OR if INPUT TYPE IS DIFFERENT THAN OUTPUT TYPE, uncomment and modify this line */

//#define ANALYSIS_OUTPUT_TYPE  type //int,uint,floats, double(*) intXX_t, uintXX_t, structs etc.. or typedefs (define new types below) 

/********* End of editable section **********/


/*DO NOT EDIT REST OF THE FILE */
#include <netgpu/Analysis/AnalysisPrototype.h>
