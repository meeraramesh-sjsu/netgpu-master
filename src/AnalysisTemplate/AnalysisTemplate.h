/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#include <netgpu/Analysis/Initializer.h> //Do not delete/edit this line

/*	ANALYSIS TEMPLATE HEADER FILE. */
/*	Fill at least uncommented LINES with appropiate values (!! Read documentation for more info)  */

/*********   Editable section   **********/

	/***** DEFINING ANALYSIS GENERAL PARAMS *****/

	/* Analysis Name: unique name here for all the program */
	#define ANALYSIS_NAME Change_me

	/* Input type: int,uint,floats, double(*) intXX_t, uintXX_t, structs etc.. or typedefs (define new types below) */
	#define ANALYSIS_INPUT_TYPE  type 

	/* Threads Per Block (unidimensional): [8-512],default:128 */
	#define ANALYSIS_TPB 128

	/***** DEFINING COMPLEX TYPES *****/

	//typedef struct{
	//	int x,y,z;
	//}mytype;


	/***** DEFINING WINDOW PARAMS *****/
	
	/* HAS_WINDOW: 1 to enable windowed analysis */
	//#define HAS_WINDOW 0 
	
	/* WINDOW_TYPE: window type */
	//#define WINDOW_TYPE TYPE
	
	/* HAS_WINDOW: window limit */
	//#define WINDOW_LIMIT 10

	
	/*****  DEFINING OUTPUT DATA TYPE  *****/

	/* If you are NOT USING PREDEFINED ANALYSIS OR if INPUT TYPE IS DIFFERENT THAN OUTPUT TYPE, uncomment and modify this line */
	//#define ANALYSIS_OUTPUT_TYPE  type 		

/********* End of editable section **********/

#include <netgpu/Analysis/AnalysisPrototype.h> //Do not delete/Edit this
