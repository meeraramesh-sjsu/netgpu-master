##Makefile sample for a netgpu-based program

#binary target name
TARGET := example

#NVCC compiler
NVCC := nvcc 

#Analysis folder
ANALYSIS_FOLDER:=Analysis

#Find all .o objects (analysis) in $(ANALYSIS_FOLDER)
#OBJS:=$(shell find ./* -name *.o)

#Libs
LIBS :=  -lnetgpu -lpcap -lpthread -lodbc -lcudart -lgomp

#Compilation flags, -XCompiler adds flags directly to host compiler
FLAGS = -pg -Xcompiler -fopenmp -g

.NOTPARALLEL:
all:analysis
	$(NVCC) $(FLAGS) Main.cpp -I/home/meera/gpudir/netgpu-master/usr/local/include $(shell find ./* -name *.o) $(LIBS) -L/home/meera/gpudir/netgpu-master/usr/local/lib -o $(TARGET) 
analysis:
	cd Analysis && make

clean:
	cd Analysis && make clean
	rm -f $(TARGET)
