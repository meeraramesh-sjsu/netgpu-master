
all:
	@rm -f .*.o
	@rm -f .*.ppph
#	@bash $(NETGPU_PATH)Tools/ppp.sh ./*.cu 
	@netgpu_ppp ./*.cu
	@echo [Compiling]
#	@nvcc -O3 *.cu -c -o .analysis.o
	@netgpu__compile_analysis ./*.cu
	@echo 
clean: 
	@rm -f .*.o
	@rm -f .*.ppph
	@echo [Cleaned]
