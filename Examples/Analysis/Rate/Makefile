
all:
	@rm -f .*.o
	@rm -f .*.ppph
	@netgpu_ppp ./*.cu 
	@echo [Compiling]
	@netgpu__compile_analysis -I./ ./*.cu
	@echo 
	@echo 
clean: 
	@rm -f .*.o
	@rm -f .*.ppph
	@echo [Cleaned]
