MAINTAINERCLEANFILES = Makefile.in

SUBDIRS=Macros

noinst_LTLIBRARIES = libanalysis-libs-gpu.la 

libanalysis_libs_gpu_la_SOURCES = Protocols.h\
				  CudaSafeAccess.h CudaSafeAccess.cu \
				  Endianness.h Endianness.cu \
				  CudaSharedMemoryOperations.h CudaSharedMemoryOperations.cu \
				InlineFiltering.h InlineFiltering.cu

#headers	
library_includedir=$(includedir)/netgpu/Analysis/Libs/Gpu

library_include_HEADERS = Protocols.h\
			  CudaSafeAccess.h\
			  Endianness.h\
			  CudaSharedMemoryOperations.h\
	   		InlineFiltering.h
	
#headers	
gpudir=$(datadir)/netgpu/include/netgpu/Analysis/Libs/Gpu

gpu_DATA = Protocols.h\
			  CudaSafeAccess.h\
			  Endianness.h\
			  CudaSharedMemoryOperations.h\
	   		InlineFiltering.h
	

#Compiling libs

LINK = $(LIBTOOL) --mode=link $(CXX) -o $@ 

.cu.o:
	$(NVCC) --host-compilation C++ --compiler-bindir=/usr/bin/gcc -c $< -o $@ $(AM_CPPFLAGS) $(DEFS) $(CFLAGS)

.cu.lo:
	$(top_srcdir)/cudalt.py $@ $(NVCC) -c  $< $(AM_CPPFLAGS) $(DEFS) $(CFLAGS)


#$(NVCC) --host-compilation C++ --compiler-bindir=/usr/bin/gcc -c $< -o $@ $($(basename $@)_CPPFLAGS) $(AM_CPPFLAGS) $(DEFS) $(CFLAGS)
#$(top_srcdir)/cudalt.py $@ $(NVCC) -c  $< $($(basename $@)_CPPFLAGS) $(AM_CPPFLAGS) $(DEFS) $(CFLAGS)
