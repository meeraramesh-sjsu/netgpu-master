# netgpu-master

This framework performs network traffic anomaly detection using GPUs. The General Purpose GPU was used to perform header checking and pattern matching on the packets. Naive pattern matching algorithm and Rabin Karp pattern matching algorithm was explored. 

Two packet processing approaches are explored: Thread level packet processing and block level packet processing. Evaluations were carried on a Tesla K80 GPU and we proved that block level packet processing outperforms thread level packet processing by a factor of 116. 

The packet processing was optimized to use shared memory and warp divergence
has been eliminated. The results prove that GPUs can be utilized to speed up Network Anomaly
detection systems and in other systems that involve signature matching algorithms.

The evaluations have been carried out by comparing CUDA version, C++ version and OpenMP version.

There are three Main branches: master branch, openMP branch, Meera branch

The master branch contains the CPU version Header checking and pattern matching algorithms used for Deep packet inspection written in C++. 

The pattern matching algorithms are:

1. Single pattern matching Rabin Karp algorithm
2. Multi pattern matching Wu Manber
3. Multi pattern matching Aho Corasick

The openMp branch contains the above pattern matching algorithms writtern in OpenMP

The Meera branch contains the above pattern matching algorithms writtern in CUDA


dependencies & requirements
------------------

Hardware
      A CUDA-enabled graphical processor unit (GPU) installed (http://www.nvidia.com/object/cuda_learn_products.html).

Software
      	GNU/Linux system. 
      	GCC 4.4 or above
      	Autotools 1.11 or above
      	CUDA libraries 7.5
      	LibPCAP (http://www.tcpdump.org/)
      	POSIX threads (lpthreads)
      	unixODBC 

	Debian packages list:
		build-essential
		automake
		autoconf
		gcc4-3 
		g++4.3 
		libpcap-dev
		unixodbc-dev	


LIBRARY Installation steps
--------------

To compile and install the framework's LIBRARY and the MODULES:

	./setup
	cd ./build
	../configure
	make 
	make install

To uninstall it, simply:

	make uninstall
