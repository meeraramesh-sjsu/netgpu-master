# netgpu-master

This framework performs network traffic anomaly detection using GPUs. The General Purpose GPU was used to perform header checking and pattern matching on the packets. Naive pattern matching algorithm and Rabin Karp pattern matching algorithm was explored. 

Two packet processing approaches are explored: Thread level packet processing and block level packet processing. Evaluations were carried on a Tesla K80 GPU and we proved that block level packet processing outperforms thread level packet processing by a factor of 116. 

The packet processing was optimized to use shared memory and warp divergence
has been eliminated. The results prove that GPUs can be utilized to speed up Network Anomaly
detection systems and in other systems that involve signature matching algorithms.

The evaluations have been carried out by comparing CUDA version, C++ version and OpenMP version.

There are three Main branches: master branch, openMP branch, CUDA branch

The master branch contains the CPU version Header checking and pattern matching algorithms used for Deep packet inspection written in C++. 

The pattern matching algorithms are:

1. Single pattern matching Rabin Karp algorithm
2. Multi pattern matching Wu Manber
3. Multi pattern matching Aho Corasick

The openMp branch contains the above pattern matching algorithms writtern in OpenMP

The CUDA branch contains the above pattern matching algorithms writtern in CUDA



