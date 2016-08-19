// Matrix Multiplication in CUDA
// ee352_prj_matrixmul.cpp
// EE 352, Spring 2011
// Name:  _______________________

// includes, system
//#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <cuda_runtime.h>


// includes, project
////////////////////////////////////////////////////////////////////////////////
// declarations, forward

#define WIDTH 10
float A [WIDTH * WIDTH] = { 0.019044f, 0.877102f, 2.991211f, 0.782342f, 1.915799f, 1.774255f, 0.583483f, 1.019837f, 1.564226f, 0.533219f, 
					        2.831812f, 2.782464f, 0.502091f, 0.798547f, 2.531510f, 2.999817f, 2.492508f, 0.124241f, 2.990021f, 2.338969f, 
							2.266732f, 1.620533f, 1.464339f, 2.766533f, 2.210883f, 1.769127f, 2.088656f, 0.641621f, 0.464278f, 1.003174f, 
							2.510727f, 1.680593f, 0.398999f, 1.143712f, 1.956267f, 1.639119f, 2.948637f, 0.287851f, 1.939970f, 1.054811f, 
							1.755669f, 0.241890f, 2.341350f, 0.894040f, 2.606037f, 1.092898f, 0.064821f, 2.446821f, 0.119205f, 1.012696f, 
							2.002686f, 0.313486f, 2.163823f, 2.888119f, 2.518967f, 0.827204f, 1.964782f, 0.871426f, 0.174139f, 0.754967f, 
							0.374462f, 2.443983f, 0.908689f, 0.622578f, 2.274056f, 2.087283f, 1.404187f, 2.506241f, 2.573717f, 1.155797f, 
							1.841090f, 0.633564f, 2.782464f, 2.044252f, 2.771203f, 1.641041f, 1.613941f, 2.243202f, 2.325511f, 1.827448f, 
							0.599414f, 0.378124f, 2.472823f, 0.249947f, 1.173559f, 0.823817f, 1.551683f, 2.006348f, 1.205237f, 1.988678f, 
							2.791162f, 0.545122f, 0.439741f, 2.863124f, 1.041627f, 1.350261f, 1.821589f, 0.925260f, 1.432569f, 0.841304f, };

float B [WIDTH * WIDTH] = { 0.167638f, 0.997955f, 2.177191f, 0.835078f, 2.157964f, 2.149815f, 1.242317f, 2.411115f, 0.113712f, 0.236396f, 
						   1.102054f, 2.525285f, 0.532212f, 0.172674f, 2.294656f, 0.215339f, 2.844447f, 0.531938f, 1.999756f, 2.742088f, 
						   2.367626f, 1.564776f, 0.357433f, 2.508988f, 2.919523f, 2.231941f, 0.745079f, 2.106418f, 2.063021f, 1.344035f, 
						   0.732261f, 1.320963f, 0.451003f, 1.402722f, 1.932463f, 2.456801f, 1.239387f, 0.237220f, 2.469161f, 1.810144f, 
						   0.036714f, 2.028504f, 0.277230f, 2.642567f, 0.360820f, 2.816981f, 0.956114f, 2.495620f, 1.068728f, 0.085330f, 
						   1.394208f, 2.191381f, 2.730918f, 2.512742f, 2.884548f, 1.362072f, 0.416944f, 2.490768f, 0.948607f, 0.852657f, 
						   0.863094f, 1.498489f, 2.588549f, 0.840388f, 1.487960f, 2.927763f, 0.184667f, 0.015107f, 2.227729f, 2.453688f, 
						   0.760094f, 2.663350f, 2.658132f, 1.543901f, 0.827387f, 0.000366f, 1.372234f, 2.200171f, 2.539109f, 1.595355f, 
						   1.082736f, 2.525925f, 0.359722f, 2.230109f, 0.417859f, 2.214087f, 0.977447f, 2.038759f, 2.879421f, 1.548479f, 
						   2.675344f, 1.269509f, 2.930876f, 0.012452f, 2.879788f, 0.513718f, 0.686392f, 2.099002f, 2.105686f, 2.745384f, };

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);

// MatrixMul kernel
__global__ void MatrixMul_GPU(float* M, float* N, float* P, int width, int* runtime)
{

    // TODO : Kernel Function
    //        C = A * B
    // --> 
    clock_t start_time = clock();
#if 1
    const unsigned int tid = threadIdx.x;
    const unsigned int col = tid%width;
    const unsigned int row = tid/width;

    int i, j;

    for(i = 0, j = 0; (i < width) && (j < width); i++, j++)
    {
        P[tid] += M[row*width+i] * N[col+j*width];
    }

#else
    
    const unsigned int tid = threadIdx.x;
    const unsigned int col = tid%width;
    const unsigned int row = tid/width;
    __shared__ float sM[WIDTH * WIDTH];
    __shared__ float sN[WIDTH * WIDTH];
    __shared__ float sP[WIDTH * WIDTH];

    sM[tid] = M[tid];
    sN[tid] = N[tid];

    __syncthreads();

    int i, j;
    for(i = 0, j = 0; (i < width) && (j < width); i++, j++)
    {
        sP[tid] += sM[row*width+i] * sN[col+j*width];
    }

    __syncthreads();

    P[tid] = sP[tid];
#endif
    clock_t stop_time = clock();

    runtime[tid] = (int)(stop_time - start_time);
    // <--    

}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {

	float  C[WIDTH*WIDTH] = {};
	float  reference[WIDTH*WIDTH] = {};

    // compute the matrix multiplication on the CPU for comparison
    computeGold(reference, A, B, WIDTH, WIDTH, WIDTH);

    // M * N on the device
    //MatrixMulOnDevice(M, N, P);
    /* Load M and N to the device*/
	float* A_gpu;
	float* B_gpu;
	float* C_gpu;
    int data_size = WIDTH * WIDTH * sizeof(float);

    cudaMalloc((void**)&A_gpu, data_size);
    cudaMalloc((void**)&B_gpu, data_size);
    cudaMalloc((void**)&C_gpu, data_size);
    cudaMemcpy(A_gpu, A, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, data_size, cudaMemcpyHostToDevice);

    int* runtime_gpu;
	int runtime_size = WIDTH * WIDTH * sizeof(int);
    int* runtime = (int*)malloc(runtime_size);
    memset(runtime, 0, runtime_size);
    cudaMalloc((void**)&runtime_gpu, runtime_size);

    // TODO : Kernel Invocation 
    //        Assign as many threads as the size of matrix in a thread block and
    //        invoke the kernel function.
    // --> 
    dim3 dimBlock(WIDTH*WIDTH, 1, 1);
    dim3 dimGrid(1, 1, 1);
    MatrixMul_GPU<<< dimGrid, dimBlock>>>(A_gpu, B_gpu, C_gpu, WIDTH, runtime_gpu);
    cudaThreadSynchronize();
    // <--                                                           

    cudaMemcpy(runtime, runtime_gpu, runtime_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(C, C_gpu, data_size, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    int elapsed_time = 0;
    for(int i = 0; i < WIDTH*WIDTH; i++)
        if(elapsed_time < runtime[i])
            elapsed_time = runtime[i];
    printf("Kernel Execution Time: %d\n", elapsed_time);

    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);
    cudaFree(runtime_gpu);

    // in this case check if the result is equivalent to the expected soluion
    bool res = 1;
	for (int i = 0; i < WIDTH*WIDTH; i++)
	{
		float diff = fabs(reference[i] - C[i]);
		if(diff > 0.001f)
		{
			res = 0;
			break;
		}
	}
	printf("Test %s\n", (res == 1) ? "PASSED" : "FAILED");

	return 0;
}

void
computeGold(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j) {
            double sum = 0;
            for (unsigned int k = 0; k < wA; ++k) {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }
            C[i * wB + j] = (float)sum;
        }

}
