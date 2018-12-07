/*
============================================================================
Filename    : implementation.cu
Author      : Lucien Michaël Iseli, Loris Pilotto
SCIPER      : 274999, 262651
============================================================================
*/

#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <cuda_runtime.h>
using namespace std;

// CPU Baseline
void array_process(double *input, double *output, int length, int iterations)
{
    double *temp;

    for(int n=0; n<(int) iterations; n++)
    {
        for(int i=1; i<length-1; i++)
        {
            for(int j=1; j<length-1; j++)
            {
                output[(i)*(length)+(j)] = (input[(i-1)*(length)+(j-1)] +
                                            input[(i-1)*(length)+(j)]   +
                                            input[(i-1)*(length)+(j+1)] +
                                            input[(i)*(length)+(j-1)]   +
                                            input[(i)*(length)+(j)]     +
                                            input[(i)*(length)+(j+1)]   +
                                            input[(i+1)*(length)+(j-1)] +
                                            input[(i+1)*(length)+(j)]   +
                                            input[(i+1)*(length)+(j+1)] ) / 9;

            }
        }
        output[(length/2-1)*length+(length/2-1)] = 1000;
        output[(length/2)*length+(length/2-1)]   = 1000;
        output[(length/2-1)*length+(length/2)]   = 1000;
        output[(length/2)*length+(length/2)]     = 1000;

        temp = input;
        input = output;
        output = temp;
    }
}


__global__
void compute_GPU(double* gpu_input, double* gpu_output, double* gpu_temp, int gpu_length){
    int x_glob = (blockIdx.x * blockDim.x) + threadIdx.x ;
    int y_glob = (blockIdx.y * blockDim.y) + threadIdx.y ;
	
	if(!(x_glob >= gpu_length-1 || x_glob <= 0 || y_glob >= gpu_length-1 || y_glob <= 0)){
    	gpu_output[(y_glob)*(gpu_length)+(x_glob)] = (gpu_input[(x_glob-1)*(gpu_length)+(y_glob-1)] +
                                                  gpu_input[(x_glob-1)*(gpu_length)+(y_glob)]   +
             	                                  gpu_input[(x_glob-1)*(gpu_length)+(y_glob+1)] +
                                                  gpu_input[(x_glob)*(gpu_length)+(y_glob-1)]   +
                     	                          gpu_input[(x_glob)*(gpu_length)+(y_glob)]     +
                                                  gpu_input[(x_glob)*(gpu_length)+(y_glob+1)]   +
                                                  gpu_input[(x_glob+1)*(gpu_length)+(y_glob-1)] +
                                                  gpu_input[(x_glob+1)*(gpu_length)+(y_glob)]   +
                                                  gpu_input[(x_glob+1)*(gpu_length)+(y_glob+1)] ) / 9;
    }
	
    __syncthreads();
	
	if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y){
    	gpu_output[(gpu_length/2-1)*gpu_length+(gpu_length/2-1)] = 1000;
    	gpu_output[(gpu_length/2)*gpu_length+(gpu_length/2-1)]   = 1000;
    	gpu_output[(gpu_length/2-1)*gpu_length+(gpu_length/2)]   = 1000;
    	gpu_output[(gpu_length/2)*gpu_length+(gpu_length/2)]     = 1000;

    	gpu_temp = gpu_input;
    	gpu_input = gpu_output;
    	gpu_output = gpu_temp;
    }
    __syncthreads();
}

// GPU Optimized function
void GPU_array_process(double *input, double *output, int length, int iterations)
{
    //Cuda events for calculating elapsed time
    cudaEvent_t cpy_H2D_start, cpy_H2D_end, comp_start, comp_end, cpy_D2H_start, cpy_D2H_end;
    cudaEventCreate(&cpy_H2D_start);
    cudaEventCreate(&cpy_H2D_end);
    cudaEventCreate(&cpy_D2H_start);
    cudaEventCreate(&cpy_D2H_end);
    cudaEventCreate(&comp_start);
    cudaEventCreate(&comp_end);

    /* Preprocessing goes here */
	double* GPU_input;
    double* GPU_output;
	double* GPU_temp;
	int* GPU_length;
	size_t size_n = length * length * sizeof(double)
	cudaMalloc((void**) &GPU_input, size_n);
    cudaMalloc((void**) &GPU_output, size_n);
	cudaMalloc((void**) &GPU_temp, size_n);
	cudaMalloc((void**) &GPU_length, sizeof(int));

    cudaEventRecord(cpy_H2D_start);
    /* Copying array from host to device goes here */
	cudaMemcpy((void*) GPU_input, (void*) &input, size_n, cudaMemcpyHostToDevice);
	cudaMemcpy((void*) GPU_output, (void*) &output, size_n, cudaMemcpyHostToDevice);
	cudaMemcpy((void*) GPU_length, (void*) length, sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(cpy_H2D_end);
    cudaEventSynchronize(cpy_H2D_end);

    //Copy array from host to device
    cudaEventRecord(comp_start);
    /* GPU calculation goes here */
	dim3 nBlks(1,1);
	dim3 thrsPerBlock(length, length);
	for(int n = 0; n < iterations; n++){
	    compute_GPU <<< nBlks, thrsPerBlock >>> (GPU_input, GPU_output, GPU_temp, GPU_length);
	}
	
    cudaEventRecord(comp_end);
    cudaEventSynchronize(comp_end);

    cudaEventRecord(cpy_D2H_start);
    /* Copying array from device to host goes here */
	cudaMemcpy((void*) &input, (void*) GPU_input, size_n, cudaMemcpyDeviceToHost);
	cudaMemcpy((void*) &output, (void*) GPU_output, size_n, cudaMemcpyDeviceToHost);

    cudaEventRecord(cpy_D2H_end);
    cudaEventSynchronize(cpy_D2H_end);

    /* Postprocessing goes here */
	cudaFree((void**) &GPU_input);
    cudaFree((void**) &GPU_output);
    cudaFree((void**) &GPU_temp);
	cudaFree((void*) &GPU_length);

    float time;
    cudaEventElapsedTime(&time, cpy_H2D_start, cpy_H2D_end);
    cout<<"Host to Device MemCpy takes "<<setprecision(4)<<time/1000<<"s"<<endl;

    cudaEventElapsedTime(&time, comp_start, comp_end);
    cout<<"Computation takes "<<setprecision(4)<<time/1000<<"s"<<endl;

    cudaEventElapsedTime(&time, cpy_D2H_start, cpy_D2H_end);
    cout<<"Device to Host MemCpy takes "<<setprecision(4)<<time/1000<<"s"<<endl;
}