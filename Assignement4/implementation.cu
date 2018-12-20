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
#include <math.h>
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
void compute_gpu(double* gpu_input, double* gpu_output, int length){
    int x_glob = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y_glob = (blockIdx.y * blockDim.y) + threadIdx.y;

    if(x_glob == length/2-1 && (y_glob == length/2 || y_glob == length/2-1) ||
       x_glob == length/2 && (y_glob == length/2 || y_glob == length/2-1) || 
       x_glob <= 0 || x_glob >= length-1 || y_glob <= 0 || y_glob >= length-1)
	    return;

    gpu_output[(x_glob)*(length)+(y_glob)] = (gpu_input[(x_glob-1)*(length)+(y_glob-1)] +
                                              gpu_input[(x_glob-1)*(length)+(y_glob)]   +
        	                              gpu_input[(x_glob-1)*(length)+(y_glob+1)] +
                                              gpu_input[(x_glob)*(length)+(y_glob-1)]   +
                                              gpu_input[(x_glob)*(length)+(y_glob)]     +
                                              gpu_input[(x_glob)*(length)+(y_glob+1)]   +
                                              gpu_input[(x_glob+1)*(length)+(y_glob-1)] +
                                              gpu_input[(x_glob+1)*(length)+(y_glob)]   +
                                              gpu_input[(x_glob+1)*(length)+(y_glob+1)] ) /9;
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
    size_t SIZE = length * length * sizeof(double);
    double* gpu_input;
    double* gpu_output;
    double* temp;
    
    cudaMalloc((void**) &gpu_input, SIZE);
    cudaMalloc((void**) &gpu_output, SIZE);
    /* End preprocessing       */

    cudaEventRecord(cpy_H2D_start);
    /* Copying array from host to device goes here */
    cudaMemcpy((void*)gpu_input,
	       (void*)input,
	       SIZE,
	       cudaMemcpyHostToDevice);
    cudaMemcpy((void*)gpu_output,
	       (void*)output,
	       SIZE,
	       cudaMemcpyHostToDevice);
    /* End copy array				   */
    cudaEventRecord(cpy_H2D_end);
    cudaEventSynchronize(cpy_H2D_end);

    //Copy array from host to device
    cudaEventRecord(comp_start);
    /* GPU calculation goes here */
    dim3 thrsPerBlock(32,32);
    dim3 nBlks(ceil(length/32),ceil(length/32));

    for(int n = 0; n <(int)iterations; n++){
	compute_gpu <<< nBlks, thrsPerBlock >>> (gpu_input, gpu_output, length);
	
	temp = gpu_input;
	gpu_input = gpu_output;
	gpu_output = temp;
    }

    /* End GPU calculation	 */
    cudaEventRecord(comp_end);
    cudaEventSynchronize(comp_end);

    cudaEventRecord(cpy_D2H_start);
    /* Copying array from device to host goes here */
    cudaMemcpy((void*)output,
	       (void*)gpu_output,
	       SIZE,
	       cudaMemcpyDeviceToHost);
    /* End copy array 				   */
    cudaEventRecord(cpy_D2H_end);
    cudaEventSynchronize(cpy_D2H_end);

    /* Postprocessing goes here */
    cudaFree((void**) &gpu_input);
    cudaFree((void**) &gpu_output);

    float time;
    cudaEventElapsedTime(&time, cpy_H2D_start, cpy_H2D_end);
    cout<<"Host to Device MemCpy takes "<<setprecision(4)<<time/1000<<"s"<<endl;

    cudaEventElapsedTime(&time, comp_start, comp_end);
    cout<<"Computation takes "<<setprecision(4)<<time/1000<<"s"<<endl;

    cudaEventElapsedTime(&time, cpy_D2H_start, cpy_D2H_end);
    cout<<"Device to Host MemCpy takes "<<setprecision(4)<<time/1000<<"s"<<endl;
}
