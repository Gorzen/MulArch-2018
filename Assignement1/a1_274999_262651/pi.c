/*
============================================================================
Filename    : pi.c
Author      : Lucien Michaël Iseli, Loris Pilotto
SCIPER		: 274999, 262651
============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include "utility.h"
#include <omp.h>

double calculate_pi (int num_threads, int samples);

int main (int argc, const char *argv[])
{

    int num_threads, num_samples;
    double pi;

    if (argc != 3) {
        printf("Invalid input! Usage: ./pi <num_threads> <num_samples> \n");
        return 1;
    } else {
        num_threads = atoi(argv[1]);
        num_samples = atoi(argv[2]);
    }

    set_clock();
    pi = calculate_pi (num_threads, num_samples);

    printf("- Using %d threads: pi = %.15g computed in %.4gs.\n", num_threads, pi, elapsed_time());

    return 0;
}


double calculate_pi (int num_threads, int samples)
{
    double pi;
    int counter = 0;
    int* counters = calloc(num_threads, sizeof(int));
    //if counters == null, error
    int numOfIt = samples/num_threads;

    omp_set_num_threads(num_threads);

    #pragma omp parallel
    {
		rand_gen gen = init_rand();
		int localCounter = 0;
		for(int i = 0; i < numOfIt; ++i) {
			double x = next_rand(gen);
			double y = next_rand(gen);
			double length = x*x + y*y;
			if(length <= 1) {
				localCounter += 1;
			}
		}
		counters[omp_get_thread_num()] = localCounter;
		free_rand(gen);
	}
	
	for(int i = 0; i < num_threads; ++i){
		counter += counters[i];
	}
	
	free(counters);

	pi = (4.0 * counter)/samples;

	return pi;
}
