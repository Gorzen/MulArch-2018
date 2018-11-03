/*
============================================================================
Filename    : integral.c
Author      : Lucien Michaël Iseli, Loris Pilotto
SCIPER		: 274999, 262651
============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "utility.h"
#include "function.c"

double integrate (int num_threads, int samples, int a, int b, double (*f)(double));

int main (int argc, const char *argv[])
{

    int num_threads, num_samples, a, b;
    double integral;

    if (argc != 5) {
        printf("Invalid input! Usage: ./integral <num_threads> <num_samples> <a> <b>\n");
        return 1;
    } else {
        num_threads = atoi(argv[1]);
        num_samples = atoi(argv[2]);
        a = atoi(argv[3]);
        b = atoi(argv[4]);
    }

    set_clock();

    /* You can use your self-defined funtions by replacing identity_f. */
    integral = integrate (num_threads, num_samples, a, b, identity_f);

    printf("- Using %d threads: integral on [%d,%d] = %.15g computed in %.4gs.\n", num_threads, a, b, integral, elapsed_time());

    return 0;
}


double integrate (int num_threads, int samples, int a, int b, double (*f)(double))
{
    double integral = 0;
    double* integrals = calloc(num_threads, sizeof(double));
    //integrals == null is error -> seg fault

    int numOfIt = samples/num_threads;

    omp_set_num_threads(num_threads);

    #pragma omp parallel
    {
		rand_gen gen = init_rand();
		double localIntegral = 0;
		
        for(int i = 0; i < numOfIt; ++i) {
            double x = a + next_rand(gen) * (b-a);
            double y = f(x);
            localIntegral = localIntegral + (b - a) * y;
        }
        
        integrals[omp_get_thread_num()] = localIntegral;
        free_rand(gen);
    }
    
    for(int i = 0; i < num_threads; ++i){
		integral += integrals[i];
	}
    
    free(integrals);

    return integral/samples;
}
