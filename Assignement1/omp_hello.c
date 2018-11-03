/******************************************************************************
* FILE: omp_hello.c
* DESCRIPTION:
*   OpenMP Example - Hello World - C/C++ Version
*   In this simple example, the master thread forks a parallel region.
*   All threads in the team obtain their unique thread number and
*   a random number and print them. Only the master thread prints the
*   total number of threads.  Two OpenMP library routines are used to
*   obtain the number of threads and each thread's number. set_clock
*   and elapsed_time functions are used to measure the time. The three
*   *_rand functions are used to manage the random number generator.
******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "utility.h"

#define NTHREADS 8

void hello_world(int nthreads){

  int tid;

  /* Fork a team of threads giving them their own copies of variables */
  #pragma omp parallel private(nthreads, tid)
  {
    rand_gen gen = init_rand();
    double rand_number = next_rand(gen);

    /* Obtain thread number */
    tid = omp_get_thread_num();
    printf("Hello World from thread = %d, this is my random number: %f\n", tid, rand_number);

    /* Only master thread does this */
    if (tid == 0) {
      printf("Master thread: Number of threads = %d\n", nthreads);
    }

    free_rand(gen);
  }  /* All threads join master thread and disband */

}

int main (int argc, char *argv[]) {
  int nthreads = NTHREADS;

  omp_set_num_threads(nthreads);

  set_clock();
  hello_world(nthreads);
  printf("- Using %d threads: Hello World took %.4gs.\n", nthreads, elapsed_time());

  return 0;
}
