
CC = gcc
CFLAGS = -std=gnu99 -O3 -fopenmp -Wall

all: pi integral test

pi: pi.c utility.h
	$(CC) $(CFLAGS) $< -o $@

integral: integral.c function.o utility.h
	$(CC) $(CFLAGS) $< -o $@

test: omp_hello.c
	$(CC) $(CFLAGS) $< -o $@
	./test

clean:
	rm -f pi integral function.o test
