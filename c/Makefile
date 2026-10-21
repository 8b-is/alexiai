CC ?= cc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra -Wpedantic
NVCC ?= $(shell which nvcc 2>/dev/null || echo /opt/cuda/bin/nvcc)
CUDA_ARCH ?= -arch=native

UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
# SSE4.1 turns on the x86_64 assembly path; without it the portable C99
# fallback compiles instead. aarch64 gets NEON as its baseline, no flags.
CFLAGS += -msse4.1
endif

all: test_kernel

test_kernel: test_kernel.c kernel.c kernel.h
	$(CC) $(CFLAGS) -o $@ test_kernel.c kernel.c -lm

test: test_kernel
	./test_kernel

bench: bench.c kernel.c kernel.h
	$(CC) $(CFLAGS) -o $@ bench.c kernel.c -lm
	./bench

cuda-test: test_kernel.c kernel.c kernel_cuda.cu kernel.h kernel_cuda.h
	$(CC) $(CFLAGS) -DGMQ_CUDA -c kernel.c -o kernel.o
	$(CC) $(CFLAGS) -DGMQ_CUDA -c test_kernel.c -o test_kernel.o
	$(NVCC) -O2 $(CUDA_ARCH) -c kernel_cuda.cu -o kernel_cuda.o
	$(NVCC) -O2 $(CUDA_ARCH) -o test_kernel_cuda test_kernel.o kernel.o kernel_cuda.o -lm
	./test_kernel_cuda

cuda-bench: bench.c kernel.c kernel_cuda.cu kernel.h kernel_cuda.h
	$(CC) $(CFLAGS) -DGMQ_CUDA -c kernel.c -o kernel.o
	$(CC) $(CFLAGS) -DGMQ_CUDA -c bench.c -o bench.o
	$(NVCC) -O2 $(CUDA_ARCH) -c kernel_cuda.cu -o kernel_cuda.o
	$(NVCC) -O2 $(CUDA_ARCH) -o bench_cuda bench.o kernel.o kernel_cuda.o -lm
	./bench_cuda 1024

clean:
	rm -f test_kernel bench test_kernel_cuda bench_cuda *.o

.PHONY: all test bench cuda-test cuda-bench clean
