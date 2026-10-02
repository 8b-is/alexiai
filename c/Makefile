CC ?= cc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra -Wpedantic

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

clean:
	rm -f test_kernel bench

.PHONY: all test clean
