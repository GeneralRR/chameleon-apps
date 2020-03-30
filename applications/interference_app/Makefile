CC ?= gcc
FLAGS_DEBUG ?= -g
FLAGS_FAST ?= -O3
THREADS ?= $$(nproc --all) 


all: clean dist

dist:
	$(MPICC) $(FLAGS_DEBUG) $(FLAGS_FAST) -std=c11 -qopenmp dist.c -o dist.exe

#dist:
#	$(CC) $(FLAGS_DEBUG) $(FLAGS_FAST) -qopenmp dist.c -o dist.exe

run:
	env OMP_NUM_THREADS=$(THREADS) ./dist.exe

clean: 
	rm -rf *.exe core* *.o *.mod output*