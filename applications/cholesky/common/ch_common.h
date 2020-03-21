#ifndef _BENCH_CHOLESKY_COMMON_
#define _BENCH_CHOLESKY_COMMON_

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sys/syscall.h>
#include <unistd.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>

#include <mkl.h>
#include <mpi.h>
#include <omp.h>

#ifdef TRACE
#include "VT.h"
#endif

#if defined(USE_TIMING)
void helper_start_timing(int tt);
void helper_end_timing(int tt, double elapsed);
#endif

// #define SPEC_RESTRICT __restrict__
#define SPEC_RESTRICT restrict

#if defined(CHAMELEON) || defined(CHAMELEON_TARGET)
#include "chameleon.h"
#ifndef my_print
#define my_print(...) chameleon_print(0, "Cholesky", mype, __VA_ARGS__);
#endif
#else
#ifndef my_print
#define my_print(...) fprintf(stderr, __VA_ARGS__);
#endif
#endif

#ifndef DPxMOD
#define DPxMOD "0x%0*" PRIxPTR
#endif

#ifndef DPxPTR
#define DPxPTR(ptr) ((int)(2*sizeof(uintptr_t))), ((uintptr_t) (ptr))
#endif

#ifndef PRINT_DEBUG
#define PRINT_DEBUG 0
#endif

#ifdef _USE_HBW
#include <hbwmalloc.h>
#endif

void dgemm_ (const char *transa, const char *transb, int *l, int *n, int *m, double *alpha,
             const void *a, int *lda, void *b, int *ldb, double *beta, void *c, int *ldc);

void dtrsm_ (char *side, char *uplo, char *transa, char *diag, int *m, int *n, double *alpha,
             double *a, int *lda, double *b, int *ldb);

void dsyrk_ (char *uplo, char *trans, int *n, int *k, double *alpha, double *a, int *lda,
             double *beta, double *c, int *ldc);

void cholesky_single(const int ts, const int nt, double* A[nt][nt]);
void cholesky_mpi(const int ts, const int nt, double *A[nt][nt], double *B, double *C[nt], int *block_rank);

void omp_potrf(double * const A, int ts, int ld);
void omp_trsm(double * SPEC_RESTRICT A, double * SPEC_RESTRICT B, int ts, int ld);
void omp_gemm(double * SPEC_RESTRICT A, double * SPEC_RESTRICT B, double * SPEC_RESTRICT C, int ts, int ld);
void omp_syrk(double * SPEC_RESTRICT A, double * SPEC_RESTRICT B, int ts, int ld);

int get_send_flags(char *send_flags, int *block_rank, int itr1_str, int itr1_end, int itr2_str, int itr2_end, int n);
void get_recv_flag(char *recv_flag, int *block_rank, int itr1_str, int itr1_end, int itr2_str, int itr2_end, int n);

void wait(MPI_Request *comm_req);

inline static void waitall(MPI_Request *comm_req, int n)
{
// #ifdef TRACE
//     static int event_waitall = -1;
//     char* event_name = "waitall";
//     if(event_waitall == -1) {
//         int ierr;
//         ierr = VT_funcdef(event_name, VT_NOCLASS, &event_waitall);
//     }
//     VT_begin(event_waitall);
// #endif
#ifdef DISABLE_TASKYIELD
  MPI_Waitall(n, comm_req, MPI_STATUSES_IGNORE);
#else
  while (1) {
    int flag = 0;
    MPI_Testall(n, comm_req, &flag, MPI_STATUSES_IGNORE);
    if (flag) break;
    (void)flag; // <-- make the Cray compiler happy
#if defined(CHAMELEON) || defined(CHAMELEON_TARGET)
    int32_t res = chameleon_taskyield();
#else
#pragma omp taskyield
#endif
  }
#endif
// #ifdef TRACE
//     VT_end(event_waitall);
// #endif
}
void reset_send_flags(char *send_flags);

#ifdef MAIN
int np;
int mype;
int num_threads;
#else
extern int np;
extern int mype;
extern int num_threads;
#endif

#endif
