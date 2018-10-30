
#include "common.h"

static void cholesky_mpi(const int ts, const int nt, double *A[nt][nt], double *B, double *C[nt],
                         int *block_rank, MPI_Win *winB, MPI_Win *winC);

double do_cholesky_mpi(const int ts, const int nt, double *A[nt][nt], double *B, double *C[nt],
                       int *block_rank)
{
    double time;
    MPI_Win *winB, *winC;

    MPI_Alloc_mem(1  * sizeof(MPI_Win), MPI_INFO_NULL, &winB);
    MPI_Alloc_mem(nt * sizeof(MPI_Win), MPI_INFO_NULL, &winC);

    MPI_Win_create(B, ts*ts*sizeof(double), sizeof(double), MPI_INFO_NULL,
                   MPI_COMM_WORLD, &winB[0]);
    for (int i = 0; i < nt; i++) {
        MPI_Win_create(C[i], ts*ts*sizeof(double), sizeof(double), MPI_INFO_NULL,
                       MPI_COMM_WORLD, &winC[i]);
    }

    MPI_Win_lock_all(0, winB[0]);
    for (int i = 0; i < nt; i++) {
        MPI_Win_lock_all(0, winC[i]);
    }

    time = get_time();
    cholesky_mpi(ts, nt, (double* (*)[nt])A, B, C, block_rank, winB, winC);
    time = get_time() - time;

    MPI_Win_unlock_all(winB[0]);
    for (int i = 0; i < nt; i++) {
        MPI_Win_unlock_all(winC[i]);
    }

    free(winB);
    free(winC);

    return time;
}

static void cholesky_mpi(const int ts, const int nt, double *A[nt][nt], double *B, double *C[nt],
                         int *block_rank, MPI_Win *winB, MPI_Win *winC)
{
    int send_cnt = 0;
    char recv_flag = 0, *send_flags = malloc(sizeof(char) * np);
    MPI_Request recv_req, *send_reqs = malloc(sizeof(MPI_Request) * np);

#pragma omp parallel
{
    for (int k = 0; k < nt; k++) {

        if (block_rank[k*nt+k] == mype) {
#pragma omp single
            do_potrf(A[k][k], ts, ts);
#pragma omp master
{
            send_cnt = 0;
            reset_send_flags(send_flags);
            send_cnt = get_send_flags(send_flags, block_rank, k, k, k+1, nt-1, nt);

            if (send_cnt != 0) {

                send_cnt = 0;
                for (int dst = 0; dst < np; dst++) {
                    if (send_flags[dst] && dst != mype) {
                        MPI_Put(A[k][k], ts*ts, MPI_DOUBLE, dst, 0, ts*ts, MPI_DOUBLE,
                                winB[0]);
                    }
                }
                for (int dst = 0; dst < np; dst++) {
                    if (send_flags[dst] && dst != mype) {
                        MPI_Win_flush(dst, winB[0]);
                        MPI_Isend(NULL, 0, MPI_CHAR, dst, k*nt+k, MPI_COMM_WORLD,
                                  &send_reqs[send_cnt++]);
                    }
                }
                testall_and_yield(send_cnt, send_reqs);
            }
}
        } else {
#pragma omp single
{
            recv_flag = 0;
            get_recv_flag(&recv_flag, block_rank, k, k, k+1, nt-1, nt);

            if (recv_flag) {

                MPI_Irecv(NULL, 0, MPI_CHAR, block_rank[k*nt+k], k*nt+k, MPI_COMM_WORLD,
                          &recv_req);

                test_and_yield(&recv_req);
            }
}
        }

#pragma omp for
        for (int i = k + 1; i < nt; i++) {

            if (block_rank[k*nt+i] == mype) {
                if (block_rank[k*nt+k] == mype) {
                    do_trsm(A[k][k], A[k][i], ts, ts);
                } else {
                    do_trsm(B, A[k][i], ts, ts);
                }
            }
        }

#pragma omp single
        for (int i = k + 1; i < nt; i++) {

            if (block_rank[k*nt+i] != mype) {

                recv_flag = 0;
                get_recv_flag(&recv_flag, block_rank, k+1, i-1, i, i, nt);
                get_recv_flag(&recv_flag, block_rank, i, i, i+1, nt-1, nt);
                get_recv_flag(&recv_flag, block_rank, i, i, i, i, nt);

                if (recv_flag) {

                    MPI_Irecv(NULL, 0, MPI_CHAR, block_rank[k*nt+i], k*nt+i, MPI_COMM_WORLD,
                              &recv_req);

                    test_and_yield(&recv_req);
                }

            } else {

                send_cnt = 0;
                reset_send_flags(send_flags);
                send_cnt += get_send_flags(send_flags, block_rank, k+1, i-1, i, i, nt);
                send_cnt += get_send_flags(send_flags, block_rank, i, i, i+1, nt-1, nt);
                send_cnt += get_send_flags(send_flags, block_rank, i, i, i, i, nt);

                if (send_cnt != 0) {

                    send_cnt = 0;
                    for (int dst = 0; dst < np; dst++) {
                        if (send_flags[dst] && dst != mype) {
                            MPI_Put(A[k][i], ts*ts, MPI_DOUBLE, dst, 0, ts*ts, MPI_DOUBLE,
                                    winC[i]);
                        }
                    }
                    for (int dst = 0; dst < np; dst++) {
                        if (send_flags[dst] && dst != mype) {
                            MPI_Win_flush(dst, winC[i]);
                            MPI_Isend(NULL, 0, MPI_CHAR, dst, k*nt+i, MPI_COMM_WORLD,
                                      &send_reqs[send_cnt++]);
                        }
                    }
                    testall_and_yield(send_cnt, send_reqs);
                }
            }

            for (int j = k + 1; j < i; j++) {
                if (block_rank[j*nt+i] == mype) {
                    if (block_rank[k*nt+i] == mype && block_rank[k*nt+j] == mype) {
#pragma omp task
                        do_gemm(A[k][i], A[k][j], A[j][i], ts, ts);
                    } else if (block_rank[k*nt+i] != mype && block_rank[k*nt+j] == mype) {
#pragma omp task
                        do_gemm(C[i], A[k][j], A[j][i], ts, ts);
                    } else if (block_rank[k*nt+i] == mype && block_rank[k*nt+j] != mype) {
#pragma omp task
                        do_gemm(A[k][i], C[j], A[j][i], ts, ts);
                    } else {
#pragma omp task
                        do_gemm(C[i], C[j], A[j][i], ts, ts);
                    }
                }
            }
            if (block_rank[i*nt+i] == mype) {
                if (block_rank[k*nt+i] == mype) {
#pragma omp task
                    do_syrk(A[k][i], A[i][i], ts, ts);
                } else {
#pragma omp task
                    do_syrk(C[i], A[i][i], ts, ts);
                }
            }
        }
    }
} /* end omp parallel */
    MPI_Barrier(MPI_COMM_WORLD);
    free(send_flags);
}
