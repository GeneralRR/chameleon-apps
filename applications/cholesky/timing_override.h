#ifndef _BENCH_CHOLESKY_TIMING_OVR_
#define _BENCH_CHOLESKY_TIMING_OVR_

#include "timing.h"

#ifdef USE_TIMING

#define wait(req) wait_impl(req, &__timing[THREAD_NUM].ts[__timer])
#define waitall(req, nreq) waitall_impl(req, nreq, &__timing[THREAD_NUM].ts[__timer])

static void wait_impl(MPI_Request *comm_req, double *timer)
{
    int comm_comp = 0;

    MPI_Test(comm_req, &comm_comp, MPI_STATUS_IGNORE);
    while (!comm_comp) {
        double yield_time = timestamp();
        #pragma omp taskyield
        *timer -= timestamp() - yield_time;
        MPI_Test(comm_req, &comm_comp, MPI_STATUS_IGNORE);
    }
}

static void waitall_impl(MPI_Request *comm_req, int nreq, double *timer)
{
    int comm_comp = 0;

    MPI_Testall(nreq, comm_req, &comm_comp, MPI_STATUS_IGNORE);
    while (!comm_comp) {
        double yield_time = timestamp();
#pragma omp taskyield
        *timer -= timestamp() - yield_time;
        MPI_Testall(nreq, comm_req, &comm_comp, MPI_STATUS_IGNORE);
    }
}
#endif

#endif // _BENCH_CHOLESKY_TIMING_