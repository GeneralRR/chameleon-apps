// number of tasks 
#ifndef NR_TASKS
#define NR_TASKS 200
#endif

#ifndef RANDOMINIT
#define RANDOMINIT 0
#endif

#ifndef RANDOMDIST
#define RANDOMDIST 1
#endif

#ifndef PARALLEL_INIT
#define PARALLEL_INIT 1
#endif

#ifndef VERY_VERBOSE
#define VERY_VERBOSE 0
#endif

#ifndef CHECK_GENERATED_TASK_ID
#define CHECK_GENERATED_TASK_ID 0
#endif

#ifndef SIMULATE_CONST_WORK
#define SIMULATE_CONST_WORK 0
#endif

#ifndef COMPILE_CHAMELEON
#define COMPILE_CHAMELEON 1
#endif

#ifndef COMPILE_TASKING
#define COMPILE_TASKING 1
#endif

#ifndef USE_TASK_ANNOTATIONS
#define USE_TASK_ANNOTATIONS 0
#endif

#ifndef USE_REPLICATION
#define USE_REPLICATION 0
#endif

#ifndef ITERATIVE_VERSION
#define ITERATIVE_VERSION 1
#endif

#ifndef NUM_ITERATIONS
#define NUM_ITERATIONS 1
#endif

#ifndef NUM_REPETITIONS
#define NUM_REPETITIONS 1
#endif

#ifndef USE_EXTERNAL_CALLBACK
#define USE_EXTERNAL_CALLBACK 0
#endif

#if !COMPILE_CHAMELEON
#undef USE_EXTERNAL_CALLBACK
#define USE_EXTERNAL_CALLBACK 0
#endif

//#define LOG(rank, str) fprintf(stderr, "#R%d: %s\n", rank, str)
#define LOG(rank, str) printf("#R%d: %s\n", rank, str)

#include <assert.h>
#include <mpi.h>
#include "chameleon.h"
#include <cstdlib>
#include <cstdio>
#include <random>
#include <iostream>
#include <string>
#include <sstream>
#include "math.h"
#include <cmath>
#include <unistd.h>
#include <sys/syscall.h>
#include <atomic>

#if CHECK_GENERATED_TASK_ID
#include <mutex>
#include <list>
#endif


#define SPEC_RESTRICT __restrict__
//#define SPEC_RESTRICT restrict

// static rank id that can also be used in other functions except main
static int my_rank_id = -1;

#if USE_EXTERNAL_CALLBACK
typedef struct my_custom_params_t {
    int val1;
    TYPE_TASK_ID task_id;
} my_custom_params_t;

void print_finish_message(void *param) {
    // parse parameter again to regular data type
    my_custom_params_t *mydata = (my_custom_params_t *) param;
    printf("#R%d (OS_TID:%ld): External task finish callback for task with id %d with value %d\n", my_rank_id, syscall(SYS_gettid), mydata->task_id, mydata->val1);
    // clean up
    free(mydata);
}
#endif

void initialize_matrix_rnd(double *mat, int matrixSize) {
	double lower_bound = 0;
	double upper_bound = 10000;
	std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
	std::default_random_engine re;

	for(int i=0; i<matrixSize*matrixSize; i++) {
		mat[i]= unif(re);
	}
}

void initialize_matrix_zero(double *mat, int matrixSize) {
	for(int i=0; i<matrixSize*matrixSize; i++) {
		mat[i]= 0;
	}
}

void initialize_matrix_test_A(double *mat, int matrixSize) {
	for(int i=0; i<matrixSize*matrixSize; i++) {
			mat[i]= 1;
    }
}

void compute_matrix_matrix(double * SPEC_RESTRICT a, double * SPEC_RESTRICT b, double * SPEC_RESTRICT c, int matrixSize) {
    // make the tasks more computational expensive by repeating this operation several times to better see effects 
    for(int iter=0;iter<NUM_REPETITIONS;iter++) {
        for(int i=0;i<matrixSize;i++) {
            for(int j=0;j<matrixSize;j++) {
                c[i*matrixSize+j]=0;
                for(int k=0;k<matrixSize;k++) {
                    c[i*matrixSize+j] += a[i*matrixSize+k] * b[k*matrixSize+j];
                }
            }
        }
    }
}

bool check_test_matrix(double *c, double val, int matrixSize) {
	for(int i=0;i<matrixSize;i++) {
		for(int j=0;j<matrixSize;j++) {
			if(fabs(c[i*matrixSize+j] - val) > 1e-3) {
				printf("#R%d (OS_TID:%ld): Error in matrix entry (%d,%d) expected:%f but value is %f\n", my_rank_id, syscall(SYS_gettid),i,j,val,c[i*matrixSize+j]);
				return false;
			}
		}
	}
	return true;
}

void compute_random_task_distribution(int *dist, int nRanks) {
	double *weights = new double[nRanks];
	
	double lower_bound = 0;
	double upper_bound = 1;
	std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
	std::default_random_engine re;
	double sum = 0;

	for(int i=0; i<nRanks; i++) {
		weights[i]= unif(re);
		sum += weights[i];
	}

	for(int i=0; i<nRanks; i++) {
		weights[i]= weights[i]/sum;
		dist[i] = weights[i]*NR_TASKS;
	}

	delete[] weights;
}

void printHelpMessage() {
    std::cout<<"Usage: mpiexec -n np ./matrixExample matrixSize [nt_(0) ... nt_(np-1)] "<<std::endl;
    std::cout<<"    Arguments: "<<std::endl;
    std::cout<<"        matrixSize: number of elements of the matrixSize x matrixSize matrices"<<std::endl;
    std::cout<<"        nt_(i): number of tasks for process i "<<std::endl;
    std::cout<<"If the number of tasks is not specified for every process, the application will generate an initial task distribution"<<std::endl; 
}

void printArray(int rank, double * SPEC_RESTRICT array, char* arr_name, int n) {
    printf("#R%d (OS_TID:%ld): %s[0-%d] at (" DPxMOD "): ", rank, syscall(SYS_gettid), arr_name, n, DPxPTR(&array[0]));
    for(int i = 0; i < n; i++) {
        printf("%f ", array[i]);
    }
    printf("\n");
}

void matrixMatrixKernel(double * SPEC_RESTRICT A, double * SPEC_RESTRICT B, double * SPEC_RESTRICT C, int matrixSize, int i) {
#if VERY_VERBOSE
    int iMyRank2;
    MPI_Comm_rank(MPI_COMM_WORLD, &iMyRank2);
    printArray(iMyRank2, A, "A", 10);
    printArray(iMyRank2, B, "B", 10);
    printArray(iMyRank2, C, "C", 10);
#endif

#if SIMULATE_CONST_WORK
    // simulate work by just touching the arrays and wait for 50 ms here
    C[matrixSize] = A[matrixSize] * B[matrixSize];
    usleep(50000);
#else
    compute_matrix_matrix(A, B, C, matrixSize);
#endif
}

int main(int argc, char **argv)
{
	int iMyRank, iNumProcs;
	int provided;
	int requested = MPI_THREAD_MULTIPLE;
	MPI_Init_thread(&argc, &argv, requested, &provided);
	MPI_Comm_size(MPI_COMM_WORLD, &iNumProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &iMyRank);
    my_rank_id = iMyRank;
	int numberOfTasks;
	int matrixSize;
	double fTimeStart, fTimeEnd;
	double wTimeCham, wTimeHost;
	bool pass = true;

#if CHECK_GENERATED_TASK_ID
    std::mutex mtx_t_ids;
    std::list<int32_t> t_ids;
#endif

#if COMPILE_CHAMELEON
    // chameleon_init();
    #pragma omp parallel
    {
        chameleon_thread_init();
    }
    // necessary to be aware of binary base addresses to calculate offset for target entry functions
    chameleon_determine_base_addresses((void *)&main);

#if USE_EXTERNAL_CALLBACK
    chameleon_external_callback_t call_back_fcn = print_finish_message;
#endif
#endif /* COMPILE_CHAMELEON */

    if(argc==2) {
        matrixSize = atoi( argv[1] );
        if(RANDOMDIST) {
            int *dist = new int[iNumProcs];
            if( iMyRank==0 ) {
                compute_random_task_distribution(dist, iNumProcs);
            }
            MPI_Bcast(dist, iNumProcs, MPI_INTEGER, 0, MPI_COMM_WORLD);
            numberOfTasks = dist[iMyRank];
            delete[] dist;
        } else {
            numberOfTasks = NR_TASKS;
        }
    } else if(argc==iNumProcs+2) {
        if(iMyRank==0) {
            LOG(iMyRank, "using user-defined initial load distribution...");    
        } 
        matrixSize = atoi( argv[1] ); 
        numberOfTasks = atoi( argv[iMyRank+2] ); 
    } else { 
        printHelpMessage();
        return 0;     
    }
	
    std::string msg = "will create "+std::to_string(numberOfTasks)+" tasks";
    LOG(iMyRank, msg.c_str());

	double **matrices_a, **matrices_b, **matrices_c;

	matrices_a = new double*[numberOfTasks];
	matrices_b = new double*[numberOfTasks];
	matrices_c = new double*[numberOfTasks];

	//allocate and initialize matrices
#if PARALLEL_INIT
    if(iMyRank == 0) {
        printf("Executing parallel init\n");
    }
    #pragma omp parallel for
#endif
	for(int i=0; i<numberOfTasks; i++) {
 		matrices_a[i] = new double[(long)matrixSize*matrixSize];
    	matrices_b[i] = new double[(long)matrixSize*matrixSize];
    	matrices_c[i] = new double[(long)matrixSize*matrixSize];
    	if(RANDOMINIT) {
    		initialize_matrix_rnd(matrices_a[i], matrixSize);
    		initialize_matrix_rnd(matrices_b[i], matrixSize);
    		initialize_matrix_zero(matrices_c[i], matrixSize);
    	}
    	else {
    		initialize_matrix_test_A(matrices_a[i], matrixSize);
    		initialize_matrix_test_A(matrices_b[i], matrixSize);
    		initialize_matrix_zero(matrices_c[i], matrixSize);
    	}
#if VERY_VERBOSE
        printArray(iMyRank, matrices_a[i], "A", 10);
        printArray(iMyRank, matrices_b[i], "B", 10);
        printArray(iMyRank, matrices_c[i], "C", 10);
#endif
    }

    MPI_Barrier(MPI_COMM_WORLD);

#if COMPILE_CHAMELEON
    fTimeStart=MPI_Wtime();
    #pragma omp parallel
    {
#if ITERATIVE_VERSION
        for(int iter = 0; iter < NUM_ITERATIONS; iter++) {
            if(iMyRank == 0) {
                #pragma omp master
                printf("Executing iteration %d ...\n", iter);
            }
#endif
    	// if(iMyRank==0) {
#if USE_REPLICATION
        std::atomic<int> replicated_cnt = 0;
        int num_to_replicate            = 10;
#endif
		#pragma omp for
    		for(int i=0; i<numberOfTasks; i++) {
				double * SPEC_RESTRICT A = matrices_a[i];
		        double * SPEC_RESTRICT B = matrices_b[i];
		        double * SPEC_RESTRICT C = matrices_c[i];
#if VERY_VERBOSE	
                printArray(iMyRank, A, "A", 10);
                printArray(iMyRank, B, "B", 10);
                printArray(iMyRank, C, "C", 10);
#endif
                // here we need to call library function to add task entry point and parameters by hand
                void* literal_matrix_size   = *(void**)(&matrixSize);
                void* literal_i             = *(void**)(&i);

                chameleon_map_data_entry_t* args = new chameleon_map_data_entry_t[5];
                args[0] = chameleon_map_data_entry_create(A, matrixSize*matrixSize*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
                args[1] = chameleon_map_data_entry_create(B, matrixSize*matrixSize*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
                args[2] = chameleon_map_data_entry_create(C, matrixSize*matrixSize*sizeof(double), CHAM_OMP_TGT_MAPTYPE_FROM);
                args[3] = chameleon_map_data_entry_create(literal_matrix_size, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                args[4] = chameleon_map_data_entry_create(literal_i, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);

                // create opaque task here
                cham_migratable_task_t *cur_task = chameleon_create_task((void *)&matrixMatrixKernel, 5, args);
#if USE_TASK_ANNOTATIONS
                chameleon_annotations_t* annotations = chameleon_create_annotation_container();
                chameleon_set_annotation_int(annotations, "Int", 42);
                chameleon_set_annotation_double(annotations, "Dbl", 42.1345);
                chameleon_set_annotation_string(annotations, "Str", "Test123");
                chameleon_set_task_annotations(cur_task, annotations);
#endif
#if USE_REPLICATION
                if(iMyRank==0) {
                    if(replicated_cnt++<num_to_replicate) {
                        int num_replication = 1;
                        int *replication_ranks = new int[num_replication];
                        replication_ranks[0] = 1;
                        chameleon_set_task_replication_info(cur_task, num_replication, replication_ranks);
                        delete[] replication_ranks;
                    }
                }
#endif
                // get the id of the last task added
                TYPE_TASK_ID last_t_id = chameleon_get_task_id(cur_task);
#if CHECK_GENERATED_TASK_ID
                printf("#R%d (OS_TID:%ld): last task that has been created: %ld\n", iMyRank, syscall(SYS_gettid), last_t_id);
                mtx_t_ids.lock();
                t_ids.push_back(last_t_id);
                mtx_t_ids.unlock();
#endif
#if USE_EXTERNAL_CALLBACK
                my_custom_params_t* tmp_struct = (my_custom_params_t*) malloc(sizeof(my_custom_params_t));
                tmp_struct->val1 = 42;
                tmp_struct->task_id = last_t_id;
                chameleon_set_callback_task_finish(cur_task, call_back_fcn, (void*)tmp_struct);
#endif
                int32_t res = chameleon_add_task(cur_task);
                // clean up again
                delete[] args;
#if USE_TASK_ANNOTATIONS
                chameleon_annotations_t* tmp_annotations = chameleon_get_task_annotations(last_t_id);
                int     found = 0; 
                int     val_annotation_int;
                double  val_annotation_dbl;
                char*   val_annotation_string;
                
                found = chameleon_get_annotation_int(tmp_annotations, "Int", &val_annotation_int);
                assert(found == 1 && val_annotation_int == 42);
                found = chameleon_get_annotation_double(tmp_annotations, "Dbl", &val_annotation_dbl);
                assert(found == 1 && val_annotation_dbl == 42.1345);
                found = chameleon_get_annotation_string(tmp_annotations, "Str", &val_annotation_string);
                assert(found == 1 && strcmp("Test123", val_annotation_string) == 0);
#endif
    		}

#if CHECK_GENERATED_TASK_ID
        #pragma omp barrier
        #pragma omp single
        {
            printf("Before Running Tasks\n");
            for (std::list<int32_t>::iterator it=t_ids.begin(); it!=t_ids.end(); ++it) {
                printf("R#%d Task with id %d finished?? ==> %d\n", iMyRank, *it, chameleon_local_task_has_finished(*it));
            }
        }
#endif
    	int res = chameleon_distributed_taskwait(0);

        #pragma omp single
        MPI_Barrier(MPI_COMM_WORLD);
#if ITERATIVE_VERSION
        }
#endif
    }
    MPI_Barrier(MPI_COMM_WORLD);

#if CHECK_GENERATED_TASK_ID
    printf("After Running Tasks\n");
    for (std::list<int32_t>::iterator it=t_ids.begin(); it!=t_ids.end(); ++it) {
        printf("R#%d Task with id %d finished?? ==> %d\n", iMyRank, *it, chameleon_local_task_has_finished(*it));
    }
#endif

    fTimeEnd=MPI_Wtime();
    wTimeCham = fTimeEnd-fTimeStart;
    if( iMyRank==0 ) {
        printf("#R%d: Computations with chameleon took %.5f\n", iMyRank, wTimeCham);
    }
    LOG(iMyRank, "Validation:");
    if(numberOfTasks>0) {
        for(int t=0; t<numberOfTasks; t++) {
            pass &= check_test_matrix(matrices_c[t], matrixSize, matrixSize);
        }
        if(pass)
            LOG(iMyRank, "TEST SUCCESS");
        else
            LOG(iMyRank, "TEST FAILED");
    }
#endif /* COMPILE_CHAMELEON */

    MPI_Barrier(MPI_COMM_WORLD);

#if COMPILE_TASKING
    fTimeStart=MPI_Wtime();
    #pragma omp parallel
    {
#if ITERATIVE_VERSION
        for(int iter = 0; iter < NUM_ITERATIONS; iter++) {
            if(iMyRank == 0) {
                #pragma omp master
                printf("Executing iteration %d ...\n", iter);
            }
#endif
		#pragma omp for
        for(int i=0; i<numberOfTasks; i++) {
            // double *A = matrices_a[i];
            // double *B = matrices_b[i];
            // double *C = matrices_c[i];
            
            // somehow target offloading is very slow when performing more that one iteration
            // #pragma omp target map(from: C[0:matrixSize*matrixSize]) map(to:matrixSize, A[0:matrixSize*matrixSize], B[0:matrixSize*matrixSize]) device(1001)
            
            // uses normal tasks to have a fair comparison
            #pragma omp task default(shared) firstprivate(i)
            {
                compute_matrix_matrix(matrices_a[i], matrices_b[i], matrices_c[i], matrixSize);
            }
        }

        #pragma omp single
        MPI_Barrier(MPI_COMM_WORLD);
#if ITERATIVE_VERSION
        }
#endif
    }
    MPI_Barrier(MPI_COMM_WORLD);
    fTimeEnd=MPI_Wtime();
    wTimeHost = fTimeEnd-fTimeStart;

    if( iMyRank==0 ) {
        printf("#R%d: Computations with normal tasking took %.5f\n", iMyRank, wTimeHost);
    }

    LOG(iMyRank, "Validation:");
    pass = true;
    if(numberOfTasks>0) {
        for(int t=0; t<numberOfTasks; t++) {
            pass &= check_test_matrix(matrices_c[t], matrixSize, matrixSize);
        }
        if(pass)
            LOG(iMyRank, "TEST SUCCESS");
        else
            LOG(iMyRank, "TEST FAILED");
    }
#endif /* COMPILE_TASKING */

#if COMPILE_TASKING && COMPILE_CHAMELEON
    if( iMyRank==0 ) {
        printf("#R%d: This corresponds to a speedup of %.5f!\n", iMyRank, wTimeHost/wTimeCham);
    }
#endif /* COMPILE_TASKING && COMPILE_CHAMELEON */  

    //deallocate matrices
    for(int i=0; i<numberOfTasks; i++) {
    	delete[] matrices_a[i];
    	delete[] matrices_b[i];
    	delete[] matrices_c[i];
    }

    delete[] matrices_a;
    delete[] matrices_b;
    delete[] matrices_c;

    MPI_Barrier(MPI_COMM_WORLD);
#if COMPILE_CHAMELEON
    #pragma omp parallel
    {
        chameleon_thread_finalize();
    }
    chameleon_finalize();
#endif
    MPI_Finalize();
    return 0;
}
