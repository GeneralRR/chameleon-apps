//#include <mpi.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/time.h>
#include <omp.h>

#define MB 1024*1024

double omp_get_wtime() {
    struct timeval currentTime;
	gettimeofday(&currentTime, NULL);
	return (double)(currentTime.tv_sec) +  (1.0E-06)*(double)(currentTime.tv_usec);
}

char **split(char *str, char div, int num_args);

char *space = "\n-------------------------------------------------\n";

typedef enum disturbance_mode_t {
    compute = 0,
    memory = 1,
    communication = 2,
} disturbance_mode;

disturbance_mode d_mode = compute;
int window_comp = 10;
int window_pause = 5;
int window_size_min = 1;
int window_size_max = 10;
bool use_random = false;
int rank_number = 0;
int use_multiple_cores = 1;
int use_ram = 1;
int seed = 0;

void compute_kernel()
{
    int i, j, v;
    int n = 500000000;
    int length = 8;
    float a[length];
    float c[length];


    for (i = 0; i < length; i++)
    {
        a[i] = i;
        c[i] = 1.0+i*0.01;
    }

    if (window_pause == 0) {
        while(true)
        {
            for (i = 0; i < n; i++)
                for (v = 0; v < length; v++)
                    a[v]=a[v]+c[v]*i;
                
            for (v = 0; v < length; v++)
                a[v]=(double)(((int)a[v])%length);
        }
    }

    double passed_time = 0.0;
    while(true)
    {
        double tmp_time = omp_get_wtime();
        for (i = 0; i < n; i++)
            for (v = 0; v < length; v++)
                a[v]=a[v]+c[v]*i;
            
        for (v = 0; v < length; v++)
            a[v]=(double)(((int)a[v])%length);

        passed_time += omp_get_wtime()-tmp_time;

        //printf("%f\n",passed_time);

        if(passed_time > window_comp)
        {
            //printf("go to sleep\n");
            passed_time -= window_comp;
            sleep(window_pause);
            //printf("wake up\n");
        }

    }
}

void memory_kernel()
{

    long long size_of_ram_usage = use_ram/use_multiple_cores/2;
    printf ("Allocating %dMB of RAM\n",size_of_ram_usage*2);

    void *p = malloc(size_of_ram_usage*MB);
    void *q = malloc(size_of_ram_usage*MB);
    int i = 0;

    if (window_pause == 0) {

        memset(p, 1, size_of_ram_usage*MB);
        memcpy(q, p, size_of_ram_usage*MB);

        while(true)
        {
            memset(p, i, size_of_ram_usage*MB);
            memcpy(q, p,  size_of_ram_usage*MB);
            //sleep(1);

            if (i == 0)
                i = 1;
            else
                i = 0;
            
        }
        return;
    }

    double passed_time = 0.0;
    while(true)
    {
        double tmp_time = omp_get_wtime();
        memset(p, i, size_of_ram_usage*MB);
        memcpy(q, p,  size_of_ram_usage*MB);

        passed_time += omp_get_wtime()-tmp_time;
        //printf("Time Passed: %f\n",passed_time);

        if (i == 0)
            i = 1;
        else
            i = 0;

        if(passed_time > window_comp)
        {
            passed_time -= window_comp;
            free(p);
            free(q);
            sleep(window_pause);
            p = malloc(size_of_ram_usage*MB);
            q = malloc(size_of_ram_usage*MB);
        }

    }
}

void communication_kernel()
{
    while(true)
    {

    }
}

int main(int argc, char *argv[])
{
    int i, k;

    int iMyRank, iNumProcs;
	int provided;
	/*int requested = MPI_THREAD_MULTIPLE;
	MPI_Init_thread(&argc, &argv, requested, &provided);
	MPI_Comm_size(MPI_COMM_WORLD, &iNumProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &iMyRank);*/
    
    for (i = 1; i < argc; i++)
    {
        char **s = split(argv[i], '=', 2);

        if(strcmp(s[0], "--type") == 0)
        {
            if(strcmp(s[1], "compute") == 0)
            {
                d_mode = 0;
            } 
            else if(strcmp(s[1], "memory") == 0)
            {
                d_mode = 1;
            } 
            else if(strcmp(s[1], "communication") == 0)
            {
                d_mode = 2;
            } else
            {
                printf(space);
                printf("--type was not set correctly!\n");
                printf("please try one of the following:\n");
                printf("\tcompute\n\tmemory\n\tcommunication");
                printf(space);
                continue;
            }
            
            printf("--type is set to: %s\n", s[1]);
        }
        else if(strcmp(s[0], "--window_comp") == 0)
        {
            int a = atoi(s[1]);
            if (a == 0 && s[1] != "0" || a < 0)
            {
                printf(space);
                printf("--window_comp was not set correctly!\n");
                printf("please only use positiv integers!");
                printf(space);
                continue;
            }
            window_comp = a;
            printf("--window_comp is set to: %s\n", s[1]);
        }
        else if(strcmp(s[0], "--window_size_max") == 0)
        {
            int a = atoi(s[1]);
            if (a == 0 && s[1] != "0" || a < 0)
            {
                printf(space);
                printf("--window_size_max was not set correctly!\n");
                printf("please only use positiv integers!");
                printf(space);
                continue;
            }
            window_size_max = a;
            printf("--window_size_max is set to: %s\n", s[1]);
        }
        else if(strcmp(s[0], "--window_size_min") == 0)
        {
            int a = atoi(s[1]);
            if (a == 0 && s[1] != "0" || a < 0)
            {
                printf(space);
                printf("--window_size_min was not set correctly!\n");
                printf("please only use positiv integers!");
                printf(space);
                continue;
            }
            window_size_min = a;
            printf("--window_size_min is set to: %s\n", s[1]);
        }
        else if(strcmp(s[0], "--window_pause") == 0)
        {
            int a = atoi(s[1]);
            if (a == 0 && s[1] != "0" || a < 0)
            {
                printf(space);
                printf("--window_pause was not set correctly!\n");
                printf("please only use positiv integers!");
                printf(space);
                continue;
            }
            window_pause = a;
            printf("--window_pause is set to: %s\n", s[1]);
        }
        else if(strcmp(s[0], "--use_multiple_cores") == 0)
        {
            int a = atoi(s[1]);
            if (a == 0 && s[1] != "0" || a < 0)
            {
                printf(space);
                printf("--use_multiple_cores was not set correctly!\n");
                printf("please only use positiv integers!");
                printf(space);
                continue;
            }
            use_multiple_cores = a;
            printf("--use_multiple_cores is set to: %s\n", s[1]);
        }
        else if(strcmp(s[0], "--use_random") == 0)
        {if(strcmp(s[1], "true") == 0)
            {
                use_random = true;
            } 
            else if(strcmp(s[1], "false") == 0){
                use_random = false;
            } 
            else
            {
                printf(space);
                printf("--use_random was not set correctly!\n");
                printf("please try one of the following:\n");
                printf("\ttrue\n\tfalse");
                printf(space);
                continue;
            }
            printf("--use_random is set to: %s\n", s[1]);
        }
        else if(strcmp(s[0], "--rank_number") == 0)
        {
            int a = atoi(s[1]);
            if (a == 0 && s[1] != "0" || a < 0)
            {
                printf(space);
                printf("--rank_number was not set correctly!\n");
                printf("please only use positiv integers!");
                printf(space);
                continue;
            }
            rank_number = a;
            printf("--rank_number is set to: %s\n", s[1]);
        }
        else if(strcmp(s[0], "--use_ram") == 0)
        {
            int a = atoi(s[1]);
            if (a == 0 && s[1] != "0" || a < 0)
            {
                printf(space);
                printf("--use_ram was not set correctly!\n");
                printf("please only use positiv integers!");
                printf(space);
                continue;
            }
            use_ram = a;
            printf("--use_ram is set to: %s\n", s[1]);
        }
        else if(strcmp(argv[i], "--help") == 0)
        {
            printf("Possible arguments are:\n");
            printf("\t--type\n");
            printf("\t--window_comp\t in seconds\n");
            printf("\t--window_pause\t in seconds\n");
            printf("\t--window_size_min\t in seconds\n");
            printf("\t--window_size_max\t in seconds\n");
            printf("\t--use_random\n");
            printf("\t--use_multiple_cores\n");
            printf("\t--rank_number\n");
            printf("\t--use_ram\t  in MB\n");
            return 0;
        }
    }
    

    if (use_random)
    {
        seed = (rank_number+1)*42;
        srand(seed);
        int random = rand();
        printf("Seed = %d, random = %d\n", seed, random);

        window_comp = random % (window_size_max-window_size_min) + window_size_min;
        printf("New window comp set to: %d\n", window_comp);
        
    }


    if (window_comp <= 0)
    {
        printf("window_comp is negative or zero, program will terminate\n");
        return 0;
    }
    int num_threads = use_multiple_cores;

#pragma omp parallel num_threads(use_multiple_cores)
{
    printf("started thread: %d\n", omp_get_thread_num());
    switch (d_mode)
    {
    case compute:
        compute_kernel();
        break;
    case memory:
        memory_kernel();
        break;
    case communication:
        communication_kernel();
        break;
    default:
        break;
    }
}

    //MPI_Finalize();
}

char **split(char *str, char div, int num_args)
{
    char **result = (char**)malloc(sizeof(char)*num_args);

    int i;
    for (i = 0; i < num_args; i++)
    {
        result[i] = (char*)malloc(strlen(str)*sizeof(char));
    }

    int j = 0;
    int k = 0;

    for(i = 0; i < strlen(str); i++)
    {
        if (str[i] == div)
        {
            j = j + 1;
            k = 0;
            if (j >= num_args) {
               return result;
            }
        }
        else
        {
            result[j][k] = str[i];
            k = k + 1;
        }
    }

    return result;
}