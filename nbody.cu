#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <driver_types.h>
#include <math.h>
#include <cuda.h>

#include "config.h"
#include "planets.h"
#include "vector.h"



#define PAIRS (NUMENTITIES*NUMENTITIES)
#define THREADS_PER_BLOCK 256
#define N2_BLOCKS (((PAIRS + THREADS_PER_BLOCK)-1)/THREADS_PER_BLOCK)
#define N_BLOCKS (((NUMENTITIES + THREADS_PER_BLOCK)-1)/THREADS_PER_BLOCK)

extern vector3 *hVel, *gVel;
extern vector3 *hPos, *gPos;
extern vector3 *accels, *gAccels, *gAccelsSummed;
extern double *mass, *gMass;
extern size_t allocSizeVecXNUMENT,allocSizeMass;



__global__ void computePairs(vector3* gAccels, vector3* gPos, double* gMass){

    long pair = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if(pair >= PAIRS) return;

    //column
    int i = pair % NUMENTITIES;
    int j = pair / NUMENTITIES;

    if(i == j){
        FILL_VECTOR(gAccels[pair], 0, 0, 0);
        return;
    }
    int k;
    vector3 distance;
    for (k=0;k<3;k++) distance[k]=gPos[i][k]-gPos[j][k];
    double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
    double magnitude=sqrt(magnitude_sq);
    if(magnitude_sq == 0) magnitude_sq = 1;
    double accelmag=-1*GRAV_CONSTANT*gMass[j]/magnitude_sq;
    FILL_VECTOR(gAccels[pair],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);


}

__global__ void accelAdd(vector3* gAccelsSummed, vector3* gAccels){
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= NUMENTITIES) return;

    vector3 accelSum = {0.,0.,0.};
    int j,k;
    for(j = 0; j<NUMENTITIES;j++){
        long idx = (long)j * NUMENTITIES + i;
        for(k = 0; k<3; k++){
            accelSum[k] += gAccels[idx][k];
        }
    }
    FILL_VECTOR(gAccelsSummed[i], accelSum[0], accelSum[1], accelSum[2]);
}

__global__ void updateBodies(vector3* gPos, vector3* gVel, vector3* gAccelsSummed){

    long i  = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUMENTITIES) return;

    int k;
    for (k=0; k<3; k++){
        gVel[i][k] += gAccelsSummed[i][k] * INTERVAL;
        gPos[i][k] += gVel[i][k] * INTERVAL;
    }
}

// represents the objects in the system.  Global variables

vector3 *hVel, *gVel;
vector3 *hPos, *gPos;
vector3 *accels, *gAccels, *gAccelsSummed;
double *mass, *gMass;

size_t allocSizeVecXNUMENT,allocSizeMass;


//initHostMemory: Create storage for numObjects entities in our system
//Parameters: numObjects: number of objects to allocate
//Returns: None
//Side Effects: Allocates memory in the hVel, hPos, and mass global variables
int initHostMemory(int numObjects)
{


    allocSizeVecXNUMENT = sizeof(vector3) * NUMENTITIES;
    allocSizeMass = sizeof(double) * NUMENTITIES;

	hVel = (vector3 *)malloc(allocSizeVecXNUMENT);
	if(hVel == NULL){
	    return 1;
	}
	hPos = (vector3 *)malloc(allocSizeVecXNUMENT);
	if(hPos == NULL){
	    return 1;
	}
	mass = (double *)malloc(allocSizeMass);
	if(mass == NULL){
		return 1;
	}
	accels = (vector3 *)malloc(allocSizeVecXNUMENT * NUMENTITIES);
	if(accels == NULL){
		return 1;
	}

	return 0;

}


int initGpuMemory(){

	cudaError_t err;
	err = cudaMalloc(&gPos, allocSizeVecXNUMENT);
	if(err != cudaSuccess){
		return 1;
	}
	err = cudaMalloc(&gMass, allocSizeMass);
	if(err != cudaSuccess){
	    //crash
		return 1;
	}

	err = cudaMalloc(&gVel, allocSizeVecXNUMENT);
	if(err != cudaSuccess){
	    //crash
		return 1;
	}
	err = cudaMalloc(&gAccels, allocSizeVecXNUMENT*NUMENTITIES);
	if(err != cudaSuccess){
	    return 1;
	}

	err = cudaMalloc(&gAccelsSummed, allocSizeVecXNUMENT);
	if(err != cudaSuccess){
	    return 1;
	}
	return 0;
}



void freeHostMemory()
{
	if(hVel)free(hVel);
	if(hPos)free(hPos);
	if(mass)free(mass);
	if(accels)free(accels);
}


void freeGpuMemory(){
	if(gPos) cudaFree(gPos);
	if(gMass) cudaFree(gMass);
	if(gVel) cudaFree(gVel);
	if(gAccels) cudaFree(gAccels);
}

void cleanupMem(){
	freeHostMemory();
	freeGpuMemory();
}
//planetFill: Fill the first NUMPLANETS+1 entries of the entity arrays with an estimation
//				of our solar system (Sun+NUMPLANETS)
//Parameters: None
//Returns: None
//Fills the first 8 entries of our system with an estimation of the sun plus our 8 planets.
void planetFill(){
	int i,j;
	double data[][7]={SUN,MERCURY,VENUS,EARTH,MARS,JUPITER,SATURN,URANUS,NEPTUNE};
	for (i=0;i<=NUMPLANETS;i++){
		for (j=0;j<3;j++){
			hPos[i][j]=data[i][j];
			hVel[i][j]=data[i][j+3];
		}
		mass[i]=data[i][6];
	}
}

//randomFill: FIll the rest of the objects in the system randomly starting at some entry in the list
//Parameters: 	start: The index of the first open entry in our system (after planetFill).
//				count: The number of random objects to put into our system
//Returns: None
//Side Effects: Fills count entries in our system starting at index start (0 based)
void randomFill(int start, int count)
{
    //might paralellize this as well
	int i, j = start;
	for (i = start; i < start + count; i++)
	{
		for (j = 0; j < 3; j++)
		{
			//?????????
			hVel[i][j] = (double)rand() / RAND_MAX * MAX_DISTANCE * 2 - MAX_DISTANCE;
			hPos[i][j] = (double)rand() / RAND_MAX * MAX_VELOCITY * 2 - MAX_VELOCITY;
			mass[i] = (double)rand() / RAND_MAX * MAX_MASS;
		}
	}
}

//printSystem: Prints out the entire system to the supplied file
//Parameters: 	handle: A handle to an open file with write access to prnt the data to
//Returns: 		none
//Side Effects: Modifies the file handle by writing to it.
void printSystem(FILE* handle){
	int i,j;
	for (i=0;i<NUMENTITIES;i++){
		fprintf(handle,"pos=(");
		for (j=0;j<3;j++){
			fprintf(handle,"%lf,",hPos[i][j]);
		}
		printf("),v=(");
		for (j=0;j<3;j++){
			fprintf(handle,"%lf,",hVel[i][j]);
		}
		fprintf(handle,"),m=%lf\n",mass[i]);
	}
}

int main(int argc, char **argv)
{
	clock_t t0=clock();
	int t_now;
	//srand(time(NULL));
	srand(1234);

	if(initHostMemory(NUMENTITIES) != 0) cleanupMem();
	if(initGpuMemory()!=0) cleanupMem();
	planetFill();
	randomFill(NUMPLANETS + 1, NUMASTEROIDS);
	printf("We have a system\n");
	//now we have a system.
	#ifdef DEBUG
	printSystem(stdout);
	#endif


	cudaError_t err;

	err = cudaMemcpy(gPos, hPos,sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
	if(err != cudaSuccess){
		cleanupMem();
		return 1;
	}
	err = cudaMemcpy(gMass,mass,sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);
	if(err != cudaSuccess){
  		cleanupMem();
    	return 1;
	}
	err = cudaMemcpy(gAccels,accels,sizeof(vector3)*NUMENTITIES*NUMENTITIES, cudaMemcpyHostToDevice);
	if(err != cudaSuccess){
  		cleanupMem();
   		return 1;
	}

	err = cudaMemcpy(gVel,hVel,sizeof(vector3)*NUMENTITIES,cudaMemcpyHostToDevice);
	if(err != cudaSuccess){
	    cleanupMem();
		return 5;
	}


	for (t_now=0;t_now<DURATION;t_now+=INTERVAL){
	    computePairs<<<N2_BLOCKS,THREADS_PER_BLOCK>>>(gAccels,gPos,gMass);
		if(cudaGetLastError() != cudaSuccess){
			cleanupMem();
			return 1;
		}
		accelAdd<<<N_BLOCKS,THREADS_PER_BLOCK>>>(gAccelsSummed,gAccels);
		if(cudaGetLastError() != cudaSuccess){
			cleanupMem();
		    return 1;
		}
		updateBodies<<<N_BLOCKS,THREADS_PER_BLOCK>>>(gPos, gVel, gAccelsSummed);
		if(cudaGetLastError()!= cudaSuccess){
			cleanupMem();
		    return 1;
		}



	}
	cudaMemcpy(hPos, gPos, sizeof(vector3)*NUMENTITIES,cudaMemcpyDeviceToHost);
	if(err != cudaSuccess){
		cleanupMem();
		return 1;
	}
	cudaMemcpy(hVel,gVel,sizeof(vector3)*NUMENTITIES,cudaMemcpyDeviceToHost);
	if(err != cudaSuccess){
  		cleanupMem();
		return 1;
	}
	cudaMemcpy(accels,gAccels,sizeof(vector3)*NUMENTITIES*NUMENTITIES,cudaMemcpyDeviceToHost);
	if(err != cudaSuccess){
  		cleanupMem();
		return 1;
	}
	clock_t t1=clock()-t0;
#ifdef DEBUG
	printSystem(stdout);
#endif
	printf("This took a total time of %f seconds\n",(double)t1/CLOCKS_PER_SEC);

	cleanupMem();

	return 0;


}
