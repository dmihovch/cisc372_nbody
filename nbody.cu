#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <driver_types.h>
#include <cuda.h>
#include <math.h>

#include "config.h"
#include "planets.h"
#include "vector.h"



#define PAIRS (NUMENTITIES*NUMENTITIES)
#define THREADS_PER_BLOCK 256
#define N_BLOCKS (((NUMENTITIES + THREADS_PER_BLOCK)-1)/THREADS_PER_BLOCK)


__global__ void computeBodyAccels(vector3* gPos, double* gMass, vector3* gAccelsSummed){
	long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
	vector3 pos;
	vector3 accel = {0.,0.,0.};
	int iLessNumEnt = 0;
	if(i < NUMENTITIES) {FILL_VECTOR(pos, gPos[i][0],gPos[i][1], gPos[i][2]); iLessNumEnt = 1;}

	__shared__ vector3 sharedPos[THREADS_PER_BLOCK];
	__shared__ double sharedMass[THREADS_PER_BLOCK];


	for(int t = 0; t < gridDim.x; t++){

		int threadIdxTemp = threadIdx.x;
		int idx = t * blockDim.x + threadIdx.x;

		if(idx < NUMENTITIES){
			FILL_VECTOR(sharedPos[threadIdxTemp], gPos[idx][0], gPos[idx][1], gPos[idx][2]);
			sharedMass[threadIdxTemp] = gMass[idx];
		} else {
			FILL_VECTOR(sharedPos[threadIdxTemp], 0., 0., 0.);
			sharedMass[threadIdxTemp] = 0.;
		}

		__syncthreads();

		if(iLessNumEnt){
			for(int j = 0; j< blockDim.x; j++){
				double distanceX = sharedPos[j][0] - pos[0];
				double distanceY = sharedPos[j][1] - pos[1];
				double distanceZ = sharedPos[j][2] - pos[2];



				double distanceSquared = distanceX*distanceX + distanceY*distanceY + distanceZ*distanceZ;

				if(distanceSquared > 0){
					double inverseDistance = 1.0 / sqrt(distanceSquared);
					double inverseDistanceCubed = inverseDistance*inverseDistance*inverseDistance;

					double force = GRAV_CONSTANT*sharedMass[j]*inverseDistanceCubed;

					accel[0] += force * distanceX;
					accel[1] += force * distanceY;
					accel[2] += force * distanceZ;

				}


			}
		}

		__syncthreads();
	}
	if(iLessNumEnt) FILL_VECTOR(gAccelsSummed[i], accel[0], accel[1], accel[2]);
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


size_t allocSizeVecXNUMENT,allocSizeMass;


//initHostMemory: Create storage for numObjects entities in our system
//Parameters: numObjects: number of objects to allocate
//Returns: None
//Side Effects: Allocates memory in the hVel, hPos, and mass global variables
int initHostMemory(vector3** hPos,vector3** hVel, double** mass)
{



    allocSizeVecXNUMENT = sizeof(vector3) * NUMENTITIES;
    allocSizeMass = sizeof(double) * NUMENTITIES;

	*hVel = (vector3 *)malloc(allocSizeVecXNUMENT);
	if(*hVel == NULL){
	    return 1;
	}
	*hPos = (vector3 *)malloc(allocSizeVecXNUMENT);
	if(*hPos == NULL){
	    return 1;
	}
	*mass = (double *)malloc(allocSizeMass);
	if(*mass == NULL){
		return 1;
	}

	return 0;

}


int initGpuMemory(vector3** gPos, vector3** gVel, double** gMass, vector3** gAccelsSummed){

	cudaError_t err;
	err = cudaMalloc(gPos, allocSizeVecXNUMENT);
	if(err != cudaSuccess){
		return 1;
	}
	err = cudaMalloc(gMass, allocSizeMass);
	if(err != cudaSuccess){
	    //crash
		return 1;
	}

	err = cudaMalloc(gVel, allocSizeVecXNUMENT);
	if(err != cudaSuccess){
	    //crash
		return 1;
	}
	err = cudaMalloc(gAccelsSummed, allocSizeVecXNUMENT);
	if(err != cudaSuccess){
	    return 1;
	}
	return 0;
}

int initMemory(vector3** hPos,vector3** hVel, double** mass, vector3** gPos, vector3** gVel, double** gMass, vector3** gAccelsSummed){
	if(initHostMemory(hPos, hVel, mass) != 0){
		return 1;
	}
	if(initGpuMemory(gPos, gVel, gMass,gAccelsSummed)!= 0){
		return 2;
	}
	return 0;
}


void freeHostMemory(vector3* hPos, vector3* hVel, double* mass)
{
	if(hVel)free(hVel);
	if(hPos)free(hPos);
	if(mass)free(mass);
}


void freeGpuMemory(vector3* gPos, vector3* gVel, double* gMass,vector3* gAccelsSummed){
	if(gPos) cudaFree(gPos);
	if(gMass) cudaFree(gMass);
	if(gVel) cudaFree(gVel);
	if(gAccelsSummed) cudaFree(gAccelsSummed);
}

void cleanupMem(vector3* hPos, vector3* hVel, double* mass, vector3* gPos, vector3* gVel, double* gMass, vector3* gAccelsSummed){
	freeHostMemory(hPos,hVel,mass);
	freeGpuMemory(gPos,gVel,gMass,gAccelsSummed);
}
//planetFill: Fill the first NUMPLANETS+1 entries of the entity arrays with an estimation
//				of our solar system (Sun+NUMPLANETS)
//Parameters: None
//Returns: None
//Fills the first 8 entries of our system with an estimation of the sun plus our 8 planets.
void planetFill(vector3* hPos, vector3* hVel, double* mass){
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
void randomFill(int start, int count, vector3* hPos, vector3* hVel, double* mass)
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
void printSystem(FILE* handle, vector3* hPos, vector3* hVel, double* mass){
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

	vector3 *hVel = NULL, *gVel = NULL;
	vector3 *hPos = NULL, *gPos = NULL;
	vector3 *gAccelsSummed = NULL;
	double *mass = NULL, *gMass = NULL;



	clock_t t0=clock();
	int t_now;
	//srand(time(NULL));
	srand(1234);

	int memerr = initMemory(&hPos,&hVel,&mass,&gPos,&gVel,&gMass,&gAccelsSummed);
	if(memerr){
		cleanupMem(hPos,hVel,mass,gPos,gVel,gMass,gAccelsSummed);
		printf("memory allocation failure\n");
		return 1;
	}
	planetFill(hPos,hVel,mass);
	randomFill(NUMPLANETS + 1, NUMASTEROIDS, hPos,hVel,mass);
	//now we have a system.
	#ifdef DEBUG
	printSystem(stdout,hPos,hVel,mass);
	#endif


	cudaError_t err;

	err = cudaMemcpy(gPos, hPos,sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
	if(err != cudaSuccess){
		cleanupMem(hPos,hVel,mass,gPos,gVel,gMass,gAccelsSummed);
		return 1;
	}
	err = cudaMemcpy(gMass,mass,sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);
	if(err != cudaSuccess){
  		cleanupMem(hPos,hVel,mass,gPos,gVel,gMass,gAccelsSummed);
    	return 1;
	}

	err = cudaMemcpy(gVel,hVel,sizeof(vector3)*NUMENTITIES,cudaMemcpyHostToDevice);
	if(err != cudaSuccess){
	    cleanupMem(hPos,hVel,mass,gPos,gVel,gMass,gAccelsSummed);
		return 1;
	}


	for (t_now=0;t_now<DURATION;t_now+=INTERVAL){


		computeBodyAccels<<<N_BLOCKS,THREADS_PER_BLOCK>>>(gPos, gMass,gAccelsSummed);
		if(cudaGetLastError() != cudaSuccess){
			cleanupMem(hPos,hVel,mass,gPos,gVel,gMass,gAccelsSummed);
			return 1;
		}

		updateBodies<<<N_BLOCKS,THREADS_PER_BLOCK>>>(gPos, gVel, gAccelsSummed);
		if(cudaGetLastError()!= cudaSuccess){
			cleanupMem(hPos,hVel,mass,gPos,gVel,gMass,gAccelsSummed);
		    return 1;
		}

	}
	cudaMemcpy(hPos, gPos, sizeof(vector3)*NUMENTITIES,cudaMemcpyDeviceToHost);
	if(err != cudaSuccess){
		cleanupMem(hPos,hVel,mass,gPos,gVel,gMass,gAccelsSummed);
		return 1;
	}
	cudaMemcpy(hVel,gVel,sizeof(vector3)*NUMENTITIES,cudaMemcpyDeviceToHost);
	if(err != cudaSuccess){
  		cleanupMem(hPos,hVel,mass,gPos,gVel,gMass,gAccelsSummed);
		return 1;
	}

	cudaDeviceSynchronize();

	clock_t t1=clock()-t0;
#ifdef DEBUG
	printSystem(stdout,hPos,hVel,mass);
#endif
	printf("This took a total time of %f seconds\n",(double)t1/CLOCKS_PER_SEC);

	cleanupMem(hPos,hVel,mass,gPos,gVel,gMass,gAccelsSummed);

	return 0;


}
