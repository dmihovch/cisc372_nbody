
#include <math.h>
#include "vector.h"
#include "config.h"
#include "compute.h"

#include <cuda.h>

__global__ void computePairs(vector3* gAccels, vector3* gPos, double* gMass){

    long pair = blockIdx.x * blockDim.x + threadIdx.x;
    if(pair >= PAIRS) return;
    int i = pair / NUMENTITIES;
    int j = pair % NUMENTITIES;

    if(i == j){
        FILL_VECTOR(gAccels[pair], 0, 0, 0);
        return;
    }
    int k;
    vector3 distance;
    for (k=0;k<3;k++) distance[k]=gPos[i][k]-gPos[j][k];
    double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
    double magnitude=sqrt(magnitude_sq);
    double accelmag=-1*GRAV_CONSTANT*gMass[j]/magnitude_sq;
    FILL_VECTOR(gAccels[pair],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);


}

__global__ void accelAdd(vector3* gAccelsSummed, vector3* gAccels){
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= NUMENTITIES) return;

    vector3 accelSum = {0.,0.,0.};
    long rowIdx = i*NUMENTITIES;
    int j,k;
    for(j = 0; j<NUMENTITIES;j++){
        long idx = rowIdx + j;
        for(k = 0; k<3; k++){
            accelSum[k] += gAccels[idx][k];
        }
    }
    FILL_VECTOR(gAccelsSummed[i], accelSum[0], accelSum[1], accelSum[2]);
}

__global__ void updateBodies(vector3* gPos, vector3* gVel, vector3* gAccelsSummed){

    long i  = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUMENTITIES) return;

    int k;
    for (k=0; k<3; k++){
        gVel[i][k] += gAccelsSummed[i][k] * INTERVAL;
        gPos[i][k] += gVel[i][k] * INTERVAL;
    }
}



//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){
	computePairs<<<N2_BLOCKS,THREADS_PER_BLOCK>>>(gAccels,gPos,gMass);

	accelAdd<<<N_BLOCKS,THREADS_PER_BLOCK>>>(gAccelsSummed,gAccels);

	updateBodies<<<N_BLOCKS,THREADS_PER_BLOCK>>>(gPos, gVel, gAccelsSummed);
}
