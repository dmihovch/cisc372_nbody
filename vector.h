#ifndef __TYPES_H__
#define __TYPES_H__

#include <stdlib.h>
#include <cuda_runtime.h>
typedef double vector3[3];
#define FILL_VECTOR(vector,a,b,c) {vector[0]=a;vector[1]=b;vector[2]=c;}
extern vector3 *hVel, *gVel;
extern vector3 *hPos, *gPos;
extern vector3 *accels, *gAccels, *gAccelsSummed;
extern double *mass, *gMass;
extern size_t allocSizeVecXNUMENT,allocSizeMass;
#endif
