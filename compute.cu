#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <stdio.h>

// i think CUDA will handle the distribution of kernels after the calls, its just about figuring out the dimensions for the blocks and declaring that
// Q: do i have one grid???


	/*Kernel 2 computes arrays*/
__global__ void computeAccels(vector3* d_accelvalues, vector3* d_Pos, double* d_mass){
	//first compute the pairwise accelerations.  Effect is on the first argument.
	// we're just going to compute the index using our massive 1d array (our matrix, with the computed indices)

	// so in a grid, the whichBlock*Threadsperblock jumps you to the right block, 
	// then you add the thread index to get to the right thread in that block. and this is a 2d grid so u do that for the y's too
	// then need to map it onto the 2d matrix of accels (really a 1d array currently)
	int i = blockIdx.x * blockDim.x + threadIdx.x; 
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	// z dimension?
	// int z=threadIdx.z;
	
	// NOTE. CURRENTLY D_ACCELVALUES IS A 1D ARRAY SO YOU CANT DOUBLE [][] INDEX LIKE ITS A 2D ARRAY. SO ALL CHECKS NEED TO BE ACCORDING
	// Q: the 1d array is why i changed the i check to num SQUARED, bc thats the full range?
	// -- okay so basically, we're not checking i against the whole array, we're seeing if i makes sense, so i may be under num*2 but be out of bounds for a row?
	if (i < NUMENTITIES && j < NUMENTITIES) { // this means that a specific thread in the allocated grids actually correspond w an entity, bc we may have extra blocks
		if (i==j) {
			FILL_VECTOR(d_accelvalues[i*NUMENTITIES+j],0,0,0); // cant use fill vector if u have k
		}
		else{
			vector3 distance;
			for (int k=0;k<3;k++) distance[k]=d_Pos[i][k]-d_Pos[j][k]; // you need to synthreads here
			double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
			double magnitude=sqrt(magnitude_sq);
			double accelmag=-1*GRAV_CONSTANT*d_mass[j]/magnitude_sq;
			FILL_VECTOR(d_accelvalues[i*NUMENTITIES+j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);// this uses z index
		}
			
		
	}

}

/*Kernel 3 does sums, and also updates values if reduction is not happening */

__global__ void sumAccelsAndUpdate(vector3* d_accelvalues, vector3* d_Pos, vector3* d_Vel){ 
//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
/*
this is NOT a thread doing the smallest component yet 
launch a thread per column and just loop thru each column
OR to optimize launch block

local thing in kernel called sum, calculate sum with single thread, and put that value in the very beginning of accels. OR a temp array 
your final sum will still be a vector3, it will be the aggregate x's,y's, z's of all the objects
if you do __shared__ then all the threads in the block will be able to access the sum and you can do reduction
if you do separate out the sumaccels and update functions then yes u need
--to have an array of vector3* d_sums where you keep track of the sums. OR, you could just overwrite them in the first column of accels

for this one, you only need j to increment, i is static, and youre walking down the column
*/
// does this actually need a start and end
// with our current 1D grid and block settings for this kernel, i could technically just = blockIdx.x
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < NUMENTITIES) {
	vector3 accel_sum={0,0,0};
	for (int j=0;j<NUMENTITIES;j++){
		for (int k=0;k<3;k++)
			accel_sum[k]+=d_accelvalues[i*NUMENTITIES+j][k];
	}

	// Store the result in the accels array??

	// for now, include update here
	//compute the new velocity based on the acceleration and time interval
	//compute the new position based on the velocity and time interval
	for (int k=0;k<3;k++){
		d_Vel[i][k]+=accel_sum[k]*INTERVAL;
		d_Pos[i][k]+=d_Vel[i][k]*INTERVAL;
	}
	
}	

}

/*Kernel 4 uses sums to do update

__global__ void update(vector3** accels, vector3* d_Pos, vector3* d_Vel){
//compute the new velocity based on the acceleration and time interval
//compute the new position based on the velocity and time interval
//Q: is there a built in way to do reductions
	for (i=0;i<NUMENTITIES;i++){

		for (k=0;k<3;k++){
			d_Vel[i][k]+=accels[k]*INTERVAL;
			d_Pos[i][k]+=d_Vel[i][k]*INTERVAL;
		}
	}
}

*/

//compute: Kernel invocations on the GPU
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){

	// define grid with x blocks
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); // does this need a z? no? bc the threads are the ones that need the z to compute each in the vector?
	dim3 numBlocks((NUMENTITIES+BLOCK_SIZE-1)/BLOCK_SIZE, (NUMENTITIES+BLOCK_SIZE-1)/BLOCK_SIZE);
	
	// grid for second kernel
	dim3 dimBlock2(1,1); // one thread per block
	dim3 numBlocks2(NUMENTITIES,1); // numentities blocks, 1D

	computeAccels<<<numBlocks, dimBlock>>>(d_accelvalues, d_Pos, d_mass);
	sumAccelsAndUpdate<<<numBlocks2, dimBlock2>>>(d_accelvalues, d_Pos, d_Vel);
	// update<<numBlocks, dimBlock<<(d_accelvalues,d_Pos, d_Vel);

	// Q: Do i need cudaDeviceSynchronize() anywhere?
	// need to fundamentally understand the relationship btwn compute.cu and nbody.cu files. what is being accessed


}
