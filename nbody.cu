#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include "planets.h"
#include "compute.h"

// represents the objects in the system.  Global variables
vector3 *hVel, *d_Vel; // the d_ values are all for the device (GPU). but while youre doing mallocmanage maybe u only need one. declare this here or in compute?
vector3 *hPos, *d_Pos;
double *mass, *d_mass; 
vector3* d_accelvalues; 


//initHostMemory: Create storage for numObjects entities in our system
//Parameters: numObjects: number of objects to allocate
//Returns: None
//Side Effects: Allocates memory in the hVel, hPos, and mass global variables
void initHostMemory(int numObjects)
{
	hVel = (vector3 *)malloc(sizeof(vector3) * numObjects);
	hPos = (vector3 *)malloc(sizeof(vector3) * numObjects); 
	mass = (double *)malloc(sizeof(double) * numObjects);
}

// KEEP THIS HERE
// you could just have the values array thats size numentities squared, and ur computing values[i*numentities+j] to index each, you dont need to store the pointers
// YOU NEED TO DECLARE A LOCAL ACCELS ARRAY TO LOOP THRU AND MALLOC FOR EACH temp_accels[i] -- okay lets just have one array

void initDeviceMemory(int numObjects){

	//declare the arrays here, but these will be allocated on the device
	// Q: declare accelvalues here?
	

	// if we want to use two arrays, i.e. store pointers in accels
	/* vector3** d_accels; */
	/* cudaMalloc(&d_accels,sizeof(vector3)*numObjects); */
	// you have to cudamalloc for the arrays, and then initialize a temp array and malloc in it and copy it up i think
	// its just that cudamalloc expects a cpu variable so you cant iterate thru ?
	// ?? sets up the pointers
	/* 
	int i,j;
	vector3* temp_accels;
	for (i=0;i<NUMENTITIES;i++)
		cudaMalloc(&temp_accels[i],sizeof(vector3)?);
	*/
	
	cudaMalloc(&d_accelvalues,sizeof(vector3)*numObjects*numObjects); // cudamalloc returns a pointer to the CPU
	cudaMalloc(&d_Pos,sizeof(vector3) * numObjects);
	cudaMalloc(&d_Vel,sizeof(vector3) * numObjects);
	// cudaMalloc(&d_mass,sizeof(double) * numObjects);

	// never need to memcpy accels

	#ifdef DEBUG
	cudaError_t cudaError = cudaMalloc(&d_mass, sizeof(double) * numObjects);
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaError));
		fflush(stdout); //?
		// Handle the error or exit the program
		exit(EXIT_FAILURE);
	}
	// else printf("Mem allocated for d_mass!!\n");
	
	#endif

}

void loadDeviceMemory(){
	// assuming the solar system is populated
	cudaMemcpy (d_Pos, hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice); 
	cudaMemcpy (d_Vel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);

	#ifdef DEBUG
	cudaError_t cudaError = cudaMemcpy (d_mass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaError));
		// Handle the error or exit the program
		exit(EXIT_FAILURE);
	}
	#endif
	
}

// OTHER FUNCTIONS: DEFINE A CALLXKERNEL FN FOR EACH KERNEL, WHERE YOU SPECIFY BLOCK SIZE AND NEEDED VARIABLES??
void freeDeviceMemory(){
	// if this was using 2 arrays, we would need a loop to iterate thru the accels array and then free
	// "and even when you free it you have to bring it down to CPU in another temp accels and loop thru and free each part and then free the accels"
	cudaFree(d_accelvalues);
	cudaFree(d_Pos);
	cudaFree(d_Vel);
	cudaFree(d_mass);
}
//freeHostMemory: Free storage allocated by a previous call to initHostMemory
//Parameters: None
//Returns: None
//Side Effects: Frees the memory allocated to global variables hVel, hPos, and mass.
void freeHostMemory()
{
	free(hVel);
	free(hPos);
	free(mass);
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
			hVel[i][j]=data[i][j+3];// why is this j+3?? bc the beginning of the vel is after the 3 dimensions of the pos in a planet in data
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
	int i, j;
	for (i = start; i < start + count; i++)
	{
		for (j = 0; j < 3; j++)
		{
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
	printf("NUMENTITIES IS %d\n", NUMENTITIES);
	clock_t t0=clock();
	int t_now;
	//srand(time(NULL));
	srand(1234);
	initHostMemory(NUMENTITIES);
	planetFill();
	randomFill(NUMPLANETS + 1, NUMASTEROIDS);
	//now we have a system.
	#ifdef DEBUG
	printSystem(stdout);
	printf("This is our solar system.\n");
	#endif
	initDeviceMemory(NUMENTITIES);
	// Now we want to load what we need from our system into the device
	loadDeviceMemory();
	for (t_now=0;t_now<DURATION;t_now+=INTERVAL){	
		compute(); // we want all the memory transfers to happen outside of this loop
	}
	// copy relevant data to host. CPU is making this call and the CUDA API is handling the GPU actions
	cudaMemcpy ( hPos, d_Pos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy ( hVel, d_Vel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
	clock_t t1=clock()-t0;
#ifdef DEBUG
	printSystem(stdout);
#endif
	printf("This took a total time of %f seconds\n",(double)t1/CLOCKS_PER_SEC);
	// free device memory here
	freeDeviceMemory();
	freeHostMemory();
}
