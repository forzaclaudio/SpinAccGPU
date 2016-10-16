/*
 * David Claudio Gonzalez
 * University of Guanajuato, 2012
 * d.claudiogonzalez@gmail.com
 * File: parameters.h 
 * Description: Contains all the definitions required 
 * to compute the spin accumulation given a magnetization
 * configuration
 */
// I/O parameters

//If readspinaccumulation == 1, the spin accumulation is read from spinfilein
//If readspinaccumulation == 0, the spin accumulation is computed
#define readspinaccumulation 0

//Name of file where magnetization is read from 
#define filein "ATW-magn-2.5nm.dat"	

//Name of file where spinaccumulation is going to be saved
#define fileout "-spin.dat"	

//Name of file where effective values (i.e. effective u, effective ubeta and 
//effectuve beta) are going to be saved
#define effvalsfileout "-eff.dat"

//Name of file containing spin accumulation data for
//calculation of effective values
#define spinfilein "ATW-magn-2.5nm-spin.dat"

//Name of log file
#define logout "-log.txt"

// Mesh information
//Number of cells along direction x
#define NX	480		
//Number of cells along direction y
#define NY	120
//Number of cells along direction z
#define NZ	1

//Size of calculation box
#define TX	1200.0
#define TY	300.0
#define TZ	5.0

//Diffusion parameters
#define u_const	1  		//nm/ns
#define D  	1.0e3		//nm^2/ns
#define tau_sd_const 	1.0e-3		//ns
#define tau_sf_const 	25.0e-3		//ns
#define unitsfactor 1e-3	//needed to scale integer arguments to real value


//Runge Kutta 4th order integration parameters
#define dt		25.0e-6	//Time step in nanoseconds
#define tmax		1.0  //Integration time in nanoseconds
#define max_diff	1.0e-9  //Convergence criterium if betta_diff diference is less that this value simulation stops
#define Nsave		1000	//Iterations between save points
#define saveflag	1	//Indicates whether save is enabled (saveflag=1) or not(saveflag=0)
#define Nwrite		1000	//Iteration between writing points (to screen and log file)
#define writeflag	1	//Indicates whether save is enabled (saveflag=1) or not(saveflag=0)

//Threads and array sizes parameteres
#define XTHREADS_PERBLOCK	32
#define	YTHREADS_PERBLOCK	32

//Define miscellaneous code for cuda debuggin
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

