/*
 * david Claudio Gonzalez
 * University of Guanajuato, 2012
 * d.claudiogonzalez@gmail.com
 * File: gpuspinAccumulationEffBeta.c
 * Description: Contains the program that sets up
 * the arguments to compute the spin accumulation
 * on a GPU
 */

//General includes
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>
#include <time.h>
#include <math.h>

//Parameters include
#include <parameters.h>

//GPU kernels include
//#include "kernelspinAccumulationEffBeta.cu"

FILE *magnin, *spinin, *outlog;
char fileinstr[255],fileoutstr[255],effvalsfileoutstr[255],logoutstr[255];

//Number of cells along direction x
int NXCUDA;
//Number of cells along direction y
int NYCUDA;
//Number of cells along direction z
int NZCUDA;
//Global values to use for simulations
double u_val=u_const;
double tau_sd_val=tau_sd_const;
double tau_sf_val=tau_sf_const;

//This should be modified in the future when new
//GPUs allow more than 1024 threads per block
//and shared memory is increased
int XBLOCKS_PERGRID, YBLOCKS_PERGRID;

//Declare pointers to arrays to store magnetization and spinaccumulation
//on CPU
double **mx,**my,**mz;
double **u_eff, **u_eff_beta_eff, **beta_eff;	//Arrays of effective values
double **deltam_x,**deltam_y,**deltam_z;
double **XCOORD,**YCOORD;
double *mem_buffer;
double *beta_diff_num;
double *beta_diff_den;

//Declare pointers to arrays to store magnetization and spinaccumulation
//on GPU as well as auxiliary arrays
double *dev_mx, *dev_my, *dev_mz;
double *dev_deltam_x, *dev_deltam_y, *dev_deltam_z;
double *dev_sm_x, *dev_sm_y, *dev_sm_z;          	//Source term
double *dev_sdex_x, *dev_sdex_y, *dev_sdex_z;    	//Exchange term
double *dev_sfrelax_x, *dev_sfrelax_y, *dev_sfrelax_z;	//Spin relaxation term
double *dev_lapl_x, *dev_lapl_y, *dev_lapl_z;		//Laplacian term
double *dev_m_x_sm_x, *dev_m_x_sm_y, *dev_m_x_sm_z;	//Array of cross product of m and sm/u
double *dev_u_eff, *dev_u_eff_beta_eff, *dev_beta_eff;	//Arrays of effective values
double *dev_tempx, *dev_tempy, *dev_tempz, *dev_beta_diff_num, *dev_beta_diff_den;	//Arrays of temporary values
double *dev_k1x, *dev_k1y, *dev_k1z;			//Arrays for storing terms of runge kutta terms
double *dev_k2x, *dev_k2y, *dev_k2z;
double *dev_k3x, *dev_k3y, *dev_k3z;
double *dev_k4x, *dev_k4y, *dev_k4z;
double *dev_d2adx2, *dev_d2bdx2, *dev_d2gdx2;
double *dev_d2ady2, *dev_d2bdy2, *dev_d2gdy2;


//Declare global variables
int NXNY;
double beta_diff=0.0;
int size_bytes;

void parse_args(int argc, char *argv[]){
int next_option;
char temp[10];//String to create filenames
/* A string listing valid short options letters. */
const char* const short_options = "sfu";
/* An array describing valid long options. */
const struct option long_options[] =
        {
        { "tausd", 0, NULL, 's' },
        { "tausf", 0, NULL, 'f' },
        { "spindriftvel", 0, NULL, 'u' },
        { NULL, 0, NULL, 0}/*Required at end of array */
};

//Create name for spin acucmulation
strcpy(fileoutstr,filein);
fileoutstr[strlen(fileoutstr)-4]='\0';
//Create name for effective values
strcpy(effvalsfileoutstr,fileoutstr);
//Create name for log values
strcpy(logoutstr,fileoutstr);

do
        {
        next_option = getopt_long(argc, argv, short_options, long_options, NULL);
        switch(next_option)
                {
                case 's': /* -s or --tausd*/
			tau_sd_val = atof(argv[optind])*unitsfactor;
                        printf("Value of tau_sd is: %f\n",tau_sf_val);
			strcpy(temp,"tau_sd");
			//Append data to name for spin accumulation
			strcat(fileoutstr,temp);
			strcat(fileoutstr,argv[optind]);
			//Append data to name for effective values
			strcat(effvalsfileoutstr,temp);
			strcat(effvalsfileoutstr,argv[optind]);
			//append datto  name for log values
			strcat(logoutstr,temp);
			strcat(logoutstr,argv[optind]);
                        break;
                case 'f': /* -f or --tausf*/
			tau_sf_val = atof(argv[optind])*unitsfactor;
                        printf("Value of tau_sf is: %f\n",tau_sf_val);
			strcpy(temp,"tau_sf");
			//Append data to name for spin accumulation
			strcat(fileoutstr,temp);
			strcat(fileoutstr,argv[optind]);
			//Append data to name for effective values
			strcat(effvalsfileoutstr,temp);
			strcat(effvalsfileoutstr,argv[optind]);
			//append datto  name for log values
			strcat(logoutstr,temp);
			strcat(logoutstr,argv[optind]);
                        break;
                case 'u': /* -u or --spindriftvel*/
			u_val = atof(argv[optind])*unitsfactor;
                        printf("Value of u is: %f\n",u_val);
			strcpy(temp,"u");
			//Append data to name for spin accumulation
			strcat(fileoutstr,temp);
			strcat(fileoutstr,argv[optind]);
			//Append data to name for effective values
			strcat(effvalsfileoutstr,temp);
			strcat(effvalsfileoutstr,argv[optind]);
			//append data to  name for log values
			strcat(logoutstr,temp);
			strcat(logoutstr,argv[optind]);
                        break;

                case '?':
                        printf("Unknown option has been found\n");
                        break;
                case -1:
                        break;
                default:
                        abort();
                }
        }
while(next_option != -1);
//Finish name for spin accumulation
strcat(fileoutstr,fileout);
//Finish name for effective values
strcat(effvalsfileoutstr,effvalsfileout);
//Finish name for log values
strcat(logoutstr,logout);
//printf("File names are: %s\n",fileoutstr);
//printf("File names are: %s\n",effvalsfileoutstr);
//printf("File names are: %s\n",logoutstr);
}


__device__ void SEL3(	double A11, double A12, double A13,
			double A21, double A22, double A23,
			double A31, double A32, double A33,
			double BFCT1, double BFCT2, double BFCT3,
			double *DFDL, double *D2FDL2, double *D3FDL3 )

{
//Computation of the first, second and third derivatives
//for points i = 0, NX+2, j = 0, NY-1
double DET2A,DET2B,DET2C;
double DET2D,DET2E,DET2F;
double DET2G,DET2H,DET2I;
double YDENOM,YNUM1,YNUM2,YNUM3;
//Determinants of 2nd order
DET2A = A22*A33 -A23*A32;
DET2B = A12*A33 -A32*A13;
DET2C = A12*A23 -A22*A13;
DET2D = A21*A33 -A31*A23;
DET2E = A11*A33 -A31*A13;
DET2F = A11*A23 -A13*A21;
DET2G = A21*A32 -A22*A31;
DET2H = A11*A32 -A12*A31;
DET2I = A11*A22 -A21*A12;

YDENOM= +A11*DET2A- A21*DET2B+ A31*DET2C;
YNUM1 = BFCT1*DET2A-BFCT2*DET2B+BFCT3*DET2C;
YNUM2 =-BFCT1*DET2D+BFCT2*DET2E-BFCT3*DET2F;
YNUM3 = BFCT1*DET2G-BFCT2*DET2H+BFCT3*DET2I;

(*DFDL) = YNUM1/YDENOM;
(*D2FDL2) = YNUM2/YDENOM;
(*D3FDL3) = YNUM3/YDENOM;
}

__device__ void SEL4(	double A11, double A12, double A13, double A14,
			double A21, double A22, double A23, double A24,
			double A31, double A32, double A33, double A34,
			double A41, double A42, double A43, double A44,
			double BFCT1, double BFCT2, double BFCT3, double BFCT4,
			double *DFDL, double *D2FDL2)
{
//Computation of the first and seconde derivatives for
//points i = 1, NX +1, j = 2, NY - 2
double DET2A,DET2B,DET2C,DET2D,DET2E;
double DET2F,DET2G,DET2H,DET2I,DET2J;
double DET3A,DET3B,DET3C,DET3D;
double DET3E,DET3F,DET3G,DET3H;
double YDENOM,YNUM1,YNUM2;

//Determinants of 2nd order
DET2A = A33*A44 -A34*A43;
DET2B = A32*A44 -A34*A42;
DET2C = A32*A43 -A33*A42;
DET2D = A13*A24 -A14*A23;
DET2E = A12*A24 -A14*A22;
DET2F = A12*A23 -A13*A22;
DET2G = A31*A44 -A34*A41;
DET2H = A31*A43 -A33*A41;
DET2I = A11*A24 -A14*A21;
DET2J = A11*A23 -A13*A21;

//Determinants of 3rd order
DET3A = + A22*DET2A - A23*DET2B + A24*DET2C;
DET3B = + A12*DET2A - A13*DET2B + A14*DET2C;
DET3C = + A42*DET2D - A43*DET2E + A44*DET2F;
DET3D = + A32*DET2D - A33*DET2E + A34*DET2F;
DET3E = + A21*DET2A - A23*DET2G + A24*DET2H;
DET3F = + A11*DET2A - A13*DET2G + A14*DET2H;
DET3G = + A41*DET2D - A43*DET2I + A44*DET2J;
DET3H = + A31*DET2D - A33*DET2I + A34*DET2J;

YDENOM = A11* DET3A-A21 *DET3B+A31* DET3C-A41* DET3D;
YNUM1 = BFCT1*DET3A-BFCT2*DET3B+BFCT3*DET3C-BFCT4*DET3D;
YNUM2 = -BFCT1*DET3E+BFCT2*DET3F-BFCT3*DET3G+BFCT4*DET3H;

(*DFDL) = YNUM1/YDENOM;
(*D2FDL2) = YNUM2/YDENOM;
}



__global__ void gterm4_RK4(double *deltam_x, double *deltam_y, double *deltam_z,
				double *k1x, double *k1y, double *k1z,
				double *k2x, double *k2y, double *k2z,
				double *k3x, double *k3y, double *k3z,
				double *k4x, double *k4y, double *k4z,
				double *tempx, double *tempy, double *tempz,
			 	int grid_width)
//Computation of 4th term of RK4 and integrated value of Spin Accummulation using global memory
{
int i,j,index;
//The last increment of two is due to the shifting of
//two array elements in the x direction in all arrays
i = blockIdx.x * blockDim.x + threadIdx.x + 2;
j = blockIdx.y * blockDim.y + threadIdx.y;
// map the two 2D indices to a single linear, 1D index
index = j * grid_width + i;

if (i > 1 && i < NX+2 && j >= 0 && j < NY)
	{
	k4x[index] = dt * deltam_x[index];
	deltam_x[index] = tempx[index] + (k1x[index] +
				(double)2.0 * (k2x[index] + k3x[index])
				+ k4x[index]) / (double) 6.0;


	k4y[index] = dt * deltam_y[index];
	deltam_y[index] = tempy[index] + (k1y[index] +
				(double)2.0 * (k2y[index] + k3y[index])
				+ k4y[index]) / (double) 6.0;

	k4z[index] = dt * deltam_z[index];
	deltam_z[index] = tempz[index] + (k1z[index] +
				(double)2.0 * (k2z[index] + k3z[index])
				+ k4z[index]) / (double) 6.0;
	}
}

__global__ void gterm3_RK4(double *deltam_x, double *deltam_y, double *deltam_z,
				double *k3x, double *k3y, double *k3z,
				double *tempx, double *tempy, double *tempz,
			 	int grid_width)
//Computation of 3rd term of RK4 using global memory
{
int i,j,index;
//The last increment of two is due to the shifting of
//two array elements in the x direction in all arrays
i = blockIdx.x * blockDim.x + threadIdx.x + 2;
j = blockIdx.y * blockDim.y + threadIdx.y;
// map the two 2D indices to a single linear, 1D index
index = j * grid_width + i;

if (i > 1 && i < NX+2 && j >= 0 && j < NY)
	{
	k3x[index] = dt * deltam_x[index];
	deltam_x[index] = tempx[index] + (double)0.5 * k3x[index];

	k3y[index] = dt * deltam_y[index];
	deltam_y[index] = tempy[index] + (double)0.5 * k3y[index];

	k3z[index] = dt * deltam_z[index];
	deltam_z[index] = tempz[index] + (double)0.5 * k3z[index];
	}
}

__global__ void gterm2_RK4(double *deltam_x, double *deltam_y, double *deltam_z,
				double *k2x, double *k2y, double *k2z,
				double *tempx, double *tempy, double *tempz,
			 	int grid_width)
//Computation of 2nd term of RK4 using global memory
{
int i,j,index;
//The last increment of two is due to the shifting of
//two array elements in the x direction in all arrays
i = blockIdx.x * blockDim.x + threadIdx.x + 2;
j = blockIdx.y * blockDim.y + threadIdx.y;
// map the two 2D indices to a single linear, 1D index
index = j * grid_width + i;

if (i > 1 && i < NX+2 && j >= 0 && j < NY)
	{
	k2x[index] = dt * deltam_x[index];
	deltam_x[index] = tempx[index] + (double)0.5 * k2x[index];

	k2y[index] = dt * deltam_y[index];
	deltam_y[index] = tempy[index] + (double)0.5 * k2y[index];

	k2z[index] = dt * deltam_z[index];
	deltam_z[index] = tempz[index] + (double)0.5 * k2z[index];
	}
}

__global__ void gterm1_RK4(double *deltam_x, double *deltam_y, double *deltam_z,
				double *k1x, double *k1y, double *k1z,
				double *tempx, double *tempy, double *tempz,
			 	int grid_width)
//Computation of 1st term of RK4 using global memory
{
int i,j,index;
//The last increment of two is due to the shifting of
//two array elements in the x direction in all arrays
i = blockIdx.x * blockDim.x + threadIdx.x + 2;
j = blockIdx.y * blockDim.y + threadIdx.y;
// map the two 2D indices to a single linear, 1D index
index = j * grid_width + i;

//Evaluate dm/dt at t = n
if (i > 1 && i < NX+2 && j >= 0 && j < NY)
	{
	k1x[index] = dt * deltam_x[index];
	deltam_x[index] = tempx[index] + (double)0.5 * k1x[index];

	k1y[index] = dt * deltam_y[index];
	deltam_y[index] = tempy[index] + (double)0.5 * k1y[index];

	k1z[index] = dt * deltam_z[index];
	deltam_z[index] = tempz[index] + (double)0.5 * k1z[index];
	}
}

__global__ void gspinaccum_backup(double *deltam_x, double *deltam_y, double *deltam_z,
				double *tempx, double *tempy, double *tempz,
			 	int grid_width)

//Computation of sf_relaxation term using global memory
{
int i,j,index;
//The last increment of two is due to the shifting of
//two array elements in the x direction in all arrays
i = blockIdx.x * blockDim.x + threadIdx.x + 2;
j = blockIdx.y * blockDim.y + threadIdx.y;
// map the two 2D indices to a single linear, 1D index
index = j * grid_width + i;
//Backup current values of spin accumulation

if (i > 1 && i < NX+2 && j >= 0 && j < NY)
	{
	tempx[index] = deltam_x[index];
	tempy[index] = deltam_y[index];
	tempz[index] = deltam_z[index];
	}
}

__global__ void gsolution(double *sfrelax_x, double *sfrelax_y, double *sfrelax_z,
				double *sm_x, double *sm_y, double *sm_z,
				double *sdex_x, double *sdex_y, double *sdex_z,
				double *lapl_x, double *lapl_y, double *lapl_z,
				double *deltam_x, double *deltam_y, double *deltam_z,
			 	int grid_width)
//Evaluation of Zhang and Li model using global memory
{
int i,j,index;
//The last increment of two is due to the shifting of
//two array elements in the x direction in all arrays
i = blockIdx.x * blockDim.x + threadIdx.x + 2;
j = blockIdx.y * blockDim.y + threadIdx.y;
// map the two 2D indices to a single linear, 1D index
index = j * grid_width + i;

if (i > 1 && i < NX+2 && j >= 0 && j < NY)
	{
	//Sign of source term is changed due to double cross product
	//used to orthogonalize
	deltam_x[index] = sfrelax_x[index] + sdex_x[index] + lapl_x[index]
				- sm_x[index];
	deltam_y[index] = sfrelax_y[index] + sdex_y[index] + lapl_y[index]
				- sm_y[index];
	deltam_z[index] = sfrelax_z[index] + sdex_z[index] + lapl_z[index]
				- sm_z[index];
	}
}

__global__ void gsf_relaxation(double tau_sf,double *sfrelax_x, double *sfrelax_y, double *sfrelax_z,
				double *deltam_x, double *deltam_y, double *deltam_z,
			 	int grid_width)
//Computation of sf_relaxation term using global memory
{
int i,j,index;
//The last increment of two is due to the shifting of
//two array elements in the x direction in all arrays
i = blockIdx.x * blockDim.x + threadIdx.x + 2;
j = blockIdx.y * blockDim.y + threadIdx.y;
// map the two 2D indices to a single linear, 1D index
index = j * grid_width + i;

if (i > 1 && i < NX+2 && j >= 0 && j < NY)
	{
	sfrelax_x[index] = - deltam_x[index] / tau_sf;
	sfrelax_y[index] = - deltam_y[index] / tau_sf;
	sfrelax_z[index] = - deltam_z[index] / tau_sf;
	}
}

__global__ void glaplacianx(double *lapl_x, double *lapl_y, double *lapl_z,
				double *d2adx2, double *d2bdx2, double *d2gdx2,
				double *deltam_x, double *deltam_y, double *deltam_z,
			 	int grid_width)
//Computation of laplacian term using global memory
{
int i,j,index;
//Indexes of different neighbours
//leftneigh1 = i-1, rightneigh1 = i+1, frontneigh1 = j+1, backneigh1 = j-1
//leftneigh2 = i-2, rightneigh2 = i+2, frontneigh2 = j+2, backneigh2 = j-2
int leftneigh1,rightneigh1;
int leftneigh2,rightneigh2;
double DELTAX;

//The last increment of two is due to the shifting of
//two array elements in the x direction in all arrays
i = blockIdx.x * blockDim.x + threadIdx.x + 2;
j = blockIdx.y * blockDim.y + threadIdx.y;
// map the two 2D indices to a single linear, 1D index
index = j * grid_width + i;
leftneigh1 = j * grid_width + (i-1);
rightneigh1 = j * grid_width + (i+1);
leftneigh2 = j * grid_width + (i-2);
rightneigh2 = j * grid_width + (i+2);

DELTAX = (double)TX/(double)NX;
if (i > 1 && i < NX+2 && j >= 0 && j < NY)
	{
	d2adx2[index] = - deltam_x[rightneigh2] / (double)12.0 + (double)4.0 * deltam_x[rightneigh1] / (double)3.0
			- (double)5.0 * deltam_x[index] / (double)2.0
			- deltam_x[leftneigh2] / (double)12.0 + (double)4.0 * deltam_x[leftneigh1] / (double)3.0;
	d2adx2[index] = d2adx2[index] / (DELTAX * DELTAX);

	d2bdx2[index] = - deltam_y[rightneigh2] / (double)12.0 + (double)4.0 * deltam_y[rightneigh1] / (double)3.0
			- (double)5.0 * deltam_y[index] / (double)2.0
			- deltam_y[leftneigh2] / (double)12.0 + (double)4.0 * deltam_y[leftneigh1] / (double)3.0;
	d2bdx2[index] = d2bdx2[index] / (DELTAX * DELTAX);

	d2gdx2[index] = - deltam_z[rightneigh2] / (double)12.0 + (double)4.0 * deltam_z[rightneigh1] / (double)3.0
			- (double)5.0 * deltam_z[index] / (double)2.0
			- deltam_z[leftneigh2] / (double)12.0 + (double)4.0 * deltam_z[leftneigh1] / (double)3.0;
	d2gdx2[index] = d2gdx2[index] / (DELTAX * DELTAX);
	}

if (i > 1 && i < NX+2 && j >= 0 && j < NY)
	{
	lapl_x[index] = (double)D * d2adx2[index];
	lapl_y[index] = (double)D * d2bdx2[index];
	lapl_z[index] = (double)D * d2gdx2[index];
	}

}

__global__ void glaplaciany(double *lapl_x, double *lapl_y, double *lapl_z,
				double *d2ady2, double *d2bdy2, double *d2gdy2,
				double *deltam_x, double *deltam_y, double *deltam_z,
			 	int grid_width)
//Computation of laplacian term using global memory
{
int i,j,index;
//Indexes of different neighbours
//leftneigh1 = i-1, rightneigh1 = i+1, frontneigh1 = j+1, backneigh1 = j-1
//leftneigh2 = i-2, rightneigh2 = i+2, frontneigh2 = j+2, backneigh2 = j-2
int frontneigh1,backneigh1;
int frontneigh2,backneigh2;

double DELTAY;

//The last increment of two is due to the shifting of
//two array elements in the x direction in all arrays
i = blockIdx.x * blockDim.x + threadIdx.x + 2;
j = blockIdx.y * blockDim.y + threadIdx.y;
// map the two 2D indices to a single linear, 1D index
index = j * grid_width + i;

frontneigh1 = (j+1) * grid_width + i;
backneigh1 = (j-1) * grid_width + i;
frontneigh2 = (j+2) * grid_width + i;
backneigh2 = (j-2) * grid_width + i;

DELTAY = (double)TY/(double)NY;

if (i > 1 && i < NX+2 && j >= 2 && j < NY-2)
	{
	d2ady2[index] = - deltam_x[frontneigh2] / (double)12.0 + (double)4.0 * deltam_x[frontneigh1] / (double)3.0
			- (double)5.0 * deltam_x[index] / (double)2.0
			- deltam_x[backneigh2] / (double)12.0 + (double)4.0 * deltam_x[backneigh1] / (double)3.0;
	d2ady2[index] = d2ady2[index] / (DELTAY * DELTAY);

	d2bdy2[index] = - deltam_y[frontneigh2] / (double)12.0 + (double)4.0 * deltam_y[frontneigh1] / (double)3.0
			- (double)5.0 * deltam_y[index] / (double)2.0
			- deltam_y[backneigh2] / (double)12.0 + (double)4.0 * deltam_y[backneigh1] / (double)3.0;
	d2bdy2[index] = d2bdy2[index] / (DELTAY * DELTAY);

	d2gdy2[index] = - deltam_z[frontneigh2] / (double)12.0 + (double)4.0 * deltam_z[frontneigh1] / (double)3.0
			- (double)5.0 * deltam_z[index] / (double)2.0
			- deltam_z[backneigh2] / (double)12.0 + (double)4.0 * deltam_z[backneigh1] / (double)3.0;
	d2gdy2[index] = d2gdy2[index] / (DELTAY * DELTAY);

}
if (i > 1 && i < NX+2 && j >= 2 && j < NY-2)
	{
	//In order to havethe correct result glaplacianx must be called first
	lapl_x[index] += (double)D * d2ady2[index];
	lapl_y[index] += (double)D * d2bdy2[index];
	lapl_z[index] += (double)D * d2gdy2[index];
	}
}

__global__ void glaplacianyboundaries(double *lapl_x, double *lapl_y, double *lapl_z,
				double *d2ady2, double *d2bdy2, double *d2gdy2,
				double *deltam_x, double *deltam_y, double *deltam_z,
			 	int grid_width)
//Computation of laplacian term using global memory
{
int i,j,index;
//Indexes of different neighbours
//leftneigh1 = i-1, rightneigh1 = i+1, frontneigh1 = j+1, backneigh1 = j-1
//leftneigh2 = i-2, rightneigh2 = i+2, frontneigh2 = j+2, backneigh2 = j-2
int frontneigh1,backneigh1;
int frontneigh2,backneigh2;

double DELTAY;
double A11,A12,A13,A14;
double A21,A22,A23,A24;
double A31,A32,A33,A34;
double A41,A42,A43,A44;

double BFCT1,BFCT2,BFCT3,BFCT4;
double DFDL,D2FDL2,D3FDL3;

//The last increment of two is due to the shifting of
//two array elements in the x direction in all arrays
i = blockIdx.x * blockDim.x + threadIdx.x + 2;
j = blockIdx.y * blockDim.y + threadIdx.y;
// map the two 2D indices to a single linear, 1D index
index = j * grid_width + i;

frontneigh1 = (j+1) * grid_width + i;
backneigh1 = (j-1) * grid_width + i;
frontneigh2 = (j+2) * grid_width + i;
backneigh2 = (j-2) * grid_width + i;

DELTAY = (double)TY/(double)NY;

if (i > 1 && i < NX+2 && j == 0)
	{
	//j = 0 mesh point after outmost down
	//Setup coefficients
	A11 = (double)2.0*DELTAY;
	A12 = A11*DELTAY;
	A13 = A11*A12/(double)3.0;
	A21 = DELTAY;
    	A22 = A21*DELTAY/(double)2.0;
    	A23 = A21*A22/(double)3.0;
    	A31 = (double)1.0;
    	A32 = -DELTAY/(double)2.0;
    	A33 = A22/(double)4.0;

	//d2deltam_x/dy2, (Lower Boundary)
	BFCT1 = deltam_x[frontneigh2] - deltam_x[index];
	BFCT2 = deltam_x[frontneigh1] - deltam_x[index];
	BFCT3 = (double)0.0;

	SEL3(A11,A12,A13,A21,A22,A23,A31,A32,A33,BFCT1,BFCT2,BFCT3,&DFDL,&D2FDL2,&D3FDL3);

	d2ady2[index] = D2FDL2;

	//d2deltam_y/dy2, (Lower Boundary)
	BFCT1 = deltam_y[frontneigh2] - deltam_y[index];
	BFCT2 = deltam_y[frontneigh1] - deltam_y[index];
	BFCT3 = (double)0.0;

	SEL3(A11,A12,A13,A21,A22,A23,A31,A32,A33,BFCT1,BFCT2,BFCT3,&DFDL,&D2FDL2,&D3FDL3);

	d2bdy2[index] = D2FDL2;

	//d2deltam_z/dy2, (Lower Boundary)
	BFCT1 = deltam_z[frontneigh2] - deltam_z[index];
	BFCT2 = deltam_z[frontneigh1] - deltam_z[index];
	BFCT3 = (double)0.0;

	SEL3(A11,A12,A13,A21,A22,A23,A31,A32,A33,BFCT1,BFCT2,BFCT3,&DFDL,&D2FDL2,&D3FDL3);

	d2gdy2[index] = D2FDL2;
	}

if (i > 1 && i < NX+2 && j == 1)
	{
	// j = 1 mesh point after outmost down
	// Setup coefficients
      	A11 = (double)1.0;
      	A12 = (double)-1.5 * DELTAY;
      	A13 = A12*A12/(double)2.0;
      	A14 = -A13*DELTAY/(double)2.0;
      	A21 = -DELTAY;
      	A22 = DELTAY*DELTAY/(double)2.0;
      	A23 = A21*A22/(double)3.0;
      	A24 = A21*A23/(double)4.0;
      	A31 = DELTAY;
      	A32 = A22;
      	A33 = -A23;
      	A34 = A24;
      	A41 = (double)2.0*DELTAY;
      	A42 = A41*DELTAY;
      	A43 = A41*A42/(double)3.0;
      	A44 = A41*A43/(double)4.0;

   	// d2deltam_x/dy2
	BFCT1 = (double)0.0;
      	BFCT2 = deltam_x[backneigh1] - deltam_x[index];
      	BFCT3 = deltam_x[frontneigh1] - deltam_x[index];
     	BFCT4 = deltam_x[frontneigh2] - deltam_x[index];

      	SEL4(A11,A12,A13,A14,A21,A22,A23,A24,A31,A32,A33,A34,A41,A42,A43,A44, BFCT1,BFCT2,BFCT3,BFCT4, &DFDL,&D2FDL2);

	d2ady2[index]=D2FDL2;

	// d2deltam_y/dy2
	BFCT1 = (double)0.0;
      	BFCT2 = deltam_y[backneigh1] - deltam_y[index];
      	BFCT3 = deltam_y[frontneigh1] - deltam_y[index];
     	BFCT4 = deltam_y[frontneigh2] - deltam_y[index];

      	SEL4(A11,A12,A13,A14,A21,A22,A23,A24,A31,A32,A33,A34,A41,A42,A43,A44, BFCT1,BFCT2,BFCT3,BFCT4, &DFDL,&D2FDL2);

	d2bdy2[index]=D2FDL2;

   	// d2deltam_z/dy2
	BFCT1 = (double)0.0;
      	BFCT2 = deltam_z[backneigh1] - deltam_z[index];
      	BFCT3 = deltam_z[frontneigh1] - deltam_z[index];
     	BFCT4 = deltam_z[frontneigh2] - deltam_z[index];

      	SEL4(A11,A12,A13,A14,A21,A22,A23,A24,A31,A32,A33,A34,A41,A42,A43,A44, BFCT1,BFCT2,BFCT3,BFCT4, &DFDL,&D2FDL2);

	d2gdy2[index]=D2FDL2;
	}

if (i > 1 && i < NX+2 && j == NY-2)
	{
	// j = Ny - 2
	// Setup coefficients

      	A11 = (double)1.0;
      	A12 = (double)1.5 * DELTAY;
      	A13 = A12*A12/(double)2.0;
      	A14 = A13*DELTAY/(double)2.0;
      	A21 = DELTAY;
      	A22 = DELTAY*DELTAY/(double)2.0;
      	A23 = A21*A22/(double)3.0;
      	A24 = A21*A23/(double)4.0;
      	A31 = -DELTAY;
      	A32 = A22;
      	A33 = -A23;
      	A34 = A24;
      	A41 = (double)-2.0*DELTAY;
      	A42 = -A41*DELTAY;
      	A43 = A41*A42/(double)3.0;
      	A44 = A41*A43/(double)4.0;

   	// d2deltam_x/dy2
	BFCT1 = (double)0.0;
      	BFCT2 = deltam_x[frontneigh1] - deltam_x[index];
      	BFCT3 = deltam_x[backneigh1] - deltam_x[index];
     	BFCT4 = deltam_x[backneigh2] - deltam_x[index];

      	SEL4(A11,A12,A13,A14,A21,A22,A23,A24,A31,A32,A33,A34,A41,A42,A43,A44, BFCT1,BFCT2,BFCT3,BFCT4, &DFDL,&D2FDL2);

	d2ady2[index]=D2FDL2;

	// d2deltam_y/dy2
	BFCT1 = (double)0.0;
      	BFCT2 = deltam_y[frontneigh1] - deltam_y[index];
      	BFCT3 = deltam_y[backneigh1] - deltam_y[index];
     	BFCT4 = deltam_y[backneigh2] - deltam_y[index];

      	SEL4(A11,A12,A13,A14,A21,A22,A23,A24,A31,A32,A33,A34,A41,A42,A43,A44, BFCT1,BFCT2,BFCT3,BFCT4, &DFDL,&D2FDL2);

	d2bdy2[index]=D2FDL2;

   	// d2deltam_z/dy2
	BFCT1 = (double)0.0;
      	BFCT2 = deltam_z[frontneigh1] - deltam_z[index];
      	BFCT3 = deltam_z[backneigh1] - deltam_z[index];
     	BFCT4 = deltam_z[backneigh2] - deltam_z[index];

      	SEL4(A11,A12,A13,A14,A21,A22,A23,A24,A31,A32,A33,A34,A41,A42,A43,A44, BFCT1,BFCT2,BFCT3,BFCT4, &DFDL,&D2FDL2);

	d2gdy2[index]=D2FDL2;
	}

if (i > 1 && i < NX+2 && j == NY-1)
	{
	//j = NY - 1 mesh point to the outmost up
      	A11 = (double)1.0;
      	A21 = -DELTAY;
      	A31 = (double)-2.0*DELTAY;
      	A12 = DELTAY/(double)2.0;
      	A22 = A12*DELTAY;
      	A32 = (double)4.0*A22;
      	A13 = A22/(double)4.0;
      	A23 = A21*A22/(double)3.0;
      	A33 = A31*A32/(double)3.0;

	//d2deltam_x/dy2, (Upper Boundary)
	BFCT1 = (double)0.0;
	BFCT2 = deltam_x[backneigh1] - deltam_x[index];
	BFCT3 = deltam_x[backneigh2] - deltam_x[index];

	SEL3(A11,A12,A13,A21,A22,A23,A31,A32,A33,BFCT1,BFCT2,BFCT3,&DFDL,&D2FDL2,&D3FDL3);

	d2ady2[index] = D2FDL2;

	//d2deltam_y/dy2,
	BFCT1 = (double)0.0;
	BFCT2 = deltam_y[backneigh1] - deltam_y[index];
	BFCT3 = deltam_y[backneigh2] - deltam_y[index];

	SEL3(A11,A12,A13,A21,A22,A23,A31,A32,A33,BFCT1,BFCT2,BFCT3,&DFDL,&D2FDL2,&D3FDL3);

	d2bdy2[index] = D2FDL2;

	//d2deltam_z/dy2,
	BFCT1 = (double)0.0;
	BFCT2 = deltam_z[backneigh1] - deltam_z[index];
	BFCT3 = deltam_z[backneigh2] - deltam_z[index];

	SEL3(A11,A12,A13,A21,A22,A23,A31,A32,A33,BFCT1,BFCT2,BFCT3,&DFDL,&D2FDL2,&D3FDL3);

	d2gdy2[index] = D2FDL2;
	}

if (i > 1 && i < NX+2 && j >= 0 && j < 2)
	{
	//In order to have the correct result glaplacianx must be called first
	lapl_x[index] += (double)D * d2ady2[index];
	lapl_y[index] += (double)D * d2bdy2[index];
	lapl_z[index] += (double)D * d2gdy2[index];
	}

if (i > 1 && i < NX+2 && j >= NY-2 && j <= NY-1)
	{
	//In order to havethe correct result glaplacianx must be called first
	lapl_x[index] += (double)D * d2ady2[index];
	lapl_y[index] += (double)D * d2bdy2[index];
	lapl_z[index] += (double)D * d2gdy2[index];
	}

}

__global__ void gsd_exchange(double tau_sd,double *sdex_x, double *sdex_y, double *sdex_z,
				double *deltam_x, double *deltam_y, double *deltam_z,
				double *mx, double *my, double *mz,
			 	int grid_width)
//Computation of sd_term using global memory
{
int i,j,index;
//The last increment of two is due to the shifting of
//two array elements in the x direction in all arrays
i = blockIdx.x * blockDim.x + threadIdx.x + 2;
j = blockIdx.y * blockDim.y + threadIdx.y;
// map the two 2D indices to a single linear, 1D index
index = j * grid_width + i;

if (i > 1 && i < NX+2 && j >= 0 && j < NY)
	{
	sdex_x[index] = - (deltam_y[index] * mz[index] - deltam_z[index] * my[index]) / tau_sd;
	sdex_y[index] = - (deltam_z[index] * mx[index] - deltam_x[index] * mz[index]) / tau_sd;
	sdex_z[index] = - (deltam_x[index] * my[index] - deltam_y[index] * mx[index]) / tau_sd;
	}
}

__global__ void gsource(double u, double *sm, double *m, int grid_width)
//Computation of source term using global memory
{
int i,j,index;
double DELTAX;
DELTAX = TX/NX;
//The last increment of two is due to the shifting of
//two array elements in the x direction in all arrays
i = blockIdx.x * blockDim.x + threadIdx.x + 2;
j = blockIdx.y * blockDim.y + threadIdx.y;
// map the two 2D indices to a single linear, 1D index
index = j * grid_width + i;

if (i > 1 && i < NX+2 && j >= 0 && j < NY)
	{
	sm[index] = u * (m[index - 2] - 8.0 * m[index - 1] +
			8.0 * m[index + 1] - m[index + 2]) / (12.0 * DELTAX);
	}
}

__global__ void gm_x_sm(double u, double *m_x_sm_x, double *m_x_sm_y, double *m_x_sm_z,
			double *mx, double *my, double *mz,
			double *sm_x, double *sm_y, double *sm_z, int grid_width)
{
int i,j,index;
//The last increment of two is due to the shifting of
//two array elements in the x direction in all arrays
i = blockIdx.x * blockDim.x + threadIdx.x + 2;
j = blockIdx.y * blockDim.y + threadIdx.y;
// map the two 2D indices to a single linear, 1D index
index = j * grid_width + i;
if (i > 1 && i < NX+2 && j >= 0 && j < NY)
	{
	//Compute x component of cross product
	m_x_sm_x[index] = my[index] * sm_z[index]/u - mz[index] * sm_y[index]/u;

	//Compute y component of cross product
	m_x_sm_y[index] = mz[index] * sm_x[index]/u - mx[index] * sm_z[index]/u;

	//Compute z component of cross product
	m_x_sm_z[index] = mx[index] * sm_y[index]/u - my[index] * sm_x[index]/u;
	}
}

__global__ void gm_x_source(double *tempx, double *tempy, double *tempz,
			double *mx, double *my, double *mz,
			double *sm_x, double *sm_y, double *sm_z, int grid_width)
{
int i,j,index;
//The last increment of two is due to the shifting of
//two array elements in the x direction in all arrays
i = blockIdx.x * blockDim.x + threadIdx.x + 2;
j = blockIdx.y * blockDim.y + threadIdx.y;
// map the two 2D indices to a single linear, 1D index
index = j * grid_width + i;
//Firstly backup current source term into a temporary array
if (i > 1 && i < NX+2 && j >= 0 && j < NY)
	{
	//Backup x component of source term
	tempx[index] = sm_x[index];

	//Backup y component of source term
	tempy[index] = sm_y[index];

	//Backup z component of source term
	tempz[index] = sm_z[index];
	}
__syncthreads();
if (i > 1 && i < NX+2 && j >= 0 && j < NY)
	{
	//Compute x component of cross product
	sm_x[index] = my[index] * tempz[index] - mz[index] * tempy[index];

	//Compute y component of cross product
	sm_y[index] = mz[index] * tempx[index] - mx[index] * tempz[index];

	//Compute z component of cross product
	sm_z[index] = mx[index] * tempy[index] - my[index] * tempx[index];
	}
}

__global__ void gu_eff( double u, double tau_sd, double *u_eff,
			double *deltam_x, double *deltam_y, double *deltam_z,
			double *m_x_sm_x, double *m_x_sm_y, double *m_x_sm_z,
			double *sm_x, double *sm_y, double *sm_z, int grid_width)
{
int i, j, index;
double partial_norm;
//The last increment of two is due to the shifting of
//two array elements in the x direction in all arrays
i = blockIdx.x * blockDim.x + threadIdx.x + 2;
j = blockIdx.y * blockDim.y + threadIdx.y;
// map the two 2D indices to a single linear, 1D index
index = j * grid_width + i;
if (i > 1 && i < NX+2 && j >= 0 && j < NY)
	{
	//First we compute the norm for the current cell note that sm must be divided
	//by u in order to get the correct value of partial x from source
	partial_norm = sm_x[index] * sm_x[index] + sm_y[index] * sm_y[index] + sm_z[index] * sm_z[index];

	partial_norm = partial_norm / (u*u);

	//Compute effective u notice that tau_sd is initially given in ps
	//so we convert to ns
	u_eff[index] = -(1.0 / tau_sd) * (deltam_x[index] * m_x_sm_x[index]
			+ deltam_y[index] * m_x_sm_y[index]
			+ deltam_z[index] * m_x_sm_z[index])/partial_norm;
	}
}

__global__ void gu_eff_beta_eff(double u, double tau_sd, double *u_eff_beta_eff,
			double *deltam_x, double *deltam_y, double *deltam_z,
			double *sm_x, double *sm_y, double *sm_z, int grid_width)
{
int i, j, index;
double partial_norm;
//The last increment of two is due to the shifting of
//two array elements in the x direction in all arrays
i = blockIdx.x * blockDim.x + threadIdx.x + 2;
j = blockIdx.y * blockDim.y + threadIdx.y;
// map the two 2D indices to a single linear, 1D index
index = j * grid_width + i;
if (i > 1 && i < NX+2 && j >= 0 && j < NY)
	{
	//First we compute the norm for the current cell note that sm must be divided
	//by u in order to get the correct value of partial x from source
	partial_norm = sm_x[index] * sm_x[index] + sm_y[index] * sm_y[index] + sm_z[index] * sm_z[index];

	partial_norm = partial_norm / (u*u);

	//Compute effective u times effective beta, notice that tau_sd
	//is initially given in ps
	u_eff_beta_eff[index] = -(1.0 / tau_sd) * (deltam_x[index] * sm_x[index]/u
			+ deltam_y[index] * sm_y[index]/u
			+ deltam_z[index] * sm_z[index]/u)/partial_norm;
	}
}

__global__ void gbeta_eff(double *beta_eff, double *u_eff, double *u_eff_beta_eff, int grid_width)
{
int i, j, index;

//The last increment of two is due to the shifting of
//two array elements in the x direction in all arrays
i = blockIdx.x * blockDim.x + threadIdx.x + 2;
j = blockIdx.y * blockDim.y + threadIdx.y;
// map the two 2D indices to a single linear, 1D index
index = j * grid_width + i;
if (i > 1 && i < NX+2 && j >= 0 && j < NY)
	{

	//Compute effective beta
	beta_eff[index] = u_eff_beta_eff[index] / u_eff[index];
	}
}

__global__ void gbeta_diff(double u, double *beta_diff_num, double *beta_diff_den, double *beta_eff,
				double *sm_x, double *sm_y, double *sm_z, int grid_width)
//This kernel computes the numerators and denominators in order to obtain the
//beta_diff value
{
//Notice that we are using less than 1024 threads because shared memory
//is limited to 16 KB
__shared__ double cachenum[(XTHREADS_PERBLOCK / 2) * (YTHREADS_PERBLOCK / 2)];
__shared__ double cacheden[(XTHREADS_PERBLOCK / 2) * (YTHREADS_PERBLOCK / 2)];
int i, j, index, cacheIndex;
double partial_norm;
//The last increment of two is due to the shifting of
//two array elements in the x direction in all arrays
i = blockIdx.x * blockDim.x + threadIdx.x + 2;
j = blockIdx.y * blockDim.y + threadIdx.y;
// map the two 2D indices to a single linear, 1D index
index = j * grid_width + i;
cacheIndex = threadIdx.y * blockDim.x + threadIdx.x;
if (i > 1 && i < NX+2 && j >= 0 && j < NY)
	{
	//First we compute the norm for the current cell note that sm must be divided
	//by u in order to get the correct value of partial x from source
	partial_norm = sm_x[index] * sm_x[index] + sm_y[index] * sm_y[index] + sm_z[index] * sm_z[index];

	partial_norm = partial_norm / (u*u);

	//every thread copies its value to the cache
        cachenum[cacheIndex] = beta_eff[index]*partial_norm;
        cacheden[cacheIndex] = partial_norm;
	}

//we wait until all threads have written their values
__syncthreads();

if (i > 1 && i < NX+2 && j >= 0 && j < NY)
	{
	beta_diff_num[index] = cachenum[cacheIndex];
	beta_diff_den[index] = cacheden[cacheIndex];
	}
}

////////////////////////////////////////
// CPU functions
////////////////////////////////////////

void read_magnetization_data(FILE *in)
{
int i,j,i0;

fscanf(in,"%i",&NXNY); //Number of points
printf("The number of points to read are: %i\n",NXNY);
for (j = 0; j < NY; j++)
	for (i = 0; i < NX; i++)
	{
	fscanf(in,"%lf %lf",&XCOORD[i][j],&YCOORD[i][j]); //Read mx coordinate
	}
printf("Initial and final coordinates read are:\n%20.15f,%20.15f\n%20.15f,%20.15f\n",
        XCOORD[0][0],YCOORD[0][0],XCOORD[NX-1][NY-1],YCOORD[NX-1][NY-1]);
//Read data
for (j = 0; j < NY ; j++)
	for (i = 0 ; i < NX ; i++)
	{
	i0 = i + 2;
	fscanf(in,"%lf %lf %lf",&mx[i0][j],&my[i0][j],&mz[i0][j]);
	}
printf("Magnetization read: \n%20.15f %20.15f %20.15f\n%20.15f %20.15f %20.15f\n",
	mx[2][0],my[2][0],mz[2][0],mx[NX+1][NY-1],my[NX+1][NY-1],mz[NX+1][NY-1]);
for (j = 0; j < NY; j++)
	{
	//Replicate values at the two left boundary cells
	mx[1][j] = mx[2][j];
	my[1][j] = my[2][j];
	mz[1][j] = mz[2][j];

	mx[0][j] = mx[2][j];
	my[0][j] = my[2][j];
	mz[0][j] = mz[2][j];

	//Replicate values at the two right boundary cells

	mx[NX+2][j] = mx[NX+1][j];
	my[NX+2][j] = my[NX+1][j];
	mz[NX+2][j] = mz[NX+1][j];

	mx[NX+3][j] = mx[NX+1][j];
	my[NX+3][j] = my[NX+1][j];
	mz[NX+3][j] = mz[NX+1][j];
	}
}

void read_spinaccumulation_data(FILE *in)
{
int i,j,i0;
fscanf(in,"%i",&NXNY); //Number of points
printf("The number of points to read are: %i\n",NXNY);

//Read data
for (j = 0; j < NY ; j++)
	for (i = 0 ; i < NX ; i++)
	{
	i0 = i + 2;
	fscanf(in,"%lf %lf %lf",&deltam_x[i0][j],&deltam_y[i0][j],&deltam_z[i0][j]);
	}
printf("Spin accumulation read: \n%20.15f %20.15f %20.15f\n%20.15f %20.15f %20.15f\n",
	deltam_x[2][0],deltam_y[2][0],deltam_z[2][0],deltam_x[NX+1][NY-1],deltam_y[NX+1][NY-1],deltam_z[NX+1][NY-1]);
for (j = 0; j < NY; j++)
	{
	//Replicate values at the two left boundary cells
	deltam_x[1][j] = deltam_x[2][j];
	deltam_y[1][j] = deltam_y[2][j];
	deltam_z[1][j] = deltam_z[2][j];
	deltam_x[0][j] = deltam_x[2][j];
	deltam_y[0][j] = deltam_y[2][j];
	deltam_z[0][j] = deltam_z[2][j];

	//Replicate values at the two right boundary cells
	deltam_x[NX+2][j] = deltam_x[NX+1][j];
	deltam_y[NX+2][j] = deltam_y[NX+1][j];
	deltam_z[NX+2][j] = deltam_z[NX+1][j];
	deltam_x[NX+3][j] = deltam_x[NX+1][j];
	deltam_y[NX+3][j] = deltam_y[NX+1][j];
	deltam_z[NX+3][j] = deltam_z[NX+1][j];
	}
}


void save_spinaccumulation_data(FILE *out)
{
int i,j,i0;
fprintf(out,"%i\n",NX*NY);
//Save spin accumulation data
for (j = 0; j < NY; j++)
	for (i = 0 ; i < NX ; i++)
	{
	i0 = i + 2;
	fprintf(out,"%20.15f%20.15f%20.15f\n",deltam_x[i0][j],deltam_y[i0][j],deltam_z[i0][j]); //Write
	}
printf("File: %s written succesfully!\n",fileoutstr);
}

void save_effectivevalues_data(FILE *out)
{
int i,j,i0;

fprintf(out,"%i\n",NX*NY);
//Save spin accumulation data
for (j = 0; j < NY; j++)
	for (i = 0 ; i < NX ; i++)
	{
	i0 = i + 2;
	fprintf(out,"%20.15f%20.15f%20.15f\n",u_eff[i0][j],u_eff_beta_eff[i0][j],beta_eff[i0][j]); //Write
	}

printf("File: %s written succesfully!\n",effvalsfileoutstr);
}

void flatten_array(double **array2D,int cols, int rows)
//This function maps a 2D array into a temporary 1D array
{
int i,j,index;
for (j = 0; j < rows; j++)
	for (i = 0; i < cols; i++)
		{
		index = j * cols + i;
		mem_buffer[index] = array2D[i][j];
		}
}

void unflatten_array(double **array2D,int cols, int rows)
{
int i,j,index;
for (j = 0; j < rows; j++)
	for (i = 0; i < cols; i++)
		{
		index = j * cols + i;
		array2D[i][j] =  mem_buffer[index];
		}
}

void initial_setup(void)
{
int i;

//Setup array dimensions in powers of 2 for optimum processing in CUDA
NXCUDA = (int)powf(2,ceilf(logf(NX)/logf(2)));
printf("NXCUDA = %i\n",NXCUDA);
NYCUDA = (int)powf(2,ceilf(logf(NY)/logf(2)));
printf("NYCUDA = %i\n",NYCUDA);
if((int)powf(2,ceilf(logf(NZ)/logf(2))) < 1)
	NZCUDA = 1;
else
        NZCUDA = (int)powf(2,ceilf(logf(NZ)/logf(2)));
//printf("NZCUDA = %i\n",NZCUDA);

//Setup optimum number of blocks
XBLOCKS_PERGRID = (int)ceil((float)NX/(float)XTHREADS_PERBLOCK);
printf("XBLOCKS_PERGRID = %i\n",XBLOCKS_PERGRID);

YBLOCKS_PERGRID = (int)ceil((float)NY/(float)YTHREADS_PERBLOCK);
printf("YBLOCKS_PERGRID = %i\n",YBLOCKS_PERGRID);

//Allocation of arrays
printf("Allocating arrays for magnetization, spin accumulation and effective data ");

//Arrays for magnetization
mx=(double **)calloc(NXCUDA,sizeof(double *));
for(i = 0; i < NXCUDA; i++)
	mx[i]=(double *)calloc(NYCUDA,sizeof(double));
printf(". ");

my=(double **)calloc(NXCUDA,sizeof(double *));
for(i = 0; i < NXCUDA; i++)
	my[i]=(double *)calloc(NYCUDA,sizeof(double));
printf(". ");

mz=(double **)calloc(NXCUDA,sizeof(double *));
for(i = 0; i < NXCUDA; i++)
	mz[i]=(double *)calloc(NYCUDA,sizeof(double));
printf(". ");

//Arrays for spin accumulation
deltam_x=(double **)calloc(NXCUDA,sizeof(double *));
for(i = 0; i < NXCUDA; i++)
	deltam_x[i]=(double *)calloc(NYCUDA,sizeof(double));
printf(". ");

deltam_y=(double **)calloc(NXCUDA,sizeof(double *));
for(i = 0; i < NXCUDA; i++)
	deltam_y[i]=(double *)calloc(NYCUDA,sizeof(double));
printf(". ");

deltam_z=(double **)calloc(NXCUDA,sizeof(double *));
for(i = 0; i < NXCUDA; i++)
	deltam_z[i]=(double *)calloc(NYCUDA,sizeof(double));
printf(". ");

//Arrays for effective values
u_eff=(double **)calloc(NXCUDA,sizeof(double *));
for(i = 0; i < NXCUDA; i++)
	u_eff[i]=(double *)calloc(NYCUDA,sizeof(double));
printf(". ");

u_eff_beta_eff=(double **)calloc(NXCUDA,sizeof(double *));
for(i = 0; i < NXCUDA; i++)
	u_eff_beta_eff[i]=(double *)calloc(NYCUDA,sizeof(double));
printf(". ");

beta_eff=(double **)calloc(NXCUDA,sizeof(double *));
for(i = 0; i < NXCUDA; i++)
	beta_eff[i]=(double *)calloc(NYCUDA,sizeof(double));
printf(". ");

//Arrays for coordinates
XCOORD=(double **)calloc(NX,sizeof(double *));
for(i = 0; i < NX; i++)
	XCOORD[i]=(double *)calloc(NY,sizeof(double));
printf(". ");

YCOORD=(double **)calloc(NX,sizeof(double *));
for(i = 0; i < NX; i++)
	YCOORD[i]=(double *)calloc(NY,sizeof(double));
printf(". ");

//Temporary array to convert among 1D and 2D before transferences between CPU and GPU
mem_buffer=(double *)calloc(NXCUDA*NYCUDA,sizeof(double));

//Array of partial computations of beta_diff
beta_diff_num = (double *)calloc( NXCUDA * NYCUDA ,sizeof(double));
beta_diff_den = (double *)calloc( NXCUDA * NYCUDA ,sizeof(double));

printf(".\tDone!\n");

//Read magnetization
if ((magnin = fopen(filein,"r")) != NULL )
	{
	read_magnetization_data(magnin);	//Open file containing magnetization
	fclose(magnin); //Close input file
	}
else
	{
	printf("Error: File %s doesn't exists!\n",filein);
	exit(1);
	}
//Read spinccumulation if required
if (readspinaccumulation == 1)
	{
	if ((spinin = fopen(spinfilein,"r")) != NULL )
		{
		read_spinaccumulation_data(spinin);	//Open file containing spinaccumulation
		fclose(spinin); //Close input file
		}
	else
		{
		printf("Error: File %s doesn't exists!\n",spinin);
		exit(1);
		}
	}

// Allocating arrays in GPU
size_bytes = NXCUDA * NYCUDA * sizeof(double); //Compute total size of arrays
printf("Allocating arrays for processing in GPU  ");
// Allocate arrays for magnetization
HANDLE_ERROR( cudaMalloc((void **)&dev_mx,size_bytes) );
printf(". ");
HANDLE_ERROR( cudaMalloc((void **)&dev_my,size_bytes) );
printf(". ");
HANDLE_ERROR( cudaMalloc((void **)&dev_mz,size_bytes) );
printf(". ");

// Allocate arrays for spinaccumulation
HANDLE_ERROR( cudaMalloc((void **)&dev_deltam_x,size_bytes) );
printf(". ");
HANDLE_ERROR( cudaMalloc((void **)&dev_deltam_y,size_bytes) );
printf(". ");
HANDLE_ERROR( cudaMalloc((void **)&dev_deltam_z,size_bytes) );
printf(". ");

// Allocate arrays for source term
HANDLE_ERROR( cudaMalloc((void **)&dev_sm_x,size_bytes) );
printf(". ");
HANDLE_ERROR( cudaMalloc((void **)&dev_sm_y,size_bytes) );
printf(". ");
HANDLE_ERROR( cudaMalloc((void **)&dev_sm_z,size_bytes) );
printf(". ");

// Allocate arrays for exchange term
HANDLE_ERROR( cudaMalloc((void **)&dev_sdex_x,size_bytes) );
printf(". ");
HANDLE_ERROR( cudaMalloc((void **)&dev_sdex_y,size_bytes) );
printf(". ");
HANDLE_ERROR( cudaMalloc((void **)&dev_sdex_z,size_bytes) );
printf(". ");
HANDLE_ERROR( cudaMalloc((void **)&dev_sm_z,size_bytes) );
printf(". ");

// Allocate arrays for spinrelaxation term
HANDLE_ERROR( cudaMalloc((void **)&dev_sfrelax_x,size_bytes) );
printf(". ");
HANDLE_ERROR( cudaMalloc((void **)&dev_sfrelax_y,size_bytes) );
printf(". ");
HANDLE_ERROR( cudaMalloc((void **)&dev_sfrelax_z,size_bytes) );
printf(". ");

// Allocate arrays for spinrelaxation term
HANDLE_ERROR( cudaMalloc((void **)&dev_lapl_x,size_bytes) );
printf(". ");
HANDLE_ERROR( cudaMalloc((void **)&dev_lapl_y,size_bytes) );
printf(". ");
HANDLE_ERROR( cudaMalloc((void **)&dev_lapl_z,size_bytes) );
printf(". ");

// Allocate arrays for cross product of m and sm/u
HANDLE_ERROR( cudaMalloc((void **)&dev_m_x_sm_x,size_bytes) );
printf(". ");
HANDLE_ERROR( cudaMalloc((void **)&dev_m_x_sm_y,size_bytes) );
printf(". ");
HANDLE_ERROR( cudaMalloc((void **)&dev_m_x_sm_z,size_bytes) );
printf(". ");

// Allocate arrays for effective values
HANDLE_ERROR( cudaMalloc((void **)&dev_u_eff,size_bytes) );
printf(". ");
HANDLE_ERROR( cudaMalloc((void **)&dev_u_eff_beta_eff,size_bytes) );
printf(". ");
HANDLE_ERROR( cudaMalloc((void **)&dev_beta_eff,size_bytes) );
printf(". ");

//Allocate arrays for temporary results
HANDLE_ERROR( cudaMalloc((void **)&dev_tempx,size_bytes) );
printf(". ");
HANDLE_ERROR( cudaMalloc((void **)&dev_tempy,size_bytes) );
printf(". ");
HANDLE_ERROR( cudaMalloc((void **)&dev_tempz,size_bytes) );
printf(". ");

//Allocate arrays for temporary results
HANDLE_ERROR( cudaMalloc((void **)&dev_k1x,size_bytes) );
printf(". ");
HANDLE_ERROR( cudaMalloc((void **)&dev_k1y,size_bytes) );
printf(". ");
HANDLE_ERROR( cudaMalloc((void **)&dev_k1z,size_bytes) );
printf(". ");

//Allocate arrays for temporary results
HANDLE_ERROR( cudaMalloc((void **)&dev_k2x,size_bytes) );
printf(". ");
HANDLE_ERROR( cudaMalloc((void **)&dev_k2y,size_bytes) );
printf(". ");
HANDLE_ERROR( cudaMalloc((void **)&dev_k2z,size_bytes) );
printf(". ");

//Allocate arrays for temporary results
HANDLE_ERROR( cudaMalloc((void **)&dev_k3x,size_bytes) );
printf(". ");
HANDLE_ERROR( cudaMalloc((void **)&dev_k3y,size_bytes) );
printf(". ");
HANDLE_ERROR( cudaMalloc((void **)&dev_k3z,size_bytes) );
printf(". ");

//Allocate arrays for temporary results
HANDLE_ERROR( cudaMalloc((void **)&dev_k4x,size_bytes) );
printf(". ");
HANDLE_ERROR( cudaMalloc((void **)&dev_k4y,size_bytes) );
printf(". ");
HANDLE_ERROR( cudaMalloc((void **)&dev_k4z,size_bytes) );
printf(". ");

//Allocate arrays for evaluation of laplacian term
HANDLE_ERROR( cudaMalloc((void **)&dev_d2adx2,size_bytes) );
printf(". ");
HANDLE_ERROR( cudaMalloc((void **)&dev_d2bdx2,size_bytes) );
printf(". ");
HANDLE_ERROR( cudaMalloc((void **)&dev_d2gdx2,size_bytes) );
printf(". ");

//Allocate arrays for evaluation of laplacian term
HANDLE_ERROR( cudaMalloc((void **)&dev_d2ady2,size_bytes) );
printf(". ");
HANDLE_ERROR( cudaMalloc((void **)&dev_d2bdy2,size_bytes) );
printf(". ");
HANDLE_ERROR( cudaMalloc((void **)&dev_d2gdy2,size_bytes) );
printf(". ");

//Allocate arrays for effective values
HANDLE_ERROR( cudaMalloc((void **)&dev_beta_diff_num, size_bytes ) );
printf(". ");
HANDLE_ERROR( cudaMalloc((void **)&dev_beta_diff_den, size_bytes ) );
printf(".\tDone!\n");

//Upload data to GPU
printf("Uploading arrays for processing in GPU  ");
flatten_array(mx,NXCUDA,NYCUDA);
HANDLE_ERROR( cudaMemcpy(dev_mx,mem_buffer,size_bytes,cudaMemcpyHostToDevice) );
printf(". ");
flatten_array(my,NXCUDA,NYCUDA);
HANDLE_ERROR( cudaMemcpy(dev_my,mem_buffer,size_bytes,cudaMemcpyHostToDevice) );
printf(". ");
flatten_array(mz,NXCUDA,NYCUDA);
HANDLE_ERROR( cudaMemcpy(dev_mz,mem_buffer,size_bytes,cudaMemcpyHostToDevice) );

//If spin accumulation is not read from a file then we have to initialize
//deltam vector to zero
if(readspinaccumulation == 1)
	{
	flatten_array(deltam_x,NXCUDA,NYCUDA);
	HANDLE_ERROR( cudaMemcpy(dev_deltam_x,mem_buffer,size_bytes,cudaMemcpyHostToDevice) );
	printf(". ");
	flatten_array(deltam_y,NXCUDA,NYCUDA);
	HANDLE_ERROR( cudaMemcpy(dev_deltam_y,mem_buffer,size_bytes,cudaMemcpyHostToDevice) );
	printf(". ");
	flatten_array(deltam_z,NXCUDA,NYCUDA);
	HANDLE_ERROR( cudaMemcpy(dev_deltam_z,mem_buffer,size_bytes,cudaMemcpyHostToDevice) );
	}
else
	{
	HANDLE_ERROR( cudaMemset(dev_deltam_x,0,size_bytes) );
	HANDLE_ERROR( cudaMemset(dev_deltam_y,0,size_bytes) );
	HANDLE_ERROR( cudaMemset(dev_deltam_z,0,size_bytes) );
	HANDLE_ERROR( cudaMemset(dev_sfrelax_x,0,size_bytes) );
	HANDLE_ERROR( cudaMemset(dev_sfrelax_y,0,size_bytes) );
	HANDLE_ERROR( cudaMemset(dev_sfrelax_z,0,size_bytes) );
	HANDLE_ERROR( cudaMemset(dev_sdex_x,0,size_bytes) );
	HANDLE_ERROR( cudaMemset(dev_sdex_y,0,size_bytes) );
	HANDLE_ERROR( cudaMemset(dev_sdex_z,0,size_bytes) );
	HANDLE_ERROR( cudaMemset(dev_lapl_x,0,size_bytes) );
	HANDLE_ERROR( cudaMemset(dev_lapl_y,0,size_bytes) );
	HANDLE_ERROR( cudaMemset(dev_lapl_z,0,size_bytes) );
	HANDLE_ERROR( cudaMemset(dev_d2adx2,0,size_bytes) );
	HANDLE_ERROR( cudaMemset(dev_d2bdx2,0,size_bytes) );
	HANDLE_ERROR( cudaMemset(dev_d2gdx2,0,size_bytes) );
	HANDLE_ERROR( cudaMemset(dev_d2ady2,0,size_bytes) );
	HANDLE_ERROR( cudaMemset(dev_d2bdy2,0,size_bytes) );
	HANDLE_ERROR( cudaMemset(dev_d2gdy2,0,size_bytes) );
	}
printf(".\tDone!\n");
}



void finalize(void)
{
int i;
//Free magnetization
for (i = 0; i < NXCUDA; i++)
	free(mx[i]);
free(mx);

for (i = 0; i < NXCUDA; i++)
	free(my[i]);
free(my);

for (i = 0; i < NXCUDA; i++)
	free(mz[i]);
free(mz);

//Free spin accumulation
for (i = 0; i < NXCUDA; i++)
	free(deltam_x[i]);
free(deltam_x);

for (i = 0; i < NXCUDA; i++)
	free(deltam_y[i]);
free(deltam_y);

for (i = 0; i < NXCUDA; i++)
	free(deltam_z[i]);
free(deltam_z);

//Free effective values arrays
for (i = 0; i < NXCUDA; i++)
	free(u_eff[i]);
free(u_eff);

for (i = 0; i < NXCUDA; i++)
	free(u_eff_beta_eff[i]);
free(u_eff_beta_eff);

for (i = 0; i < NXCUDA; i++)
	free(beta_eff[i]);
free(beta_eff);

//Free coordinate arrays
for (i = 0; i < NX; i++)
	free(XCOORD[i]);
free(XCOORD);

for (i = 0; i < NX; i++)
	free(YCOORD[i]);
free(YCOORD);

//Free temporary array buffer
free(mem_buffer);

//Free vector with partial data of beta_diff
free(beta_diff_num);
free(beta_diff_den);

//Free arrays in GPU global memory
cudaFree(dev_mx);
cudaFree(dev_my);
cudaFree(dev_mz);

cudaFree(dev_deltam_x);
cudaFree(dev_deltam_y);
cudaFree(dev_deltam_z);

cudaFree(dev_sm_x);
cudaFree(dev_sm_y);
cudaFree(dev_sm_z);

cudaFree(dev_sdex_x);
cudaFree(dev_sdex_y);
cudaFree(dev_sdex_z);

cudaFree(dev_sfrelax_x);
cudaFree(dev_sfrelax_y);
cudaFree(dev_sfrelax_z);

cudaFree(dev_lapl_x);
cudaFree(dev_lapl_y);
cudaFree(dev_lapl_z);

cudaFree(dev_m_x_sm_x);
cudaFree(dev_m_x_sm_y);
cudaFree(dev_m_x_sm_z);

cudaFree(dev_u_eff);
cudaFree(dev_u_eff_beta_eff);
cudaFree(dev_beta_eff);

cudaFree(dev_beta_diff_num);
cudaFree(dev_beta_diff_den);

cudaFree(dev_tempx);
cudaFree(dev_tempy);
cudaFree(dev_tempz);

cudaFree(dev_k1x);
cudaFree(dev_k1y);
cudaFree(dev_k1z);

cudaFree(dev_k2x);
cudaFree(dev_k2y);
cudaFree(dev_k2z);

cudaFree(dev_k3x);
cudaFree(dev_k3y);
cudaFree(dev_k3z);

cudaFree(dev_k4x);
cudaFree(dev_k4y);
cudaFree(dev_k4z);

cudaFree(dev_d2adx2);
cudaFree(dev_d2bdx2);
cudaFree(dev_d2gdx2);

cudaFree(dev_d2ady2);
cudaFree(dev_d2bdy2);
cudaFree(dev_d2gdy2);

}

void update_CPUspinaccumulation(void)
{
HANDLE_ERROR( cudaMemcpy(mem_buffer,dev_deltam_x,size_bytes,cudaMemcpyDeviceToHost) );
unflatten_array(deltam_x,NXCUDA,NYCUDA);

HANDLE_ERROR( cudaMemcpy(mem_buffer,dev_deltam_y,size_bytes,cudaMemcpyDeviceToHost) );
unflatten_array(deltam_y,NXCUDA,NYCUDA);

HANDLE_ERROR( cudaMemcpy(mem_buffer,dev_deltam_z,size_bytes,cudaMemcpyDeviceToHost) );
unflatten_array(deltam_z,NXCUDA,NYCUDA);
}

void effective_values(void)
{
//A conventional division of work in blocks with 1024 (32 in x and 32
// 32 in y) threads each, is used in most of the calculations that are not using shared
//data or dimensions are less than 16 kB
dim3 blocks(XBLOCKS_PERGRID,YBLOCKS_PERGRID);
dim3 threads(XTHREADS_PERBLOCK,YTHREADS_PERBLOCK);

//A further division of work is carried out
//when using 1024 threads is not possible because shared data requirements
//are higher than 16 kB per kernel
dim3 blocks2(XBLOCKS_PERGRID*2,YBLOCKS_PERGRID*2);
dim3 threads2(XTHREADS_PERBLOCK/2,YTHREADS_PERBLOCK/2);

int i,i0,j,index;
double num = 0.0, den = 0.0;

gm_x_sm<<<blocks,threads>>>(u_val,dev_m_x_sm_x, dev_m_x_sm_y, dev_m_x_sm_z,
				dev_mx, dev_my, dev_mz,
				dev_sm_x, dev_sm_y, dev_sm_z, NXCUDA);

gu_eff<<<blocks,threads>>>(u_val,tau_sd_val,dev_u_eff,
			dev_deltam_x, dev_deltam_y, dev_deltam_z,
			dev_m_x_sm_x, dev_m_x_sm_y, dev_m_x_sm_z,
			dev_sm_x, dev_sm_y, dev_sm_z, NXCUDA);

gu_eff_beta_eff<<<blocks,threads>>>(u_val,tau_sd_val,dev_u_eff_beta_eff,
			dev_deltam_x, dev_deltam_y, dev_deltam_z,
			dev_sm_x, dev_sm_y, dev_sm_z, NXCUDA);

gbeta_eff<<<blocks,threads>>>(dev_beta_eff, dev_u_eff, dev_u_eff_beta_eff, NXCUDA);

//Notice that an alternative size of blocks and threads is used
gbeta_diff<<<blocks2,threads2>>>(u_val,dev_beta_diff_num, dev_beta_diff_den,
				dev_beta_eff, dev_sm_x, dev_sm_y, dev_sm_z, NXCUDA);

HANDLE_ERROR( cudaMemcpy(beta_diff_num, dev_beta_diff_num, size_bytes, cudaMemcpyDeviceToHost) );
HANDLE_ERROR( cudaMemcpy(beta_diff_den, dev_beta_diff_den, size_bytes, cudaMemcpyDeviceToHost) );
for(j = 0; j < NY; j++)
	for(i = 0; i < NX; i++)
	{
	i0 = i + 2;
	index = j * NXCUDA + i0;
	num += beta_diff_num[index];
	den += beta_diff_den[index];
	}
beta_diff = num / den;
printf("Diffusive beta: %20.15e\n", beta_diff);
fprintf(outlog,"Diffusive beta: %20.15e\n", beta_diff);

HANDLE_ERROR( cudaMemcpy(mem_buffer,dev_u_eff,size_bytes,cudaMemcpyDeviceToHost) );
unflatten_array(u_eff,NXCUDA,NYCUDA);

HANDLE_ERROR( cudaMemcpy(mem_buffer,dev_u_eff_beta_eff,size_bytes,cudaMemcpyDeviceToHost) );
unflatten_array(u_eff_beta_eff,NXCUDA,NYCUDA);

HANDLE_ERROR( cudaMemcpy(mem_buffer,dev_beta_eff,size_bytes,cudaMemcpyDeviceToHost) );
unflatten_array(beta_eff,NXCUDA,NYCUDA);

}

void initial_calculations(void)
{
//A conventional division of work in blocks with 1024 (32 in x and 32
// 32 in y) threads each, is used in most of the calculations that are not using shared
//data or dimensions are less than 16 kB
dim3 blocks(XBLOCKS_PERGRID,YBLOCKS_PERGRID);
dim3 threads(XTHREADS_PERBLOCK,YTHREADS_PERBLOCK);

//A further division of work is carried out
//when using 1024 threads is not possible because shared data requirements
//are higher than 16 kB per kernel
dim3 blocks2(XBLOCKS_PERGRID*2,YBLOCKS_PERGRID*2);
dim3 threads2(XTHREADS_PERBLOCK/2,YTHREADS_PERBLOCK/2);

//Compute x component of source term
gsource<<<blocks,threads>>>(u_val,dev_sm_x, dev_mx, NXCUDA);

//Compute y component of source term
gsource<<<blocks,threads>>>(u_val,dev_sm_y, dev_my, NXCUDA);

//Compute z component of source term
gsource<<<blocks,threads>>>(u_val,dev_sm_z, dev_mz, NXCUDA);

//Project source term on magnetization components by computing
//a cross product twice
gm_x_source<<<blocks,threads>>>(dev_tempx, dev_tempy, dev_tempz,
				dev_mx, dev_my, dev_mz,
				dev_sm_x, dev_sm_y, dev_sm_z, NXCUDA);

gm_x_source<<<blocks,threads>>>(dev_tempx, dev_tempy, dev_tempz,
				dev_mx, dev_my, dev_mz,
				dev_sm_x, dev_sm_y, dev_sm_z, NXCUDA);

}


void rk4_integ(void)
{
//A conventional division of work in blocks with 1024 (32 in x and 32
// 32 in y) threads each, is used in most of the calculations that are not using shared
//data or dimensions are less than 16 kB
dim3 blocks(XBLOCKS_PERGRID,YBLOCKS_PERGRID);
dim3 threads(XTHREADS_PERBLOCK,YTHREADS_PERBLOCK);

//A further division of work is carried out
//when using 1024 threads is not possible because shared data requirements
//are higher than 16 kB per kernel
dim3 blocks2(XBLOCKS_PERGRID*2,YBLOCKS_PERGRID*2);
dim3 threads2(XTHREADS_PERBLOCK/2,YTHREADS_PERBLOCK/2);

gspinaccum_backup<<<blocks,threads>>>(dev_deltam_x, dev_deltam_y, dev_deltam_z,
                                dev_tempx, dev_tempy, dev_tempz,
                                NXCUDA);

//This call computes all the required terms to compute a solution
gsf_relaxation<<<blocks,threads>>>(tau_sf_val,dev_sfrelax_x, dev_sfrelax_y, dev_sfrelax_z,
		dev_deltam_x, dev_deltam_y, dev_deltam_z,
		NXCUDA);

gsd_exchange<<<blocks,threads>>>(tau_sd_val,dev_sdex_x, dev_sdex_y, dev_sdex_z,
				dev_deltam_x, dev_deltam_y, dev_deltam_z,
				dev_mx, dev_my, dev_mz,
			 	NXCUDA);

glaplacianx<<<blocks,threads>>>(dev_lapl_x, dev_lapl_y, dev_lapl_z,
				dev_d2adx2, dev_d2bdx2, dev_d2gdx2,
				dev_deltam_x, dev_deltam_y, dev_deltam_z,
			 	NXCUDA);

glaplacianyboundaries<<<blocks,threads>>>(dev_lapl_x, dev_lapl_y, dev_lapl_z,
				dev_d2ady2, dev_d2bdy2, dev_d2gdy2,
				dev_deltam_x, dev_deltam_y, dev_deltam_z,
			 	NXCUDA);
glaplaciany<<<blocks2,threads2>>>(dev_lapl_x, dev_lapl_y, dev_lapl_z,
				dev_d2ady2, dev_d2bdy2, dev_d2gdy2,
				dev_deltam_x, dev_deltam_y, dev_deltam_z,
			 	NXCUDA);

//This call evaluates dm / dt at t = n
gsolution<<<blocks,threads>>>(dev_sfrelax_x, dev_sfrelax_y, dev_sfrelax_z,
		dev_sm_x, dev_sm_y, dev_sm_z,
		dev_sdex_x, dev_sdex_y, dev_sdex_z,
		dev_lapl_x, dev_lapl_y, dev_lapl_z,
		dev_deltam_x, dev_deltam_y, dev_deltam_z,
		NXCUDA);

//This call evaluates the first term of the RK4 integrator
gterm1_RK4<<<blocks,threads>>>(dev_deltam_x, dev_deltam_y, dev_deltam_z,
                                dev_k1x, dev_k1y, dev_k1z,
                                dev_tempx, dev_tempy, dev_tempz,
                                NXCUDA);

//This call computes all the required terms to compute a solution
gsf_relaxation<<<blocks,threads>>>(tau_sf_val,dev_sfrelax_x, dev_sfrelax_y, dev_sfrelax_z,
		dev_deltam_x, dev_deltam_y, dev_deltam_z,
		NXCUDA);

gsd_exchange<<<blocks,threads>>>(tau_sd_val,dev_sdex_x, dev_sdex_y, dev_sdex_z,
				dev_deltam_x, dev_deltam_y, dev_deltam_z,
				dev_mx, dev_my, dev_mz,
			 	NXCUDA);

glaplacianx<<<blocks,threads>>>(dev_lapl_x, dev_lapl_y, dev_lapl_z,
				dev_d2adx2, dev_d2bdx2, dev_d2gdx2,
				dev_deltam_x, dev_deltam_y, dev_deltam_z,
			 	NXCUDA);

glaplacianyboundaries<<<blocks,threads>>>(dev_lapl_x, dev_lapl_y, dev_lapl_z,
				dev_d2ady2, dev_d2bdy2, dev_d2gdy2,
				dev_deltam_x, dev_deltam_y, dev_deltam_z,
			 	NXCUDA);

glaplaciany<<<blocks2,threads2>>>(dev_lapl_x, dev_lapl_y, dev_lapl_z,
				dev_d2ady2, dev_d2bdy2, dev_d2gdy2,
				dev_deltam_x, dev_deltam_y, dev_deltam_z,
			 	NXCUDA);

//This call evaluates (dm + 1/2 k1) / dt at t = dt * 1/2
gsolution<<<blocks,threads>>>(dev_sfrelax_x, dev_sfrelax_y, dev_sfrelax_z,
		dev_sm_x, dev_sm_y, dev_sm_z,
		dev_sdex_x, dev_sdex_y, dev_sdex_z,
		dev_lapl_x, dev_lapl_y, dev_lapl_z,
		dev_deltam_x, dev_deltam_y, dev_deltam_z,
		NXCUDA);

//This call evaluates the first term of the RK4 integrator
gterm2_RK4<<<blocks,threads>>>(dev_deltam_x, dev_deltam_y, dev_deltam_z,
                                dev_k2x, dev_k2y, dev_k2z,
                                dev_tempx, dev_tempy, dev_tempz,
                                NXCUDA);

//This call computes all the required terms to compute a solution
gsf_relaxation<<<blocks,threads>>>(tau_sf_val,dev_sfrelax_x, dev_sfrelax_y, dev_sfrelax_z,
		dev_deltam_x, dev_deltam_y, dev_deltam_z,
		NXCUDA);

gsd_exchange<<<blocks,threads>>>(tau_sd_val,dev_sdex_x, dev_sdex_y, dev_sdex_z,
				dev_deltam_x, dev_deltam_y, dev_deltam_z,
				dev_mx, dev_my, dev_mz,
			 	NXCUDA);

glaplacianx<<<blocks,threads>>>(dev_lapl_x, dev_lapl_y, dev_lapl_z,
				dev_d2adx2, dev_d2bdx2, dev_d2gdx2,
				dev_deltam_x, dev_deltam_y, dev_deltam_z,
			 	NXCUDA);

glaplacianyboundaries<<<blocks,threads>>>(dev_lapl_x, dev_lapl_y, dev_lapl_z,
				dev_d2ady2, dev_d2bdy2, dev_d2gdy2,
				dev_deltam_x, dev_deltam_y, dev_deltam_z,
			 	NXCUDA);

glaplaciany<<<blocks2,threads2>>>(dev_lapl_x, dev_lapl_y, dev_lapl_z,
				dev_d2ady2, dev_d2bdy2, dev_d2gdy2,
				dev_deltam_x, dev_deltam_y, dev_deltam_z,
			 	NXCUDA);

//This call evaluates (dm + 1/2 k2) / dt at t = dt * 1/2
gsolution<<<blocks,threads>>>(dev_sfrelax_x, dev_sfrelax_y, dev_sfrelax_z,
		dev_sm_x, dev_sm_y, dev_sm_z,
		dev_sdex_x, dev_sdex_y, dev_sdex_z,
		dev_lapl_x, dev_lapl_y, dev_lapl_z,
		dev_deltam_x, dev_deltam_y, dev_deltam_z,
		NXCUDA);

//This call evaluates the first term of the RK4 integrator
gterm3_RK4<<<blocks,threads>>>(dev_deltam_x, dev_deltam_y, dev_deltam_z,
                                dev_k3x, dev_k3y, dev_k3z,
                                dev_tempx, dev_tempy, dev_tempz,
                                NXCUDA);

//This call computes all the required terms to compute a solution
gsf_relaxation<<<blocks,threads>>>(tau_sf_val,dev_sfrelax_x, dev_sfrelax_y, dev_sfrelax_z,
		dev_deltam_x, dev_deltam_y, dev_deltam_z,
		NXCUDA);

gsd_exchange<<<blocks,threads>>>(tau_sd_val,dev_sdex_x, dev_sdex_y, dev_sdex_z,
				dev_deltam_x, dev_deltam_y, dev_deltam_z,
				dev_mx, dev_my, dev_mz,
			 	NXCUDA);

glaplacianx<<<blocks,threads>>>(dev_lapl_x, dev_lapl_y, dev_lapl_z,
				dev_d2adx2, dev_d2bdx2, dev_d2gdx2,
				dev_deltam_x, dev_deltam_y, dev_deltam_z,
			 	NXCUDA);

glaplacianyboundaries<<<blocks,threads>>>(dev_lapl_x, dev_lapl_y, dev_lapl_z,
				dev_d2ady2, dev_d2bdy2, dev_d2gdy2,
				dev_deltam_x, dev_deltam_y, dev_deltam_z,
			 	NXCUDA);

glaplaciany<<<blocks2,threads2>>>(dev_lapl_x, dev_lapl_y, dev_lapl_z,
				dev_d2ady2, dev_d2bdy2, dev_d2gdy2,
				dev_deltam_x, dev_deltam_y, dev_deltam_z,
			 	NXCUDA);

//This call evaluates (dm + 1/2 k2) / dt at t = dt * 1/2
gsolution<<<blocks,threads>>>(dev_sfrelax_x, dev_sfrelax_y, dev_sfrelax_z,
		dev_sm_x, dev_sm_y, dev_sm_z,
		dev_sdex_x, dev_sdex_y, dev_sdex_z,
		dev_lapl_x, dev_lapl_y, dev_lapl_z,
		dev_deltam_x, dev_deltam_y, dev_deltam_z,
		NXCUDA);

//This call evaluates the first term of the RK4 integrator
gterm4_RK4<<<blocks,threads>>>(dev_deltam_x, dev_deltam_y, dev_deltam_z,
                                dev_k1x, dev_k1y, dev_k1z,
                                dev_k2x, dev_k2y, dev_k2z,
                                dev_k3x, dev_k3y, dev_k3z,
                                dev_k4x, dev_k4y, dev_k4z,
                                dev_tempx, dev_tempy, dev_tempz,
                                NXCUDA);
}



int main (int argc, char *argv[])
{
int iteration = 0;
int iwrite = 0;
double sim_time = 0.0;
float elapsedTime;
FILE *out;

cudaEvent_t start,stop;
HANDLE_ERROR( cudaEventCreate( &start));
HANDLE_ERROR( cudaEventCreate( &stop));
//Check arguments for simulation
parse_args(argc,argv);

//Open file containing log data
outlog = fopen(logoutstr,"w");
fprintf(outlog,"Time(ns)\n");

initial_setup();

HANDLE_ERROR( cudaEventRecord( start,0 ));
initial_calculations();
do
	{
	//Increase counter
	iteration++;

	//If saving is enabled increase iteration in 1
	iwrite += writeflag;

	//Compute time in ns
	sim_time = (double)dt * (double)iteration;
	rk4_integ();
	if (iwrite == Nwrite)
		{
		printf("%20.15f\n",sim_time);
		fprintf(outlog,"%20.15f\n",sim_time);

		//Reset iwrite counte
		iwrite = 0;
		}
	}
while(tmax > sim_time);

HANDLE_ERROR( cudaEventRecord(stop));
HANDLE_ERROR( cudaEventSynchronize(stop));
HANDLE_ERROR( cudaEventElapsedTime(&elapsedTime, start, stop));

printf("Calculation completed in %3.1f ms\n", elapsedTime);
fprintf(outlog,"Calculation completed in %3.1f ms\n", elapsedTime);
printf("Simulated time %f\n", sim_time);
fprintf(outlog,"Simulated time %f\n", sim_time);

HANDLE_ERROR( cudaEventDestroy(start));
HANDLE_ERROR( cudaEventDestroy(stop));

update_CPUspinaccumulation();

effective_values();
//Save effective values

if ((out = fopen(effvalsfileoutstr,"w")) != NULL )
	{
	save_effectivevalues_data(out);
	fclose(out); //Close input file
	}
else
	{
	printf("Error: File %s doesn't exists!\n",effvalsfileoutstr);
	exit(1);
	}

//Check whether spin accumulation was computed or red from file
if(!readspinaccumulation)
	{
	//Save computed spin accumulation
	out = fopen(fileoutstr,"w"); //File containing original data
	save_spinaccumulation_data(out);
	fclose(out);
	}

fclose(outlog);
finalize();
return 0;
}
