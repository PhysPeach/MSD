//positionFile format
//diam1 diam2...
//
//t K U p1_x p1_y p1_vx p1_vy p2_x...
#include <cuda.h>
#include <iostream>
#include <fstream>
#include <sstream>

#define NT 1024
#define NB 512

typedef unsigned int uint;

template <typename T>
T pow_tmp(T x, int y){
	T xx = (T)1;
	if (y > 0){
		for (int i = 1; i <= y; i++) {
			xx *= x;
		}
	}
	else{
		for (int i = -1; i >= y; i--) {
			xx /= x;
		}
	}

	return xx;
}

//---settings---//
const uint D = 2;
uint ID;
uint NP;
double T;
double tmax;
//--------------//

const double pi = 4 * atan(1.0);
__global__ void xx0t(double *xx0t, double *x, double L, uint l, uint d){
    uint i_block = blockIdx.x;
    uint i_local = threadIdx.x;
    uint i_global = i_block * blockDim.x + i_local;
    double x12;
    for(uint i = i_global; i < l; i += NB*NT){
        x12 = x[i]-x[i%d];
        if(x12 < -0.5 * L) x12 += L;
        if(x12 > 0.5 * L) x12 -= L;
            
        xx0t[i] = x12 * x12;
    }
}
__global__ void reductionMsd(double *out, double *xx0t, uint l){
    uint i_block = blockIdx.x;
    uint i_local = threadIdx.x;
    uint i_global = i_block * blockDim.x + i_local;

    __shared__ double f[NT];

    uint remain, reduce;
    uint ib = i_block;
    for(uint i = i_global; i < l; i += NB*NT){
        f[i_local] = xx0t[i];
        __syncthreads();

        for(uint j = NT; j > 1; j = remain){
            reduce = j >> 1;
            remain = j - reduce;
            if((i_local < reduce) && (i + remain < l)){
                f[i_local] += f[i_local+remain];
            }
            __syncthreads();
        }
        if(i_local == 0){
            out[ib] = f[0];
        }
        __syncthreads();
        ib += NB;
    }
}

int main(int argc, char** argv){
    ID = atoi(argv[1]);
    NP = atoi(argv[2]);
	T = atof(argv[3]);
    uint timescale= atoi(argv[4]);
    tmax = pow_tmp(2., timescale);
    
    std::cout << "---settings---" << std::endl;
    std::cout << "ID: [1, " << ID << "]" << std::endl;
    std::cout << "D: " << D << std::endl;
    std::cout << "NP: " << NP << std::endl;
    std::cout << "T: " << T << std::endl;
    std::cout << "timescale: " << timescale << std::endl;
    std::cout << "--------------" << std::endl << std::endl;

    //Variables
    double *diam;
    double *x, *x_dev;
    double *xx0t_dev[2];
    double *t, *dt;
    diam = new double[NP];

    const double a0 = 1.;
    const double a1 = a0 * 1;
    const double a2 = a0 * 1.4;
    double dnsty = 0.8;
    double L = sqrt((double)NP/dnsty);

    uint Nt;

    //find dt, diam
    std::ostringstream positionName0;
	positionName0 << "../../pos/N"<< argv[2] << "/T" << argv[3] << "/posBD_N" << argv[2] << "_T" << argv[3] << "_id1.data";
	std::ifstream positionFile;
	positionFile.open(positionName0.str().c_str());
	std::cout << "Loading " << positionName0.str() << " for find dt, diam" << std::endl;
	double t1, t2, DA, DK, DU, DX, DV;
	for (uint n = 0; n < NP; n++) {
		positionFile >> diam[n];
	}
	positionFile >> t1 >> DK >> DU;
	for (int n = 0; n < NP; n++) {
        positionFile >> DX >> DX >> DV >> DV;
	}
	positionFile >> t2;
    positionFile.close();
    std::cout << "dt = " << t2-t1 << std::endl;
    Nt = 0;
	double ttmp = 10 * (t2-t1);
	while (ttmp < tmax) {
		ttmp *= 1.1;
		Nt++;
    }

    //newMemory
    x = new double[ID*Nt*NP*D];
    t = new double[Nt];
	dt = new double[Nt - 1];
    cudaMalloc((void**)&x_dev, ID * Nt * NP * D * sizeof(double));
    cudaMalloc((void**)&xx0t_dev[0], ID * Nt * NP * D * sizeof(double));
    cudaMalloc((void**)&xx0t_dev[1], ID * Nt * NP * D * sizeof(double));

    //loadFile
    for (short i = 0; i < ID; i++){
        // positionFile: t pi_x pi_y...
        std::ostringstream positionName;
        positionName << "../../pos/N"<< argv[2] << "/T" << argv[3];
        positionName << "/posBD_N" << argv[2] << "_T" << argv[3] << "_id" << i+1 << ".data";
	    positionFile.open(positionName.str().c_str());
		std::cout << "Loading " << positionName.str() << "..." << std::endl;
    	for (int n = 0; n < NP; n++) {
	    	positionFile >> DA;
    	}
		for (int nt = 0; nt < Nt; nt++){
    			positionFile >> t[nt] >> DK >> DU;
			for (int n = 0; n < NP; n++){
                positionFile >> x[nt*ID*NP*D + i*NP*D + n*D];
                positionFile >> x[nt*ID*NP*D + i*NP*D + n*D + 1];
                positionFile >> DV >> DV;
			}
		}
		positionFile.close();
		std::cout << " -> done" << std::endl;
    }
    cudaMemcpy(x_dev, x, Nt*ID*NP*D * sizeof(double), cudaMemcpyHostToDevice);

	for (short nt = 1; nt <= Nt - 1; nt++){
		dt[nt - 1] = t[nt] - t[0];
    }
    
    //analise
    std::cout << "Recording msd_N" << argv[2] << "_T_" << argv[3] << "..." << std::endl;
    std::ostringstream msdName;
	msdName << "./data/msd_N" << argv[2] << "_T" << argv[3] << ".data";
	std::ofstream msdFile;
    msdFile.open(msdName.str().c_str());
    double *msd;
    msd = new double[Nt-1];

    xx0t<<<NB,NT>>>(xx0t_dev[0], x_dev, L, Nt*ID*NP*D, ID*NP*D);
    uint flip;
    for(uint nt = 1; nt <= Nt - 1; nt++){
        flip = 0;
        for(uint l = ID*NP*D; l > 1; l = (l+NT-1)/NT){
            flip = !flip;
            reductionMsd<<<NB,NT>>>(&xx0t_dev[flip][nt*ID*NP*D], &xx0t_dev[!flip][nt*ID*NP*D], l);
        }
        cudaMemcpy(&msd[nt - 1], &xx0t_dev[flip][nt*ID*NP*D], sizeof(double), cudaMemcpyDeviceToHost);
    }
    for (int nt = 1; nt <= Nt - 1; nt++){
		msdFile << dt[nt - 1] << " " << msd[nt - 1]/(ID*NP) << std::endl;
    }
    delete[] msd;
    msdFile.close();
    
    
    //deleteMemory
    delete[] diam;
    delete[] x;
    delete[] t;
    delete[] dt;
    cudaFree(x_dev);
    cudaFree(xx0t_dev[0]);
    cudaFree(xx0t_dev[1]);
    return 0;
}