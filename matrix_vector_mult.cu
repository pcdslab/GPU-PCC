/*
Copyright (C)Taban Eslami and Muaaz Gul Awan and Fahad Saeed  
This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/
#include<iostream>
#include<cuda_profiler_api.h>
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "memory.h"
#include <cuda_runtime_api.h>
#include <ctime>
#include<stdio.h>
#include<stdlib.h>
#include <string.h>
#include <iomanip>
#include"cuda.h"
#include "dirent.h"
#include <fstream>
#include <stack>
#include<sstream>
using namespace std;
int main(int argc, char *argv[])
{

clock_t t1,t2;
int  m ,n ,k;
n = atoi(argv[1]);
m = atoi(argv[2]);
long long cor_size = n-1;
cor_size *=n;
cor_size /=2;
cout<<" \n ---corsize---\n"<<cor_size<<"\n\n";
float * bold1 = new float [m*n];
float * bold3 = new float [cor_size];

string name = "/home/taban/Correlation_codes/code_wang/100000_500/";
stringstream sstm; 
ifstream myReadFile;
sstm.str("");
sstm << name <<"data1"<<".txt";
string ad = sstm.str();
myReadFile.open(ad.c_str());
int l;
for (  k = 0; k < n; k++)

   for ( l = 0; l < m; l++)

       {
        myReadFile>>bold1[l*n+k];
        }
/////////////////////Pre-processing/////////////////////////

t1=clock();
float* BOLD_t = new float [n*m];
for (int i = 0; i < m; i ++)

                for (int j = 0; j < n; j++)

                {

                        BOLD_t[j * m + i] = bold1[i * n + j];

               }

	for (int i = 0; i < n; i++)

                {

                        float* row = BOLD_t + i * m;

                        double sum1 = 0, sum2 = 0;

                        for (int l = 0; l < m; l++)

                        {

                                sum1 += row[l];

                        }

                        sum1 /= m;

                        for (int l = 0; l < m; l++)

                        {

                                sum2 += (row[l] - sum1) * (row[l] - sum1);

                        }

                                   sum2 = sqrt(sum2);
	        for (int l = 0; l < m; l++)
		        {
	        	    if(sum2!=0)
                		row[l] = (row[l] - sum1) / sum2;
		            if(sum2==0)
        		        row[l] = 0;
		        }

                }


////////////////////////////////////////////////////////////
cudaError_t cudaStat;
float * dev_bold1,*dev_bold3;
cublasStatus_t stat;
cublasHandle_t handle;
stat = cublasCreate(&handle) ;
const float alpha = 1.0;
const float beta = 0;
cudaStat = cudaMalloc ((void**)&dev_bold1,sizeof(float)*m*n);
cudaMemcpy( dev_bold1, BOLD_t, sizeof(float) * m  * n, cudaMemcpyHostToDevice);

////////////////getting size of 80% of pgu memory
 size_t free;
    size_t total;
    cudaMemGetInfo(&free,&total);
    long long available_mem = free;
    available_mem-=(sizeof(float)*m*n);
    available_mem/=sizeof(float);
    available_mem*=4;
    available_mem/=5;
 if(cor_size<available_mem)
                cudaMalloc ((void**)&dev_bold3,sizeof(float)*cor_size);
        if(cor_size>=available_mem)
                cudaMalloc ((void**)&dev_bold3,sizeof(float)*available_mem);
////////////////
long long so_far=0,so_far_temp=0;
int iter_num = cor_size/available_mem;
float * out3 = dev_bold3;
if(iter_num==0)
{
	for(int i=0;i<n;i++)
	{
		stat = cublasSgemv(handle,CUBLAS_OP_T,m,(n-(i+1)),&alpha, dev_bold1+((i+1)*m), m, dev_bold1+(i*m),1, &beta,out3,1);
		out3+=(n-(i+1));
	}
	cudaDeviceSynchronize();
	cudaMemcpy(bold3,dev_bold3, sizeof(float)*(cor_size), cudaMemcpyDeviceToHost);
}
  else if(iter_num>0)
        {
		 for(int i=0;i<n;i++)
        {
		if((so_far_temp+(n-(i+1)))>available_mem)
			{
			cout<<" *********"<<so_far_temp<<"\n";
			cudaMemcpy(bold3+so_far,dev_bold3, sizeof(float)*(so_far_temp), cudaMemcpyDeviceToHost);
			so_far+=so_far_temp;
			so_far_temp=0;
			out3=dev_bold3;
			}
                stat = cublasSgemv(handle,CUBLAS_OP_T,m,(n-(i+1)),&alpha, dev_bold1+((i+1)*m), m, dev_bold1+(i*m),1, &beta,out3,1);
                out3+=(n-(i+1));
	//	so_far+=(n-(i+1));
		so_far_temp+=(n-(i+1));
	
        }
		if(so_far_temp>0)
		{
		 cudaMemcpy(bold3+so_far,dev_bold3, sizeof(float)*(so_far_temp), cudaMemcpyDeviceToHost);
		}
		
	}
t2= clock();

cout<<"time:   "<<double(t2-t1)/CLOCKS_PER_SEC;


cudaFree(dev_bold1);
cudaFree(dev_bold3);
delete[]bold1;
delete[]bold3;
delete[]BOLD_t;
myReadFile.close();
//cudaProfilerStop();
cublasDestroy(handle);

return 0;
}
