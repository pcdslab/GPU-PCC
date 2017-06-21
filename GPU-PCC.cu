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
#include<math.h>
#include <thrust/scan.h>
#define PER 1
#define div 16
using namespace std;
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void ker(float * a, float * c,int n, int m,long long so_far)
{
	float sum=0;
	float2 f1,f2;
        long i,j,tempj,tempi,iindex,jindex;//check with long long as well
	int group_id=threadIdx.x/div;
	int group_index=threadIdx.x%div;
	int group = blockDim.x/div;
	long long indexx=(blockIdx.x*group + group_id);
	indexx+=so_far;
	long long t_0 =4*n;
	t_0 = t_0 * (n-1) - 7;
 	t_0 =floor((sqrt((double)(((-8.0)* indexx)+t_0)))/2.0 - 0.5);
 	i = n-2-t_0;//index i (based on equation 5 of the paper)
	t_0 = n-1;
        t_0 *=n;
        t_0 /=2;
 	j = indexx+ i + 1 -  t_0 + (n-i)*((n-i)-1)/2; //index j (based on equation 6 of the paper)

	int m_prime=m/32;
	int temp=0,temp1=0;
	
	int flag=1;
	if(i>=(n-1) ||j>=n)
		flag=0;

		if(flag==1)
        	{
	        int tt=m_prime*32;
        	int remain=(m-tt);
                 iindex= i*(m/2);
                 jindex= j*(m/2);
                 tempj = jindex + group_index;
                 tempi = iindex + group_index;	
                       	 sum = 0;
               		 temp1=0;
			#pragma unroll
	                for(temp = 0 ; temp<m_prime ; temp++)
        	        {
				f1=reinterpret_cast<float2*>(a)[tempi+temp1];
				f2=reinterpret_cast<float2*>(a)[tempj+temp1];		
                        	sum+= (f1.x * f2.x);
				sum+= (f1.y * f2.y);
				temp1 += div;
	                }
	
			if(m_prime*32!=m)//in case length og time series is not multiple of 32
                        {
	                               if(group_index*2<remain)
                                        {
                                             f1=reinterpret_cast<float2*>(a)[tempi+temp1];
		                             f2=reinterpret_cast<float2*>(a)[tempj+temp1];
  		                             sum += (f1.x * f2.x);
					     sum += (f1.y * f2.y);
                                        }
			}


		
	 	
		#pragma unroll   //computing global sum
       		 for(temp =8;temp>0;temp/=2)
	        {
        	       sum += __shfl_down(sum,temp);
	        }
		
        	if(group_index==0&&indexx<t_0)
	        {        
			c[indexx-so_far]= sum;
	        }
	}



}
int main(int argc, char *argv[])
{
  
    clock_t t1,t3,test1,test2;
  
    int n,m,flag=0;
    n = atoi(argv[1]);
    m = atoi(argv[2]);
   
    if(m%2==1)
	
	{
		flag=1;
		m++;
	}
 
    size_t free;
    size_t total;
    cudaMemGetInfo(&free,&total);
    long long available_mem = free;//getting available memory space of GPU
    available_mem-=(sizeof(float)*m*n);
    available_mem/=sizeof(float);
    available_mem*=4;
    available_mem/=5;//computing 80% of empty memory space in gpu
   
    long long cor_size = n-1;
    cor_size *=n;
    cor_size /=2;
    cout<<" \n Correlation size is: \n"<<cor_size<<"\n\n";

    float * bold1 = new float [m*n];
    memset(bold1, 0, sizeof(float)*m*n);
    float * bold3 = new float [cor_size];
    memset(bold3, 0, sizeof(float) *cor_size);	
    string name = "/home/taban/Correlation_codes/code_wang/100000_500/";//path to dataset
    stringstream sstm;
    ifstream myReadFile;
    sstm.str("");
    sstm << name<<"data1.txt";//name of dataset
    string ad = sstm.str();
    myReadFile.open(ad.c_str());
    int i,j;
    if(flag==0)
    {
       for (i = 0; i< n; i++){
           for(j = 0; j <m;j++)
           {
               myReadFile>>bold1[i*m+j];
           }
	
       }
    }
    if(flag==1)//If the length of time series is not multiple of 2, a 0 will be added to the end 
    {
        for ( i = 0; i< n; i++){
            for(j = 0; j <m-1;j++)
            {
                 myReadFile>>bold1[i*m+j];

	    }
       	bold1[i*m+j]=0;
       }
   }
    myReadFile.close();
   
 t1 = clock(); //timing starts here

if(flag==0){ //normalizing when length of vector is even
    for (int i = 0; i < n; i++)
    {
        float * row = bold1 + i * m;
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
}

if(flag==1)// normalizing when the length of vector is odd
{ 

 for (int i = 0; i < n; i++)
    {
        float * row = bold1 + i * m;
        double sum1 = 0, sum2 = 0;
        for (int l = 0; l < m-1; l++)
        {
            sum1 += row[l];
        }
        sum1 /= (m-1);
        for (int l = 0; l < (m-1); l++)
        {
            sum2 += (row[l] - sum1) * (row[l] - sum1);
        }

        sum2 = sqrt(sum2);

        for (int l = 0; l < (m-1); l++)
        {
            if(sum2!=0)
                row[l] = (row[l] - sum1) / sum2;
            if(sum2==0)
                row[l] = 0;
        }
    }



}

	float * dev_bold1,*dev_bold3;
	cudaMalloc ((void**)&dev_bold1,sizeof(float)*m*n);
	cudaMemcpy( dev_bold1, bold1, sizeof(float) * m* n, cudaMemcpyHostToDevice);
	if(cor_size<available_mem)
                cudaMalloc ((void**)&dev_bold3,sizeof(float)*cor_size);
        if(cor_size>=available_mem)
                cudaMalloc ((void**)&dev_bold3,sizeof(float)*available_mem);
	int block_size=512;
	int group = block_size/div;
        int iter_num = cor_size/available_mem;
	if(iter_num==0)
	{
		long long grid_size = 1+((cor_size-1)/group);  
		ker<<<grid_size,block_size>>>(dev_bold1,dev_bold3,n,m,0);
	        cudaDeviceSynchronize();
		gpuErrchk( cudaPeekAtLastError() );
	         gpuErrchk(cudaMemcpy(bold3, dev_bold3, sizeof(float)*cor_size, cudaMemcpyDeviceToHost));
	}
    else if(iter_num>0)
	{
	     long long smaller,so_far=0,temp_cor_size=cor_size;
    	     for(int index_iter=0;index_iter<iter_num+1;index_iter++)
    	     {
		cout<<"\n"<<index_iter;
        	 if(temp_cor_size>=available_mem)
	         smaller = available_mem;
	         if(temp_cor_size<available_mem)
        	 smaller = temp_cor_size;
	         long grid_size = 1+((smaller-1)/group);
	 	 ker<<<grid_size,block_size>>>(dev_bold1,dev_bold3,n,m,so_far);
	       	 cudaDeviceSynchronize();
		 gpuErrchk( cudaPeekAtLastError() );
		 cudaMemcpy(bold3+so_far, dev_bold3, sizeof(float)*(smaller), cudaMemcpyDeviceToHost);
        	 temp_cor_size -= available_mem;
         	 so_far+=smaller;
    	     }
	}
	t3 = clock();
	cout<<"elapsed time is: \n"<<(double)(t3-t1)/CLOCKS_PER_SEC ;

	cudaFree(dev_bold1);
	cudaFree(dev_bold3);
	delete[]bold1;
	delete[]bold3;
	return 0;
}


