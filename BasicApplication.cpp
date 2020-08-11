// BasicOpenCLApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Chrono.h"
#include <math.h>
#include <iostream>
#include <immintrin.h>
#include <vector>
#include <thread>
#include <mutex>

#define NB_THREADS  4
#define float8 __m128
#define mul8(a,b) _mm256_mul_ps(a,b)
#define sub8(a,b) _mm256_sub_ps(a,b)
#define add8(a,b) _mm256_add_ps(a,b)
#define div8(a,b) _mm256_div_ps(a,b)
#define set8(x) _mm256_set1_ps(x)
#define rcp8(a) _mm256_rcp_ps(a)


void SaveBMP(char *fname, unsigned char *image, int width, int height, int componentPerPixel=1, int reverseColor=0)
{
	FILE *destination;
    int i,j;
	int *pt;
	char name[512],hdr[0x36];
	unsigned char *imsource=new unsigned char [width*height*3];
	//int al=(ImageSize*3)%4;
	
	if (componentPerPixel==1)
		for (i=0;i<width*height*3;i++)
			imsource[i]=image[i/3];
	else 
		for (i=0;i<width*height*3;i++)
			imsource[i]=image[i];
	if (reverseColor)
		for (j=0;j<height;j++)
			for (i=0;i<width;i++)
			{
				unsigned char aux;
				aux=imsource[3*(i+width*j)];
				imsource[3*(i+width*j)]=imsource[3*(i+width*j)+2];
				imsource[3*(i+width*j)+2]=aux;
			}
	strcpy(name,fname);
	i=(int)strlen(name);
	if (!((i>4)&&(name[i-4]=='.')&&(name[i-3]=='b')&&(name[i-2]=='m')&&(name[i-1]=='p')))
	{
		name[i]='.';
		name[i+1]='b';
		name[i+2]='m';
		name[i+3]='p';
		name[i+4]=0;
	}
	if ((destination=fopen(name, "wb"))==NULL) 
		perror("erreur de creation de fichier\n");
    hdr[0]='B';
    hdr[1]='M';
	pt=(int *)(hdr+2);// file size
	*pt=0x36+width*height*3;
	pt=(int *)(hdr+6);//reserved
	*pt=0x0;
	pt=(int *)(hdr+10);// image address
	*pt=0x36;
	pt=(int *)(hdr+14);// size of [0E-35]
	*pt=0x28;
	pt=(int *)(hdr+0x12);// Image width
	*pt=width;
	pt=(int *)(hdr+0x16);// Image heigth
	*pt=height;
	pt=(int *)(hdr+0x1a);// color planes
	*pt=1;
	pt=(int *)(hdr+0x1c);// bit per pixel
	*pt=24;
	for (i=0x1E;i<0x36;i++) 
		hdr[i]=0;
	fwrite(hdr,0x36,1,destination);
	fwrite (imsource,width*height*3,1,destination);
    fclose(destination);
	delete[] imsource;
}

typedef struct { float real; float im; } complex;

complex add(complex a, complex b)
{
	complex res;
	res.real = a.real + b.real;
	res.im = a.im + b.im;
	return res;
}
complex sub(complex a, complex b)
{
	complex res;
	res.real = a.real - b.real;
	res.im = a.im - b.im;
	return res;
}

complex mul(complex a, complex b)
{
	complex res;
	res.real = a.real*b.real - a.im*b.im;
	res.im = a.real*b.im + a.im*b.real;
	return res;
}

float squaredNorm(complex c)
{
	return c.real*c.real + c.im*c.im;
}

int Iterate(complex c)
{
	const int max_iterations = 255;
	complex z,a,l;
	a.real = 0.91;
	a.im = 0.;
	z = c;
	l.real = 4;
	l.im = 0;
	int i = 1;
	while (i < max_iterations)
	{
		z = mul(l, mul(z,sub(a,z)));

		if (squaredNorm(z) > 128)
			break;
		i += 2;
	}
	return (min(i, max_iterations));
}


__m256 Iterate_SIMD(__m256 SetCReal, __m256 SetCIm)
{
	//Setting up relevant variable
	const int max_iterations = 255;
	__m256 SetAReal = _mm256_set1_ps(0.91);
	__m256 SetAIm = _mm256_set1_ps(0.);

	__m256 SetZReal = SetCReal;
	__m256 SetZIm = SetCIm;
	__m256 Answer = _mm256_set1_ps(0.);

	__m256 SetLReal = _mm256_set1_ps(4.);
	__m256 SetLIm = _mm256_set1_ps(0.);

	__m256 MaxIt = _mm256_set1_ps(max_iterations);


	int i = 1;
	int a = 1;
	int b = 1;
	int c = 1;
	int d = 1;
	int e = 1;
	int f = 1;
	int g = 1;

	//Batch Process 

	while (i < max_iterations || a < max_iterations || b < max_iterations || c < max_iterations || d < max_iterations || e < max_iterations || f < max_iterations || g < max_iterations)	//Computes the initial batch together
	{
		//The next part of the equation will need sub to already exist so we seperate out the equation into multiple parts.
		__m256 tempSubR = sub8(SetAReal,SetZReal);
		__m256 tempSubIm = sub8(SetAIm, SetZIm);

		//replicating the function of the multiply method
		__m256 tempMulR = sub8(mul8(SetZReal, tempSubR), mul8(SetZIm, tempSubIm));
		__m256 tempMulIm = add8(mul8(SetZReal, tempSubIm), mul8(SetZIm, tempSubR));

		SetZReal = sub8(mul8(SetLReal, tempMulR ),mul8(SetLIm, tempMulIm));

		SetZIm = add8(mul8(SetLReal, tempMulIm), mul8(SetLIm , tempMulR));

		//Itterate for each pixel
		if ((SetZReal.m256_f32[0] * SetZReal.m256_f32[0] + SetZIm.m256_f32[0] * SetZIm.m256_f32[0]) > 128)
			break;
		i += 2;
		

		if ((SetZReal.m256_f32[1] * SetZReal.m256_f32[1] + SetZIm.m256_f32[1] * SetZIm.m256_f32[1]) > 128)
			break;
		a += 2;

		if ((SetZReal.m256_f32[2] * SetZReal.m256_f32[2] + SetZIm.m256_f32[2] * SetZIm.m256_f32[2]) > 128)
			break;
		b += 2;

		if ((SetZReal.m256_f32[3] * SetZReal.m256_f32[3] + SetZIm.m256_f32[3] * SetZIm.m256_f32[3]) > 128)
			break;
		c += 2;

		if ((SetZReal.m256_f32[4] * SetZReal.m256_f32[4] + SetZIm.m256_f32[4] * SetZIm.m256_f32[4]) > 128)
			break;
		d += 2;

		if ((SetZReal.m256_f32[5] * SetZReal.m256_f32[5] + SetZIm.m256_f32[5] * SetZIm.m256_f32[5]) > 128)
			break;
		e += 2;

		if ((SetZReal.m256_f32[6] * SetZReal.m256_f32[6] + SetZIm.m256_f32[6] * SetZIm.m256_f32[6]) > 128)
			break;
		f += 2;

		if ((SetZReal.m256_f32[7] * SetZReal.m256_f32[7] + SetZIm.m256_f32[7] * SetZIm.m256_f32[7]) > 128)
			break;
		g += 2;
		
	}

	//Now we need to keep track of seperate variables, we could continue batch processing but there would come
	// some point of trade-off per say, as were always operating on 8 pixels at a time if we were to continue batch
	// processing we would be performing calculations on some position within our set that no longer needs operation on, for example
	// if iterator i were to be the first to reach 255 and we were to continue batch procesing for the next 7 iterators then we no longer need to perform a calculation on
	// the spot corresponding to i in our set of 8. As each condition variable gets bigger than max iterations we get more unneeded calculations.
	// As such we stop batch procesing and calculate for each condition variable in place.
	//
	// The start value is unknown and can be uneven, therefore we dont unroll these while loops
	

	//Same as the batch process but for a single space
		while (i < max_iterations) {
			float tempSubR;
			float tempSubIm;
			float tempMulR;
			float tempMulIm;

			tempSubR = (0.91 - SetZReal.m256_f32[0]);
			tempSubIm = (0 - SetZIm.m256_f32[0]);

			tempMulR = SetZReal.m256_f32[0] * tempSubR - SetZIm.m256_f32[0] * tempSubIm;
			tempMulIm = SetZReal.m256_f32[0] * tempSubIm + SetZIm.m256_f32[0] * tempSubR;

			SetZReal.m256_f32[0] = 4 * tempMulR - 0 * tempMulIm;

			SetZIm.m256_f32[0] = 4 * tempMulIm + 0 * tempMulR;

			complex zE;
			zE.real = SetZReal.m256_f32[0];
			zE.im = SetZIm.m256_f32[0];
			if (squaredNorm(zE) > 128)
				break;
			i += 2;

		}
		Answer.m256_f32[0] = i;

		while (a < max_iterations) {
			float tempSubR;
			float tempSubIm;
			float tempMulR;
			float tempMulIm;

			tempSubR = (SetAReal.m256_f32[1] - SetZReal.m256_f32[1]);
			tempSubIm = (SetAIm.m256_f32[1] - SetZIm.m256_f32[1]);

			tempMulR = SetZReal.m256_f32[1] * tempSubR - SetZIm.m256_f32[1] * tempSubIm;
			tempMulIm = SetZReal.m256_f32[1] * tempSubIm + SetZIm.m256_f32[1] * tempSubR;

			SetZReal.m256_f32[1] = SetLReal.m256_f32[1] * tempMulR - SetLIm.m256_f32[1] * tempMulIm;
			SetZIm.m256_f32[1] = SetLReal.m256_f32[1] * tempMulIm + SetLIm.m256_f32[1] * tempMulR;

			complex zE;
			zE.real = SetZReal.m256_f32[1];
			zE.im = SetZIm.m256_f32[1];
			if (squaredNorm(zE) > 128)
				break;
			a += 2;

		}
		Answer.m256_f32[1] = a;

		while (b < max_iterations) {
			float tempSubR;
			float tempSubIm;
			float tempMulR;
			float tempMulIm;

			tempSubR = (SetAReal.m256_f32[2] - SetZReal.m256_f32[2]);
			tempSubIm = (SetAIm.m256_f32[2] - SetZIm.m256_f32[2]);

			tempMulR = SetZReal.m256_f32[2] * tempSubR - SetZIm.m256_f32[2] * tempSubIm;
			tempMulIm = SetZReal.m256_f32[2] * tempSubIm + SetZIm.m256_f32[2] * tempSubR;

			SetZReal.m256_f32[2] = SetLReal.m256_f32[2] * tempMulR - SetLIm.m256_f32[2] * tempMulIm;
			SetZIm.m256_f32[2] = SetLReal.m256_f32[2] * tempMulIm + SetLIm.m256_f32[2] * tempMulR;

			complex zE;
			zE.real = SetZReal.m256_f32[2];
			zE.im = SetZIm.m256_f32[2];
			if (squaredNorm(zE) > 128)
				break;
			b += 2;
		}
		Answer.m256_f32[2] = b;


		while (c < max_iterations) {
			float tempSubR;
			float tempSubIm;
			float tempMulR;
			float tempMulIm;

			tempSubR = (SetAReal.m256_f32[3] - SetZReal.m256_f32[3]);
			tempSubIm = (SetAIm.m256_f32[3] - SetZIm.m256_f32[3]);

			tempMulR = SetZReal.m256_f32[3] * tempSubR - SetZIm.m256_f32[3] * tempSubIm;
			tempMulIm = SetZReal.m256_f32[3] * tempSubIm + SetZIm.m256_f32[3] * tempSubR;

			SetZReal.m256_f32[3] = SetLReal.m256_f32[3] * tempMulR - SetLIm.m256_f32[3] * tempMulIm;
			SetZIm.m256_f32[3] = SetLReal.m256_f32[3] * tempMulIm + SetLIm.m256_f32[3] * tempMulR;

			complex zE;
			zE.real = SetZReal.m256_f32[3];
			zE.im = SetZIm.m256_f32[3];
			if (squaredNorm(zE) > 128)
				break;
			c += 2;

		}
		Answer.m256_f32[3] = c;


		while (d < max_iterations) {
			float tempSubR;
			float tempSubIm;
			float tempMulR;
			float tempMulIm;

			tempSubR = (SetAReal.m256_f32[4] - SetZReal.m256_f32[4]);
			tempSubIm = (SetAIm.m256_f32[4] - SetZIm.m256_f32[4]);

			tempMulR = SetZReal.m256_f32[4] * tempSubR - SetZIm.m256_f32[4] * tempSubIm;
			tempMulIm = SetZReal.m256_f32[4] * tempSubIm + SetZIm.m256_f32[4] * tempSubR;

			SetZReal.m256_f32[4] = SetLReal.m256_f32[4] * tempMulR - SetLIm.m256_f32[4] * tempMulIm;

			SetZIm.m256_f32[4] = SetLReal.m256_f32[4] * tempMulIm + SetLIm.m256_f32[4] * tempMulR;
			complex zE;
			zE.real = SetZReal.m256_f32[4];
			zE.im = SetZIm.m256_f32[4];
			if (squaredNorm(zE) > 128)
				break;
			d += 2;

		}
		Answer.m256_f32[4] = d;


		while (e < max_iterations) {
			float tempSubR;
			float tempSubIm;
			float tempMulR;
			float tempMulIm;

			tempSubR = (SetAReal.m256_f32[5] - SetZReal.m256_f32[5]);
			tempSubIm = (SetAIm.m256_f32[5] - SetZIm.m256_f32[5]);

			tempMulR = SetZReal.m256_f32[5] * tempSubR - SetZIm.m256_f32[5] * tempSubIm;
			tempMulIm = SetZReal.m256_f32[5] * tempSubIm + SetZIm.m256_f32[5] * tempSubR;

			SetZReal.m256_f32[5] = SetLReal.m256_f32[5] * tempMulR - SetLIm.m256_f32[5] * tempMulIm;
			SetZIm.m256_f32[5] = SetLReal.m256_f32[5] * tempMulIm + SetLIm.m256_f32[5] * tempMulR;

			complex zE;
			zE.real = SetZReal.m256_f32[5];
			zE.im = SetZIm.m256_f32[5];
			if (squaredNorm(zE) > 128)
				break;
			e += 2;

		}
		Answer.m256_f32[5] = e;

		while (f < max_iterations) {
			float tempSubR;
			float tempSubIm;
			float tempMulR;
			float tempMulIm;

			tempSubR = (SetAReal.m256_f32[6] - SetZReal.m256_f32[6]);
			tempSubIm = (SetAIm.m256_f32[6] - SetZIm.m256_f32[6]);

			tempMulR = SetZReal.m256_f32[6] * tempSubR - SetZIm.m256_f32[6] * tempSubIm;
			tempMulIm = SetZReal.m256_f32[6] * tempSubIm + SetZIm.m256_f32[6] * tempSubR;

			SetZReal.m256_f32[6] = SetLReal.m256_f32[6] * tempMulR - SetLIm.m256_f32[6] * tempMulIm;
			SetZIm.m256_f32[6] = SetLReal.m256_f32[6] * tempMulIm + SetLIm.m256_f32[6] * tempMulR;

			complex zE;
			zE.real = SetZReal.m256_f32[6];
			zE.im = SetZIm.m256_f32[6];
			if (squaredNorm(zE) > 128)
				break;
			f += 2;

		}
		Answer.m256_f32[6] = f;


		while (g < max_iterations) {
			float tempSubR;
			float tempSubIm;
			float tempMulR;
			float tempMulIm;

			tempSubR = (SetAReal.m256_f32[7] - SetZReal.m256_f32[7]);
			tempSubIm = (SetAIm.m256_f32[7] - SetZIm.m256_f32[7]);

			tempMulR = SetZReal.m256_f32[7] * tempSubR - SetZIm.m256_f32[7] * tempSubIm;
			tempMulIm = SetZReal.m256_f32[7] * tempSubIm + SetZIm.m256_f32[7] * tempSubR;

			SetZReal.m256_f32[7] = SetLReal.m256_f32[7] * tempMulR - SetLIm.m256_f32[7] * tempMulIm;
			SetZIm.m256_f32[7] = SetLReal.m256_f32[7] * tempMulIm + SetLIm.m256_f32[7] * tempMulR;

			complex zE;
			zE.real = SetZReal.m256_f32[7];
			zE.im = SetZIm.m256_f32[7];
			if (squaredNorm(zE) > 128)
				break;
			g += 2;

		}
		Answer.m256_f32[7] = g;

		//return min between our itterators and max itterations
		__m256 Answer2 = _mm256_min_ps(Answer, MaxIt);

		return (Answer2);
	}


void SimpleFractalDrawing(unsigned char *image, int dim[2],float range[2][2])
{
	for (int j=0;j<dim[1];j++)
		for (int i=0;i<dim[0];i++)
		{
			complex c;
			c.real=range[0][0]+(i+0.5)*(range[0][1]-range[0][0])/dim[0]; //Create x coordinates within the range [range[0][0] .. range[0][1]] 
			c.im=range[1][0]+(j+0.5)*(range[1][1]-range[1][0])/dim[1]; //Create x coordinates within the range [range[1][0] .. range[1][1]] 

			float f = 2 * Iterate(c);
			if (f > 255.)
				f = 255.;
			image[j*dim[0]+i]=f; 
		
		}
}
void SimpleFractalDrawing_SIMD(unsigned char* image, int dim[2], float range[2][2])
{
	//Setup of needed variables
	__m256 SetRange00 = _mm256_set1_ps(range[0][0]);
	__m256 SetRange01 = _mm256_set1_ps(range[0][1]);
	__m256 SetRange10 = _mm256_set1_ps(range[1][0]);
	__m256 SetRange11 = _mm256_set1_ps(range[1][1]);
	__m256 SetDim = _mm256_set1_ps(dim[1]);
	__m256 SetHalf = _mm256_set1_ps(0.5);

	for (int j = 0; j < dim[1]; j++) //For each y
	{
		int actualX = 0;
		int xIncrement = 0;
		__m256 SetY = _mm256_set1_ps(j); //create new set of 8


		//Threads fill here
		for (int Xi = 0; Xi < dim[0]; Xi += 8) //For each 8 x
		{
			__m256 SetX;

			for (int i = 0; i < 8; i++)	//Store next 8 pixels
			{
				SetX.m256_f32[i] = xIncrement;
				xIncrement++;
			}
			//Call our operations
			__m256 SetCReal = add8(mul8(add8(SetX, SetHalf), div8(sub8(SetRange01, SetRange00), SetDim)), SetRange00);
			__m256 SetCIm = add8(mul8(add8(SetY, SetHalf), div8(sub8(SetRange11, SetRange10), SetDim)), SetRange10);
			__m256 Double = _mm256_set1_ps(2);
			__m256 SetF = mul8(Double, Iterate_SIMD(SetCReal, SetCIm));

			//For each in set of 9
			for (int i = 0; i < 8; i++)	
			{
				float f = SetF.m256_f32[i];
				if (SetF.m256_f32[i] > 255.)
					f = 255.;
				image[j * dim[0] + actualX] = f;
				actualX++; //For position in image
			}	
		}
	}
}
unsigned char* imageSIMDMT = new unsigned char[1024 * 1024];

void SimpleFractalDrawing_SIMD_MTAux(int start, int end) {
	//setting up needed variables
	
	int dimMT[2] = { 1024,1024 };
	float rangeMT[2][2] = { {-0.003,0.008},{-0.0002,0.0005} };
	__m256 SetRange00 = _mm256_set1_ps(rangeMT[0][0]);
	__m256 SetRange01 = _mm256_set1_ps(rangeMT[0][1]);
	__m256 SetRange10 = _mm256_set1_ps(rangeMT[1][0]);
	__m256 SetRange11 = _mm256_set1_ps(rangeMT[1][1]);
	__m256 SetDim = _mm256_set1_ps(dimMT[1]);
	__m256 SetHalf = _mm256_set1_ps(0.5);

	for (int j = start; j < end; j++) //For each y
	{
		int actualX = 0;
		int xIncrement = 0;
		__m256 SetY = _mm256_set1_ps(j); //New set of y relevant to each x
		

		for (int Xi = 0; Xi < dimMT[0]; Xi += 8) //For each 8 x
		{
			__m256 SetX;

			for (int i = 0; i < 8; i++)	//Store next 8 pixels
			{
				SetX.m256_f32[i] = xIncrement;
				xIncrement++;
			}
			//Use our operations
			__m256 SetCReal = add8(mul8(add8(SetX, SetHalf), div8(sub8(SetRange01, SetRange00), SetDim)), SetRange00);
			__m256 SetCIm = add8(mul8(add8(SetY, SetHalf), div8(sub8(SetRange11, SetRange10), SetDim)), SetRange10);
			__m256 Double = _mm256_set1_ps(2);
			__m256 SetF = mul8(Double, Iterate_SIMD(SetCReal, SetCIm));

			//For each pixel in answer set
			for (int i = 0; i < 8; i++)
			{
				float f = SetF.m256_f32[i];
				if (SetF.m256_f32[i] > 255.)
					f = 255.;
				imageSIMDMT[j * dimMT[0] + actualX] = f;
				actualX++; //Used for positioning in image
			}
		}
	}

}

unsigned char* imageMT = new unsigned char[1024 * 1024];

void SimpleFractalDrawing_MTAux(int start, int end) {
	int dimMT[2] = { 1024,1024 };
	float rangeMT[2][2] = { {-0.003,0.008},{-0.0002,0.0005} };

	for (int j = start; j < end; j++) {
		for (int i = 0; i < dimMT[0]; i++)
		{
			complex c;
			c.real = rangeMT[0][0] + (i + 0.5) * (rangeMT[0][1] - rangeMT[0][0]) / dimMT[0]; //Create x coordinates within the range [range[0][0] .. range[0][1]] 
			c.im = rangeMT[1][0] + (j + 0.5) * (rangeMT[1][1] - rangeMT[1][0]) / dimMT[1]; //Create x coordinates within the range [range[1][0] .. range[1][1]] 

			float f = 2 * Iterate(c);
			if (f > 255.)
				f = 255.;
			imageMT[j * dimMT[0] + i] = f;


		}
	}

}

void SimpleFractalDrawing_MT(int dim[2], float range[2][2])
{
	//Work to balance workload across threads
	int workload = dim[1] / NB_THREADS;
	int start = 0;
	int end = dim[1];
	int newEnd = start + workload;
	std::thread t[NB_THREADS];

	//initialise threads
	for (int k = 0; k < NB_THREADS; k++) {
		//initialise each thread
		t[k] = std::thread(SimpleFractalDrawing_MTAux, start, newEnd);
		//start new range
		start = start + workload;
		newEnd = start + workload;
	}
	//Join
	for (int i = 0; i < NB_THREADS; i++) {
		t[i].join();
	}
}



void SimpleFractalDrawing_SIMD_MT(int dim[2], float range[2][2])
{
	//Work to balance workload across threads
	int workload = dim[1] / NB_THREADS;
	int start = 0;
	int end = dim[1];
	int newEnd = start + workload;
	std::thread t[NB_THREADS];

	//initialise threads
	for (int k = 0; k < NB_THREADS; k++) {

		t[k] = std::thread(SimpleFractalDrawing_SIMD_MTAux, start, newEnd);
		
		//State new range of y to calculate for each thread
		start = start + workload;
		newEnd = start + workload;
	}
	for (int i = 0; i < NB_THREADS; i++) {
		t[i].join();
	}
}

//this is the process for each pixel


//For MT Methods range is defined in-method
int main(int argc, char* argv[])
{
	Chrono c;
	int dims[2]={1024,1024};
	float range[2][2] = { {-0.003,0.008},{-0.0002,0.0005} };
	float range2[2][2] = { {-1.4,0.6},{-1.1,1.3} };
	unsigned char *image=new unsigned char[dims[0]*dims[1]];

	c.InitChrono();
	SimpleFractalDrawing(image,dims,range); //largest 64bit prime
	c.PrintElapsedTime_us("Time for simd algorithm in micro seconds\n");
	SaveBMP("fractal.bmp", image, dims[0], dims[1]);
	for (int i = 0; i < dims[0] * dims[1]; i++)
		image[i] = 127; //resetting image to grey

	c.InitChrono();
	SimpleFractalDrawing_MT(dims, range);
	c.PrintElapsedTime_us("Time for MT algorithm in micro seconds\n");
	SaveBMP("fractal_MT.bmp", imageMT, dims[0], dims[1]);
	for (int i = 0; i < dims[0] * dims[1]; i++)

	c.InitChrono();
	SimpleFractalDrawing_SIMD(image, dims,range);
	c.PrintElapsedTime_us("Time for SIMD algorithm in micro seconds\n");
	SaveBMP("fractalSIMD.bmp",image,dims[0],dims[1]);
	for (int i = 0; i < dims[0] * dims[1]; i++)


	c.InitChrono();
	SimpleFractalDrawing_SIMD_MT(dims, range);
	c.PrintElapsedTime_us("Time for SIMD MT algorithm in micro seconds\n");
	SaveBMP("fractal_SIMD_MT.bmp", imageSIMDMT, dims[0], dims[1]);
	delete[] image;
}


/*
results:
Computer/processor details: Acer Aspire 5 - Intel Core i5-8265U 4 Core
CPU 1 thread  :	 1506649 ms
CPU MT        :  442229 ms
SIMD_1_Thread :	 139641	ms
SIMD_MT		  :	 53157 ms


*/
