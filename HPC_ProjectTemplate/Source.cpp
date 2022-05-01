#include <iostream>
#include <math.h>
#include <stdlib.h>
#include<string.h>
#include<msclr\marshal_cppstd.h>
#include <ctime>// include this header 
#pragma once
#include <mpi.h>
#using <mscorlib.dll>
#using <System.dll>
#using <System.Drawing.dll>
#using <System.Windows.Forms.dll>

using namespace std;
using namespace msclr::interop;
//#define imagelen = 16;

int* inputImage(int* w, int* h, System::String^ imagePath) //put the size of image in w & h
{
	int* input;


	int OriginalImageWidth, OriginalImageHeight;

	//*********************************************************Read Image and save it to local arrayss*************************	
	//Read Image and save it to local arrayss

	System::Drawing::Bitmap BM(imagePath);

	OriginalImageWidth = BM.Width;
	OriginalImageHeight = BM.Height;
	*w = BM.Width;
	*h = BM.Height;
	int *Red = new int[BM.Height * BM.Width];
	int *Green = new int[BM.Height * BM.Width];
	int *Blue = new int[BM.Height * BM.Width];
	input = new int[BM.Height*BM.Width];
	for (int i = 0; i < BM.Height; i++)
	{
		for (int j = 0; j < BM.Width; j++)
		{
			System::Drawing::Color c = BM.GetPixel(j, i);

			Red[i * BM.Width + j] = c.R;
			Blue[i * BM.Width + j] = c.B;
			Green[i * BM.Width + j] = c.G;

			input[i*BM.Width + j] = ((c.R + c.B + c.G) / 3); //gray scale value equals the average of RGB values

		}

	}
	return input;
}


void createImage(int* image, int width, int height, int index)
{
	System::Drawing::Bitmap MyNewImage(width, height);


	for (int i = 0; i < MyNewImage.Height; i++)
	{
		for (int j = 0; j < MyNewImage.Width; j++)
		{
			//i * OriginalImageWidth + j
			if (image[i*width + j] < 0)
			{
				image[i*width + j] = 0;
			}
			if (image[i*width + j] > 255)
			{
				image[i*width + j] = 255;
			}
			System::Drawing::Color c = System::Drawing::Color::FromArgb(image[i*MyNewImage.Width + j], image[i*MyNewImage.Width + j], image[i*MyNewImage.Width + j]);
			MyNewImage.SetPixel(j, i, c);
		}
	}
	MyNewImage.Save("..//Data//Output//outputRes" + index + ".png");
	cout << "result Image Saved " << index << endl;
}


int main()
{
	int ImageWidth = 4, ImageHeight = 4;

	int start_s, stop_s, TotalTime = 0;

	System::String^ imagePath;
	std::string img;
	img = "..//Data//Input//test.png";

	imagePath = marshal_as<System::String^>(img);
	int* imageData = inputImage(&ImageWidth, &ImageHeight, imagePath);
	int imageSize = ImageWidth * ImageHeight;


	start_s = clock();
	/**************************************************************/
	MPI_Init(NULL, NULL);

	int size, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); 

	int scale_val = 255;
	int imagesublength = imageSize / size;
	double local_intensity[256] = { 0 }, intensity[256] = { 0 };
	int* subimage = new int[imagesublength];
	double intensityP[256];
	double* probhist = new double[256 / size];
	double* probability = new double[256 / size];
	double total_no_of_pixels = 0;

	// divide image

	MPI_Scatter(imageData, imagesublength, MPI_INT, subimage, imagesublength, MPI_INT, 0, MPI_COMM_WORLD);

	for (int i = 0; i < imagesublength; i++) {
		local_intensity[subimage[i]]++;
	}

	MPI_Reduce(&local_intensity, &intensity, 256, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);


	if (rank == 0)
	{
		for (int i = 0; i < 256; i++)
		{
			total_no_of_pixels += intensity[i];
		}
	}

	//Broadcast To ALL Machines

	MPI_Bcast(&total_no_of_pixels, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//Scattering the global histogram array inorder that each machine compute the probability

	MPI_Scatter(&intensity, 256 / size, MPI_DOUBLE, probhist, 256 / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	for (int i = 0; i < 256 / size; i++)
	{
		probability[i] = probhist[i] / total_no_of_pixels;
	}

	//After Probability Calculations, Let's Gather them all together

	MPI_Gather(probability, 256 / size, MPI_DOUBLE, &intensityP, 256 / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//Cumulative Probability

	if (rank == 0) {
		

		double sum = 0;
		double cump[256];
		double cumpr[256];

	
		for (int i = 0; i < 256; i++) {
			sum += intensityP[i];
			cump[i] = sum ;
			cumpr[i] = floor((cump[i] * scale_val));
		}


		for (int i = 0; i < imageSize; i++) {
			imageData[i] = cumpr[imageData[i]];
		}
	}

	free(subimage);

	MPI_Finalize();
	/**************************************************************/
	stop_s = clock();
	TotalTime += (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000;
	createImage(imageData, ImageWidth, ImageHeight, 1);
	cout << "time: " << TotalTime << endl;

	free(imageData);
	return 0;

}



