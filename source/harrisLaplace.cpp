#include "imgFeat.h"

int round(double f)
{ 
	if ((int)f+0.5>f) 
		return (int)f; 
	else 
		return (int)f + 1;   
}

void feat::detectHarrisLaplace(const Mat& imgSrc, Mat& imgDst)
{
	Mat gray;
	if (imgSrc.channels() == 3)
	{
		cvtColor(imgSrc, gray, CV_BGR2GRAY);
	}
	else
	{
		gray = imgSrc.clone();
	}
	gray.convertTo(gray, CV_64F);

	/* 尺度设置*/
	double dSigmaStart = 1.5;
	double dSigmaStep = 1.2;
	int iSigmaNb = 13;

	vector<double> dvecSigma(iSigmaNb);
	for (int i = 0; i < iSigmaNb; i++)
	{
		dvecSigma[i] = dSigmaStart + i*dSigmaStep;
	}
	vector<Mat> harrisArray(iSigmaNb);
	
	for (int i = 0; i < iSigmaNb; i++)
	{
		double iSigmaI = dvecSigma[i];
		double iSigmaD = 0.7 * iSigmaI;

		int iKernelSize = 6*round(iSigmaD) + 1;
		/*微分算子*/
		Mat dx(1, iKernelSize, CV_64F);
		for (int k =0; k < iKernelSize; k++)
		{
			int pCent = (iKernelSize - 1) / 2;
			int x = k - pCent;
			dx.at<double>(0,i) = x * exp(-x*x/(2*iSigmaD*iSigmaD))/(iSigmaD*iSigmaD*iSigmaD*sqrt(2*CV_PI));
		}
	
		Mat dy = dx.t();
		Mat Ix,Iy;
		/*图像微分*/
		filter2D(gray, Ix, CV_64F, dx);
		filter2D(gray, Iy, CV_64F, dy);

		Mat Ix2,Iy2,Ixy;
		Ix2 = Ix.mul(Ix);
		Iy2 = Iy.mul(Iy);
		Ixy = Ix.mul(Iy);

		int gSize = 6*round(iSigmaI) + 1;
		Mat gaussKernel = getGaussianKernel(gSize, iSigmaI);
		filter2D(Ix2, Ix2, CV_64F, gaussKernel);
		filter2D(Iy2, Iy2, CV_64F, gaussKernel);
		filter2D(Ixy, Ixy, CV_64F, gaussKernel);

		/*自相关矩阵*/
		double alpha = 0.06;
		Mat detM = Ix2.mul(Iy2) - Ixy.mul(Ixy);
		Mat trace = Ix2 + Iy2;
		Mat cornerStrength = detM - alpha * trace.mul(trace);

		

		double maxStrength;
		minMaxLoc(cornerStrength, NULL, &maxStrength, NULL, NULL);
		Mat dilated;
		Mat localMax;
		dilate(cornerStrength, dilated, Mat());
		compare(cornerStrength, dilated, localMax, CMP_EQ);
	

		Mat cornerMap;
		double qualityLevel = 0.2;
		double thresh = qualityLevel * maxStrength;
		cornerMap = cornerStrength > thresh;
		bitwise_and(cornerMap, localMax, cornerMap);
		harrisArray[i] = cornerMap.clone();	
	}

	/*计算尺度归一化Laplace算子*/
	vector<Mat> laplaceSnlo(iSigmaNb);
	for (int i = 0; i < iSigmaNb; i++)
	{
		double iSigmaL = dvecSigma[i];
		Size kSize = Size(6 * floor(iSigmaL) +1, 6 * floor(iSigmaL) +1);
		Mat hogKernel = getHOGKernel(kSize,iSigmaL);
		filter2D(gray, laplaceSnlo[i], CV_64F, hogKernel);
		laplaceSnlo[i] *= (iSigmaL * iSigmaL);
	}
	
	/*检测每个特征点在某一尺度LOG相应是否达到最大*/
	Mat corners(gray.size(), CV_8U, Scalar(0));
	for (int i = 0; i < iSigmaNb; i++)
	{
		for (int r = 0; r < gray.rows; r++)
		{
			for (int c = 0; c < gray.cols; c++)
			{
				if (i ==0)
				{
					if (harrisArray[i].at<uchar>(r,c) > 0 && laplaceSnlo[i].at<double>(r,c) > laplaceSnlo[i + 1].at<double>(r,c))
					{
						corners.at<uchar>(r,c) = 255;
					}
				}
				else if(i == iSigmaNb -1)
				{
					if (harrisArray[i].at<uchar>(r,c) > 0 && laplaceSnlo[i].at<double>(r,c) > laplaceSnlo[i - 1].at<double>(r,c))
					{
						corners.at<uchar>(r,c) = 255;
					}
				}
				else
				{
					if (harrisArray[i].at<uchar>(r,c) > 0 &&
					laplaceSnlo[i].at<double>(r,c) > laplaceSnlo[i + 1].at<double>(r,c) &&
					laplaceSnlo[i].at<double>(r,c) > laplaceSnlo[i - 1].at<double>(r,c))
					{
						corners.at<uchar>(r,c) = 255;
					}
				}
			}
		}
	}
	imgDst = corners.clone();
	
}

void feat::detectHarrisLaplace(Mat& imgSrc, Mat& imgDst,double alpha)
{
	Mat gray;
	if (imgSrc.channels() == 3)
	{
		cvtColor(imgSrc, gray, CV_BGR2GRAY);
	}
	else
	{
		gray = imgSrc.clone();
	}
	gray.convertTo(gray, CV_64F);

	/* 尺度设置*/
	double dSigmaStart = 0.5;
	double dSigmaStep = 1.4;
	int iSigmaNb = 13;

	vector<double> dvecSigma(iSigmaNb);
	for (int i = 0; i < iSigmaNb; i++)
	{
		dvecSigma[i] = dSigmaStart * pow(dSigmaStep,i+1);
	}
	vector<Mat> harrisArray(iSigmaNb);
	Mat gray1;
	for (int i = 0; i < iSigmaNb; i++)
	{
		double iSigmaI = dvecSigma[i];
		double iSigmaD = 0.7 * iSigmaI;

		int iKernelSize = 6*round(iSigmaD) + 1;
		
		//这里可能造成不断的对gray进行高斯模糊，而不是对每一个初始的gray进行不同的高斯模糊
		GaussianBlur(gray,gray1,cv::Size(iKernelSize,iKernelSize),iSigmaD,iSigmaD);
		Mat xKernel = (Mat_<double>(1,3) << -1, 0, 1);
		Mat yKernel = xKernel.t();

		Mat Ix,Iy;
		filter2D(gray1, Ix, CV_64F, xKernel);
		filter2D(gray1, Iy, CV_64F, yKernel);

		Mat Ix2,Iy2,Ixy;
		Ix2 = Ix.mul(Ix);
		Iy2 = Iy.mul(Iy);
		Ixy = Ix.mul(Iy);

		int gSize = 6*round(iSigmaI) + 1;
		Mat gaussKernel = getGaussianKernel(gSize, iSigmaI);
		Mat gaussKernelD=gaussKernel*(iSigmaD*iSigmaD);
		filter2D(Ix2, Ix2, CV_64F, gaussKernelD);
		filter2D(Iy2, Iy2, CV_64F, gaussKernelD);
		filter2D(Ixy, Ixy, CV_64F, gaussKernelD);

		/*自相关矩阵*/
		Mat detM = Ix2.mul(Iy2) - Ixy.mul(Ixy);
		Mat trace = Ix2 + Iy2;
		Mat cornerStrength = detM - alpha * trace.mul(trace);

		double maxStrength;
		minMaxLoc(cornerStrength, NULL, &maxStrength, NULL, NULL);
		Mat dilated;
		Mat localMax;
		dilate(cornerStrength, dilated, Mat());
		compare(cornerStrength, dilated, localMax, CMP_EQ);


		Mat cornerMap;
		double qualityLevel = 0.01;
		double thresh = qualityLevel * maxStrength;
		cornerMap = cornerStrength > thresh;
		bitwise_and(cornerMap, localMax, cornerMap);
		harrisArray[i] = cornerMap.clone();
	}

	/*计算尺度归一化Laplace算子*/
	/*vector<Mat> laplaceSnlo(iSigmaNb);
	for (int i = 0; i < iSigmaNb; i++)
	{
		double iSigmaI = dvecSigma[i];
		int iKernelSize = 6*round(iSigmaI) + 1;

		Mat gray2;
		GaussianBlur(gray,gray2,cv::Size(iKernelSize,iKernelSize),iSigmaI,iSigmaI);
		Mat xKernel = (Mat_<double>(1,3) << 1,-2, 1);
		Mat yKernel = xKernel.t();

		Mat Ixx,Iyy;
		filter2D(gray2, Ixx, CV_64F, xKernel);
		filter2D(gray2, Iyy, CV_64F, yKernel);

		add(Ixx,Iyy,laplaceSnlo[i]);
		laplaceSnlo[i] *= (iSigmaI * iSigmaI);
	}*/

	/*检测每个特征点在某一尺度LOG相应是否达到最大*/

	/*计算尺度归一化Laplace算子*/
	vector<Mat> laplaceSnlo(iSigmaNb);
	for (int i = 0; i < iSigmaNb; i++)
	{
		double iSigmaL = dvecSigma[i];
		Size kSize = Size(6 * floor(iSigmaL) +1, 6 * floor(iSigmaL) +1);
		Mat hogKernel = getHOGKernel(kSize,iSigmaL);
		filter2D(gray, laplaceSnlo[i], CV_64F, hogKernel);
		laplaceSnlo[i] *= (iSigmaL * iSigmaL);
	}
	Mat corners(gray.size(), CV_8U, Scalar(0));
	
	for (int r = 0; r < gray.rows; r++)
	{
		for (int c = 0; c < gray.cols; c++)
		{
			for (int i = 1; i < iSigmaNb-1; i++)
			{

				if (harrisArray[i].at<uchar>(r,c) > 0 &&
					laplaceSnlo[i].at<double>(r,c) > laplaceSnlo[i + 1].at<double>(r,c) &&
					laplaceSnlo[i].at<double>(r,c) > laplaceSnlo[i - 1].at<double>(r,c))
				{
					//corners.at<uchar>(r,c) = 255;
					circle(imgSrc, Point(c, r), 3*round(dvecSigma[i]), Scalar(0, 255, 0), 0);
				}

			}	
		}
	}
	//imgDst = corners.clone();

}

