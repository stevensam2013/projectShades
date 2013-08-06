#pragma once

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <stdio.h>

#include "findEyeCenter.h"
#include "constants.h"

using namespace std;
using namespace cv;

class facialAnalysis
{
private:

	//The input image
	Mat m_inputImage;
	Mat m_processedImage;

	//The area containing the face
	Rect m_face;

	//Pupil related data
	Point m_leftPupilPosition;
	Point m_rightPupilPosition;
	int m_interPupillaryDistance;

public:
	facialAnalysis(Mat inputImage);
	~facialAnalysis(void);
	int detectFace();
	void findPupils();
	Point getLeftPupil();
	Point getRightPupil();
	int getPupillaryDistance();

};

