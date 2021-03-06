#pragma once

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>

#include "findEyeCenter.h"
#include "constants.h"

using namespace std;
using namespace cv;

class facialAnalysis
{
private:
	//Flags
	bool m_drawGlasses;

	String m_status;
	String m_faceCascadeName;
	//The input image
	Mat m_inputImage;
	Mat m_processedImage;

	//The area containing the face
	Rect m_face;
	bool m_faceDetected;
	int m_faceHeight;
	int m_faceTop;
	int m_faceBottom;
	int m_faceWidth;
	int m_faceLeft;
	int m_faceRight;

	//Pupil related data
	Point m_leftPupilPosition;
	Point m_rightPupilPosition;
	int m_interPupillaryDistance;

	//Nose measurementa
	int m_noseBridgeMeasure;
	int m_noseBaseMeasure;
	int m_noseLength;

	//Distance between earlobe and eye socket.
	int m_earToEyeMeasure;

public:
	facialAnalysis(Mat inputImage, string directory);
	~facialAnalysis(void);
	int detectFace();
	void findPupils();
	Point getLeftPupil();
	Point getRightPupil();
	int getPupillaryDistance();
	int getFaceTop();
	int getFaceBottom();
	int getFaceLeft();
	int getFaceRight();
	int getFaceWidth();
	bool drawGlasses();
	void addGlasses(Mat specsImage);
	Mat getProcessedImage();
	String getStatus();
	void measureNoseBridge();
	void measureFace();
	void skinFilter(Mat inputImage, Mat outputImage);
	int findFaceLeft();
	int findFaceTop();
	int findFaceBottom();
	int findFaceRight();
	Mat cannyEdgeDetector(Mat inputImage);
};

