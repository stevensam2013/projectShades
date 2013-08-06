#include "facialAnalysis.h"
#include <math.h>


facialAnalysis::facialAnalysis(Mat inputImage)
{
	//Set the member matrix to the input image
	m_inputImage = inputImage;

	if(detectFace())
	{
		findPupils();
	}
}


facialAnalysis::~facialAnalysis(void)
{
}

int facialAnalysis::detectFace()
{
	vector<Rect> faces;
	vector<cv::Mat> rgbChannels(3);
	Mat processedImage;
	String face_cascade_name = "haarcascade_frontalface_alt.xml";
	CascadeClassifier face_cascade;

	//Load the cascade for face detection
	if( !face_cascade.load( face_cascade_name ) )
	{
		printf("Error loading cascade\n");
	};

	//Split the input image into three images (RGB), select one and blur it
	split(m_inputImage, rgbChannels);
	processedImage = rgbChannels[1];

	//Detect faces
	face_cascade.detectMultiScale( processedImage, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(150, 150) );

	if (faces.size() > 0)
	{
		//set the member variables
		m_processedImage = processedImage;
		m_face =  faces[0];
		return 1;
	}
	else
	{
		return 0;
	}
}

void facialAnalysis::findPupils()
{
	Mat faceImage = m_processedImage(m_face);
	double interPupillaryDistance;

	//Blur the facial image if this is set
	if (kSmoothFaceImage)
	{
		double sigma = kSmoothFaceFactor * m_face.width;
		GaussianBlur( faceImage, faceImage, cv::Size( 0, 0 ), sigma);
	}

	//Find eye regions
	int eye_region_width = m_face.width * (kEyePercentWidth/100.0);
	int eye_region_height = m_face.width * (kEyePercentHeight/100.0);
	int eye_region_top = m_face.height * (kEyePercentTop/100.0);
	cv::Rect leftEyeRegion(m_face.width*(kEyePercentSide/100.0),
						eye_region_top,eye_region_width,eye_region_height);
	cv::Rect rightEyeRegion(m_face.width - eye_region_width - m_face.width*(kEyePercentSide/100.0),
						eye_region_top,eye_region_width,eye_region_height);


	//-- Find Eye Centers
	m_leftPupilPosition = findEyeCenter(faceImage,leftEyeRegion,"Left Eye");
	m_rightPupilPosition = findEyeCenter(faceImage,rightEyeRegion,"Right Eye");

	//Add the offset for the position of the face in the origonal image
	m_rightPupilPosition.x += m_face.x + rightEyeRegion.x;
	m_rightPupilPosition.y += m_face.y + rightEyeRegion.y;
	m_leftPupilPosition.x += m_face.x + leftEyeRegion.x;
	m_leftPupilPosition.y += m_face.y + leftEyeRegion.y;

	interPupillaryDistance = sqrt( double( (pow(m_leftPupilPosition.x - m_rightPupilPosition.x, 2.0)) + (pow(m_leftPupilPosition.y - m_rightPupilPosition.y, 2.0)) ));

	m_interPupillaryDistance = (int)(interPupillaryDistance+0.5);
	
}

Point facialAnalysis::getLeftPupil()
{
	return m_leftPupilPosition;
}

Point facialAnalysis::getRightPupil()
{
	return m_rightPupilPosition;
}

int facialAnalysis::getPupillaryDistance()
{
	return m_interPupillaryDistance;
}
