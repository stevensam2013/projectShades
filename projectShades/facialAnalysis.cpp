#include "facialAnalysis.h"
#include <math.h>


facialAnalysis::facialAnalysis(Mat inputImage, String directory)
{
	m_faceDetected = false;

	//Set the member matrix to the input image
	m_inputImage = inputImage;
	m_faceCascadeName = directory + "haarcascade_frontalface_alt.xml";

	if(detectFace())
	{
		m_faceDetected = true;
		findPupils();
		measureNoseBridge();
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
	CascadeClassifier face_cascade;

	//Load the cascade for face detection
	if( !face_cascade.load( m_faceCascadeName ) )
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

Mat facialAnalysis::getProcessedImage()
{
	return m_processedImage;
}

void facialAnalysis::addGlasses(Mat specsImage)
{
	int left = 9999, right = -1, top = 9999, bottom = -1;
	Mat greySpecs;
	Scalar colour;
	Vec3b pixel;
	Mat croppedSpecs;
	Point specsCentre, specsTopLeft;

	//If no face was detected, then exit this function.
	if (!m_faceDetected)
	{
		return;
	}

	//Get the ROI of the specs image
	cvtColor(specsImage, greySpecs, CV_BGR2GRAY);

	//Iterate through the pixels and set the extreme values of the frames within the image.
	for(int y = 0; y < greySpecs.rows; y++)
	{
		for(int x = 0; x < greySpecs.cols; x++)
		{
			colour = greySpecs.at<uchar>(Point(x, y));
			if(colour.val[0] < 240)
			{
				if(x < left)
				{left = x;}

				if(x > right)
				{right = x;}

				if(y < top)
				{top = y;}

				if(y > bottom)
				{bottom = y;}
			}
		}
	}

	//Crop the Specs
	Rect specsRect(Point(left,top), Point(right, bottom));
	croppedSpecs = specsImage(specsRect).clone();

	//Resize the specs
	resize(croppedSpecs, croppedSpecs, Size(0,0), double(2.2*getPupillaryDistance())/croppedSpecs.cols, double(2.2*getPupillaryDistance())/croppedSpecs.cols); 

	//Rotate the glasses
	//Rotate the image of the glasses so they match the line between the pupils.

	//Merge the images
	specsCentre = Point(((m_leftPupilPosition.x - m_rightPupilPosition.x)/2) + m_rightPupilPosition.x, ((m_leftPupilPosition.y - m_rightPupilPosition.y)/2) + m_rightPupilPosition.y);
	specsTopLeft.x = specsCentre.x - (croppedSpecs.cols/2);
	specsTopLeft.y = specsCentre.y - (croppedSpecs.rows/2);

	cvtColor(croppedSpecs, greySpecs, CV_BGR2GRAY);
	m_inputImage.copyTo(m_processedImage);

	for(int y = 0; y < croppedSpecs.rows; y++)
	{
		for(int x = 0; x < croppedSpecs.cols; x++)
		{
			colour = greySpecs.at<uchar>(Point(x, y));
			if(colour.val[0] < 150 && specsTopLeft.x+x < m_processedImage.cols)
			{
				m_processedImage.at<Vec3b>(Point(specsTopLeft.x + x, specsTopLeft.y + y)) = croppedSpecs.at<Vec3b>(Point(x, y));
			}
			
		}
	}

	//Show the image for debug
	//namedWindow( "Superimpose window", CV_WINDOW_AUTOSIZE );// Create a window for display.
    //imshow( "Superimpose window", m_processedImage );                   // Show our image inside it.

	//waitKey(0); 
}

void facialAnalysis::measureNoseBridge()
{
	Mat noseBridgeArea, tempInputImage;
	int left, right, top, bottom, width, height;
	Point noseCentre;
	Scalar colour;
	Vec3b pixel;
	vector<Mat> rgbChannels(3);

	noseCentre = Point(((m_leftPupilPosition.x - m_rightPupilPosition.x)/2) + m_rightPupilPosition.x, ((m_leftPupilPosition.y - m_rightPupilPosition.y)/2) + m_rightPupilPosition.y);

	left = m_leftPupilPosition.x;
	top = m_leftPupilPosition.y - 2;
	right = m_rightPupilPosition.x;
	bottom = m_rightPupilPosition.y + 60;
	width = right - left;
	height = bottom - top;


	//Crop the bridge area
	Rect noseBridgeRect(Point(left,top), Point(right, bottom));
	noseBridgeArea = m_inputImage(noseBridgeRect).clone();
	//resize(noseBridgeArea, noseBridgeArea, Size(0,0), 10, 10);
	//line(noseBridgeArea, Point(noseBridgeArea.cols/2, 0), Point(noseBridgeArea.cols/2, noseBridgeArea.rows), 555);  

	//Canny(noseBridgeArea, noseBridgeArea, 1, 200);
	noseBridgeArea.copyTo(tempInputImage);

	split(noseBridgeArea, rgbChannels);

	rgbChannels[1].copyTo(tempInputImage);

	/*
	cvtColor(noseBridgeArea, noseBridgeArea, CV_BGR2GRAY);
	cvtColor(tempInputImage, tempInputImage, CV_BGR2GRAY);
	//Iterate through the pixels and set the extreme values of the frames within the image.
	
	for(int y = 0; y < noseBridgeArea.rows; y++)
	{
		for(int x = 1; x < noseBridgeArea.cols; x++)
		{
			//colour = noseBridgeArea.at<uchar>(Point(x, y));
			colour = 10*abs(noseBridgeArea.at<uchar>(Point(x-1, y)) - noseBridgeArea.at<uchar>(Point(x, y)));

			tempInputImage.at<uchar>(Point(x,y)) = colour.val[0];
		}
	}
	
	GaussianBlur(noseBridgeArea, noseBridgeArea, Size(3,3), 0);

	Scharr(noseBridgeArea, tempInputImage, CV_8U, 0, 1);

	*/

	resize(noseBridgeArea, noseBridgeArea, Size(0,0), 10, 10);
	resize(tempInputImage, tempInputImage, Size(0,0), 10, 10);

	//Show the image for debug
	//namedWindow( "Nose Bridge window1", CV_WINDOW_AUTOSIZE );// Create a window for display.
    //imshow( "Nose Bridge window1", noseBridgeArea );  

	//Show the image for debug
	//namedWindow( "Nose Bridge window", CV_WINDOW_AUTOSIZE );// Create a window for display.
    //imshow( "Nose Bridge window", tempInputImage );                   // Show our image inside it.

	//waitKey(0); 
}

void facialAnalysis::measureFace()
{
	
}