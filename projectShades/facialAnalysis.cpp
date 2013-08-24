#include "facialAnalysis.h"
#include <math.h>


facialAnalysis::facialAnalysis(Mat inputImage, String directory)
{
	Mat tempImage;

	m_drawGlasses = true;
	m_faceDetected = false;

	m_status = "";

	//Set the member matrix to the input image
	m_inputImage = inputImage;
	m_faceCascadeName = directory + "haarcascade_frontalface_alt.xml";

	if(detectFace())
	{
		m_faceDetected = true;
		findPupils();
		measureNoseBridge();
		skinFilter(m_inputImage, tempImage);
		m_inputImage.copyTo(m_processedImage);
		measureFace();
	}
	else
	{
		m_status.append("No face detected, ");
	}

	if (m_status == "")
	{
		m_status = "OK";
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

	
	cvtColor(specsImage, greySpecs, CV_BGR2GRAY);

	//Get the ROI of the specs image
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
	
	//Resize based on face width -- Needs work
	resize(croppedSpecs, croppedSpecs, Size(0,0), double(m_faceWidth*1.1)/croppedSpecs.cols, double(m_faceWidth)/croppedSpecs.cols); 

	//Resize based on pupil distance -- Temporary solutuion
	//resize(croppedSpecs, croppedSpecs, Size(0,0), double(2.2*m_interPupillaryDistance)/croppedSpecs.cols, double(2.2*m_interPupillaryDistance)/croppedSpecs.cols); 

	//Rotate the glasses
	//Rotate the image of the glasses so they match the line between the pupils.

	//Merge the images
	//specsCentre = Point(((m_leftPupilPosition.x - m_rightPupilPosition.x)/2) + m_rightPupilPosition.x, ((m_leftPupilPosition.y - m_rightPupilPosition.y)/2) + m_rightPupilPosition.y);
	specsCentre = Point((m_faceLeft + m_faceRight)/2, ((m_leftPupilPosition.y + m_rightPupilPosition.y)/2)+croppedSpecs.rows/5);
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
	//m_faceTop = findFaceTop();
	m_faceBottom = findFaceBottom();
	m_faceLeft = findFaceLeft();
	m_faceRight = findFaceRight();
	m_faceWidth = m_faceRight - m_faceLeft;
}

int facialAnalysis::getFaceLeft()
{
	return m_faceLeft;
}

int facialAnalysis::getFaceRight()
{
	return m_faceRight;
}

int facialAnalysis::getFaceWidth()
{
	return m_faceWidth;
}

int facialAnalysis::getFaceTop()
{
	return m_faceTop;
}

int facialAnalysis::getFaceBottom()
{
	return m_faceBottom;
}

void facialAnalysis::skinFilter(Mat inputImage, Mat outputImage)
{
	Mat chromianceImage;
	Mat skinMask;
	vector<Mat> chromianceChannels(3);
	
	//Create a local scope working copy of the image
	inputImage.copyTo(outputImage);

	Scalar Cr, Cb, H, S, V;
	

	//Convert the input image into the CYrYb colourspace and set the skinMask image to be an 8bit image.
	cvtColor(outputImage, chromianceImage, CV_BGR2YCrCb);
	//cvtColor(outputImage, chromianceImage, CV_BGR2HSV);
	cvtColor(outputImage, skinMask, CV_BGR2GRAY);

	split(chromianceImage, chromianceChannels);

	blur(chromianceImage, chromianceImage, Size(5,5));

	//inRange(chromianceImage, Scalar(0, 10, 60), Scalar(20, 150, 255), skinMask);
	inRange(chromianceImage, Scalar(0, 133, 77), Scalar(255, 174, 127), skinMask);

	/*
	for(int y = 0; y < chromianceImage.rows; y++)
	{
		for(int x = 1; x < chromianceImage.cols; x++)
		{
			//colour = noseBridgeArea.at<uchar>(Point(x, y));
			Cr = chromianceChannels[1].at<uchar>(Point(x, y));
			Cb = chromianceChannels[2].at<uchar>(Point(x, y));

			if((Cr.val[0] >= 133) && (Cr.val[0] <= 174) && (Cb.val[0] >= 77) && (Cb.val[0] <= 127))
			{
				skinMask.at<uchar>(Point(x,y)) = 255;
			}
			else
			{
				skinMask.at<uchar>(Point(x,y)) = 0;
			} 
			
		}
	}
	*/

	//Show the image for debug
    //namedWindow( "Chromiance window", CV_WINDOW_AUTOSIZE );// Create a window for display.
	//imshow( "Chromiance window", skinMask);                   // Show our image inside it.

	//waitKey(0);
	skinMask.copyTo(outputImage);
}

int facialAnalysis::findFaceTop()
{
	Mat tempImage;
	vector<Mat> rgbChannels(3);
	int eyeHeight, lowerHairLine, hairLine, upperHairLine, change, maxChange;
	int average [3], prevAverage [3];
	Vec3b pixel;

	//Create working copy of the input image
	m_inputImage.copyTo(tempImage);

	eyeHeight = (m_rightPupilPosition.y + m_leftPupilPosition.y)/2;
	lowerHairLine = eyeHeight - (abs(m_rightPupilPosition.x - m_leftPupilPosition.x)*0.7);
	upperHairLine = eyeHeight - (abs(m_rightPupilPosition.x - m_leftPupilPosition.x)*1.5);

	
	//GaussianBlur(tempImage, tempImage, Size(3,3), 0);
	split(tempImage, rgbChannels);

	average[0] = 0;
	average[1] = 0;
	average[2] = 0;

	maxChange = 0;

	for(int y = lowerHairLine; y >= upperHairLine; y--)
	{
		prevAverage[0] = average[0];
		prevAverage[1] = average[1];
		prevAverage[2] = average[2];

		average[0] = 0;
		average[1] = 0;
		average[2] = 0;

		for(int x = m_leftPupilPosition.x; x < m_rightPupilPosition.x; x++)
		{
			average[0] = average[0] + rgbChannels[0].at<uchar>(Point(x, y));
			average[1] = average[1] + rgbChannels[1].at<uchar>(Point(x, y));
			average[2] = average[2] + rgbChannels[2].at<uchar>(Point(x, y));
		}

		average[0] = average[0]/(m_rightPupilPosition.x - m_leftPupilPosition.x);
		average[1] = average[1]/(m_rightPupilPosition.x - m_leftPupilPosition.x);
		average[2] = average[2]/(m_rightPupilPosition.x - m_leftPupilPosition.x);

		change = (average[0] - prevAverage[0])*(average[0] - prevAverage[0]) + (average[1] - prevAverage[1])*(average[1] - prevAverage[1]) + (average[2] - prevAverage[2])*(average[2] - prevAverage[2]);
		change = sqrt(double(change));

		if (change > 15 && change > maxChange && y != lowerHairLine)
		{
			maxChange = change;
			hairLine = y;
			
		}
		
	}

	//line(tempImage, Point(m_leftPupilPosition.x, lowerHairLine), Point(m_rightPupilPosition.x, lowerHairLine), 222);
	//line(tempImage, Point(m_leftPupilPosition.x, upperHairLine), Point(m_rightPupilPosition.x, upperHairLine), 222);
	//line(tempImage, Point(m_leftPupilPosition.x, hairLine), Point(m_rightPupilPosition.x, hairLine), 555);

	//Show the image for debug
    //namedWindow( "Face top window", CV_WINDOW_AUTOSIZE );// Create a window for display.
	//imshow( "Face top window", tempImage);                   // Show our image inside it.

	//waitKey(0);
	
	return hairLine;
}

int facialAnalysis::findFaceLeft()
{
	Mat tempImage;
	vector<Mat> rgbChannels(3);
	int eyeHeight, lowerSideLine, sideLine, upperSideLine, change, maxChange;
	int average [3], prevAverage [3];
	Vec3b pixel;

	//Create working copy of the input image
	m_inputImage.copyTo(tempImage);

	lowerSideLine = m_leftPupilPosition.x - (m_interPupillaryDistance*0.6);
	upperSideLine = m_leftPupilPosition.x - (m_interPupillaryDistance*0.35);

	if (lowerSideLine < 0)
	{
		lowerSideLine = 0;
	}

	//Set default side position in case one is not found... Perhaps this should error and update the status instead?
	//sideLine = (lowerSideLine + upperSideLine)/2;
	sideLine = 1;
	split(tempImage, rgbChannels);

	average[0] = 0;
	average[1] = 0;
	average[2] = 0;

	maxChange = 0;

	for(int x = lowerSideLine; x <= upperSideLine; x++)
	{
		prevAverage[0] = average[0];
		prevAverage[1] = average[1];
		prevAverage[2] = average[2];

		average[0] = 0;
		average[1] = 0;
		average[2] = 0;

		for(int y = m_leftPupilPosition.y; y > m_leftPupilPosition.y-(m_interPupillaryDistance/4); y--)
		{
			average[0] = average[0] + rgbChannels[0].at<uchar>(Point(x, y));
			average[1] = average[1] + rgbChannels[1].at<uchar>(Point(x, y));
			average[2] = average[2] + rgbChannels[2].at<uchar>(Point(x, y));
		}

		average[0] = average[0]/(m_interPupillaryDistance/4);
		average[1] = average[1]/(m_interPupillaryDistance/4);
		average[2] = average[2]/(m_interPupillaryDistance/4);

		change = (average[0] - prevAverage[0])*(average[0] - prevAverage[0]) + (average[1] - prevAverage[1])*(average[1] - prevAverage[1]) + (average[2] - prevAverage[2])*(average[2] - prevAverage[2]);
		change = sqrt(double(change));

		if (change > 15 && change > maxChange && x != lowerSideLine)
		{
			maxChange = change;
			sideLine = x;
			
		}
		
	}

	//line(tempImage, Point(lowerSideLine, m_leftPupilPosition.y-(m_interPupillaryDistance/2)), Point(lowerSideLine, m_leftPupilPosition.y), 555);
	//line(tempImage, Point(upperSideLine, m_leftPupilPosition.y-(m_interPupillaryDistance/2)), Point(upperSideLine, m_leftPupilPosition.y), 555);
	//line(tempImage, Point(sideLine, m_leftPupilPosition.y-(m_interPupillaryDistance/2)), Point(sideLine, m_leftPupilPosition.y), 555);

	//Show the image for debug
    //namedWindow( "Face left window", CV_WINDOW_AUTOSIZE );// Create a window for display.
	//imshow( "Face left window", tempImage);                   // Show our image inside it.

	//waitKey(0);
	if(sideLine == 1)
	{
		m_status.append("Left side not detected, ");
		m_drawGlasses = false;
	}

	return sideLine;
}

int facialAnalysis::findFaceRight()
{
	Mat tempImage;
	vector<Mat> rgbChannels(3);
	int eyeHeight, lowerSideLine, sideLine, upperSideLine, change, maxChange;
	int average [3], prevAverage [3];
	Vec3b pixel;

	//Create working copy of the input image
	m_inputImage.copyTo(tempImage);

	lowerSideLine = m_rightPupilPosition.x + (m_interPupillaryDistance*0.35);
	upperSideLine = m_rightPupilPosition.x + (m_interPupillaryDistance*0.6);
	
	if (upperSideLine > tempImage.cols)
	{
		upperSideLine = tempImage.cols;
	}

	//Set default side position in case one is not found... Perhaps this should error and update the status instead?
	//sideLine = (lowerSideLine + upperSideLine)/2;
	sideLine = 1;

	split(tempImage, rgbChannels);

	average[0] = 0;
	average[1] = 0;
	average[2] = 0;

	maxChange = 0;

	for(int x = lowerSideLine; x <= upperSideLine; x++)
	{
		prevAverage[0] = average[0];
		prevAverage[1] = average[1];
		prevAverage[2] = average[2];

		average[0] = 0;
		average[1] = 0;
		average[2] = 0;

		for(int y = m_rightPupilPosition.y; y > m_rightPupilPosition.y-(m_interPupillaryDistance/4); y--)
		{
			average[0] = average[0] + rgbChannels[0].at<uchar>(Point(x, y));
			average[1] = average[1] + rgbChannels[1].at<uchar>(Point(x, y));
			average[2] = average[2] + rgbChannels[2].at<uchar>(Point(x, y));
		}

		average[0] = average[0]/(m_interPupillaryDistance/4);
		average[1] = average[1]/(m_interPupillaryDistance/4);
		average[2] = average[2]/(m_interPupillaryDistance/4);

		change = (average[0] - prevAverage[0])*(average[0] - prevAverage[0]) + (average[1] - prevAverage[1])*(average[1] - prevAverage[1]) + (average[2] - prevAverage[2])*(average[2] - prevAverage[2]);
		change = sqrt(double(change));

		if (change > 15 && change > maxChange && x != lowerSideLine)
		{
			maxChange = change;
			sideLine = x;
			
		}
		
	}

	//line(tempImage, Point(lowerSideLine, m_leftPupilPosition.y-(m_interPupillaryDistance/2)), Point(lowerSideLine, m_leftPupilPosition.y), 555);
	//line(tempImage, Point(upperSideLine, m_leftPupilPosition.y-(m_interPupillaryDistance/2)), Point(upperSideLine, m_leftPupilPosition.y), 555);
	//line(tempImage, Point(sideLine, m_rightPupilPosition.y-(m_interPupillaryDistance/2)), Point(sideLine, m_rightPupilPosition.y), 555);

	//Show the image for debug
    //namedWindow( "Face left window", CV_WINDOW_AUTOSIZE );// Create a window for display.
	//imshow( "Face left window", tempImage);                   // Show our image inside it.

	//waitKey(0);
	
	if(sideLine == 1)
	{
		m_status.append("Right side not detected, ");
		m_drawGlasses = false;
	}
	return sideLine;
}

int facialAnalysis::findFaceBottom()
{
	Mat tempImage;
	vector<Mat> rgbChannels(3);
	int eyeHeight, lowerChinLine, chinLine, upperChinLine, change, maxChange;
	int average [3], prevAverage [3];
	Vec3b pixel;

	//Create working copy of the input image
	m_inputImage.copyTo(tempImage);

	eyeHeight = (m_rightPupilPosition.y + m_leftPupilPosition.y)/2;
	lowerChinLine = eyeHeight + (abs(m_rightPupilPosition.x - m_leftPupilPosition.x)*1.5);
	upperChinLine = eyeHeight + (abs(m_rightPupilPosition.x - m_leftPupilPosition.x)*2);

	chinLine = 1;

	if (upperChinLine > tempImage.rows)
	{
		upperChinLine = tempImage.rows-1;
	}
	
	//GaussianBlur(tempImage, tempImage, Size(3,3), 0);
	split(tempImage, rgbChannels);

	average[0] = 0;
	average[1] = 0;
	average[2] = 0;

	maxChange = 0;

	
	for(int y = lowerChinLine; y <= upperChinLine; y++)
	{
		prevAverage[0] = average[0];
		prevAverage[1] = average[1];
		prevAverage[2] = average[2];

		average[0] = 0;
		average[1] = 0;
		average[2] = 0;

		for(int x = m_leftPupilPosition.x; x < m_rightPupilPosition.x; x++)
		{
			average[0] = average[0] + rgbChannels[0].at<uchar>(Point(x, y));
			average[1] = average[1] + rgbChannels[1].at<uchar>(Point(x, y));
			average[2] = average[2] + rgbChannels[2].at<uchar>(Point(x, y));
		}

		average[0] = average[0]/(m_rightPupilPosition.x - m_leftPupilPosition.x);
		average[1] = average[1]/(m_rightPupilPosition.x - m_leftPupilPosition.x);
		average[2] = average[2]/(m_rightPupilPosition.x - m_leftPupilPosition.x);

		change = (average[0] - prevAverage[0])*(average[0] - prevAverage[0]) + (average[1] - prevAverage[1])*(average[1] - prevAverage[1]) + (average[2] - prevAverage[2])*(average[2] - prevAverage[2]);
		change = sqrt(double(change));

		if (change > 15 && change > maxChange && y != lowerChinLine)
		{
			maxChange = change;
			chinLine = y;
			
		}
		
	}

	//line(tempImage, Point(m_leftPupilPosition.x, lowerChinLine), Point(m_rightPupilPosition.x, lowerChinLine), 222);
	//line(tempImage, Point(m_leftPupilPosition.x, upperChinLine), Point(m_rightPupilPosition.x, upperChinLine), 222);
	//line(tempImage, Point(m_leftPupilPosition.x, chinLine), Point(m_rightPupilPosition.x, chinLine), 555);
	
	//Show the image for debug
    //namedWindow( "Face bottom window", CV_WINDOW_AUTOSIZE );// Create a window for display.
	//imshow( "Face bottom window", tempImage);                   // Show our image inside it.

	//waitKey(0);
	
	if(chinLine == 1)
	{
		m_status.append("Chin line not detected, ");
	}

	return chinLine;
}

Mat facialAnalysis::cannyEdgeDetector(Mat inputImage)
{
	Mat tempImage, greyImage;

	inputImage.copyTo(tempImage);

	cvtColor(tempImage, greyImage, CV_BGR2GRAY);

	blur( greyImage, greyImage, Size(3,3) );

	Canny( greyImage, greyImage, 35,35*3, 3 );

	//Show the image for debug
    //namedWindow( "Canny window", CV_WINDOW_AUTOSIZE );// Create a window for display.
	//imshow( "Canny window", greyImage);                   // Show our image inside it.

	//waitKey(0);

	return greyImage;
}

bool facialAnalysis::drawGlasses()
{
	return m_drawGlasses;
}

String facialAnalysis::getStatus()
{
	return m_status;
}