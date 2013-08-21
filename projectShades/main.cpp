#include <stdio.h>
#include <fstream>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "facialAnalysis.h"
#include "superImpose.h"
#include "xmlDocument.h"

using namespace std;
using namespace cv;



int main(int argc,char *argv[])
{
	string filename, imgName, processedFilename, debugFilename, specsFilename, xmlFilename;
	Mat inputImage, debugImage, specsImage;
	xmlDocument outputXml;
	ofstream xmlFile;


	ostringstream stringStream;

	string exeDir;
	exeDir = argv[0];
		
	exeDir = exeDir.substr(0, exeDir.find_last_of("\\/")+1);
	
	if( exeDir.find_last_of("\\/") == exeDir.npos)
	{
		exeDir = "";
	}


	if(argc != 3)
	{
		cout << "This program requires the specs image and the captured image prefix!\n\nDebug Mode\n\n";
		//return -1;
	}

	
	if(argc != 3)
	{
		filename = exeDir + "face";
		specsFilename = exeDir + "specs2";

		cout << filename << endl;
		cout << specsFilename << endl;
		//return -1;
	}
	else
	{
		//Set the name of the spectacles image
		specsFilename = argv[1];

		//Set the name of the captured webcam image
		filename = argv[2];

		cout << "\n" << filename << "\n";
	}

	imgName = filename;
	filename += "_1.jpg";
	specsFilename += ".jpg";

	cout << specsFilename;

	inputImage = imread(filename, CV_LOAD_IMAGE_COLOR);   // Read the file
	specsImage = imread(specsFilename, CV_LOAD_IMAGE_COLOR);   // Read the file
	

    if(!inputImage.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the captured image" << std::endl ;
        return -1;
    }

	if(!specsImage.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the specs image" << std::endl ;
        return -1;
    }

	//Analyse image and determine facial characteristics
	facialAnalysis myFacialAnalysis(inputImage, exeDir);


	//Add content to XML
	stringStream << myFacialAnalysis.getPupillaryDistance();
	outputXml.addElement("interPupilaryDistance", stringStream.str());

	//Write XML file
	xmlFilename = imgName + ".xml";
	xmlFile.open (xmlFilename);
	xmlFile << outputXml.getHeader() << outputXml.getBody() << outputXml.getFooter();
	xmlFile.close();

	//superimpose specs on face
	processedFilename = imgName + "_processed.jpg";
	myFacialAnalysis.addGlasses(specsImage);
	imwrite( processedFilename, myFacialAnalysis.getProcessedImage());

	//Debug image
	debugImage = inputImage;
	circle(debugImage, myFacialAnalysis.getLeftPupil(), 3, 9999);
	circle(debugImage, myFacialAnalysis.getRightPupil(), 3, 9999);
	line(debugImage, myFacialAnalysis.getLeftPupil(), myFacialAnalysis.getRightPupil(), 9999,1);
	
	
	debugFilename = imgName + "_debug.jpg";
	imwrite( debugFilename, debugImage );

	/*
    namedWindow( "Debug window", CV_WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Debug window", debugImage );                   // Show our image inside it.
	
	namedWindow( "Processed window", CV_WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Processed window", specsImage );                   // Show our image inside it.

    waitKey(0);                                          // Wait for a keystroke in the window
	*/
    return 0;
}