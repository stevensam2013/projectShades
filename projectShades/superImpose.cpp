#include "superImpose.h"


superImpose::superImpose(void)
{
}


superImpose::~superImpose(void)
{
}


Mat superImpose::cropSpecs(Mat inputImage)
{
	namedWindow( "Superimpose window", CV_WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Superimpose window", inputImage );                   // Show our image inside it.

	waitKey(0); 

	return inputImage;
}
