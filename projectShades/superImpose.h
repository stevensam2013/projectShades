#pragma once

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>

using namespace std;
using namespace cv;

class superImpose
{
private:

public:
	superImpose(void);
	~superImpose(void);
	Mat cropSpecs(Mat inputImage);
};

