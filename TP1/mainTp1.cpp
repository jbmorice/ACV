#include <iostream>
#include <string>
#include <sstream>
#include <assert.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

// MACRO pour avoir le nom d'une variable.
#define getName(var) #var

using namespace cv;


//=======================================================================================
// convert int into string
//=======================================================================================

string toString(int i) // convert int to string
{
    std::stringstream value;
    value << i;
    return value.str();
}

//=======================================================================================
// computeHistogram
//=======================================================================================
void computeHistogram(const Mat& inputComponent, Mat& myHist)
{
	/// Establish the number of bins
	int histSize = 256;
	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 } ;
	const float* histRange = { range };
	bool uniform = true;
	bool accumulate = false;

	/// Compute the histograms:
	calcHist( &inputComponent, 1, 0, Mat(), myHist, 1, &histSize, &histRange, uniform, accumulate );
}

//=======================================================================================
// computeEntropy
//
// Help found on : http://stackoverflow.com/questions/24930134/entropy-for-a-gray-image-in-opencv
//
//=======================================================================================
float computeEntropy(const Mat& inputComponent)
{
	// Create Hist
	Mat myHist;
	computeHistogram(inputComponent,myHist);

	// Stats of Image
	myHist /= inputComponent.total();

	//Computed Entropy
    Mat logP;
    cv::log(myHist,logP);
    float entropy = -1*sum(myHist.mul(logP))[0];

    return entropy;

}

//=======================================================================================
// displayHistogram
//=======================================================================================
Mat displayHistogram(const Mat& myHist)
{
	// Establish the number of bins
	int histSize = 256;
	// Draw one histogram
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound( (double) hist_w/histSize );
	Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
	/// Normalize the result to [ 0, histImage.rows ]
	Mat myHistNorm;
	normalize(myHist, myHistNorm, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

	/// Draw for each channel
	for( int i = 1; i < histSize; i++ )
	{
		line( histImage, Point( bin_w*(i-1), hist_h - cvRound(myHistNorm.at<float>(i-1)) ) , Point( bin_w*(i), hist_h - cvRound(myHistNorm.at<float>(i)) ), Scalar( 255, 255, 255), 2, 8, 0 );
	}
	/// Display
	namedWindow("Display Histo", CV_WINDOW_AUTOSIZE );
	imshow("Display Histo", histImage );
	cvWaitKey();
	return histImage;
}

//=======================================================================================
// Mat norm_0_255(InputArray _src)
// Create and return normalized image
//=======================================================================================
Mat norm_0_255(InputArray _src) {
 Mat src = _src.getMat();
 // Create and return normalized image:
 Mat dst;
 switch(src.channels()) {
	case 1:
		normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
	src.copyTo(dst);
	break;
 }
 return dst;
}

//=======================================================================================
// EQM
//=======================================================================================
double eqm(const Mat & img1, const Mat & img2)
{
	assert(img1.cols == img2.cols && img1.rows == img2.rows);
	double result = 0;
	for(int i = 0; i < img1.rows; i++) {
		for(int j = 0; j < img1.cols; j++) {
			double pixel1 = (double) img1.at<unsigned char>(i, j);
			double pixel2 = (double) img2.at<unsigned char>(i, j);
			result += ( pixel1 - pixel2) * (pixel1 - pixel2);

		}
	}
 return result/(img1.rows * img1.cols);
}

//=======================================================================================
// psnr
//=======================================================================================
double psnr(const Mat & imgSrc, const Mat & imgDeg)
{
	assert(imgSrc.cols == imgDeg.cols && imgSrc.rows == imgDeg.rows);
	// d c'est la dynamique = ensemble des valeurs possibles par pixel
	int d = 255;
	return 10 * log10((d * d) / eqm(imgSrc, imgDeg));
}

//=======================================================================================
// distortionMap
//=======================================================================================
void distortionMap(const vector<Mat> & imgSrc, const vector<Mat> & imgDeg, Mat & distoMap)
{
	std::vector<Mat> distoMapSplit;
	Mat result;
	for(int i = 0; i < 3; i++) {
		result = ((imgSrc[i] - imgDeg[i]) + 155) / 2;
		distoMapSplit.push_back(result);
	}
	merge(distoMapSplit, distoMap);

}

//=======================================================================================
// convert a image from BGR to YCrCb
//=======================================================================================
void BGRtoYCrCb(const Mat & imgSrc, Mat & imgOut)
{
	cvtColor(imgSrc, imgOut, CV_BGR2YCrCb);
}

//=======================================================================================
// recupération d'image
//=======================================================================================
std::vector<Mat>  recupImage(int argc, char** argv)
{
	std::vector<Mat> images;
	for(int i = 1; i < argc ; i++)
	{
		Mat inputImage;
		inputImage = imread(argv[i], CV_LOAD_IMAGE_COLOR);
		if(!inputImage.data ) { // Check for invalid input
    		std::cout <<  "Could not open or find the image " << argv[1] << std::endl ;
			waitKey(0); // Wait for a keystroke in the window
 		 }
		images.push_back(inputImage);
	}
	return images;
}


//=======================================================================================
//=======================================================================================
// MAIN
//=======================================================================================
//=======================================================================================
int main(int argc, char** argv){
	  
	if (argc < 2){
	    std::cout << "No image data... At least one argument is required! \n";
	    return -1;
	}

	std::cout<< "--------Récupération Image--------" << std::endl;

	std::vector<Mat> imagesBGR = recupImage(argc, argv);
	std::vector<Mat> imagesYCrCb;


  	// Conversion en YCrCb
  	for(int i = 0; i < imagesBGR.size() ; i++)
	{
		Mat imgYCrCb;
		BGRtoYCrCb(imagesBGR[i],imgYCrCb);
		imagesYCrCb.push_back(imgYCrCb);
	}

	std::cout<< "--------Split des Images--------" << std::endl;

	std::vector<std::vector<Mat> > imagesSplit;
	for(int i = 0; i < imagesYCrCb.size(); i++)
	{
		std::vector<Mat> imgSplit;
		split(imagesYCrCb[i],imgSplit);
		imagesSplit.push_back(imgSplit);
	}

	std::cout<< "-------- Compute : Distortion Map, EQM, PSNR --------" << std::endl;

	for(int i = 0; i < imagesSplit.size(); i++)
	{

		std::cout << "Image " << i << std::endl;
		
		if(i != 0)
		{
			std::cout << "EQM : " << eqm(imagesSplit[0][0],imagesSplit[i][0]) << std::endl;
			std::cout << "PSNR : " << psnr(imagesSplit[0][0],imagesSplit[i][0]) << std::endl;
		}

		std::cout << "Entropy : " << computeEntropy(imagesSplit[i][0]) << std::endl;

		Mat myHist;
		computeHistogram(imagesSplit[i][0],myHist);
		imwrite ( "ImageRes/hist_"+ toString(i)+".jpg" , displayHistogram(myHist));

		if(i != 0)
		{
			Mat distoMap;
			distortionMap(imagesSplit[0], imagesSplit[i], distoMap);
			imshow("Distortion Map", distoMap);
		}

		std::cout << "----------------" << std::endl;

		waitKey();
	}
	
	
  return 0;
}
