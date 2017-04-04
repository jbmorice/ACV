#include <iostream>
#include <assert.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

using namespace cv;

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
// Help find on : http://stackoverflow.com/questions/24930134/entropy-for-a-gray-image-in-opencv
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
    float entropy = -1*sum(myHist.mul(logP)).val[0];

    std::cout << "Entropy : "<<entropy << std::endl;
    return entropy;

}

//=======================================================================================
// displayHistogram
//=======================================================================================
void displayHistogram(const Mat& myHist)
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
	imshow("coucou", distoMapSplit[0]);
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

  std::cout << "Nombre argument" << argc << std::endl;
  std::vector<Mat> images = recupImage(argc, argv);


  Mat inputImageSrc1;

  // Ouvrir l'image d'entr�e et v�rifier que l'ouverture du fichier se d�roule normalement
  inputImageSrc1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);

  imshow("InputImageSrcBGR", inputImageSrc1);
  

  if(!inputImageSrc1.data ) { // Check for invalid input
    std::cout <<  "Could not open or find the image " << argv[1] << std::endl ;
		waitKey(0); // Wait for a keystroke in the window
    return -1;
  }

	Mat inputImageSrc2;
	inputImageSrc2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);

  if(!inputImageSrc2.data ) { // Check for invalid input
    std::cout <<  "Could not open or find the image " << argv[2] << std::endl ;
		waitKey(0); // Wait for a keystroke in the window
    return -1;
  }

 
	Mat imageYCrCb1;
	BGRtoYCrCb(inputImageSrc1, imageYCrCb1);

	Mat imageYCrCb2;
	BGRtoYCrCb(inputImageSrc2, imageYCrCb2);

	std::vector<Mat> image1Split;
	split(imageYCrCb1, image1Split);

	std::vector<Mat> image2Split;
	split(imageYCrCb2, image2Split);

  // Visualiser l'image
   imshow("InputImageSrcBGR", inputImageSrc1);
	 imshow("Image1 Y", image1Split[0]);
	 imshow("Image1 Cr", image1Split[1]);
	 imshow("Image1 Cb", image1Split[2]);
	// #TODO Sauver ces images pour le rapport

	std::cout << "EQM : " << eqm(image1Split[0], image2Split[0]) << '\n';
	std::cout << "PSNR : " << psnr(image1Split[0], image2Split[0]) << '\n';
	

	Mat distoMap;
	distortionMap(image1Split, image2Split, distoMap);
	imshow("Distortion Map", distoMap);
	waitKey();

  return 0;
}
