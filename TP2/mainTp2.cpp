#include <iostream>
#include <string>
#include <sstream>
#include <assert.h>
#include <math.h>
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
// nombre d'échantillon d'un histogramme en Mat
//=======================================================================================

int nbSamplesHisto(const Mat& inputHisto) // convert int to string
{
	int res = 0;
    for(int nbRows = 0; nbRows < inputHisto.rows; nbRows++)
    {
    	res += inputHisto.at<float>(nbRows);
    }
    return res;
}

//=======================================================================================
// moyenne d'un histogramme en Mat
//=======================================================================================

float meanHisto(const Mat& inputHisto) // convert int to string
{
	float res = 0;
	float diviseur = nbSamplesHisto(inputHisto);
    for(int nbRows = 0; nbRows < inputHisto.rows; nbRows++)
    {
    	res += inputHisto.at<float>(nbRows) * nbRows;
    }
    return res/diviseur;
}

//=======================================================================================
// moyenne d'un histogramme en Mat
//=======================================================================================

float standartDeviationHisto(const Mat& inputHisto) // convert int to string
{
	float res = 0;
	float diviseur = nbSamplesHisto(inputHisto);
	float moyenne = meanHisto(inputHisto);
    for(int nbRows = 0; nbRows < inputHisto.rows; nbRows++)
    {
    	res += pow((nbRows - moyenne),2) * inputHisto.at<float>(nbRows);
    }
    return sqrt(res/diviseur);
}

//=======================================================================================
// moyenne d'un histogramme en Mat
//=======================================================================================

float kurtosisHisto(const Mat& inputHisto) // convert int to string
{
	float res = 0;
	float diviseur = nbSamplesHisto(inputHisto);
	float moyenne = meanHisto(inputHisto);
	float standartDeviation = standartDeviationHisto(inputHisto);
    for(int nbRows = 0; nbRows < inputHisto.rows; nbRows++)
    {
    	res += pow(((nbRows)-moyenne)/standartDeviation,4) * inputHisto.at<float>(nbRows);
    }
    return res/diviseur;
}

//=======================================================================================
// computeHistogram
//=======================================================================================
Mat absoluteMat(const Mat & mat) {
    Mat res(mat.size(), CV_32FC1);
    for(int i = 0; i < mat.rows; i++) {
        for(int j = 0; j < mat.cols; j++) {
            // std::cout << "i : " << i << " j : " << j << " val : " << mat.at<float>(i, j) << '\n';
            res.at<float>(i, j) = abs(mat.at<float>(i, j));
        }
    }
    return res;
}

void computeHistogram(const Mat& inputComponent, Mat& myHist)
{
	/// Establish the number of bins
	int histSize = 256;
	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 } ;
	const float* histRange = { range };
	bool uniform = true;
	bool accumulate = false;

	// Compute the histograms:
    Mat absInputComponent = abs(inputComponent);

    // // Snippet to check if two matrices are differents
    // Mat diff = absInputComponent != absInputComponent;
    // if (countNonZero(diff) == 0) {
    //     std::cout << "diff" << '\n';
    // }
    // else {
    //     std::cout << "pareil" << '\n';
    // }

	calcHist( &absInputComponent, 1, 0, Mat(), myHist, 1, &histSize, &histRange, uniform, accumulate );
}

//=======================================================================================
// computeEntropy
//=======================================================================================
float computeEntropy(const Mat& inputComponent)
{
	// Create Hist
	Mat myHist;
	computeHistogram(inputComponent,myHist);

	// Stats of Image
	myHist /= inputComponent.total();

    float entropy = 0;
    for(int i = 0; i < myHist.rows; i++) {
        if(myHist.at<float>(i) != 0) {
            entropy += -myHist.at<float>(i) * log2(myHist.at<float>(i));
        }
    }

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
		result = ((imgSrc[i] - imgDeg[i]) + 255) / 2;
		distoMapSplit.push_back(result);
	}
	merge(distoMapSplit, distoMap);

}

//=======================================================================================
// discrete cosine transform
//=======================================================================================
void computeDCT(const vector<Mat> & imgIn, vector<Mat> & imgOut)
{
    for(int i = 0; i < 3; i++) {
        // Mat dctRes(imgIn[i].size(), CV_32F);
        Mat dctRes;
        dct(imgIn[i], dctRes);
        imgOut.push_back(dctRes);
    }

}

void computeInverseDCT(const vector<Mat> & imgIn, vector<Mat> & imgOut)
{
    for(int i = 0; i < 3; i++) {
        // Mat iDctRes(imgIn[i].size(), CV_32F);
        Mat iDctRes;
        idct(imgIn[i], iDctRes);
        imgOut.push_back(iDctRes);
    }

}

void visualizeDCT(const vector<Mat> & img)
{
    for(int k = 0; k < 3; k++) {
        double maxVal;
        minMaxLoc(img[k], NULL, &maxVal, NULL, NULL);

        Mat res(img[k].size(), CV_32FC1);

        for(int i = 0; i < img[k].rows; i++) {
            for(int j = 0; j < img[k].cols; j++) {
                res.at<float>(i, j) = log(1 + fabs(img[k].at<float>(i, j))) / log(1 + maxVal) * 255;
            }
        }

        Mat in(res.size(), CV_8UC1);
        res.convertTo(in, CV_8UC1);
        Mat out(res.size(), CV_8UC1);
        applyColorMap(in, out, COLORMAP_JET);
        imshow("DCT", out);
        waitKey();

    }

}

void visualizeDCTHistograms(vector<Mat> & imgIn) {
    for(int k = 0; k < 3; k++) {
        Mat hist;

        computeHistogram(norm_0_255(imgIn[k]), hist);
        displayHistogram(hist);

        // #TODO les coeffs sont signés, il faudra modifier le calcul de H
        std::cout << "Entropy DCT canal " << toString(k) << " : " << computeEntropy(imgIn[k]) << '\n';
    }
}

//=======================================================================================
// convert a image from BGR to YCrCb
//=======================================================================================
void BGRtoYCrCb(const Mat & imgSrc, Mat & imgOut)
{
	cvtColor(imgSrc, imgOut, CV_BGR2YCrCb);
}

//=======================================================================================
// convert a image from YCrCb to BGR
//=======================================================================================
void YCrCbtoBGR(const Mat & imgSrc, Mat & imgOut)
{
	cvtColor(imgSrc, imgOut, CV_YCrCb2BGR);
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

    Mat inputImage;
    inputImage = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    if(!inputImage.data ) { // Check for invalid input
        std::cout <<  "Could not open or find the image " << argv[1] << std::endl;
        waitKey(0); // Wait for a keystroke in the window
     }

	std::cout<< "-------- Compute : DCT --------" << std::endl;

    // Convertion de l'image source de uchar vers float 32 bits
    Mat img32F(inputImage.size(), CV_32F);
    inputImage.convertTo(img32F, CV_32F);

    // Conversion de BGR float 32 bits vers YCrCb float 32 bits
    Mat imgYCrCb32F(img32F.size(), CV_32F);
    BGRtoYCrCb(img32F, imgYCrCb32F);

    // Decomposition des canaux
    std::vector<Mat> imgYCrCb32FSplit;
    split(imgYCrCb32F, imgYCrCb32FSplit);

    // Affichage du canal Y
    imshow("Canal Y", norm_0_255(imgYCrCb32FSplit[0]));
    waitKey();

    // Calcul de la DCT
    std::vector<Mat> dctImgYCrCb32FSplit;
    computeDCT(imgYCrCb32FSplit, dctImgYCrCb32FSplit);

    // Affichage des coeffs de la DCT
    visualizeDCT(dctImgYCrCb32FSplit);

    // Calcul de la DCT inverse
    std::vector<Mat> iDctImgYCrCb32FSplit;
    computeInverseDCT(dctImgYCrCb32FSplit, iDctImgYCrCb32FSplit);

    // Affichage du canal Y de la DCT inverse
    imshow("Canal Y inverse", norm_0_255(iDctImgYCrCb32FSplit[0]));
    waitKey();

    // Fusion des canaux YCrCb
    Mat iDctImgYCrCb32F;
    merge(iDctImgYCrCb32FSplit, iDctImgYCrCb32F);

    // Conversion de YCrCb vers BGR (float 32 bits)
    Mat iDctImgBGR32F;
    YCrCbtoBGR(iDctImgYCrCb32F, iDctImgBGR32F);

    // Conversion en uchar pour l'affichage
    Mat iDctImg;
    iDctImgBGR32F.convertTo(iDctImg, CV_8UC3);

    // Affichage de la DCT inverse
    imshow("IDCT", iDctImg);
	waitKey();

    // std::vector<Mat> inputImageSplit;
    // split(inputImage, inputImageSplit);
    //
    // std::vector<Mat> iDctImageSplit;
    // split(iDctImg, iDctImageSplit);
    //
    // std::cout << "EQM : " << eqm(inputImageSplit[0], iDctImageSplit[0]) << '\n';
    // std::cout << "PSNR : " << psnr(inputImageSplit[0], iDctImageSplit[0]) << '\n';

    return 0;
}
