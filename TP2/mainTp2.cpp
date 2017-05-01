#include <iostream>
#include <string>
#include <sstream>
#include <assert.h>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

using namespace cv;

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
	waitKey();
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
void computeDCT(const vector<Mat> & imgIn, vector<Mat> & imgOut, bool inverse = false)
{
    for(int i = 0; i < 3; i++) {
        Mat dctRes(imgIn[i].size(), CV_32FC1);

        if(!inverse) {
            dct(imgIn[i], dctRes);
        }
        if (inverse) {
            idct(imgIn[i], dctRes);
        }

        imgOut.push_back(dctRes);
    }

}

//=======================================================================================
// inverse discrete cosine transform
//=======================================================================================
void computeInverseDCT(const vector<Mat> & imgIn, vector<Mat> & imgOut)
{
    computeDCT(imgIn, imgOut, true);
}

//=======================================================================================
// convert a matrix of dct coeficients to a visualizable image
//=======================================================================================
Mat visualizeDCT(const Mat & img)
{
    double maxVal;
    minMaxLoc(img, NULL, &maxVal, NULL, NULL);

    Mat res(img.size(), CV_32FC1);

    for(int i = 0; i < img.rows; i++) {
        for(int j = 0; j < img.cols; j++) {
            res.at<float>(i, j) = log(1 + fabs(img.at<float>(i, j))) / log(1 + maxVal) * 255;
        }
    }

    Mat in(res.size(), CV_8UC1);
    res.convertTo(in, CV_8UC1);
    Mat out(res.size(), CV_8UC1);
    applyColorMap(in, out, COLORMAP_JET);

    return out;
}

//=======================================================================================
// convert a vector of matrices of dct coeficients to a vector of visualizable images
//=======================================================================================
void visualizeDCT(const vector<Mat> & img)
{
    for(int k = 0; k < 3; k++) {
        Mat res;
        res = visualizeDCT(img[k]);

        imshow("DCT", res);
        waitKey();

    }

}

void visualizeDCTHistograms(vector<Mat> & imgIn) {
    for(int k = 0; k < 3; k++) {
        Mat hist;

        computeHistogram(norm_0_255(imgIn[k]), hist);
        displayHistogram(hist);

        std::cout << "Entropy DCT canal " << std::to_string(k) << " : " << computeEntropy(imgIn[k]) << '\n';
    }
}

//=======================================================================================
// nullify dct coeficients according to several masks
//=======================================================================================
void nullifyCoefficients(const vector<Mat> & imgIn, vector<Mat> & imgOut, int i = 0)
{

    int x, y, width, height;

    int nbRows = imgIn[0].rows;
    int nbCols = imgIn[0].cols;

  Rect mask;

  switch (i) {
      case 0:
      {
          x = (nbCols/2);
          y = (nbRows/2);
          width = nbCols - x;
          height = nbRows - y;

          mask = Rect(x, y, width, height);

          break;
      }

      case 1:
      {
          x = (nbCols/2);
          y = 0;
          width = nbCols - x;
          height = nbRows - y;

          mask = Rect(x, y, width, height);

          break;
      }

      case 2:
      {
          x = 0;
          y = (nbRows/2);
          width = nbCols - x;
          height = nbRows - y;

          mask = Rect(x, y, width, height);

          break;
      }

      case 3:
      {
          x = 0;
          y = (nbRows/2);
          width = (nbCols/2);
          height = nbRows - y;

          mask = Rect(x, y, width, height);

          break;
      }
  }

  for(int k = 0; k < 3; k ++){
      Mat res;
      imgIn[k].copyTo(res);
      res(mask).setTo(0);
      imgOut.push_back(res);
  }
}

//=======================================================================================
// discrete cosine transform on 8x8 blocks
//=======================================================================================
void computeBlockDCT(const vector<Mat> & imgIn, vector<Mat> & imgOut, bool inverse = false) {
    for(int k = 0; k < 3; k++) {
        Mat res(imgIn[k].size(), CV_32FC1);

        for(int i = 0; i < imgIn[k].rows; i += 8) {
            for(int j = 0; j < imgIn[k].cols; j += 8) {
                Rect window(i, j, 8, 8);
                Mat block = imgIn[k](window);

                if(!inverse) {
                    dct(block, res(window));
                }
                if (inverse) {
                    idct(block, res(window));
                }

            }
        }

        imgOut.push_back(res);
    }
}

//=======================================================================================
// inverse discrete cosine transform on 8x8 blocks
//=======================================================================================
void computeInverseBlockDCT(const vector<Mat> & imgIn, vector<Mat> & imgOut)
{
    computeBlockDCT(imgIn, imgOut, true);
}

//=======================================================================================
// apply a nullifying mask on dct coeficients in 8x8 blocks
//=======================================================================================
void applyBlockMask(const vector<Mat> & imgIn, vector<Mat> & imgOut) {
    Mat_<float> m1(8, 8);
    m1 << 1, 1, 1, 1, 1, 0, 0, 0,
          1, 1, 1, 1, 0, 0, 0, 0,
          1, 1, 1, 0, 0, 0, 0, 0,
          1, 1, 0, 0, 0, 0, 0, 0,
          1, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0;
    Mat mask = m1;

    for(int k = 0; k < 3; k++) {
        Mat res(imgIn[k].size(), CV_32FC1);

        for(int i = 0; i < imgIn[k].rows; i += 8) {
            for(int j = 0; j < imgIn[k].cols; j += 8) {
                Rect window(i, j, 8, 8);
                Mat block = imgIn[k](window);

                res(window) = block.mul(mask);

            }
        }

        imgOut.push_back(res);
    }
}

//=======================================================================================
// apply a JPEG transformation on dct coeficients in 8x8 blocks
//=======================================================================================
void applyBlockTransform(const vector<Mat> & imgIn, vector<Mat> & imgOut, bool inverse = false) {
    Mat_<float> m1(8, 8);
    m1 << 16, 11, 10, 16, 24, 40, 51, 61,
          12, 12, 14, 19, 26, 58, 60, 55,
          14, 13, 16, 24, 40, 57, 69, 56,
          14, 17, 22, 29, 51, 87, 80, 62,
          18, 22, 37, 56, 68, 109, 103, 77,
          24, 35, 55, 64, 81, 104, 113, 92,
          49, 64, 78, 87, 103, 121, 120, 101,
          72, 92, 95, 98, 112, 100, 103, 99;
    Mat mask = m1;

    for(int k = 0; k < 3; k++) {
        Mat res(imgIn[k].size(), CV_32FC1);

        for(int i = 0; i < imgIn[k].rows; i += 8) {
            for(int j = 0; j < imgIn[k].cols; j += 8) {
                Rect window(i, j, 8, 8);
                Mat block = imgIn[k](window);

                for (int i_mask = 0; i_mask < mask.rows; i_mask++) {
                    for (int j_mask = 0; j_mask < mask.rows; j_mask++) {
                        if (!inverse) {
                            res(window).at<float>(i_mask, j_mask) = round(block.at<float>(i_mask, j_mask) / mask.at<float>(i_mask, j_mask));
                        }
                        if (inverse) {
                            res(window).at<float>(i_mask, j_mask) = block.at<float>(i_mask, j_mask) * mask.at<float>(i_mask, j_mask);
                        }
                    }
                }
            }
        }

        imgOut.push_back(res);
    }
}

//=======================================================================================
// apply inverse JPEG transformation on dct coeficients in 8x8 blocks
//=======================================================================================
void applyInverseBlockTransform(const vector<Mat> & imgIn, vector<Mat> & imgOut)
{
    applyBlockTransform(imgIn, imgOut, true);
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

    Mat src;
    src = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    if(!src.data ) { // Check for invalid input
        std::cout <<  "Could not open or find the image " << argv[1] << std::endl;
        waitKey(0); // Wait for a keystroke in the window
    }

    std::vector<Mat> srcVector;
    split(src, srcVector);

    // Convertion de l'image source de uchar vers float 32 bits
    Mat src32F(src.size(), CV_32FC3);
    src.convertTo(src32F, CV_32FC3);

    // Conversion de BGR float 32 bits vers YCrCb float 32 bits
    Mat srcYCrCb(src.size(), CV_32FC3);
    BGRtoYCrCb(src32F, srcYCrCb);

    // Decomposition des canaux
    std::vector<Mat> imgVector;
    split(srcYCrCb, imgVector);

    bool save = false;

    if (save) {

        for (int i = 0; i < 3; i++) {
            Mat hist;
            computeHistogram(norm_0_255(imgVector[i]), hist);
            std::cout << "Entropy DCT canal " << std::to_string(i) << " : " << computeEntropy(imgVector[i]) << '\n';
            imwrite("ImageRes/orig_histo_"+ std::to_string(i) +".jpg", displayHistogram(hist));

        }

    }

    std::cout<< "-------- TP2 : 2D Discrete Cosine Transform --------" << std::endl;
    std::cout << "1. DCT and inverse DCT" << std::endl;
    std::cout << "2. DCT and inverse DCT with nullified coefficients" << std::endl;
    std::cout << "3. 8x8 block DCT (and inverse)" << std::endl;
    std::cout << "4. 8x8 block DCT (and inverse) with simple binary mask" << std::endl;
    std::cout << "5. 8x8 block DCT (and inverse) with JPEG ponderation matrix" << std::endl;
    std::cout << "0. Exit program" << std::endl;
    std::cout << "Choose an action to perform : ";

    int choice;
    std::cin >> choice;

    switch (choice) {
        case 0:
        {
            std::cout << "Exiting program..." << '\n';
        }

        case 1:
        {
            std::vector<Mat> dctVector;
            computeDCT(imgVector, dctVector);
            visualizeDCT(dctVector);
            visualizeDCTHistograms(dctVector);

            std::vector<Mat> idctVector;
            computeInverseDCT(dctVector, idctVector);

            Mat finalImg;
            merge(idctVector, finalImg);
            YCrCbtoBGR(finalImg, finalImg);
            finalImg.convertTo(finalImg, CV_8UC3);
            imshow("IDCT", finalImg);
            waitKey();

            std::vector<Mat> finalImgVector;
            split(finalImg, finalImgVector);

            std::cout << "EQM : " << eqm(srcVector[0], finalImgVector[0]) << std::endl;
            std::cout << "PSNR : " << psnr(srcVector[0], finalImgVector[0]) << std::endl;

            if (save) {
                std::cout << "Saving..." << '\n';

                for (int i = 0; i < 3; i++) {
                    imwrite("ImageRes/dct_"+ std::to_string(i) +".jpg", visualizeDCT(dctVector[i]));
                    Mat hist;
                    computeHistogram(norm_0_255(dctVector[i]), hist);
                    imwrite("ImageRes/dct_histo_"+ std::to_string(i) +".jpg", displayHistogram(hist));

                }

                imwrite("ImageRes/idct_result.jpg", finalImg);

                std::cout << "Finished" << '\n';

            }

            dctVector.clear();
            idctVector.clear();
            finalImg.release();
            finalImgVector.clear();

            break;
        }

        case 2:
        {
            std::vector<Mat> dctVector;
            computeDCT(imgVector, dctVector);

            std::cout << "Which mask do you want to use ?" << std::endl;
            std::cout << "1. Bottom right hand corner" << std::endl;
            std::cout << "2. Right half" << std::endl;
            std::cout << "3. Bottom half" << std::endl;
            std::cout << "4. Bottom left hand corner" << std::endl;

            int maskIndex;
            std::cin >> maskIndex;

            std::vector<Mat> modDctVector;
            nullifyCoefficients(dctVector, modDctVector, maskIndex - 1);

            visualizeDCT(modDctVector);
            visualizeDCTHistograms(modDctVector);

            std::vector<Mat> idctVector;
            computeInverseDCT(modDctVector, idctVector);

            Mat finalImg;
            merge(idctVector, finalImg);
            YCrCbtoBGR(finalImg, finalImg);
            finalImg.convertTo(finalImg, CV_8UC3);
            imshow("IDCT", finalImg);
            waitKey();

            std::vector<Mat> finalImgVector;
            split(finalImg, finalImgVector);

            std::cout << "EQM : " << eqm(srcVector[0], finalImgVector[0]) << std::endl;
            std::cout << "PSNR : " << psnr(srcVector[0], finalImgVector[0]) << std::endl;

            if (save) {
                std::cout << "Saving..." << '\n';

                for (int i = 0; i < 3; i++) {
                    imwrite("ImageRes/dct_masked"+ std::to_string(maskIndex - 1) +"_"+ std::to_string(i) +".jpg", visualizeDCT(modDctVector[i]));
                    Mat hist;
                    computeHistogram(norm_0_255(modDctVector[i]), hist);
                    imwrite("ImageRes/dct_masked"+ std::to_string(maskIndex - 1) +"_histo_"+ std::to_string(i) +".jpg", displayHistogram(hist));

                }

                imwrite("ImageRes/idct_masked"+ std::to_string(maskIndex - 1) +"_result.jpg", finalImg);

                Mat distoMap;
                distortionMap(srcVector, finalImgVector, distoMap);
                imwrite("ImageRes/idct_masked"+ std::to_string(maskIndex - 1) +"_disto.jpg", distoMap);

                std::cout << "Finished" << '\n';

            }

            dctVector.clear();
            idctVector.clear();
            finalImg.release();
            finalImgVector.clear();

            break;
        }

        case 3:
        {
            std::vector<Mat> dctVector;
            computeBlockDCT(imgVector, dctVector);
            visualizeDCT(dctVector);
            visualizeDCTHistograms(dctVector);

            std::vector<Mat> idctVector;
            computeInverseBlockDCT(dctVector, idctVector);

            Mat finalImg;
            merge(idctVector, finalImg);
            YCrCbtoBGR(finalImg, finalImg);
            finalImg.convertTo(finalImg, CV_8UC3);
            imshow("IDCT", finalImg);
            waitKey();

            std::vector<Mat> finalImgVector;
            split(finalImg, finalImgVector);

            std::cout << "EQM : " << eqm(srcVector[0], finalImgVector[0]) << std::endl;
            std::cout << "PSNR : " << psnr(srcVector[0], finalImgVector[0]) << std::endl;

            if (save) {
                std::cout << "Saving..." << '\n';

                for (int i = 0; i < 3; i++) {
                    imwrite("ImageRes/blockdct_"+ std::to_string(i) +".jpg", visualizeDCT(dctVector[i]));
                    Mat hist;
                    computeHistogram(norm_0_255(dctVector[i]), hist);
                    imwrite("ImageRes/blockdct_histo_"+ std::to_string(i) +".jpg", displayHistogram(hist));

                }

                imwrite("ImageRes/blockidct_result.jpg", finalImg);

                std::cout << "Finished" << '\n';

            }

            dctVector.clear();
            idctVector.clear();
            finalImg.release();
            finalImgVector.clear();

            break;
        }

        case 4:
        {
            std::vector<Mat> dctVector;
            computeBlockDCT(imgVector, dctVector);

            std::vector<Mat> modDctVector;
            applyBlockMask(dctVector, modDctVector);

            visualizeDCT(modDctVector);
            visualizeDCTHistograms(modDctVector);

            std::vector<Mat> idctVector;
            computeInverseBlockDCT(modDctVector, idctVector);

            Mat finalImg;
            merge(idctVector, finalImg);
            YCrCbtoBGR(finalImg, finalImg);
            finalImg.convertTo(finalImg, CV_8UC3);
            imshow("IDCT", finalImg);
            waitKey();

            std::vector<Mat> finalImgVector;
            split(finalImg, finalImgVector);

            std::cout << "EQM : " << eqm(srcVector[0], finalImgVector[0]) << std::endl;
            std::cout << "PSNR : " << psnr(srcVector[0], finalImgVector[0]) << std::endl;

            if (save) {
                std::cout << "Saving..." << '\n';

                for (int i = 0; i < 3; i++) {
                    imwrite("ImageRes/blockdct_masked_"+ std::to_string(i) +".jpg", visualizeDCT(modDctVector[i]));
                    Mat hist;
                    computeHistogram(norm_0_255(modDctVector[i]), hist);
                    imwrite("ImageRes/blockdct_masked_histo_"+ std::to_string(i) +".jpg", displayHistogram(hist));

                }

                imwrite("ImageRes/blockidct_masked_result.jpg", finalImg);

                Mat distoMap;
                distortionMap(srcVector, finalImgVector, distoMap);
                imwrite("ImageRes/blockidct_masked_disto.jpg", distoMap);

                std::cout << "Finished" << '\n';

            }

            dctVector.clear();
            idctVector.clear();
            finalImg.release();
            finalImgVector.clear();

            break;
        }

        case 5:
        {
            std::vector<Mat> dctVector;
            computeBlockDCT(imgVector, dctVector);

            std::vector<Mat> modDctVector;
            applyBlockTransform(dctVector, modDctVector);

            visualizeDCT(modDctVector);
            visualizeDCTHistograms(modDctVector);

            std::vector<Mat> iModDctVector;
            applyInverseBlockTransform(modDctVector, iModDctVector);

            std::vector<Mat> idctVector;
            computeInverseBlockDCT(iModDctVector, idctVector);

            Mat finalImg;
            merge(idctVector, finalImg);
            YCrCbtoBGR(finalImg, finalImg);
            finalImg.convertTo(finalImg, CV_8UC3);
            imshow("IDCT", finalImg);
            waitKey();

            std::vector<Mat> finalImgVector;
            split(finalImg, finalImgVector);

            std::cout << "EQM : " << eqm(srcVector[0], finalImgVector[0]) << std::endl;
            std::cout << "PSNR : " << psnr(srcVector[0], finalImgVector[0]) << std::endl;

            if (save) {
                std::cout << "Saving..." << '\n';

                for (int i = 0; i < 3; i++) {
                    imwrite("ImageRes/blockdct_transform_"+ std::to_string(i) +".jpg", visualizeDCT(modDctVector[i]));
                    Mat hist;
                    computeHistogram(norm_0_255(modDctVector[i]), hist);
                    imwrite("ImageRes/blockdct_transform_histo_"+ std::to_string(i) +".jpg", displayHistogram(hist));

                }

                imwrite("ImageRes/blockidct_transform_result.jpg", finalImg);

                Mat distoMap;
                distortionMap(srcVector, finalImgVector, distoMap);
                imwrite("ImageRes/blockidct_transform_disto.jpg", distoMap);

                std::cout << "Finished" << '\n';

            }

            dctVector.clear();
            idctVector.clear();
            finalImg.release();
            finalImgVector.clear();

            break;
        }

        default:
            std::cout << "This is not a valid action. Please try again." << std::endl;
            break;
    }

    return 0;
}
