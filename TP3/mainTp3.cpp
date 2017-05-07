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

string intToString(int i) // convert int to string
{
    std::stringstream value;
    value << i;
    return value.str();
}


string savoirPredi(int i) // convert int to string
{
    switch(i)
			{
				case 1:
					return "MICDmono";
					break;
				case 2:
					return "MICDbi";
					break;
				case 3:
					return "MICDA";
					break;	
				default :
					return "MICDmono";
					break;
	}
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
            entropy += myHist.at<float>(i) * log2(myHist.at<float>(i));            
        }
    }

    return -1 * entropy;

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
float eqm(const Mat & img1, const Mat & img2)
{
	assert(img1.cols == img2.cols && img1.rows == img2.rows);
	float result = 0;
	for(int i = 0; i < img1.rows; i++) {
		for(int j = 0; j < img1.cols; j++) {
			float pixel1 =  img1.at<float>(i, j);
			float pixel2 =  img2.at<float>(i, j);
			result += ( pixel1 - pixel2) * (pixel1 - pixel2);

		}
	}
 return result/(img1.rows * img1.cols);
}

//=======================================================================================
// psnr
//=======================================================================================
float psnr(const Mat & imgSrc, const Mat & imgDeg)
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
	distoMap = imgSrc[0] - imgDeg[0] + 128;
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
// Prédictions MICD
//=======================================================================================
float MICD_mono(const Mat & img, int i, int j)
{
	if(j == 0)
	{
		return 128;
	}
	else
	{
		return img.at<float>(i,j-1);
	}
}

//=======================================================================================
// Prédictions MICD
//=======================================================================================
float MICD_bi(const Mat & img, int i, int j)
{
	if(i == 0 && j == 0)
	{
		return 128;
	}
	else if(i == 0)
	{
		return ((128 + img.at<float>(i,j-1))/2);
	}
	else if(j == 0)
	{
		return ((128 + img.at<float>(i-1,j))/2);
	}
	else
	{
		return ((img.at<float>(i,j-1) + img.at<float>(i-1,j))/2);
	}
}

//=======================================================================================
// Prédictions MICDA
//=======================================================================================
float MICDA(const Mat & img, int i, int j)
{
	if(i == 0 && j == 0)
	{
		return 128;
	}
	else if(i == 0)
	{
		if(0 < abs(img.at<float>(i,j-1)-128))
		{
			return img.at<float>(i,j-1);
		}
		else
		{
			return 128;
		}
	}
	else if(j == 0)
	{
		if(abs(img.at<float>(i-1,j)-128) < 0)
		{
			return 128;
		}
		else
		{
			return img.at<float>(i-1,j);
		}
	}
	else
	{
		if(abs(img.at<float>(i-1,j)-img.at<float>(i-1,j-1)) < abs(img.at<float>(i,j-1)-img.at<float>(i-1,j-1)))
		{
			return img.at<float>(i,j-1);
		}
		else
		{
			return img.at<float>(i-1,j);
		}
	}
}

//=======================================================================================
// Coding
//=======================================================================================
void coding(const Mat & img, Mat & imgPred, int choixPredi, int choixQuantif = 1)
{
	Mat imgDecode(img.rows,img.cols,CV_32FC1);

	int prediction;
	int predictionDequantif;

	switch(choixPredi)
			{
				case 1:
					std::cout << "-------- MICD_mono --------" << std::endl;
					break;
				case 2:
					std::cout << "-------- MICD_bi --------" << std::endl;
					break;
				case 3:
					std::cout << "-------- MICDA --------" << std::endl;
					break;	
				default :
					std::cout << "-------- MICD_mono --------" << std::endl;
					break;
	}

	for(int i =0; i < imgPred.rows; ++i)
	{
		for(int j = 0; j < imgPred.cols; ++j)
		{
 			switch(choixPredi)
			{
				case 1:
					prediction = MICD_mono(imgDecode,i,j);
					break;
				case 2:
					prediction = MICD_bi(imgDecode,i,j);
					break;
				case 3:
					prediction = MICDA(imgDecode,i,j);
					break;	
				default :
					prediction = MICD_mono(imgDecode,i,j);
					break;
			}
			imgPred.at<float>(i,j) = floor((img.at<float>(i,j) - prediction)/choixQuantif + 0.5);
			predictionDequantif = imgPred.at<float>(i,j) * choixQuantif;		
			imgDecode.at<float>(i,j) = predictionDequantif + prediction;
		}
	}

	for(int i =0; i < imgPred.rows; ++i)
	{
		for(int j = 0; j < imgPred.cols; ++j)
		{
			imgPred.at<float>(i,j) += 128;
		}
	}
}

//=======================================================================================
// Decoding
//=======================================================================================
void decoding( Mat & img, Mat & imgDecode, int choixPredi, int choixQuantif = 1)
{

	int prediction;

	for(int i =0; i < img.rows; ++i)
	{
		for(int j = 0; j < img.cols; ++j)
		{
			img.at<float>(i,j) = img.at<float>(i,j) - 128;
		}
	}

	for(int i =0; i < img.rows; i++)
	{
		for(int j = 0; j < img.cols; j++)
		{
			switch(choixPredi)
			{
				case 1:
					prediction = MICD_mono(imgDecode,i,j);
					break;
				case 2:
					prediction = MICD_bi(imgDecode,i,j);
					break;
				case 3:
					prediction = MICDA(imgDecode,i,j);
					break;
				default :
					prediction = MICD_mono(imgDecode,i,j);
					break;
			}
			imgDecode.at<float>(i,j) = choixQuantif*img.at<float>(i,j) + prediction ;
		}
	}
}


//=======================================================================================
// Coding
//=======================================================================================
void codingCompetitif(const Mat & img, Mat & erreur, Mat & imgPred , int choixQuantif = 1)
{
	Mat imgDecode(img.rows,img.cols,CV_32FC1);

	int prediction;
	int predictionDequantif;

	for(int i =0; i < imgPred.rows; ++i)
	{
		for(int j = 0; j < imgPred.cols; ++j)
		{
 			int mono = MICD_mono(imgDecode,i,j);
 			int bi = MICD_bi(imgDecode,i,j);
 			int micda = MICDA(imgDecode,i,j);
 			int absmono = abs(img.at<float>(i,j) - mono);
 			int absbi = abs(img.at<float>(i,j) - bi);
 			int absmicda = abs(img.at<float>(i,j) - micda);
 			
 			if(absmono < absbi && absmono < absmicda)
 			{
 				prediction = MICD_mono(imgDecode,i,j);
				erreur.at<float>(i,j) = 1;
 			}
 			else if(absbi < absmicda)
 			{
 				prediction = MICD_bi(imgDecode,i,j);
				erreur.at<float>(i,j) = 2;
 			}
 			else
 			{
 				prediction = MICDA(imgDecode,i,j);
				erreur.at<float>(i,j) = 3;
 			}
 			
			imgPred.at<float>(i,j) = floor((img.at<float>(i,j) - prediction)/choixQuantif + 0.5);
			predictionDequantif = imgPred.at<float>(i,j) * choixQuantif;		
			imgDecode.at<float>(i,j) = predictionDequantif + prediction;
		}
	}

	for(int i =0; i < imgPred.rows; ++i)
	{
		for(int j = 0; j < imgPred.cols; ++j)
		{
			imgPred.at<float>(i,j) += 128;
		}
	}
}

//=======================================================================================
// Decoding
//=======================================================================================
void decodingCompetitif(Mat & img, Mat & erreur,Mat & imgDecode, int choixQuantif = 1)
{

	std::cout << "Entropie du choix :" << computeEntropy(erreur) << std::endl;

	int prediction;

	for(int i =0; i < img.rows; ++i)
	{
		for(int j = 0; j < img.cols; ++j)
		{
			img.at<float>(i,j) = img.at<float>(i,j) - 128;
		}
	}

	for(int i =0; i < img.rows; i++)
	{
		for(int j = 0; j < img.cols; j++)
		{

			if( erreur.at<float>(i,j) == 1)
			{
				prediction = MICD_mono(imgDecode,i,j);
			}
			else if (erreur.at<float>(i,j) == 2)
			{
				prediction = MICD_bi(imgDecode,i,j);
			}
			else
			{
				prediction = MICDA(imgDecode,i,j);
			}


			imgDecode.at<float>(i,j) = choixQuantif * img.at<float>(i,j) + prediction ;

		}
	}

	imwrite("ImageRes/erreur.jpg", norm_0_255(erreur));
}

//=======================================================================================
// Boucle fermee simple
//=======================================================================================
void boucleFermeeSimple(const Mat & imagesSplit)
{
	std::cout<< "-------- Codage des images --------" << "\n \n" << std::endl;

	Mat imagesCodees;

	int choixQuantif;
	std::cout << "Choix pas de Quantif" << std::endl;
	std::cin >> choixQuantif;

	imwrite ( "ImageRes/Imageoriginale.jpg" , imagesSplit);

	for(int choixPredi = 1; choixPredi < 4; choixPredi++)
	{

		Mat imgPred(imagesSplit.rows,imagesSplit.cols,CV_32FC1);
		coding(imagesSplit, imgPred,choixPredi,choixQuantif);

		imwrite ( "ImageRes/Imagecodee"+ savoirPredi(choixPredi) + "Q" + intToString(choixQuantif) +".jpg" , norm_0_255(imgPred));

		std::cout<< "-------- Histogramme carte d'erreur --------" << std::endl;
		Mat histogram;
		computeHistogram(imgPred,histogram);
		imwrite("ImageRes/HistogrammeErreur"+ savoirPredi(choixPredi) +"Q" + intToString(choixQuantif) +".jpg", displayHistogram(histogram));
		std::cout << "Entropie de l'erreur : " << computeEntropy(imgPred) << std::endl;


		std::cout<< "-------- Decodage des images --------" << std::endl;

		Mat imgDecode(imgPred.rows,imgPred.cols,CV_32FC1);
		decoding(imgPred, imgDecode,choixPredi, choixQuantif);

		imwrite ( "ImageRes/Imagedecodee"+ savoirPredi(choixPredi) +"Q" + intToString(choixQuantif) +".jpg" , imgDecode);

		std::cout<< "-------- Calcul du PSNR --------" << std::endl;
		std::cout << "EQM : " << eqm(imagesSplit,imgDecode) << std::endl;
		std::cout << "PSNR : " << psnr(imagesSplit,imgDecode) << "\n \n" << std::endl;
	}

}



//=======================================================================================
// Boucle fermee avec compétition
//=======================================================================================
void boucleFermeeCompetition(const Mat & imagesSplit)
{
	std::cout<< "-------- Codage des images --------" << "\n \n" << std::endl;

	Mat imagesCodees;

	int choixQuantif;
	std::cout << "Choix pas de Quantif" << std::endl;
	std::cin >> choixQuantif;

	imwrite ( "ImageRes/Imageoriginale.jpg" , imagesSplit);

	Mat imgPred(imagesSplit.rows,imagesSplit.cols,CV_32FC1);
	Mat imgErreur(imagesSplit.rows,imagesSplit.cols,CV_32FC1);
	codingCompetitif(imagesSplit,imgErreur, imgPred,choixQuantif);

	imwrite ( "ImageRes/ImagecodeeCompetitifQ" + intToString(choixQuantif) +".jpg" , norm_0_255(imgPred));

	std::cout<< "-------- Histogramme carte d'erreur --------" << std::endl;
	Mat histogram;
	computeHistogram(imgPred,histogram);
	imwrite("ImageRes/HistogrammeErreurCompetitifQ" + intToString(choixQuantif) +".jpg", displayHistogram(histogram));
	std::cout << "Entropie de l'erreur : " << computeEntropy(imgPred) << std::endl;


	std::cout<< "-------- Decodage des images --------" << std::endl;

	Mat imgDecode(imagesSplit.rows,imagesSplit.cols,CV_32FC1);
	decodingCompetitif(imgPred, imgErreur, imgDecode, choixQuantif);

	imwrite ( "ImageRes/ImagedecodeeCompetitifQ" + intToString(choixQuantif) +".jpg" , imgDecode);

	std::cout<< "-------- Calcul du PSNR --------" << std::endl;
	std::cout << "EQM : " << eqm(imagesSplit,imgDecode) << std::endl;
	std::cout << "PSNR : " << psnr(imagesSplit,imgDecode) << "\n \n" << std::endl;
	
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

	Mat imagesBGR;

	imagesBGR = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	if(!imagesBGR.data ) { // Check for invalid input
    	std::cout <<  "Could not open or find the image " << argv[1] << std::endl ;
		waitKey(0); // Wait for a keystroke in the window
 	}

	Mat imagesYCrCb;

  	// Conversion en YCrCb
	Mat imgYCrCb;
	BGRtoYCrCb(imagesBGR,imgYCrCb);

	Mat imgYCrCb32F(imagesBGR.size(),CV_32FC3);
	imgYCrCb.convertTo(imgYCrCb32F,CV_32FC3);

	std::cout<< "--------Split des Images--------" << std::endl;

	std::vector<Mat> imgSplit;
	split(imgYCrCb32F,imgSplit);

	std::cout << "Entropie de l'image originale : " << computeEntropy(imgSplit[0]) << std::endl;

	boucleFermeeSimple(imgSplit[0]);
	boucleFermeeCompetition(imgSplit[0]);


  return 0;
}
