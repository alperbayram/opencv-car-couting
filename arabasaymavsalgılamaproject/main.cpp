// main.cpp

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include<iostream>
#include<conio.h>           // windows kullanmayanlar bu satýrý kaldýrabilir.

#include "Blob.h"

#define SHOW_STEPS           

using namespace std;

// global variables ///////////////////////////////////////////////////////////////////////////////

//renklerin derecelerini ayarlanýr
const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);

/////////////////////////   fonksiyon prototipleri ////////////////////////////////////////////////////////////////////////////
void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob>& existingBlobs, std::vector<Blob>& currentFrameBlobs);
void addBlobToExistingBlobs(Blob& currentFrameBlob, std::vector<Blob>& existingBlobs, int& intIndex);
void addNewBlob(Blob& currentFrameBlob, std::vector<Blob>& existingBlobs);
double distanceBetweenPoints(cv::Point point1, cv::Point point2);
void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName);
void drawAndShowContours(cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName);
bool checkIfBlobsCrossedTheLine(std::vector<Blob>& blobs, int& intHorizontalLinePosition, int& carCount);
void drawBlobInfoOnImage(std::vector<Blob>& blobs, cv::Mat& imgFrame2Copy);
void drawCarCountOnImage(int& carCount, cv::Mat& imgFrame2Copy);

///////////////////////////////////////////////////////////////////////////////////////////////////
int main(void) {

    cv::VideoCapture capVideo;

    cv::Mat imgFrame1;
    cv::Mat imgFrame2;

    std::vector<Blob> blobs;

    cv::Point crossingLine[2];

    int carCount = 0;

    capVideo.open("video2.mp4"); //videonun konumunun ve isminin girilmesi

    if (!capVideo.isOpened()) {                                                 //video dosyasý açýlamýyorsa
        std::cout << "video bulunamadý...video yükleyiniz..." << std::endl << std::endl;      //hata mesajýný göster
        _getch();                   // Windows kullanmýyorsa bu satýrý deðiþtirmek veya kaldýrmak gerekebilir
        return(0);                                                              // and exit program
    }

    if (capVideo.get(CV_CAP_PROP_FRAME_COUNT) < 2) { //eðer 2 kareden kýsa bir video ise uyarý mesajý gönder
        std::cout << "hata: video dosyasýnýn en az iki karesi olmalýdýr";
        _getch();                   //Windows kullanmýyorsa bu satýrý deðiþtirmek veya kaldýrmak gerekebilir
    }

    capVideo.read(imgFrame1);
    capVideo.read(imgFrame2);

    int intHorizontalLinePosition = (int)std::round((double)imgFrame1.rows * 0.35); //yatay çzigi konumu

    crossingLine[0].x = 300;
    crossingLine[0].y = intHorizontalLinePosition;

    crossingLine[1].x = 950;
    crossingLine[1].y = intHorizontalLinePosition;

    char chCheckForEscKey = 0;

    bool blnFirstFrame = true;

    int frameCount = 2;

    while (capVideo.isOpened() && chCheckForEscKey != 27) {

        std::vector<Blob> currentFrameBlobs; //gerekli çerçeveleri oluþturur

        cv::Mat imgFrame1Copy = imgFrame1.clone(); //birinci çerçeve görüntüsü
        cv::Mat imgFrame2Copy = imgFrame2.clone(); //ikinci çerçeve "

        cv::Mat imgDifference;  //fark komutu
        cv::Mat imgThresh;     //

        cv::cvtColor(imgFrame1Copy, imgFrame1Copy, CV_BGR2GRAY); //gri seviyey çevirme görüntüyü
        cv::cvtColor(imgFrame2Copy, imgFrame2Copy, CV_BGR2GRAY);

        cv::GaussianBlur(imgFrame1Copy, imgFrame1Copy, cv::Size(5, 5), 0);  //gauss komutu
        cv::GaussianBlur(imgFrame2Copy, imgFrame2Copy, cv::Size(5, 5), 0);

        cv::absdiff(imgFrame1Copy, imgFrame2Copy, imgDifference);  //fark komutu

        cv::threshold(imgDifference, imgThresh, 30, 255.0, CV_THRESH_BINARY); //eþik bulma komuutu

        cv::imshow("imgThresh", imgThresh); //eþiklenmiþ görüntüüü

        cv::Mat structuringElement3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::Mat structuringElement7x7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
        cv::Mat structuringElement15x15 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));

        for (unsigned int i = 0; i < 2; i++) {
            cv::dilate(imgThresh, imgThresh, structuringElement5x5);
            cv::dilate(imgThresh, imgThresh, structuringElement5x5);
            cv::erode(imgThresh, imgThresh, structuringElement5x5);
        }

        cv::Mat imgThreshCopy = imgThresh.clone();

        std::vector<std::vector<cv::Point> > contours;

        cv::findContours(imgThreshCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        drawAndShowContours(imgThresh.size(), contours, "imgContours");

        std::vector<std::vector<cv::Point> > convexHulls(contours.size());

        for (unsigned int i = 0; i < contours.size(); i++) {
            cv::convexHull(contours[i], convexHulls[i]);
        }

        drawAndShowContours(imgThresh.size(), convexHulls, "imgConvexHulls"); //dýþbükey görüntülerin elde edilmesi

        for (auto& convexHull : convexHulls) {
            Blob possibleBlob(convexHull);

            if (possibleBlob.currentBoundingRect.area() > 400 &&
                possibleBlob.dblCurrentAspectRatio > 0.2 &&
                possibleBlob.dblCurrentAspectRatio < 4.0 &&
                possibleBlob.currentBoundingRect.width > 30 &&
                possibleBlob.currentBoundingRect.height > 30 &&
                possibleBlob.dblCurrentDiagonalSize > 60.0 &&
                (cv::contourArea(possibleBlob.currentContour) / (double)possibleBlob.currentBoundingRect.area()) > 0.50) {
                currentFrameBlobs.push_back(possibleBlob);
            }
        }

        drawAndShowContours(imgThresh.size(), currentFrameBlobs, "geçerli kare bloblarý"); //bloblarý çizer

        if (blnFirstFrame == true) {
            for (auto& currentFrameBlob : currentFrameBlobs) {
                blobs.push_back(currentFrameBlob);
            }
        }
        else {
            matchCurrentFrameBlobsToExistingBlobs(blobs, currentFrameBlobs); //geçerli Çerçeve Bloblarýný Mevcut Bloblarla Eþleþtir
        }

        drawAndShowContours(imgThresh.size(), blobs, "blob görüntüsü");

        imgFrame2Copy = imgFrame2.clone();          // yukarýdaki iþlemde önceki kare 2 kopyasýný deðiþtirdiðimiz için kare 2'nin baþka bir kopyasýný al

        drawBlobInfoOnImage(blobs, imgFrame2Copy);

        bool blnAtLeastOneBlobCrossedTheLine = checkIfBlobsCrossedTheLine(blobs, intHorizontalLinePosition, carCount); //Bloblarýn Çizgiyi Aþýp Geçmediðini kontrol edin

        if (blnAtLeastOneBlobCrossedTheLine == true) {  //En az bir damla çizgi geçti ise
            cv::line(imgFrame2Copy, crossingLine[0], crossingLine[1], SCALAR_GREEN, 2);
        }
        else {
            cv::line(imgFrame2Copy, crossingLine[0], crossingLine[1], SCALAR_RED, 2);
        }

        drawCarCountOnImage(carCount, imgFrame2Copy);

        cv::imshow("imgFrame2Copy", imgFrame2Copy); //2 fotoun kopyasý

        //cv::waitKey(0);                 
        // þimdi bir sonraki tekrarlamaya hazýrlanýyoruz

        currentFrameBlobs.clear();

        imgFrame1 = imgFrame2.clone();           // kare 1'i kare 2'nin olduðu yere klonla

        if ((capVideo.get(CV_CAP_PROP_POS_FRAMES) + 1) < capVideo.get(CV_CAP_PROP_FRAME_COUNT)) {
            capVideo.read(imgFrame2);
        }
        else {
            std::cout << "video bitti\n";
            break;
        }

        blnFirstFrame = false;
        frameCount++;
        chCheckForEscKey = cv::waitKey(1);
    }

    if (chCheckForEscKey != 27) {               // kullanýcý esc tuþuna basmadýysa (yani videonun sonuna ulaþtýk)
        cv::waitKey(0);                         // "video bitti" mesajýnýn gösterilmesine izin vermek için pencereleri açýk tutun
    }
    // Eðer kullanýcý esc tuþuna basmýþsa, pencereleri açýk tutmamýz gerekmediðini, programýn pencereleri kapatacak þekilde bitmesine izin verebileceðimizi unutmayýn

    return(0);
}

////////////////////1 FONKSÝYON = geçerli Çerçeve Bloblarýný Mevcut Bloblarla Eþleþtir  ////////////////////////////////////////////////
void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob>& existingBlobs, std::vector<Blob>& currentFrameBlobs) { //geçerli Çerçeve Bloblarýný Mevcut Bloblarla Eþleþtir

    for (auto& existingBlob : existingBlobs) {

        existingBlob.blnCurrentMatchFoundOrNewBlob = false;

        existingBlob.predictNextPosition(); //Sonraki Konumu tahmin et
    }

    for (auto& currentFrameBlob : currentFrameBlobs) { //geçerli Çerçeve Bloblarý

        int intIndexOfLeastDistance = 0; //En Küçük Mesafe Endeksi
        double dblLeastDistance = 100000.0;

        for (unsigned int i = 0; i < existingBlobs.size(); i++) {

            if (existingBlobs[i].blnStillBeingTracked == true) { //Hala Ýzleniyor

                double dblDistance = distanceBetweenPoints(currentFrameBlob.centerPositions.back(), existingBlobs[i].predictedNextPosition); //Noktalar Arasý Mesafe

                if (dblDistance < dblLeastDistance) {
                    dblLeastDistance = dblDistance;
                    intIndexOfLeastDistance = i;
                }
            }
        }

        if (dblLeastDistance < currentFrameBlob.dblCurrentDiagonalSize * 0.5) {
            addBlobToExistingBlobs(currentFrameBlob, existingBlobs, intIndexOfLeastDistance);//Mevcut Bloblara Blob ekle
        }
        else {
            addNewBlob(currentFrameBlob, existingBlobs); //yeni blob
        }

    }

    for (auto& existingBlob : existingBlobs) {

        if (existingBlob.blnCurrentMatchFoundOrNewBlob == false) { 
            existingBlob.intNumOfConsecutiveFramesWithoutAMatch++;
        }

        if (existingBlob.intNumOfConsecutiveFramesWithoutAMatch >= 5) {
            existingBlob.blnStillBeingTracked = false;
        }

    }

}

///////////////////////////////  2. FONKSÝYON MEVCUT BULAPLARI EKLER  ///////////////////////////////////////////////
void addBlobToExistingBlobs(Blob& currentFrameBlob, std::vector<Blob>& existingBlobs, int& intIndex) {

    existingBlobs[intIndex].currentContour = currentFrameBlob.currentContour;
    existingBlobs[intIndex].currentBoundingRect = currentFrameBlob.currentBoundingRect;

    existingBlobs[intIndex].centerPositions.push_back(currentFrameBlob.centerPositions.back());

    existingBlobs[intIndex].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize;
    existingBlobs[intIndex].dblCurrentAspectRatio = currentFrameBlob.dblCurrentAspectRatio;

    existingBlobs[intIndex].blnStillBeingTracked = true;
    existingBlobs[intIndex].blnCurrentMatchFoundOrNewBlob = true;
}

///////////////////////////// 3. FONKSÝÝYON  YENÝ BLOB EKLER /////////////////////////////////////////////////////
void addNewBlob(Blob& currentFrameBlob, std::vector<Blob>& existingBlobs) {

    currentFrameBlob.blnCurrentMatchFoundOrNewBlob = true;

    existingBlobs.push_back(currentFrameBlob);
}

//////////////////////////////// 4. NOKTALAR ARASI MESAFEYE DEÐER ATAMA YAPAR /////////////////////////////////////////////////////
double distanceBetweenPoints(cv::Point point1, cv::Point point2) {

    int intX = abs(point1.x - point2.x);
    int intY = abs(point1.y - point2.y);

    return(sqrt(pow(intX, 2) + pow(intY, 2)));
}

//////////////////////////////  5. FONKSÝYON  /////////////////////////////////////////////////////
void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName) {
    cv::Mat image(imageSize, CV_8UC3, SCALAR_BLACK);

    cv::drawContours(image, contours, -1, SCALAR_WHITE, -1);

    cv::imshow(strImageName, image); // kontür GÖRÜNTÜSÜNÜN ÇIKTISINI VERÝR.
}

/////////////////////////////// 6. FONKSÝYON ARAÇLARIN DIÞ HADLARINI ÇÝZER ///////////////////////////////////////////////////
void drawAndShowContours(cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName) {

    cv::Mat image(imageSize, CV_8UC3, SCALAR_BLACK);

    std::vector<std::vector<cv::Point> > contours;

    for (auto& blob : blobs) {
        if (blob.blnStillBeingTracked == true) {
            contours.push_back(blob.currentContour);
        }
    }

    cv::drawContours(image, contours, -1, SCALAR_WHITE, -1);

    cv::imshow(strImageName, image); //FÝLTRE GÖRÜNTÜSÜNÜN ÇIKTISINI VERÝR.
}

//////////////////////////////// 7. DEÐER DÖNDÜRME  ///////////////////////////////////////////////////

//ARAÇIN GEÇÝP GEÇMEDÝÐÝNÝ DÖNDÜRÜYOR 

bool checkIfBlobsCrossedTheLine(std::vector<Blob>& blobs, int& intHorizontalLinePosition, int& carCount) {
    bool blnAtLeastOneBlobCrossedTheLine = false;

    for (auto blob : blobs) {

        if (blob.blnStillBeingTracked == true && blob.centerPositions.size() >= 2) {
            int prevFrameIndex = (int)blob.centerPositions.size() - 2;
            int currFrameIndex = (int)blob.centerPositions.size() - 1;

            if (blob.centerPositions[prevFrameIndex].y > intHorizontalLinePosition && blob.centerPositions[currFrameIndex].y <= intHorizontalLinePosition) { //Yatay Çizgi Konumundan geçmiþ ise arttýr
                carCount++; //sayýyý arttýr
                blnAtLeastOneBlobCrossedTheLine = true;
            }
        }

    }

    return blnAtLeastOneBlobCrossedTheLine;
}

/////////////////////////////// 8.ARAÇLARI KALIP ÝÇÝNDE TUTMA KIRMIZI ÞEKÝLDE BLOBLARI GÖSTERME /////////////////////////////////////////////
void drawBlobInfoOnImage(std::vector<Blob>& blobs, cv::Mat& imgFrame2Copy) { //Görüntüdeki blob Bilgi Çiz

    for (unsigned int i = 0; i < blobs.size(); i++) {

        if (blobs[i].blnStillBeingTracked == true) {
            cv::rectangle(imgFrame2Copy, blobs[i].currentBoundingRect, SCALAR_RED, 2);

        
        }
    }
}

////////////////////////   9. FONKSÝYON ARABA SAYISINI BELÝRTEN ÇIKTI GÖSTERME   ///////////////////////////////////////
void drawCarCountOnImage(int& carCount, cv::Mat& imgFrame2Copy) { //resim üzerinde araba sayýsý çizmek

    int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
    double dblFontScale = (imgFrame2Copy.rows * imgFrame2Copy.cols) / 450000.0;
    int intFontThickness = (int)std::round(dblFontScale * 1);


    cv::Size textSize = cv::getTextSize(std::to_string(carCount), intFontFace, dblFontScale, intFontThickness, 0);
    cv::putText(imgFrame2Copy, "gecen araba sayisi=" + std::to_string(carCount), cv::Point(400, 50), intFontFace, dblFontScale, SCALAR_GREEN, intFontThickness);

}
