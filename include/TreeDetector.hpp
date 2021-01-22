//
// Created by Gianmarco Bortolami on 08/07/2020.
//

#ifndef EXAM_TREEDETECTOR_HPP
#define EXAM_TREEDETECTOR_HPP

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/features2d.hpp>


// C L A S S   T O   D E T E C T   T R E E S   O N   A   S E T   O F   I M A G E S

class TreeDetector {

public:
    //   V A R I A B L E S
    std::vector<cv::Mat> benchmarkImgs, trainingImgs, treesDetectedImgs;

    //   C O N S T R U C T O R
    TreeDetector();
    TreeDetector(std::vector<cv::Mat> someInputImgs, std::vector<cv::Mat> someTrainingImgs);

    //   M E T H O D S
    void loadBenchmarkImgs(cv::String directory, cv::String pattern);
    void loadTrainingImgs(cv::String directory, cv::String pattern);
    void detectAndComputeFeatures(cv::Mat inputImg, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
    void goodMatchedFeaturesBetweenBenchmarkImgAndTrainingImgs(std::vector<cv::KeyPoint> benchmarkKeypoints, cv::Mat benchmarkDescriptors, std::vector<cv::Mat> trainingDescriptors, std::vector<cv::KeyPoint>& goodKeypointsOfImg);
    void removeBackgroundFromImage(cv::Mat inputImg, cv::Mat& inputImgWithoutBackground);
    void splitInRegionsTheImage(cv::Mat inputImgWithoutBackground, cv::Mat& regionedInputImg);
    void filterRegionsWithoutTrees(std::vector<cv::KeyPoint> goodKeypointsOfImg, cv::Mat regionedInputImg, cv::Mat inputImgWithoutBackground, cv::Mat& blobTreesImg);
    void boundTreesOnImage(cv::Mat inputImg, cv::Mat blobTreesImg);

    void detect();
    void show();


};

// E X T E R N A L   F U N C T I O N S
float minDistanceAmongDMatches(std::vector<cv::DMatch> matches);
int avg(std::vector<int> numbers);


#endif //EXAM_TREEDETECTOR_HPP
