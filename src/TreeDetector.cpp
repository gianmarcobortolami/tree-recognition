//
// Created by Gianmarco Bortolami on 08/07/2020.
//

#include "../include/TreeDetector.hpp"

TreeDetector::TreeDetector() {}

TreeDetector::TreeDetector(std::vector<cv::Mat> someInputImgs, std::vector<cv::Mat> someTrainingImgs) {
    benchmarkImgs = someInputImgs;
    trainingImgs = someTrainingImgs;
}

void TreeDetector::loadBenchmarkImgs(cv::String directory, cv::String pattern) {

    std::cout << "Loading benchmark images..." << std::endl;

    // Load filenames of all input images
    std::vector<cv::String> filenames;
    cv::utils::fs::glob(directory, pattern, filenames);

    // Push back the images retrieved on the vector of input images
    for (int i = 0; i < filenames.size(); ++i) {
        cv::Mat img = cv::imread(filenames[i]);
        if (!img.data)
            std::cout << "Image " << filenames[i] << " not loaded." << std::endl;
        else
            benchmarkImgs.push_back(img);
    }

    std::cout << benchmarkImgs.size() << "/" << filenames.size() << " done." << std::endl << std::endl;

    // Resize the images if they are too large
    for (int i = 0; i < benchmarkImgs.size(); ++i) {
        if (benchmarkImgs[i].cols > 1000)
            cv::resize(benchmarkImgs[i], benchmarkImgs[i], cv::Size(), 0.5, 0.5);
    }

    std::cout << "Benchmark images resized." << std::endl << std::endl;

}


void TreeDetector::loadTrainingImgs(cv::String directory, cv::String pattern) {

    std::cout << "Loading training images..." << std::endl;

    // Load filenames of all training images
    std::vector<cv::String> filenames;
    cv::utils::fs::glob(directory, pattern, filenames);

    // Push back the images retrieved on the vector of training images
    for (int i = 0; i < filenames.size(); ++i) {
        cv::Mat img = cv::imread(filenames[i]);
        if (!img.data)
            std::cout << "Image " << filenames[i] << " not loaded." << std::endl;
        else
            trainingImgs.push_back(img);
    }

    std::cout << trainingImgs.size() << "/" << filenames.size() << " done." << std::endl << std::endl;

}


void TreeDetector::detectAndComputeFeatures(cv::Mat inputImg, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {

    // Detect and compute the keypoints of a image
    cv::Ptr<cv::ORB> orb = cv::ORB::create(3000);
    orb->detect(inputImg, keypoints);
    orb->compute(inputImg, keypoints, descriptors);

    // Show the keypoints of the image
    /*cv::Mat output;
    cv::drawKeypoints(img, keypoints, output);
    cv::imshow("Keypoints of image", output);
    cv::waitKey(0);
    cv::destroyAllWindows();*/

}

void TreeDetector::goodMatchedFeaturesBetweenBenchmarkImgAndTrainingImgs(std::vector<cv::KeyPoint> benchmarkKeypoints, cv::Mat benchmarkDescriptors, std::vector<cv::Mat> trainingDescriptors, std::vector<cv::KeyPoint>& goodKeypointsOfImg) {

    /// MATCH THE FEATURES OF BENCHMARK IMAGE AND TRAINING IMAGES
    std::vector<cv::DMatch> goodMatchesOfImg;
    // For each training image
    for (int i = 0; i < trainingImgs.size(); ++i) {
        // Match descriptors
        cv::Ptr<cv::BFMatcher> bfMatcher = cv::BFMatcher::create(cv::NORM_HAMMING);
        std::vector<cv::DMatch> matches;
        bfMatcher->match(benchmarkDescriptors, trainingDescriptors[i], matches);

        // Find good matches
        //std::vector<cv::DMatch> goodMatches; // NB: Useful to draw the good matches of each single image
        // If they have some matches, then filter out the matches with less importance
        if (!matches.empty()) {
            float minDistance = minDistanceAmongDMatches(matches);

            for (int k = 0; k < matches.size(); ++k) {
                if (matches[k].distance < 1.4f * minDistance) {
                    //goodMatches.push_back(matches[k]);
                    goodMatchesOfImg.push_back(matches[k]);
                }
            }
        }
    }

    /// GATHER ALL GOOD MATCHED KEYPOINTS OF BENCHMARK IMAGE
    // Extract keypoints with good match on training images
    for (int j = 0; j < goodMatchesOfImg.size(); ++j) {
        goodKeypointsOfImg.push_back(benchmarkKeypoints[goodMatchesOfImg[j].queryIdx]);
    }

}

void TreeDetector::removeBackgroundFromImage(cv::Mat inputImg, cv::Mat& inputImgWithoutBackground) {

    /// SMOOTH THE IMAGE
    cv::Mat smoothImg;
    cv::bilateralFilter(inputImg, smoothImg, 7, 300, 300);

    /// SEGMENT THE ENTIRE IMAGE
    // Take the length of array where do K-Means
    unsigned int singleLineSize = smoothImg.rows * smoothImg.cols;
    // Choose the number of clusters
    unsigned int K = 15;
    // Convert into HSV color space to perform better the segmentation
    cv::cvtColor(smoothImg, smoothImg, cv::COLOR_BGR2HSV);
    // Tranform image in an 1D array
    cv::Mat data = smoothImg.reshape(1, singleLineSize);
    data.convertTo(data, CV_32F);

    // Perform K-Means and make the image with clusters
    std::vector<int> labels;
    cv::Mat1f colors;
    cv::kmeans(data, K, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.), 2,
               cv::KMEANS_PP_CENTERS, colors);
    for (unsigned int i = 0; i < singleLineSize; i++) {
        data.at<float>(i, 0) = colors(labels[i], 0);
        data.at<float>(i, 1) = colors(labels[i], 1);
        data.at<float>(i, 2) = colors(labels[i], 2);
    }

    // Re-convert and re-transform the HSV-1D array in a BGR image
    cv::Mat segmentedImg = data.reshape(3, smoothImg.rows);
    segmentedImg.convertTo(segmentedImg, CV_8U);
    cv::cvtColor(segmentedImg, segmentedImg, cv::COLOR_HSV2BGR);

    /// THRESHOLD THE NON GREEN ZONES SEGMENTED
    // Choose the thresholds to apply (all shades of green/brown)
    cv::Scalar H = cv::Scalar(26, 85); // lower and upper bound for Hue
    cv::Scalar S = cv::Scalar(40, 255); // lower and upper bound for Saturation
    cv::Scalar V = cv::Scalar(0, 230); // lower and upper bound for Value

    // Convert BGR 2 HSV
    cv::cvtColor(segmentedImg, segmentedImg, cv::COLOR_BGR2HSV);
    cv::cvtColor(inputImg, inputImg, cv::COLOR_BGR2HSV);

    // Extract a mask
    cv::Mat mask;
    cv::inRange(segmentedImg, cv::Scalar(H[0], S[0], V[0]), cv::Scalar(H[1], S[1], V[1]), mask);
    //cv::imshow("Mask", mask);

    // Apply a morphological operator to the mask
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(14, 40),
                                                cv::Point(6, 20)); // 7, 19 - 4, 10
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, element);

    // Convert HSV 2 BGR
    cv::cvtColor(segmentedImg, segmentedImg, cv::COLOR_HSV2BGR);
    cv::cvtColor(inputImg, inputImg, cv::COLOR_HSV2BGR);

    // Apply the mask to the temp segmented and input image
    cv::bitwise_and(inputImg, inputImg, inputImgWithoutBackground, mask);

}

void TreeDetector::splitInRegionsTheImage(cv::Mat inputImgWithoutBackground, cv::Mat& regionedInputImg) {

    /// SEGMENT THE GREEN ZONES RETRIEVED
    // Choose the number of clusters
    const unsigned int K_tree = 30;
    // Take the length of array where do K-Means
    const unsigned int singleLineSize = inputImgWithoutBackground.rows * inputImgWithoutBackground.cols;

    // Convert into HSV color space to perform better the segmentation
    cv::cvtColor(inputImgWithoutBackground, inputImgWithoutBackground, cv::COLOR_BGR2HSV);
    // Tranform image in an 1D array
    cv::Mat data = inputImgWithoutBackground.reshape(1, singleLineSize);
    data.convertTo(data, CV_32F);

    // Push back the 2d position of the pixels on data not belonging in the background
    cv::Mat filteredData;
    cv::Mat positions;

    for (int row = 0, point = 0; row < inputImgWithoutBackground.rows; ++row) {
        for (int col = 0; col < inputImgWithoutBackground.cols; ++col) {
            // If the point does not belong to the background, it is added to the filtered data
            if (data.at<float>(point, 0) != 0 || data.at<float>(point, 1) != 0 || data.at<float>(point, 2) != 0) {
                // Emphasize the clustering on the position
                int weightOfPositions = 3;
                float rowNormalized = (row * 256 * weightOfPositions) / inputImgWithoutBackground.rows;
                float colNormalized = (col * 256 * weightOfPositions) / inputImgWithoutBackground.cols;
                cv::Mat_<float> position = (cv::Mat_<float>(1, 2) << rowNormalized, colNormalized);

                // Concat the position to the filtered data vector
                cv::Mat newRow;
                cv::hconcat(data.row(point), position, newRow);
                filteredData.push_back(newRow);

            }
            point++;
        }
    }

    // Perform K-Means and make the image with clusters (iff the image has green zones)
    std::vector<int> regionsOfImg;
    cv::Mat1f colors;
    if (!filteredData.empty()) {
        cv::kmeans(filteredData, K_tree, regionsOfImg,
                   cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.), 2,cv::KMEANS_PP_CENTERS, colors);

        for (int j = 0, k = 0; j < singleLineSize; j++) {
            if (data.at<float>(j, 0) != 0 || data.at<float>(j, 1) != 0 || data.at<float>(j, 2) != 0) {
                data.at<float>(j, 0) = colors(regionsOfImg[k], 0);
                data.at<float>(j, 1) = colors(regionsOfImg[k], 1);
                data.at<float>(j, 2) = colors(regionsOfImg[k++], 2);
            }
        }
    }

    // Re-convert and re-transform the HSV-1D array in a BGR image
    cv::Mat segmentedTreeImg = data.reshape(3, inputImgWithoutBackground.rows);
    segmentedTreeImg.convertTo(segmentedTreeImg, CV_8U);
    cv::cvtColor(segmentedTreeImg, segmentedTreeImg, cv::COLOR_HSV2BGR);

    /// MAP REGIONS ON BENCHMARK IMAGE TO CREATE A REGIONS IMAGE
    // Make a matrix with the same dimensions of associated image (having only 30 regions (i.e. values) signed char is enough)
    regionedInputImg = cv::Mat(segmentedTreeImg.rows, segmentedTreeImg.cols, CV_8SC1);
    // Maps regions on the benchmark image
    for (int row = 0, point = 0; row < segmentedTreeImg.rows; ++row) {
        for (int col = 0; col < segmentedTreeImg.cols; ++col) {
            // If the point correspond to the background set it to -1
            if (segmentedTreeImg.at<cv::Vec3b>(row,col)[0] == 0 && segmentedTreeImg.at<cv::Vec3b>(row,col)[1] == 0 && segmentedTreeImg.at<cv::Vec3b>(row,col)[2] == 0) {
                regionedInputImg.at<char>(row, col) = -1;
            } else {
                // otherwise set it with the number of associated region
                regionedInputImg.at<char>(row,col) = regionsOfImg[point];
                point++;
            }
        }
    }

}

void TreeDetector::filterRegionsWithoutTrees(std::vector<cv::KeyPoint> goodKeypointsOfImg, cv::Mat regionedInputImg, cv::Mat inputImgWithoutBackground, cv::Mat &blobTreesImg) {

    /// CALCULATE HOW MANY TREE KEYPOINTS THE REGIONS HAVE
    // Initialize all counter to 0
    const unsigned int K_tree = 30;
    std::vector<int> counterKeypointsInRegionsOfImg(K_tree, 0);

    // For each good keypoint of the image
    for (int i = 0; i < goodKeypointsOfImg.size(); ++i) {

        // Take the coords of the keypoint
        cv::Point2f coords = goodKeypointsOfImg[i].pt;
        // Maps it to the regioned image and take the region in which belongs
        char keypointRegion = regionedInputImg.at<char>(coords);
        // Increase the counter of region in which belongs
        counterKeypointsInRegionsOfImg[keypointRegion]++;

    }

    /// FILTER OUT REGIONS OF IMAGE WITHOUT TREES
    // Initialize output image with the same characterstics of associated input image
    cv::Mat treesImg = cv::Mat(inputImgWithoutBackground.rows, inputImgWithoutBackground.cols, inputImgWithoutBackground.type());

    // Compute the average number of good keypoints belonging on the regions
    int avgNumberOfKeypointsOnRegions = avg(counterKeypointsInRegionsOfImg);

    // For each point of the image
    for (int row = 0, point = 0; row < inputImgWithoutBackground.rows; ++row) {
        for (int col = 0; col < inputImgWithoutBackground.cols; ++col) {
            // If the point belongs to the background, set it to black
            if (regionedInputImg.at<char>(row, col) == -1)
                treesImg.at<cv::Vec3b>(row, col) = cv::Vec3b(0,0,0);
            else
            if ((counterKeypointsInRegionsOfImg[regionedInputImg.at<char>(row, col)] > 40) && (counterKeypointsInRegionsOfImg[regionedInputImg.at<char>(row, col)] > avgNumberOfKeypointsOnRegions))
                treesImg.at<cv::Vec3b>(row, col) = inputImgWithoutBackground.at<cv::Vec3b>(row,col);
            else
                treesImg.at<cv::Vec3b>(row, col) = cv::Vec3b(0,0,0);
        }
    }

    // Transform the region found in blobs
    cv::inRange(treesImg, cv::Scalar(0,0,0), cv::Scalar(0,0,0), blobTreesImg);
    cv::bitwise_not(blobTreesImg, blobTreesImg);
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(41, 41), cv::Point(20, 20));
    cv::dilate(blobTreesImg, blobTreesImg, element);

}

void TreeDetector::boundTreesOnImage(cv::Mat inputImg, cv::Mat blobTreesImg) {

    /// DETECT CENTERS AND AREAS OF BLOBS
    // Set up the detector with custom parameters
    cv::SimpleBlobDetector::Params params;
    // Filter by Area: YES
    params.filterByArea = true;
    params.minArea = 1000;
    params.maxArea = inputImg.cols * inputImg.rows;
    // Filter by Color: NO
    params.filterByColor = false;
    params.blobColor = 255; // detect white blobs
    // Filter by Circularity: NO
    params.filterByCircularity = false;
    // Filter by Convexity: NO
    params.filterByConvexity = false;
    // Filter by Inertia: NO
    params.filterByInertia = false;

    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);

    // Detect blobs
    std::vector<cv::KeyPoint> blobKeypoints;
    detector->detect(blobTreesImg, blobKeypoints);

    /// DRAW THE RECTANGLES OVER THE TREES
    cv::Mat treesDetectedImg = inputImg.clone();

    // Draw detected blobs as red rectangle
    for (int j = 0; j < blobKeypoints.size(); ++j) {
        // Coords and dimension
        cv::Point2f topLeftCorner = blobKeypoints[j].pt;
        int size = blobKeypoints[j].size;
        // Make the rectangle
        cv::Rect rect(topLeftCorner.x-round(size/2), topLeftCorner.y-round(size/2), size, size);
        // Draw the rectangle on the image
        cv::rectangle(treesDetectedImg, rect, cv::Scalar(0, 0, 255));
    }

    treesDetectedImgs.push_back(treesDetectedImg);

}

void TreeDetector::detect() {

    /// DETECT THE KEYPOINTS AND COMPUTE THE DESCRIPTORS OF BENCHMARK IMAGES
    std::cout << "Detecting keypoints on benchmark images...    ";
    std::vector<std::vector<cv::KeyPoint> > benchmarkKeypoints;
    std::vector<cv::Mat> benchmarkDescriptors;
    for (int i = 0; i < benchmarkImgs.size(); ++i) {
        std::vector<cv::KeyPoint> tempKeypoints;
        cv::Mat tempDescriptors;
        detectAndComputeFeatures(benchmarkImgs[i], tempKeypoints, tempDescriptors);
        benchmarkKeypoints.push_back(tempKeypoints);
        benchmarkDescriptors.push_back(tempDescriptors);
    }
    std::cout << "Done." << std::endl << std::endl;

    /// DETECT THE KEYPOINTS AND COMPUTE THE DESCRIPTORS OF TRAINING IMAGES
    std::cout << "Detecting keypoints on training images...    ";
    std::vector<std::vector<cv::KeyPoint> > trainingKeypoints;
    std::vector<cv::Mat> trainingDescriptors;
    for (int i = 0; i < trainingImgs.size(); ++i) {
        std::vector<cv::KeyPoint> tempKeypoints;
        cv::Mat tempDescriptors;
        detectAndComputeFeatures(trainingImgs[i], tempKeypoints, tempDescriptors);
        trainingKeypoints.push_back(tempKeypoints);
        trainingDescriptors.push_back(tempDescriptors);
    }
    std::cout << "Done." << std::endl << std::endl;


    /// DETECT TREES ON EACH BENCHMARK IMAGES
    for (int i = 0; i < benchmarkImgs.size(); ++i) {

        // Extract the good keypoints of the benchmark image
        std::cout << "Selecting best matched keypoints for image " << i << "...    ";
        std::vector<cv::KeyPoint> goodKeypointsOfBenchmarkImg;
        goodMatchedFeaturesBetweenBenchmarkImgAndTrainingImgs(benchmarkKeypoints[i], benchmarkDescriptors[i], trainingDescriptors, goodKeypointsOfBenchmarkImg);
        std::cout << "Done." << std::endl << std::endl;

        // Remove the background from the benchmark image
        std::cout << "Removing background on image " << i << "...    ";
        cv::Mat benchmarkImgWithoutBackground;
        removeBackgroundFromImage(benchmarkImgs[i], benchmarkImgWithoutBackground);
        std::cout << "Done." << std::endl << std::endl;

        // Split the benchmark image without background in green regions
        std::cout << "Splitting in regions the image " << i << "...    ";
        cv::Mat regionedBenchmarkImg;
        splitInRegionsTheImage(benchmarkImgWithoutBackground, regionedBenchmarkImg);
        std::cout << "Done." << std::endl << std::endl;

        // Filter out the green regions without trees
        std::cout << "Filtering out the regions without trees of image " << i << "...    ";
        cv::Mat blobTreesImg;
        filterRegionsWithoutTrees(goodKeypointsOfBenchmarkImg, regionedBenchmarkImg, benchmarkImgWithoutBackground, blobTreesImg);
        std::cout << "Done." << std::endl << std::endl;

        // Bound the blob with a green rectangle
        std::cout << "Bounding the trees of image " << i << "...    ";
        boundTreesOnImage(benchmarkImgs[i], blobTreesImg);
        std::cout << "Done." << std::endl << std::endl;

    }

}

void TreeDetector::show() {

    for (int i = 0; i < treesDetectedImgs.size(); ++i) {
        std::cout << "Showing image " << i << " with trees bounded..." << std::endl;
        cv::imshow("Trees detected", treesDetectedImgs[i]);
        cv::waitKey(0);
    }

}


//   E X T E R N A L   F U N C T I O N S

float minDistanceAmongDMatches(std::vector<cv::DMatch> matches) {

    float minDistance = matches[0].distance;
    for (int i=1; i<matches.size(); ++i) {
        if (matches[i].distance < minDistance)
            minDistance = matches[i].distance;
    }
    return minDistance;

}

int avg(std::vector<int> numbers) {
    // If is empty return -1, as alert
    if (numbers.empty())
        return -1;

    // Compute the average of non zero terms
    int sum = 0;
    int numberOfNonZero = 0;
    for (int i = 0; i < numbers.size(); ++i)
        if (numbers[i] != 0) {
            sum += numbers[i];
            numberOfNonZero++;
        }

    if (numberOfNonZero == 0)
        return -1;
    return std::round(sum/numberOfNonZero);
}