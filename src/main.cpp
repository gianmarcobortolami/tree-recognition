//
// Created by Gianmarco Bortolami on 08/07/2020.
//

#include "../include/TreeDetector.hpp"

int main(int argc, const char * argv[]) {

    // Create an empty object of Tree Detector
    TreeDetector td;

    // Load benchmark and training images
    td.loadBenchmarkImgs("../data/Benchmark_step_1", "Figure *.jpg");
    td.loadTrainingImgs("../data/Training_images", "Training *.jpg");

    // Detect the trees on benchmark images
    td.detect();
    // Show the trees found
    td.show();

    std::cout<< "Program well terminated.";
    return 0;
}