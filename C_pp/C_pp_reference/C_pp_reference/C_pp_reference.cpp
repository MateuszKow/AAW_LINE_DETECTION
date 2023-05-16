#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <chrono>
using namespace cv;
using namespace std;
int main(int argc, char** argv)
{
    // Declare the output variables
    auto start = std::chrono::high_resolution_clock::now();
    Mat dst, cdst, cdstP;
    const char* default_file = "..//..//..//images//78.jpg";
    const char* filename = argc >= 2 ? argv[1] : default_file;
    // Loads an image
    Mat src = imread(samples::findFile(filename), IMREAD_GRAYSCALE);
    // Check if image is loaded fine
    if (src.empty()) {
        printf(" Error opening image\n");
        printf(" Program Arguments: [image_name -- default %s] \n", default_file);
        return -1;
    }
    // Edge detection
    auto start_canny = std::chrono::high_resolution_clock::now();
    Canny(src, dst, 50, 200, 3);
    auto end_canny = std::chrono::high_resolution_clock::now();
    // Copy edges to the images that will display the results in BGR
    cvtColor(dst, cdst, COLOR_GRAY2BGR);
    cdstP = cdst.clone();
    // Standard Hough Line Transform
    vector<Vec2f> lines; // will hold the results of the detection
    auto start_Hough = std::chrono::high_resolution_clock::now();
    HoughLines(dst, lines, 1, CV_PI / 180, 250, 0, 0); // runs the actual detection
    auto end_Hough = std::chrono::high_resolution_clock::now();
    // Draw the lines
    for (size_t i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    auto duration_canny = std::chrono::duration_cast<std::chrono::milliseconds>(end_canny - start_canny);
    auto duration_Hough = std::chrono::duration_cast<std::chrono::milliseconds>(end_Hough - start_Hough);
    std::cout <<"Czas trwania programu [ms]:" << duration.count() << std::endl;
    std::cout << "Czas trwania Canny [ms]:" << duration_canny.count() << std::endl;
    std::cout << "Czas trwania Hough [ms]:" << duration_Hough.count() << std::endl;
    imshow("Source", src);
    imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst);
    // Wait and Exit
    waitKey();
    return 0;
}
