/**********************************************************************
Copyright �2014 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

�   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
�   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#ifndef SOBEL_FILTER_IMAGE_H_
#define SOBEL_FILTER_IMAGE_H_

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "CLUtil.hpp"
#include "SDKBitMap.hpp"
#include <opencv2/highgui.hpp>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <cstdlib>



#define SAMPLE_VERSION "AMD-APP-SDK-v2.9-1.599.1"

#define INPUT_IMAGE "SobelFilterImage_Input.bmp"
#define OUTPUT_IMAGE "SobelFilterImage_Output.bmp"

#define GROUP_SIZE 256

using namespace appsdk;

/**
* SobelFilterImage
* Class implements OpenCL Sobel Filter sample using Images
*/


// class CLCommandArgsHough
//     : public CLCommandArgs
// {
//     public: 
//         unsigned int binarize_threshold;
//         CLCommandArgsHough() : CLCommandArgs() 
//         { 
//             binarize_threshold = 128;
//         }
// };


class SobelFilterImage
{
        cl_double setupTime;                /**< time taken to setup OpenCL resources and building kernel */
        cl_double kernelTime;               /**< time taken to run kernel and read result back */
        cl_uchar4* inputImageData;          /**< Input bitmap data to device */
        cl_uchar4* outputImageData;         /**< Output from device */
        
        cv::VideoCapture cap;
        cv::Mat frame;
        int deviceID;
        int apiID;
        cv::Mat imageMat;        

        cl::Context context;                            /**< CL context */
        std::vector<cl::Device> devices;                /**< CL device list */
        std::vector<cl::Device> device;                 /**< CL device to be used */
        std::vector<cl::Platform> platforms;            /**< list of platforms */
        cl::Image2D inputImage2D;
        cl::Image2D erodeImage2D;                           /**< CL Input image2d */
        cl::Image2D outputImage2D;
        cl::Image2D sinusoid;
        cl::Image3D outputImage3D;                       /**< CL Output image2d */
        cl::Image2D acum_2D;
        cl::Buffer outputBuffer_rho;
        cl::Buffer rho_buffer;
        cl::Buffer to_display_buffer_in;
        cl::Buffer to_display_buffer_out;
        cl::Buffer out_acumulator;
        
        cl::CommandQueue commandQueue;                  /**< CL command queue */
        cl::Program program;
        // std::vector<cl_float>;                            /**< CL program  */
        cl::Kernel kernel;
        cl::Kernel kernel2;
        cl::Kernel hough;
        cl::Kernel hough_buffer;
        cl::Kernel to_display;
        cl::Kernel hough_image3D;
        cl::Kernel accumulator_kernel;
        unsigned int bufferSize;                                   /**< CL kernel */

        cl_uchar* verificationOutput;       /**< Output array for reference implementation */

        SDKBitMap inputBitmap;   /**< Bitmap class object */
        uchar4* pixelData;       /**< Pointer to image data */
        cl_uint pixelSize;                  /**< Size of a pixel in BMP format> */
        cl_uint width;                      /**< Width of image */
        cl_uint height;                      /**< Width of image */
        cl_uint width_accumulator;
        cl_uint height_accumulator;                          /**< Height of image */
        
        cl_uint binarize_threshold;         /**< Binarization threshold value */                     
        cl_uint theta_resolution; 
        cl_uint threshold;         /**< Resolution of theta */

        cl_bool byteRWSupport;
        size_t kernelWorkGroupSize;
        size_t kernel2WorkGroupSize;   
        size_t houghWorkGroupSize;
        size_t hough_image3DWorkGroupSize;
        size_t accumulator_kernelWorkGroupSize;
        size_t to_displayGroupSize;       /**< Group Size returned by kernel */
        size_t hough_bufferWorkGroupSize;
        size_t blockSizeX;                  /**< Work-group size in x-direction */
        size_t blockSizeY;                  /**< Work-group size in y-direction */
        int iterations;                     /**< Number of iterations for kernel execution */
        int imageSupport;
        int* to_display_table;

        SDKTimer    *sampleTimer;      /**< SDKTimer object */

    public:

        CLCommandArgs   *sampleArgs;   /**< CLCommand argument class */
        int camera;

        /**
        * Read bitmap image and allocate host memory
        * @param inputImageName name of the input file
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int readInputImage(std::string inputImageName);

        /**
        * Write to an image file
        * @param outputImageName name of the output file
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int writeOutputImage(std::string outputImageName);

        /**
        * Constructor
        * Initialize member variables
        */
        SobelFilterImage()
            : inputImageData(NULL),
              outputImageData(NULL),
              verificationOutput(NULL),
              byteRWSupport(true)
        {
            sampleArgs = new CLCommandArgs();
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
            pixelSize = sizeof(uchar4);
            pixelData = NULL;
            blockSizeX = GROUP_SIZE;
            blockSizeY = 1;
            iterations = 1;
            imageSupport = 0;
        }

        ~SobelFilterImage()
        {
        }

        /**
        * Allocate image memory and Load bitmap file
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setupSobelFilterImage();

        /**
         * Override from SDKSample, Generate binary image of given kernel
         * and exit application
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int genBinaryImage();

        /**
        * OpenCL related initialisations.
        * Set up Context, Device list, Command Queue, Memory buffers
        * Build CL kernel program executable
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setupCL();

        /**
        * Set values for kernels' arguments, enqueue calls to the kernels
        * on to the command queue, wait till end of kernel execution.
        * Get kernel start and end time if timing is enabled
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int runCLKernels();

        /**
        * Reference CPU implementation of Binomial Option
        * for performance comparison
        */
        void sobelFilterImageCPUReference();

        /**
        * Override from SDKSample. Print sample stats.
        */
        void printStats();

        /**
        * Override from SDKSample. Initialize
        * command line parser, add custom options
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int initialize();

        /**
        * Override from SDKSample, adjust width and height
        * of execution domain, perform all sample setup
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setup();

        /**
        * Override from SDKSample
        * Run OpenCL Sobel Filter
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int run();

        /**
        * Override from SDKSample
        * Cleanup memory allocations
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int cleanup();

        /**
        * Override from SDKSample
        * Verify against reference implementation
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int verifyResults();

        // 
        // 
        int initialize_camera();
        int read_camera();
        int convert_frame_to_image();
        int camera_show();
};

#endif // SOBEL_FILTER_IMAGE_H_
