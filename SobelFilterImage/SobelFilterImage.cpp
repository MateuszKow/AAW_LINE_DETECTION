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

#include <opencv2/imgproc.hpp>
#include "SobelFilterImage.hpp"
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <cstdlib>

void ShowImage(cv::Mat image, int *tab, int HEIGHT, int WIDTH, int RHO) {
    
    // flip the image (OpenCV is origin at top left)
    cv::flip(image, image, 0);
        // for each line parameters (rho, theta) draw a line on binary image
        for (int i = 0; i < (int)((2*(HEIGHT + WIDTH) * RHO)); i++) {
        if (tab[2 * i] != 5000) {
            int length = tab[2 * i+1] - (HEIGHT+WIDTH);
            int angle = (tab[2 * i])-90;
            if (angle == -180){
                angle = 0;
            }
            float angleRad = angle * CV_PI / 180.0;
            float x0 = std::cos(angleRad) * length;
            float y0 = std::sin(angleRad) * length;
            cv::Point pt1(cvRound(x0 + 1000 * (-std::sin(angleRad))), cvRound(y0 + 1000 * (std::cos(angleRad))));
            cv::Point pt2(cvRound(x0 - 1000 * (-std::sin(angleRad))), cvRound(y0 - 1000 * (std::cos(angleRad))));
            cv::line(image, pt1, pt2, cv::Scalar(255, 0, 255), 1, cv::LINE_AA);
        }
    }
    // write the output image
    cv::imwrite("Output.bmp", image);
}

int
SobelFilterImage::readInputImage(std::string inputImageName)
{
    // load input bitmap image
    inputBitmap.load(inputImageName.c_str());

    // error if image did not load
    if(!inputBitmap.isLoaded())
    {
        std::cout << "Failed to load input image!" << std::endl;
        return SDK_FAILURE;
    }

    // get width and height of input image
    height = inputBitmap.getHeight();
    width = inputBitmap.getWidth();
    // allocate memory for input & output image data  */

    // allocate memory for input & output image data  */
    inputImageData  = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));

    // error check
    CHECK_ALLOCATION(inputImageData, "Failed to allocate memory! (inputImageData)");


    // allocate memory for output image data
    outputImageData = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));

    

    // error check
    CHECK_ALLOCATION(outputImageData,
                     "Failed to allocate memory! (outputImageData)");


    // initialize the Image data to NULL
    memset(outputImageData, 0, width * height * pixelSize);

    // get the pointer to pixel data
    pixelData = inputBitmap.getPixels();

    // error check
    CHECK_ALLOCATION(pixelData, "Failed to read pixel Data!");

    // Copy pixel data into inputImageData
    memcpy(inputImageData, pixelData, width * height * pixelSize);

    // allocate memory for verification output
    verificationOutput = (cl_uchar*)malloc(width * height * pixelSize);

    // error check
    CHECK_ALLOCATION(verificationOutput,
                     "verificationOutput heap allocation failed!");

    // initialize the data to NULL
    memset(verificationOutput, 0, width * height * pixelSize);

    return SDK_SUCCESS;

}


int
SobelFilterImage::writeOutputImage(std::string outputImageName)
{
    // copy output image data back to original pixel data
    memcpy(pixelData, outputImageData, width * height * pixelSize);

    // write the output bmp file
    if(!inputBitmap.write(outputImageName.c_str()))
    {
        std::cout << "Failed to write output image!" << std::endl;
        return SDK_FAILURE;
    }

    return SDK_SUCCESS;
}

int
SobelFilterImage::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("SobelFilterImage_Kernels.cl");
    binaryData.flagsStr = std::string("");
    if(sampleArgs->isComplierFlagsSpecified())
    {
        binaryData.flagsFileName = std::string(sampleArgs->flags.c_str());
    }

    binaryData.binaryName = std::string(sampleArgs->dumpBinary.c_str());
    int status = generateBinaryImage(binaryData);
    return status;
}


int
SobelFilterImage::setupCL()
{
    cl_int err = CL_SUCCESS;
    cl_device_type dType;

    if(sampleArgs->deviceType.compare("cpu") == 0)
    {
        dType = CL_DEVICE_TYPE_CPU;
    }
    else //deviceType = "gpu"
    {
        dType = CL_DEVICE_TYPE_GPU;
        if(sampleArgs->isThereGPU() == false)
        {
            std::cout << "GPU not found. Falling back to CPU device" << std::endl;
            dType = CL_DEVICE_TYPE_CPU;
        }
    }

    /*
     * Have a look at the available platforms and pick either
     * the AMD one if available or a reasonable default.
     */
    err = cl::Platform::get(&platforms);
    CHECK_OPENCL_ERROR(err, "Platform::get() failed.");

    std::vector<cl::Platform>::iterator i;
    if(platforms.size() > 0)
    {
        if(sampleArgs->isPlatformEnabled())
        {
            i = platforms.begin() + sampleArgs->platformId;
        }
        else
        {
            for(i = platforms.begin(); i != platforms.end(); ++i)
            {
                if(!strcmp((*i).getInfo<CL_PLATFORM_VENDOR>().c_str(),
                           "Advanced Micro Devices, Inc."))
                {
                    break;
                }
            }
        }
    }

    cl_context_properties cps[3] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)(*i)(),
        0
    };

    context = cl::Context(dType, cps, NULL, NULL, &err);
    CHECK_OPENCL_ERROR(err, "Context::Context() failed.");

    devices = context.getInfo<CL_CONTEXT_DEVICES>(&err);
    CHECK_OPENCL_ERROR(err, "Context::getInfo() failed.");

    std::cout << "Platform :" << (*i).getInfo<CL_PLATFORM_VENDOR>().c_str() << "\n";
    int deviceCount = (int)devices.size();
    int j = 0;
    for (std::vector<cl::Device>::iterator i = devices.begin(); i != devices.end();
            ++i, ++j)
    {
        std::cout << "Device " << j << " : ";
        std::string deviceName = (*i).getInfo<CL_DEVICE_NAME>();
        std::cout << deviceName.c_str() << "\n";
    }
    std::cout << "\n";

    if (deviceCount == 0)
    {
        std::cout << "No device available\n";
        return SDK_FAILURE;
    }

    if(validateDeviceId(sampleArgs->deviceId, deviceCount))
    {
        std::cout << "validateDeviceId() failed" << std::endl;
        return SDK_FAILURE;
    }


    // Check for image support
    imageSupport = devices[sampleArgs->deviceId].getInfo<CL_DEVICE_IMAGE_SUPPORT>
                   (&err);
    CHECK_OPENCL_ERROR(err, "Device::getInfo() failed.");

    // If images are not supported then return
    if(!imageSupport)
    {
        OPENCL_EXPECTED_ERROR("Images are not supported on this device!");
    }

    commandQueue = cl::CommandQueue(context, devices[sampleArgs->deviceId], 0,
                                    &err);
    CHECK_OPENCL_ERROR(err, "CommandQueue::CommandQueue() failed.");
    /*
    * Create and initialize memory objects
    */
   // Create memory objects for input Image
    inputImage2D = cl::Image2D(context,
                               CL_MEM_READ_ONLY,
                               cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
                               width,
                               height,
                               0,
                               NULL,
                               &err);
    CHECK_OPENCL_ERROR(err, "Image2D::Image2D() failed. (inputImage2D)");


    // Create memory objects for output Image
    outputImage2D = cl::Image2D(context,
                                CL_MEM_WRITE_ONLY,
                                cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
                                width,
                                height,
                                0,
                                0,
                                &err);
    CHECK_OPENCL_ERROR(err, "Image2D::Image2D() failed. (outputImage2D)");

    // Create memory objects for eroded Image
    erodeImage2D = cl::Image2D(context,
                                CL_MEM_READ_WRITE,
                                cl::ImageFormat(CL_RGBA,CL_UNSIGNED_INT8),
                                width,
                                height,
                                0,
                                0,
                                &err);
    CHECK_OPENCL_ERROR(err, "Image2D::Image2D() failed. (erodeImage2D)");

    rho_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, number_theta*width*height * sizeof(int));
    out_acumulator = cl::Buffer(context, CL_MEM_READ_WRITE, 2*number_theta*(width+height) * sizeof(int));
    to_display_buffer_out = cl::Buffer(context, CL_MEM_READ_WRITE, 2*2*number_theta*(width+height) * sizeof(int));

    device.push_back(devices[sampleArgs->deviceId]);

    // create a CL program using the kernel source
    SDKFile kernelFile;
    std::string kernelPath = getPath();

    if(sampleArgs->isLoadBinaryEnabled())
    {
        kernelPath.append(sampleArgs->loadBinary.c_str());
        if(kernelFile.readBinaryFromFile(kernelPath.c_str()) != SDK_SUCCESS)
        {
            std::cout << "Failed to load kernel file : " << kernelPath << std::endl;
            return SDK_FAILURE;
        }
        cl::Program::Binaries programBinary(1,std::make_pair(
                                                (const void*)kernelFile.source().data(),
                                                kernelFile.source().size()));

        program = cl::Program(context, device, programBinary, NULL, &err);
        CHECK_OPENCL_ERROR(err, "Program::Program(Binary) failed.");

    }
    else
    {
        kernelPath.append("SobelFilterImage_Kernels.cl");
        if(!kernelFile.open(kernelPath.c_str()))
        {
            std::cout << "Failed to load kernel file : " << kernelPath << std::endl;
            return SDK_FAILURE;
        }

        // create program source
        cl::Program::Sources programSource(1,
                                           std::make_pair(kernelFile.source().data(),
                                                   kernelFile.source().size()));

        // Create program object
        program = cl::Program(context, programSource, &err);
        CHECK_OPENCL_ERROR(err, "Program::Program() failed.");

    }

    std::string flagsStr = std::string("");

    // Get additional options
    if(sampleArgs->isComplierFlagsSpecified())
    {
        SDKFile flagsFile;
        std::string flagsPath = getPath();
        flagsPath.append(sampleArgs->flags.c_str());
        if(!flagsFile.open(flagsPath.c_str()))
        {
            std::cout << "Failed to load flags file: " << flagsPath << std::endl;
            return SDK_FAILURE;
        }
        flagsFile.replaceNewlineWithSpaces();
        const char * flags = flagsFile.source().c_str();
        flagsStr.append(flags);
    }

    if(flagsStr.size() != 0)
    {
        std::cout << "Build Options are : " << flagsStr.c_str() << std::endl;
    }

    err = program.build(device, flagsStr.c_str());

    if(err != CL_SUCCESS)
    {
        if(err == CL_BUILD_PROGRAM_FAILURE)
        {
            std::string str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[sampleArgs->deviceId]);

            std::cout << " \n\t\t\tBUILD LOG\n";
            std::cout << " ************************************************\n";
            std::cout << str << std::endl;
            std::cout << " ************************************************\n";
        }
    }
    CHECK_OPENCL_ERROR(err, "Program::build() failed.");

    // Create kernel
    kernel = cl::Kernel(program, "erode",  &err);
    CHECK_OPENCL_ERROR(err, "Kernel::Kernel() failed.");
    kernel2 = cl::Kernel(program, "erode_diff",  &err);
    CHECK_OPENCL_ERROR(err, "Kernel::Kernel() failed.");
    hough_buffer = cl::Kernel(program, "hough_transform_buffer",  &err);
    CHECK_OPENCL_ERROR(err, "Kernel::Kernel() failed.");
    accumulator_kernel = cl::Kernel(program, "accumulator",  &err);
    CHECK_OPENCL_ERROR(err, "Kernel::Kernel() failed.");
    to_display = cl::Kernel(program, "to_display",  &err);
    CHECK_OPENCL_ERROR(err, "Kernel::Kernel() failed.");

    // Check group size against group size returned by kernel
    kernelWorkGroupSize = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>
                          (devices[sampleArgs->deviceId], &err);
    CHECK_OPENCL_ERROR(err, "Kernel::getWorkGroupInfo()  failed.");

    kernel2WorkGroupSize = kernel2.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>
                          (devices[sampleArgs->deviceId], &err);
    CHECK_OPENCL_ERROR(err, "Kernel::getWorkGroupInfo()  failed.");

    accumulator_kernelWorkGroupSize = accumulator_kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>
                          (devices[sampleArgs->deviceId], &err);
    CHECK_OPENCL_ERROR(err, "Kernel::getWorkGroupInfo()  failed.");

    hough_bufferWorkGroupSize = hough_buffer.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>
                          (devices[sampleArgs->deviceId], &err);
                          
    CHECK_OPENCL_ERROR(err, "Kernel::getWorkGroupInfo()  failed.");

    to_displayGroupSize = to_display.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>
                          (devices[sampleArgs->deviceId], &err);
    CHECK_OPENCL_ERROR(err, "Kernel::getWorkGroupInfo()  failed.");


    // Check local group sizes for all kernels
    if((blockSizeX * blockSizeY) > kernelWorkGroupSize)
    {
        if(!sampleArgs->quiet)
        {
            std::cout << "Out of Resources!" << std::endl;
            std::cout << "Group Size specified : "
                      << blockSizeX * blockSizeY << std::endl;
            std::cout << "Max Group Size supported on the kernel : "
                      << kernelWorkGroupSize << std::endl;
            std::cout << "Falling back to " << kernelWorkGroupSize << std::endl;
        }

        if(blockSizeX > kernelWorkGroupSize)
        {
            blockSizeX = kernelWorkGroupSize;
            blockSizeY = 1;
        }
    }

    if((blockSizeX * blockSizeY) > kernel2WorkGroupSize)
    {
        if(!sampleArgs->quiet)
        {
            std::cout << "Out of Resources!" << std::endl;
            std::cout << "Group Size specified : "
                      << blockSizeX * blockSizeY << std::endl;
            std::cout << "Max Group Size supported on the kernel : "
                      << kernel2WorkGroupSize << std::endl;
            std::cout << "Falling back to " << kernel2WorkGroupSize << std::endl;
        }

        if(blockSizeX > kernel2WorkGroupSize)
        {
            blockSizeX = kernel2WorkGroupSize;
            blockSizeY = 1;
        }
    }

    if((blockSizeX * blockSizeY) > hough_bufferWorkGroupSize)
    {
        if(!sampleArgs->quiet)
        {
            std::cout << "Out of Resources!" << std::endl;
            std::cout << "Group Size specified : "
                      << blockSizeX * blockSizeY << std::endl;
            std::cout << "Max Group Size supported on the kernel : "
                      << hough_bufferWorkGroupSize << std::endl;
            std::cout << "Falling back to " << hough_bufferWorkGroupSize << std::endl;
        }

        if(blockSizeX > hough_bufferWorkGroupSize)
        {
            blockSizeX = hough_bufferWorkGroupSize;
            blockSizeY = 1;
        }
    }

    if((blockSizeX * blockSizeY) > accumulator_kernelWorkGroupSize)
    {
        if(!sampleArgs->quiet)
        {
            std::cout << "Out of Resources!" << std::endl;
            std::cout << "Group Size specified : "
                      << blockSizeX * blockSizeY << std::endl;
            std::cout << "Max Group Size supported on the kernel : "
                      << accumulator_kernelWorkGroupSize << std::endl;
            std::cout << "Falling back to " << accumulator_kernelWorkGroupSize << std::endl;
        }

        if(blockSizeX > accumulator_kernelWorkGroupSize)
        {
            blockSizeX = accumulator_kernelWorkGroupSize;
            blockSizeY = 1;
        }
    }

    if((blockSizeX * blockSizeY) > to_displayGroupSize)
    {
        if(!sampleArgs->quiet)
        {
            std::cout << "Out of Resources!" << std::endl;
            std::cout << "Group Size specified : "
                      << blockSizeX * blockSizeY << std::endl;
            std::cout << "Max Group Size supported on the kernel : "
                      << to_displayGroupSize << std::endl;
            std::cout << "Falling back to " << to_displayGroupSize << std::endl;
        }

        if(blockSizeX > to_displayGroupSize)
        {
            blockSizeX = to_displayGroupSize;
            blockSizeY = 1;
        }
    }



    return SDK_SUCCESS;
}

int
SobelFilterImage::runCLKernels()
{
    cl_int status;

    cl::size_t<3> origin;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;

    cl::size_t<3> region;
    region[0] = width;
    region[1] = height;
    region[2] = 1;

    // Set angle_resolution due to the number of theta
    theta_resolution = (uint) 180/number_theta;
    
    // Write input data to image
    cl::Event writeEvt2;
    status = commandQueue.enqueueWriteImage(
                 inputImage2D,
                 CL_TRUE,
                 origin,
                 region,
                 0,
                 0,
                 inputImageData,
                 NULL,
                 &writeEvt2);
    CHECK_OPENCL_ERROR(status,
                       "CommandQueue::enqueueWriteImage failed. (inputImage2D)");

    status = commandQueue.finish();
    CHECK_OPENCL_ERROR(status, "cl::CommandQueue.finish failed.");

    cl_int eventStatus2 = CL_QUEUED;
    while(eventStatus2 != CL_COMPLETE)
    {
        status = writeEvt2.getInfo<cl_int>(
                     CL_EVENT_COMMAND_EXECUTION_STATUS,
                     &eventStatus2);
        CHECK_OPENCL_ERROR(status,
                           "cl:Event.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS) failed.");

    }

    // Write zeros to accumulator
    int* zera = new int[2*number_theta*(width+height)];
    memset(zera, 0, 2*number_theta*(width+height)*sizeof(int));

    status = commandQueue.enqueueWriteBuffer(
                 out_acumulator, 
                 CL_TRUE, 
                 0, 
                 2*number_theta*(width+height) * sizeof(int), 
                 zera);
    status = commandQueue.finish();
    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueReadBuffer failed.");

    delete zera;

    // Set appropriate arguments to the kernels
    

    status = kernel.setArg(0, inputImage2D);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (inputImageBuffer)");

    status = kernel.setArg(1, erodeImage2D);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (erodeImage2D)");

    status = kernel.setArg(2, binarize_threshold);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (binarize_threshold)");

    status = kernel2.setArg(0, erodeImage2D);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (erodeImage2D)");

    status = kernel2.setArg(1, outputImage2D);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (outputImageBuffer)");

    status = hough_buffer.setArg(0, outputImage2D);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (outputImage2D)");

    status = hough_buffer.setArg(1, rho_buffer);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (rho_buffer)");

    status = hough_buffer.setArg(2, theta_resolution);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (theta_resolution)");

    status = hough_buffer.setArg(3, width);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (width)");

    status = hough_buffer.setArg(4, height);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (height)");

    status = accumulator_kernel.setArg(0, rho_buffer);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (rho_buffer)");

    status = accumulator_kernel.setArg(1, out_acumulator);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (out_acumulator)");

    status = to_display.setArg(0, out_acumulator);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (out_acumulator)");

    status = to_display.setArg(1, to_display_buffer_out);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (to_display_buffer_out)");

    status = to_display.setArg(2, threshold_edge);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (threshhold)");

    status = to_display.setArg(3, number_theta);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (width)");

   
    /*
    * Enqueue a kernel run call.
    */
    cl::NDRange globalThreads(width, height);
    cl::NDRange localThreads(blockSizeX, blockSizeY);

    cl::Event ndrEvt;
    status = commandQueue.enqueueNDRangeKernel(
                 kernel,
                 cl::NullRange,
                 globalThreads,
                 cl::NullRange,
                 0,
                 &ndrEvt);
    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueNDRangeKernel() failed.");

    status = commandQueue.finish();
    CHECK_OPENCL_ERROR(status, "cl::CommandQueue.finish failed.");

    eventStatus2 = CL_QUEUED;
    while(eventStatus2 != CL_COMPLETE)
    {
        status = ndrEvt.getInfo<cl_int>(
                     CL_EVENT_COMMAND_EXECUTION_STATUS,
                     &eventStatus2);
        CHECK_OPENCL_ERROR(status,
                           "cl:Event.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS) failed.");
    }

    cl::Event ndrEvt2;
    status = commandQueue.enqueueNDRangeKernel(
                 kernel2,
                 cl::NullRange,
                 globalThreads,
                 cl::NullRange,
                 0,
                 &ndrEvt2);
    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueNDRangeKernel() failed.");

    status = commandQueue.finish();
    CHECK_OPENCL_ERROR(status, "cl::CommandQueue.finish failed.");

    cl_int eventStatus = CL_QUEUED;
    while(eventStatus != CL_COMPLETE)
    {
        status = ndrEvt2.getInfo<cl_int>(
                     CL_EVENT_COMMAND_EXECUTION_STATUS,
                     &eventStatus);
        CHECK_OPENCL_ERROR(status,
                           "cl:Event.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS) failed.");
    }
    cl::Event ndrEvt4;
    status = commandQueue.enqueueNDRangeKernel(
                 hough_buffer,
                 cl::NullRange,
                 globalThreads,
                 localThreads,
                 0,
                 &ndrEvt4);
    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueNDRangeKernel() failed.");

    status = commandQueue.finish();
    CHECK_OPENCL_ERROR(status, "cl::CommandQueue.finish failed.");

    eventStatus = CL_QUEUED;
    while(eventStatus != CL_COMPLETE)
    {
        status = ndrEvt4.getInfo<cl_int>(
                     CL_EVENT_COMMAND_EXECUTION_STATUS,
                     &eventStatus);
        CHECK_OPENCL_ERROR(status,
                           "cl:Event.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS) failed.");
    }


    cl::NDRange globalThreads2(height*width*number_theta);
    cl::NDRange localThreads2(1);

    cl::Event ndrEvt10;
    status = commandQueue.enqueueNDRangeKernel(
                accumulator_kernel,
                cl::NullRange,
                globalThreads2,
                localThreads2,
                0,
                &ndrEvt10);
    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueNDRangeKernel() failed.");

    status = commandQueue.finish();
    CHECK_OPENCL_ERROR(status, "cl::CommandQueue.finish failed.");

    eventStatus = CL_QUEUED;
    while(eventStatus != CL_COMPLETE)
    {
        status = ndrEvt10.getInfo<cl_int>(
                    CL_EVENT_COMMAND_EXECUTION_STATUS,
                    &eventStatus);
        CHECK_OPENCL_ERROR(status,
                        "cl:Event.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS) failed.");
    }

    cl::NDRange globalThreads3(2*number_theta*(width+height));
    cl::NDRange localThreads3(1);

    cl::Event ndrEvt5;
    status = commandQueue.enqueueNDRangeKernel(
                 to_display,
                 cl::NullRange,
                 globalThreads3,
                 localThreads3,
                 0,
                 &ndrEvt5);
    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueNDRangeKernel() failed.");

    status = commandQueue.finish();
    CHECK_OPENCL_ERROR(status, "cl::CommandQueue.finish failed.");
    eventStatus = CL_QUEUED;
    while(eventStatus != CL_COMPLETE)
    {
        status = ndrEvt5.getInfo<cl_int>(
                     CL_EVENT_COMMAND_EXECUTION_STATUS,
                     &eventStatus);
        CHECK_OPENCL_ERROR(status,
                           "cl:Event.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS) failed.");
    }

    // Enqueue Read Image
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;


    region[0] = width;
    region[1] = height;
    region[2] = 1;

    cl::Event readEvt;
    status = commandQueue.enqueueReadImage(
                 outputImage2D,
                 CL_TRUE,
                 origin,
                 region,
                 0,
                 0,
                 outputImageData,
                 NULL,
                 &readEvt);
    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueReadImage failed.");
    status = commandQueue.finish();
    CHECK_OPENCL_ERROR(status, "cl::CommandQueue.finish failed.");

    eventStatus = CL_QUEUED;
    while(eventStatus != CL_COMPLETE)
    {
        status = readEvt.getInfo<cl_int>(
                     CL_EVENT_COMMAND_EXECUTION_STATUS,
                     &eventStatus);
        CHECK_OPENCL_ERROR(status,
                           "cl:Event.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS) failed.");

    }
    //read rhos and thetas from buffer
    to_display_table = new int[2*2*number_theta*(width+height)];

    status = commandQueue.enqueueReadBuffer(
                 to_display_buffer_out,
                 CL_TRUE,
                 0,
                 2*2*number_theta*(width+height) * sizeof(int),
                 to_display_table,
                 NULL,
                 &readEvt);
    status = commandQueue.finish();
    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueReadBuffer failed.");

    // create an image object
    imageMat.create(height, width, CV_8UC4);

    // Enqueue Read Image
    status = commandQueue.enqueueReadImage(
                 outputImage2D,
                 CL_TRUE,
                 origin,
                 region,
                 0,
                 0,
                 imageMat.data,
                 NULL,
                 &readEvt);
    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueReadImage failed.");
    status = commandQueue.finish();
    
    CHECK_OPENCL_ERROR(status, "cl::CommandQueue.finish failed.");

    eventStatus = CL_QUEUED;
    while(eventStatus != CL_COMPLETE)
    {
        status = readEvt.getInfo<cl_int>(
                     CL_EVENT_COMMAND_EXECUTION_STATUS,
                     &eventStatus);
        CHECK_OPENCL_ERROR(status,
                           "cl:Event.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS) failed.");

    }
    return SDK_SUCCESS;
}



int
SobelFilterImage::initialize()
{
    // Call base class Initialize to get default configuration
    CHECK_ERROR(sampleArgs->initialize(), SDK_SUCCESS,
                "OpenCL resource initialization failed");

    Option* iteration_option = new Option;
    if(!iteration_option)
    {
        error("Memory Allocation error.\n");
        return SDK_FAILURE;
    }
    iteration_option->_sVersion = "i";
    iteration_option->_lVersion = "iterations";
    iteration_option->_description = "Number of iterations to execute kernel";
    iteration_option->_type = CA_ARG_INT;
    iteration_option->_value = &iterations;
    sampleArgs->AddOption(iteration_option);
    delete iteration_option;
    return SDK_SUCCESS;
}

int SobelFilterImage::read_camera()
{
    // grab new frame from camera
    cap.read(frame);
    if (frame.empty()) {
        std::cerr << "ERROR! blank frame grabbed\n"<<std::endl<<std::flush;
        return SDK_FAILURE;
    }
    return SDK_SUCCESS;
}

int SobelFilterImage::initialize_camera()
{
    // open the default camera, use something different from 0 otherwise;
    deviceID = 0;
    apiID = cv::CAP_ANY;
    // open selected camera using selected API
    cap.open(deviceID, apiID);
    // check if we succeeded
    if (!cap.isOpened()) {
        std::cerr << "ERROR! Unable to open camera\n"<<std::flush;
            return SDK_FAILURE;
    }
    if (read_camera() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }
    // get the frame size
    width = frame.cols;
    height = frame.rows;
    
    return SDK_SUCCESS;
}



int
SobelFilterImage::setup()
{
    // Allocate host memory and read input image
    std::string filePath = getPath() + std::string(INPUT_IMAGE);
    if(readInputImage(filePath) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // create and initialize timers
    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    int status = setupCL();
    if (status != SDK_SUCCESS)
    {
        return status;
    }

    sampleTimer->stopTimer(timer);
    // Compute setup time
    setupTime = (double)(sampleTimer->readTimer(timer));

    return SDK_SUCCESS;

}

int
SobelFilterImage::run()
{
    if(!byteRWSupport)
    {
        return SDK_SUCCESS;
    }

    // Warm up
    for(int i = 0; i < 2 && iterations != 1; i++)
    {
        // Set kernel arguments and run kernel
        if(runCLKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

    }

    std::cout << "Executing kernel for " << iterations
              << " iterations" <<std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    // create and initialize timers
    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    for(int i = 0; i < iterations; i++)
    {
        // Set kernel arguments and run kernel
        if(runCLKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

    }

    sampleTimer->stopTimer(timer);
    // Compute kernel time
    kernelTime = (double)(sampleTimer->readTimer(timer)) / iterations;

    // write the output image to bitmap file
    if(writeOutputImage(OUTPUT_IMAGE) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    return SDK_SUCCESS;
}

int
SobelFilterImage::cleanup()
{

    // release program resources (input memory etc.)
    FREE(inputImageData);
    FREE(outputImageData);
    FREE(verificationOutput);

    return SDK_SUCCESS;
}


void
SobelFilterImage::sobelFilterImageCPUReference()
{
    // x-axis gradient mask
    const int kx[][3] =
    {
        { 1, 2, 1},
        { 0, 0, 0},
        { -1, -2, -1}
    };

    // y-axis gradient mask
    const int ky[][3] =
    {
        { 1, 0, -1},
        { 2, 0, -2},
        { 1, 0, -1}
    };

    int gx = 0;
    int gy = 0;

    // pointer to input image data
    cl_uchar *ptr = (cl_uchar*)malloc(width * height * pixelSize);
    memcpy(ptr, inputImageData, width * height * pixelSize);

    // each pixel has 4 uchar components
    int w = width * 4;

    int k = 1;

    // apply filter on each pixel (except boundary pixels)
    for(int i = 0; i < (int)(w * (height - 1)) ; i++)
    {
        if(i < (k + 1) * w - 4 && i >= 4 + k * w)
        {
            gx =  kx[0][0] **(ptr + i - 4 - w)
                  + kx[0][1] **(ptr + i - w)
                  + kx[0][2] **(ptr + i + 4 - w)
                  + kx[1][0] **(ptr + i - 4)
                  + kx[1][1] **(ptr + i)
                  + kx[1][2] **(ptr + i + 4)
                  + kx[2][0] **(ptr + i - 4 + w)
                  + kx[2][1] **(ptr + i + w)
                  + kx[2][2] **(ptr + i + 4 + w);

            gy =  ky[0][0] **(ptr + i - 4 - w)
                  + ky[0][1] **(ptr + i - w)
                  + ky[0][2] **(ptr + i + 4 - w)
                  + ky[1][0] **(ptr + i - 4)
                  + ky[1][1] **(ptr + i)
                  + ky[1][2] **(ptr + i + 4)
                  + ky[2][0] **(ptr + i - 4 + w)
                  + ky[2][1] **(ptr + i + w)
                  + ky[2][2] **(ptr + i + 4 + w);

            float gx2 = pow((float)gx, 2);
            float gy2 = pow((float)gy, 2);

            double temp = sqrt(gx2 + gy2) / 2.0;

            // Saturate
            if(temp > 255)
            {
                temp = 255;
            }
            if(temp < 0)
            {
                temp = 0;
            }

            *(verificationOutput + i) = (cl_uchar)(temp);
        }

        // if reached at the end of its row then incr k
        if(i == (k + 1) * w - 5)
        {
            k++;
        }
    }
    FREE(ptr);
}


int
SobelFilterImage::verifyResults()
{
    if(!byteRWSupport)
    {
        return SDK_SUCCESS;
    }

    if(sampleArgs->verify)
    {
        // reference implementation
        sobelFilterImageCPUReference();

        size_t size = width * height * sizeof(cl_uchar4);

        cl_uchar4 *verificationData = (cl_uchar4*)malloc(size);
        memcpy(verificationData, verificationOutput, size);

        cl_uint error = 0;
        for(cl_uint y = 0; y < height; y++)
        {
            for(cl_uint x = 0; x < width; x++)
            {
                int c = x + y * width;
                // Only verify pixels inside the boundary
                if((x >= 1 && x < (width - 1) && y >= 1 && y < (height - 1)))
                {
                    error += outputImageData[c].s[0]-verificationData[c].s[0];
                }
            }
        }

        // compare the results and see if they match
        if(!error)
        {
            std::cout << "Passed!\n" << std::endl;
            FREE(verificationData);
            return SDK_SUCCESS;
        }
        else
        {
            std::cout << "Failed\n" << std::endl;
            FREE(verificationData);
            return SDK_FAILURE;
        }
    }

    return SDK_SUCCESS;
}

void
SobelFilterImage::printStats()
{
    if(sampleArgs->timing)
    {
        std::string strArray[4] =
        {
            "Width",
            "Height",
            "Time(sec)",
            "[Transfer+Kernel]Time(sec)"
        };
        std::string stats[4];

        sampleTimer->totalTime = setupTime + kernelTime;

        stats[0] = toString(width, std::dec);
        stats[1] = toString(height, std::dec);
        stats[2] = toString(sampleTimer->totalTime, std::dec);
        stats[3] = toString(kernelTime, std::dec);

        printStatistics(strArray, stats, 4);
    }
}

int SobelFilterImage::camera_show()
{
    // flip image
    cv::flip(imageMat, imageMat, 0);
 
    // Draw lines on the image
for (int i = 0; i < (int)((2*(height + width) * number_theta)); i++) {
        if (to_display_table[2 * i] != 5000) { //
            int length = to_display_table[2 * i+1] - (height+width);   
            int angle = (to_display_table[2 * i])-90;  
            if (angle == -180){
                angle = 0;
            }
            float angleRad = angle * CV_PI / 180.0;
            float x0 = std::cos(angleRad) * length;
            float y0 = std::sin(angleRad) * length;
            cv::Point pt1(cvRound(x0 + 1000 * (-std::sin(angleRad))), cvRound(y0 + 1000 * (std::cos(angleRad))));
            cv::Point pt2(cvRound(x0 - 1000 * (-std::sin(angleRad))), cvRound(y0 - 1000 * (std::cos(angleRad))));
            cv::line(imageMat, pt1, pt2, cv::Scalar(255, 0, 255), 1, cv::LINE_AA);
        }
    }
    // flip image and show it
    cv::flip(imageMat, imageMat, 0);
    cv::imshow("Obrazek", imageMat);
    int key=cv::waitKey(1);
    // return false - program will quit
    if (key == 27|| key == 'q'){
        return false;
        std::cout<<"esc"<<std::endl;
    }
        

    return true;
}


int SobelFilterImage::convert_frame_to_image()
{
    // write frame to inputImageData
    for (int i = 0; i < width*height; i++)
    {
        inputImageData[i].s[0]=frame.data[i*3];
        inputImageData[i].s[1]=frame.data[i*3+1];
        inputImageData[i].s[2]=frame.data[i*3+2];
        inputImageData[i].s[3]=0;
    }

    return SDK_SUCCESS;
}

int
main(int argc, char * argv[])
{
    SobelFilterImage clSobelFilterImage;
    // parameters to be initialized
    std::cout<<"Czy chcesz wlaczyc kamere? 1-tak 0-nie"<<std::endl;
    std::cin>>clSobelFilterImage.camera;
    std::cout<<"Podaj wartosc progu binaryzacji"<<std::endl;
    std::cin>>clSobelFilterImage.binarize_threshold;
    std::cout<<"Podaj liczbe minimalnych pikseli tworzacych krawedz"<<std::endl;
    std::cin>>clSobelFilterImage.threshold_edge;
    std::cout<<"Podaj liczbe katow na ktore bedzie dzielona przestrzen"<<std::endl;
    std::cin>>clSobelFilterImage.number_theta;
    
    if(clSobelFilterImage.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clSobelFilterImage.sampleArgs->parseCommandLine(argc, argv))
    {
        return SDK_FAILURE;
    }

    if(clSobelFilterImage.sampleArgs->isDumpBinaryEnabled())
    {
        return clSobelFilterImage.genBinaryImage();
    }
    else
    {
        // Setup
        if(clSobelFilterImage.camera)
        {
            int check_status = clSobelFilterImage.initialize_camera();
            CHECK_ERROR(check_status, SDK_SUCCESS, "Initializing camera Failed!\n");
        }
        int status = clSobelFilterImage.setup();
        if(status != SDK_SUCCESS)
        {
            return status;
        }
        // Run
        else{
        if(clSobelFilterImage.camera){
            std::cout<<"cos";
            status = true;
            while(status==true){
            status = clSobelFilterImage.read_camera();
            CHECK_ERROR(status, SDK_SUCCESS, "reading camera Failed");

            status = clSobelFilterImage.convert_frame_to_image();
            status = clSobelFilterImage.run();
            status = clSobelFilterImage.camera_show();
            }
        }
        else{
            clSobelFilterImage.run();
            ShowImage(clSobelFilterImage.imageMat, clSobelFilterImage.to_display_table, clSobelFilterImage.height, clSobelFilterImage.width, clSobelFilterImage.number_theta);

        }
        }
        // VerifyResults
        if(clSobelFilterImage.verifyResults() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        // Cleanup
        if(clSobelFilterImage.cleanup() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        clSobelFilterImage.printStats();
    }

    return SDK_SUCCESS;
}