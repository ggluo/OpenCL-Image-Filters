#define CL_HPP_ENABLE_EXCEPTIONS

#include <opencv/opencv.hpp>
#include <CL/opencl.hpp>
#include <iostream>
#include <fstream>
#include <vector>



// Helper function to load the OpenCL kernel source code
std::string loadKernel(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open kernel source file." << std::endl;
        exit(1);
    }
    return std::string(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
}

// Generate Gaussian kernel in 1D
std::vector<float> generateGaussianKernel(int kernelSize, float sigma) {
    std::vector<float> kernel(kernelSize * kernelSize);
    int halfSize = kernelSize / 2;
    float sum = 0.0f;
    
    for (int i = -halfSize; i <= halfSize; i++) {
        for (int j = -halfSize; j <= halfSize; j++) {
            float value = std::exp(-(i * i + j * j) / (2 * sigma * sigma));
            kernel[(i + halfSize) * kernelSize + (j + halfSize)] = value;
            sum += value;
        }
    }

    for (float &value : kernel) {
        value /= sum;
    }

    return kernel;
}

int main(int argc, char** argv) {
    // Check if the user provided the path to an image
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <Image Path>" << std::endl;
        return -1;
    }

    // Load the image using OpenCV
    cv::Mat src = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "Error: Could not open or find the image." << std::endl;
        return -1;
    }

    cv::imshow("Original Image", src);
    // Convert the image to RGBA format
    cv::cvtColor(src, src, cv::COLOR_BGR2RGBA);

    // Define Gaussian kernel parameters
    int kernelSize = 15;
    float sigma = 3.0f;
    std::vector<float> gaussianKernel = generateGaussianKernel(kernelSize, sigma);
    printf("Gaussian kernel generated\n");
    try {
        // Get platform and device information
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        printf("Platforms: %d\n", platforms.size());
        cl::Platform platform = platforms.front();

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        cl::Device device = devices.front();

        // Create an OpenCL context
        cl::Context context(device);

        // Create a command queue
        cl::CommandQueue queue(context, device);

        // Load and compile the OpenCL kernel
        std::string kernelSource = loadKernel("kernel.cl");
        cl::Program::Sources sources;
        sources.push_back({kernelSource.c_str(), kernelSource.length()});
        cl::Program program(context, sources);
        printf("Program created\n");
        cl_int err = program.build({device}, "-cl-opt-disable");

        
        if (err == CL_BUILD_PROGRAM_FAILURE) {
            // Get the build log
            std::string buildlog;
            program.getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &buildlog);
            std::cerr << "Error during compilation:\n" << buildlog << std::endl;
            return -1;
        } else if (err != CL_SUCCESS) {
            std::cerr << "Error building program: " << err << std::endl;
            return -1;
        }

        printf("Program built\n");
        // Create OpenCL images
        cl::ImageFormat format(CL_RGBA, CL_UNORM_INT8);
        cl::Image2D srcImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, format, src.cols, src.rows, 0, src.data);
        cl::Image2D dstImage(context, CL_MEM_WRITE_ONLY, format, src.cols, src.rows);


        // Create a buffer for the Gaussian kernel
        cl::Buffer kernelBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * gaussianKernel.size(), gaussianKernel.data());

        // Set up the OpenCL kernel
        cl::Kernel kernel(program, "gaussian_filter");
        kernel.setArg(0, srcImage);
        kernel.setArg(1, dstImage);
        kernel.setArg(2, kernelBuffer);
        kernel.setArg(3, kernelSize);

        // Execute the kernel
        cl::NDRange global(src.cols, src.rows);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);

        // Read the output image back to the host
        std::vector<unsigned char> output(src.rows * src.cols * 4);
        queue.enqueueReadImage(dstImage, CL_TRUE, {0, 0, 0}, {src.cols, src.rows, 1}, 0, 0, output.data());

        // Convert the output to an OpenCV Mat
        cv::Mat dst(src.rows, src.cols, CV_8UC4, output.data());

        // Convert the image back to BGR for display
        cv::cvtColor(dst, dst, cv::COLOR_RGBA2BGR);

        // Display the original and the filtered image
        cv::imshow("Gaussian Filtered Image", dst);
        cv::imwrite("Gaussian_Filtered_Image.jpg", dst);

        // Set up the OpenCL kernel for the bilateral filter
        cl::Kernel bilateralKernel(program, "bilateral_filter");
        bilateralKernel.setArg(0, srcImage);
        bilateralKernel.setArg(1, dstImage);
        bilateralKernel.setArg(2, 2.0f);
        bilateralKernel.setArg(3, 0.1f);

        // Execute the kernel
        queue.enqueueNDRangeKernel(bilateralKernel, cl::NullRange, global, cl::NullRange);
        queue.enqueueReadImage(dstImage, CL_TRUE, {0, 0, 0}, {src.cols, src.rows, 1}, 0, 0, output.data());

        // Convert the output to an OpenCV Mat
        cv::Mat bilateral(src.rows, src.cols, CV_8UC4, output.data());

        // Convert the image back to BGR for display
        cv::cvtColor(bilateral, bilateral, cv::COLOR_RGBA2BGR);

        // Display the bilateral filtered image
        cv::imshow("Bilateral Filtered Image", bilateral);
        cv::imwrite("Bilateral_Filtered_Image.jpg", bilateral);


        // Set up the OpenCL kernel for the sharpening filter
        cl::Kernel sharpeningKernel(program, "sharpen_filter");
        sharpeningKernel.setArg(0, srcImage);
        sharpeningKernel.setArg(1, dstImage);

        // Execute the kernel
        queue.enqueueNDRangeKernel(sharpeningKernel, cl::NullRange, global, cl::NullRange);
        queue.enqueueReadImage(dstImage, CL_TRUE, {0, 0, 0}, {src.cols, src.rows, 1}, 0, 0, output.data());

        // Convert the output to an OpenCV Mat
        cv::Mat sharpened(src.rows, src.cols, CV_8UC4, output.data());

        // Convert the image back to BGR for display
        cv::cvtColor(sharpened, sharpened, cv::COLOR_RGBA2BGR);

        // Display the sharpened image
        cv::imshow("Sharpened Image", sharpened);
        cv::imwrite("Sharpened_Image.jpg", sharpened);


        // Set up the OpenCL kernel for the median filter
        cl::Kernel medianKernel(program, "median_filter");
        medianKernel.setArg(0, srcImage);
        medianKernel.setArg(1, dstImage);
        medianKernel.setArg(2, 3);

        // Execute the kernel
        queue.enqueueNDRangeKernel(medianKernel, cl::NullRange, global, cl::NullRange);
        queue.enqueueReadImage(dstImage, CL_TRUE, {0, 0, 0}, {src.cols, src.rows, 1}, 0, 0, output.data());

        // Convert the output to an OpenCV Mat
        cv::Mat median(src.rows, src.cols, CV_8UC4, output.data());

        // Convert the image back to BGR for display
        cv::cvtColor(median, median, cv::COLOR_RGBA2BGR);

        // Display the median filtered image
        cv::imshow("Median Filtered Image", median);
        cv::imwrite("Median_Filtered_Image.jpg", median);

        // Wait indefinitely until a key is pressed
        cv::waitKey(0);

    } catch (cl::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << "(" << e.err() << ")" << std::endl;
        return -1;
    }

    return 0;
}
