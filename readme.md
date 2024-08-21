
# OpenCL Image Filters Examples

This repository provides an implementation of various image processing filters using OpenCL and OpenCV. The filters include:

- **Gaussian Filter**: For denoising images.
- **Sharpening Filter**: For enhancing image edges.
- **Bilateral Filter**: For edge-preserving smoothing.
- **Mean Filter**: For averaging neighboring pixels.

## Requirements

### Software

- **OpenCL**: Ensure you have the OpenCL runtime installed. For example, the NVIDIA or AMD drivers should include OpenCL support.
- **OpenCV**: For image loading and saving. Install OpenCV as described below.
- **C++ Compiler**: A C++ compiler supporting C++11 or later.

### Installing Dependencies

#### OpenCL

1. **On Ubuntu**:
   ```bash
   sudo apt-get update
   sudo apt-get install opencl-headers ocl-icd-opencl-dev intel-opencl-icd
   ```

2. **On macOS**:
   OpenCL is included with the system. Ensure you have the latest Xcode command line tools.

3. **On Windows**:
   Download and install the appropriate OpenCL runtime from the GPU manufacturer (NVIDIA, AMD, Intel).

#### OpenCV

1. **On Ubuntu**:
   ```bash
   sudo apt-get update
   sudo apt-get install libopencv-dev
   ```

2. **On macOS**:
   ```bash
   brew install opencv
   ```

3. **On Windows**:
   Download and install OpenCV from the [official website](https://opencv.org/releases/). Set up the environment variables and include directories as instructed.

## Building the Project

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ggluo/OpenCL-Image-Filters.git
   cd OpenCL-Image-Filters
   ```

2. **Compile the Code**:

   Ensure you have OpenCL and OpenCV libraries and headers. Compile with:

   ```bash
   g++ -o image_filters main.cpp -lOpenCL `pkg-config --cflags --libs opencv4`
   ```

   Adjust include paths and library paths as needed.

## Running the Example

1. **Prepare Your Image Data**:
   
   Place your input image in the project directory or adjust the file paths in the code. For example, use an image named `input.jpg` or use the provided `org.png`.

2. **Run the Compiled Binary**:
   ```bash
   ./image_filters ./misc/org.png
   ```

   The program will process the image using the selected filter and save the output as `output.jpg`.

## Kernel Files

The [`kernel.cl`](./kernel.cl) file contains the OpenCL kernel code for the image processing filters. The kernels include:

### Mean Filter

Averages the pixel values in a 3x3 neighborhood. 

```cpp
kernel.setArg(2, 3); // 3x3 kernel
```

### Gaussian Filter

Blurs the image using a Gaussian kernel for noise reduction.

```cpp
float kernel[3][3] = {
    {1/16.0, 2/16.0, 1/16.0},
    {2/16.0, 4/16.0, 2/16.0},
    {1/16.0, 2/16.0, 1/16.0}
};
```

### Sharpening Filter

Enhances edges by applying a sharpening kernel.

```cpp
float kernel[3][3] = {
    { 0, -1,  0},
    {-1,  5, -1},
    { 0, -1,  0}
};
```

### Bilateral Filter

Smooths the image while preserving edges using spatial and range weights.

```cpp
float sigma_s = 2.0f; // Spatial variance
float sigma_r = 0.1f; // Range variance
```
