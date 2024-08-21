__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void gaussian_filter(__read_only image2d_t srcImage, __write_only image2d_t dstImage, __constant float *g_kernel, int kernelSize) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    int width = get_image_width(srcImage);
    int height = get_image_height(srcImage);

    float4 color = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    int halfSize = kernelSize / 2;

    for (int i = -halfSize; i <= halfSize; i++) {
        for (int j = -halfSize; j <= halfSize; j++) {
            int2 neighborPos = (int2)(clamp(pos.x + i, 0, width - 1), clamp(pos.y + j, 0, height - 1));
            float4 neighborColor = read_imagef(srcImage, sampler, neighborPos);
            color += neighborColor * g_kernel[(i + halfSize) * kernelSize + (j + halfSize)];
        }
    }

    write_imagef(dstImage, pos, color);
}


__kernel void bilateral_filter(__read_only image2d_t srcImage,
                               __write_only image2d_t dstImage,
                               const float sigma_s, // spatial
                               const float sigma_r) { // range
    int2 coords = (int2)(get_global_id(0), get_global_id(1));
    float4 centerPixel = read_imagef(srcImage, sampler, coords);

    float4 sum = (float4)(0.0f);
    float weightSum = 0.0f;

    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            int2 neighborCoords = coords + (int2)(i, j);
            float4 neighborPixel = read_imagef(srcImage, sampler, neighborCoords);

            float spatialDist = (float)(i*i + j*j);
            float rangeDist = length(centerPixel - neighborPixel);

            float weight = exp(-spatialDist / (2 * sigma_s * sigma_s) - rangeDist / (2 * sigma_r * sigma_r));
            sum += neighborPixel * weight;
            weightSum += weight;
        }
    }

    write_imagef(dstImage, coords, sum / weightSum);
}


__kernel void sharpen_filter(__read_only image2d_t srcImage,
                             __write_only image2d_t dstImage) {
    int2 coords = (int2)(get_global_id(0), get_global_id(1));
    
    float weight[3][3] = {
        { 0, -1,  0},
        {-1,  5, -1},
        { 0, -1,  0}
    };

    float4 sum = (float4)(0.0f);
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            float4 pixel = read_imagef(srcImage, sampler, coords + (int2)(i, j));
            sum += pixel * weight[i + 1][j + 1];
        }
    }

    write_imagef(dstImage, coords, clamp(sum, 0.0f, 1.0f)); // Clamping to valid range
}

__kernel void median_filter(__read_only image2d_t srcImage,
                          __write_only image2d_t dstImage,
                          const int kernelSize) {
    int2 coords = (int2)(get_global_id(0), get_global_id(1));
    int halfSize = kernelSize / 2;

    // Initialize sum
    float4 sum = (float4)(0.0f);

    // Iterate over the kernel
    for (int i = -halfSize; i <= halfSize; i++) {
        for (int j = -halfSize; j <= halfSize; j++) {
            // Read the pixel value from the image
            float4 pixel = read_imagef(srcImage, sampler, coords + (int2)(i, j));
            // Sum up the pixel values
            sum += pixel;
        }
    }

    // Compute the average
    float4 average = sum / (float)(kernelSize * kernelSize);

    // Write the result to the destination image
    write_imagef(dstImage, coords, average);
}
