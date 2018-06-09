#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "lodepng.h"

__device__ int clamp(int value, int lo, int hi) {
    return (value < lo ? lo : value > hi ? hi : value);
}

__global__ void img_proc(const unsigned char* IMG_IN, unsigned char* IMG_OUT, int W, int H)
{
    /* Kernel that is executed for each pixel (x,y) in the image
     * and does the following:
     * compute the average value of the 5x5 neighborhood of (x,y):
     * sum up all IMG_IN pixels with horizontal indices x-2, ..., x+2 and
     * vertical indices y-2, ..., y+2, then divide by 25 and write
     * the result to IMG_OUT.
     * All indices are within valid range (at borders,
     * neighborhoods are clipped: if x is less than 0 then set it to 0, and similar
     * for the other borders).
     */
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < W && y < H) {
        float avg = 0.0f;
        for (int range1 = -2; range1 < +2; ++range1) {
            int yy = clamp(y + range1, 0, H - 1);
            for (int range2 = -2; range2 <= +2; ++range2) {
                int xx = clamp(x + range2, 0, W - 1);
                avg += IMG_IN[yy * W + xx];
            }
        }
        avg /= 25.0f;
        IMG_OUT[y * W + x] = avg;
    }
}

int main(void) // int argc, const char* argv[]
{
    unsigned char *img_in, *img_out;
    unsigned char *IMG_IN, *IMG_OUT;
    int W, H;

    /* Load image into img_in array; set W and H */
    {
        std::vector<unsigned char> image;
        unsigned int width, height;
        unsigned error = lodepng::decode(image, width, height, "img_in.png", LCT_GREY, 8);
        if (error) {
            fprintf(stderr, "png decoder error %d: %s\n", error, lodepng_error_text(error));
            exit(1);
        }
        W = width;
        H = height;
        img_in = (unsigned char*)malloc(W * H);
        img_out = (unsigned char*)malloc(W * H);
        memcpy(img_in, &(image[0]), W * H);
    }

    /* Allocate images on GPU */
    cudaMalloc(&IMG_IN, W * H);
    cudaMalloc(&IMG_OUT, W * H);

    /* Copy img_in from main memory to GPU memory */
    cudaMemcpy(IMG_IN, img_in, W * H, cudaMemcpyHostToDevice);

    /* Launch kernel that computes result and writes it to IMG_OUT */
    dim3 threads_in_block(32, 32, 1); // Just an example; should work fine
    dim3 blocks_in_grid(W / threads_in_block.x + (W % threads_in_block.x != 0),
                        H / threads_in_block.y + (H % threads_in_block.y != 0), 1);
    img_proc<<<blocks_in_grid, threads_in_block>>>(IMG_IN, IMG_OUT, W, H);

    /* Copy result from GPU memory to main memory */
    cudaMemcpy(img_out, IMG_OUT, W * H, cudaMemcpyDeviceToHost);

    /* Release GPU arrays */
    cudaFree(IMG_IN);
    cudaFree(IMG_OUT);

    /* Save image */
    {
        std::vector<unsigned char> image(W * H);
        memcpy(&(image[0]), img_out, W * H);
        unsigned error = lodepng::encode("img_out.png", image, W, H, LCT_GREY, 8);
        if (error) {
            fprintf(stderr, "png encoder error %d: %s\n", error, lodepng_error_text(error));
            exit(1);
        }
    }

    /* Release arrays */
    free(img_in);
    free(img_out);

    return 0;
}
