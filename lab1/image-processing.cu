#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "lodepng.h"

__global__ void img_proc(const unsigned char* IMG_IN, unsigned char* IMG_OUT, int W, int H)
{
    /* TODO: write a kernel that is executed for each pixel (x,y) in the image
     * and does the following:
     * compute the average value of the 5x5 neighborhood of (x,y):
     * sum up all IMG_IN pixels with horizontal indices x-2, ..., x+2 and
     * vertical indices y-2, ..., y+2, then divide by 25 and write
     * the result to IMG_OUT.
     * Make sure all indices are within valid range (at borders, you can simply
     * clip your neighborhood: if x is less than 0 then set it to 0, and similar
     * for the other borders).
     */
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
    TODO;

    /* Copy img_in from main memory to GPU memory */
    TODO;

    /* Launch kernel that computes result and writes it to IMG_OUT */
    dim3 threads_in_block(32, 32, 1); // Just an example; should work fine
    TODO;

    /* Copy result from GPU memory to main memory */
    TODO;

    /* Release GPU arrays */
    TODO;

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
    TODO;

    return 0;
}
