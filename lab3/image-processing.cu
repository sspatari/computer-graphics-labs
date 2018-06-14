#include <cuda.h>
#include <cuda_runtime.h>

#include "lodepng.h"

#define CUDA_CHECK(E) \
{ \
    cudaError_t e = E ; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "%s:%d: CUDA error : %s\n", \
                __FILE__, __LINE__, \
                cudaGetErrorString(e)); \
        exit(1); \
    } \
}

#define BLOCK_SIZE 32
#define FILTER_SIZE 5

__device__ int clamp(int value, int lo, int hi)
{
    return (value < lo ? lo : value > hi ? hi : value);
}

__global__ void img_proc(const unsigned char * __restrict__ IMG_IN, unsigned char* IMG_OUT, int W, int H)
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
        unsigned short avg = 0;
        for (int r1 = -FILTER_SIZE / 2; r1 <= FILTER_SIZE / 2; ++r1) {
            int yy = clamp(y + r1, 0, H - 1);
            for (int r2 = -FILTER_SIZE / 2; r2 <= FILTER_SIZE / 2; ++r2) {
                int xx = clamp(x + r2, 0, W - 1);
                avg += __ldg(IMG_IN + yy * W + xx);
            }
        }
        avg /= FILTER_SIZE * FILTER_SIZE;
        IMG_OUT[y * W + x] = avg;
    }
}

__global__ void img_proc_step_1(const unsigned char* IMG_IN, unsigned short* TMP, int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < W && y < H) {
        unsigned short sum = 0;
        for (int r = -FILTER_SIZE / 2; r <= FILTER_SIZE / 2; ++r) {
            int xx = clamp(x + r, 0, W - 1);
            sum += IMG_IN[y * W + xx];
        }
        TMP[y * W + x] = sum;
    }
}

__global__ void img_proc_step_2(const unsigned short* __restrict__ TMP, unsigned char* IMG_OUT, int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < W && y < H) {
        unsigned short avg = 0;
        for (int r = -FILTER_SIZE / 2; r <= FILTER_SIZE / 2; ++r) {
            int yy = clamp(y + r, 0, H - 1);
            avg += __ldg(TMP + yy * W + x);
        }
        avg /= FILTER_SIZE * FILTER_SIZE;
        IMG_OUT[y * W + x] = avg;
    }
}
__host__ float get_milliseconds(cudaEvent_t &start, cudaEvent_t &stop)
{
    cudaEventSynchronize(stop); //Wait until the completion of all device work preceding the most recent call to cudaEventRecord(stop)
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds;
}

int main() // int argc, const char* argv[]
{
    unsigned char *img_in, *img_out;
    unsigned char *IMG_IN, *IMG_OUT;
    unsigned short *TMP;
    int W, H;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

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
    CUDA_CHECK(cudaMalloc(&IMG_IN, W * H));
    CUDA_CHECK(cudaMalloc(&IMG_OUT, W * H));

    /* Copy img_in from main memory to GPU memory */
    CUDA_CHECK(cudaMemcpy(IMG_IN, img_in, W * H, cudaMemcpyHostToDevice));

    /* Determine block and grid dimensions */
    dim3 threads_in_block(BLOCK_SIZE, BLOCK_SIZE, 1); // Just an example; should work fine
    dim3 blocks_in_grid((W + threads_in_block.x - 1) / threads_in_block.x,
                        (H + threads_in_block.y - 1) / threads_in_block.y, 1);

    /*
     * Approach 1
     */
    /* Launch kernel that computes result and writes it to IMG_OUT */
    for (int loop = 0; loop < 2; ++loop) {
        cudaEventRecord(start);
        img_proc<<<blocks_in_grid, threads_in_block>>>(IMG_IN, IMG_OUT, W, H);
        cudaEventRecord(stop);
        CUDA_CHECK(cudaGetLastError());
        if(loop == 1)
            fprintf(stderr, "%g milliseconds\n", get_milliseconds(start, stop));
    }
    /* Copy result from GPU memory to main memory */
    CUDA_CHECK(cudaMemcpy(img_out, IMG_OUT, W * H, cudaMemcpyDeviceToHost));

    /* Save image */
    {
        std::vector<unsigned char> image(W * H);
        memcpy(&(image[0]), img_out, W * H);
        unsigned error = lodepng::encode("img_out1.png", image, W, H, LCT_GREY, 8);
        if (error) {
            fprintf(stderr, "png encoder error %d: %s\n", error, lodepng_error_text(error));
            exit(1);
        }
    }

    /*
     * Approach 2
     */
    /* Allocate TMP array on GPU */
    CUDA_CHECK(cudaMalloc(&TMP, W * H * sizeof(unsigned short)));

    /* Launch kernel that computes result and writes it to IMG_OUT */
    for (int loop = 0; loop < 2; ++loop) {
        cudaEventRecord(start);
        img_proc_step_1<<<blocks_in_grid, threads_in_block>>>(IMG_IN, TMP, W, H);
        img_proc_step_2<<<blocks_in_grid, threads_in_block>>>(TMP, IMG_OUT, W, H);
        cudaEventRecord(stop);
        CUDA_CHECK(cudaGetLastError());
        if (loop == 1)
            fprintf(stderr, "%g milliseconds\n", get_milliseconds(start, stop));
    }

    /* Release TMP GPU array */
    CUDA_CHECK(cudaFree(TMP));

    /* Copy result from GPU memory to main memory */
    CUDA_CHECK(cudaMemcpy(img_out, IMG_OUT, W * H, cudaMemcpyDeviceToHost));

    /* Save image */
    {
        std::vector<unsigned char> image(W * H);
        memcpy(&(image[0]), img_out, W * H);
        unsigned error = lodepng::encode("img_out2.png", image, W, H, LCT_GREY, 8);
        if (error) {
            fprintf(stderr, "png encoder error %d: %s\n", error, lodepng_error_text(error));
            exit(1);
        }
    }

    /* Destroy the events */
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    /* Release GPU arrays */
    CUDA_CHECK(cudaFree(IMG_IN));
    CUDA_CHECK(cudaFree(IMG_OUT));

    /* Release arrays */
    free(img_in);
    free(img_out);

    return 0;
}
