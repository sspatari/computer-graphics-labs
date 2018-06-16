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

__global__ void img_proc_v1(const unsigned char* __restrict__ IMG_IN, unsigned char* IMG_OUT, int W, int H)
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

__global__ void img_proc_v2_step_1(const unsigned char* IMG_IN, unsigned short* TMP, int W, int H)
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

__global__ void img_proc_v2_step_2(const unsigned short* __restrict__ TMP, unsigned char* IMG_OUT, int W, int H)
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

__global__ void img_proc_v3_step_1(const unsigned char* const __restrict__ IMG_IN, unsigned short* TMP, int W, int H)
{
    __shared__ unsigned char block_shared_mem[BLOCK_SIZE][BLOCK_SIZE];

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int inner_block_width = BLOCK_SIZE - 2 * (FILTER_SIZE / 2);
    int x = blockIdx.x * inner_block_width + threadIdx.x - (FILTER_SIZE / 2);

    /* Part 1: save pixel values in shared memory */
    if (x < W + FILTER_SIZE / 2 && y < H) {
        int xx = clamp(x, 0, W - 1);
        block_shared_mem[threadIdx.y][threadIdx.x] = __ldg(IMG_IN + y * W + xx);
    }
    __syncthreads();

    if (x < W && y < H && threadIdx.x >= FILTER_SIZE / 2 && threadIdx.x < BLOCK_SIZE - FILTER_SIZE / 2) {
        unsigned short sum = 0;
        for (int r = -FILTER_SIZE / 2; r <= FILTER_SIZE / 2; ++r) {
            sum += block_shared_mem[threadIdx.y][threadIdx.x + r];
        }
        TMP[y * W + x] = sum;
    }
}

__global__ void img_proc_v3_step_2(const unsigned short* TMP, unsigned char* IMG_OUT, int W, int H)
{
    __shared__ unsigned short block_shared_mem[BLOCK_SIZE][BLOCK_SIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int inner_block_height = BLOCK_SIZE - 2 * (FILTER_SIZE / 2);
    int y = blockIdx.y * inner_block_height + threadIdx.y - (FILTER_SIZE / 2);

    /* Part 1: save pixel values in shared memory */
    if (x < W && y < H + FILTER_SIZE / 2) {
        int yy = clamp(y, 0, H - 1);
        block_shared_mem[threadIdx.y][threadIdx.x] = TMP[yy * W + x];
    }
    __syncthreads();

    if (x < W && y < H && threadIdx.y >= FILTER_SIZE / 2 && threadIdx.y < BLOCK_SIZE - FILTER_SIZE / 2) {
        unsigned short sum = 0;
        for (int r = -FILTER_SIZE / 2; r <= FILTER_SIZE / 2; ++r) {
            sum += block_shared_mem[threadIdx.y + r][threadIdx.x];
        }
        IMG_OUT[y * W + x] = sum / (FILTER_SIZE * FILTER_SIZE);
    }
}

__global__ void img_proc_v4(const unsigned char* __restrict__ IMG_IN, unsigned char* IMG_OUT, int W, int H)
{
    __shared__ unsigned char block_shared_mem[BLOCK_SIZE][BLOCK_SIZE];

    int inner_block_size = BLOCK_SIZE - 2 * (FILTER_SIZE / 2);
    int x = blockIdx.x * inner_block_size+ threadIdx.x - (FILTER_SIZE / 2);
    int y = blockIdx.y * inner_block_size+ threadIdx.y - (FILTER_SIZE / 2);

    /* Part 1: save pixel values in shared memory */
    if (x < W + FILTER_SIZE / 2 && y < H + FILTER_SIZE / 2) {
        int xx = clamp(x, 0, W - 1);
        int yy = clamp(y, 0, H - 1);
        block_shared_mem[threadIdx.y][threadIdx.x] = __ldg(IMG_IN + yy * W + xx);
    }
    __syncthreads();

    if (x < W && y < H && threadIdx.x >= FILTER_SIZE / 2 && threadIdx.x < BLOCK_SIZE - FILTER_SIZE / 2 && threadIdx.y >= FILTER_SIZE / 2 && threadIdx.y < BLOCK_SIZE - FILTER_SIZE / 2) {
        unsigned short sum = 0;
        for (int r1 = -FILTER_SIZE / 2; r1 <= FILTER_SIZE / 2; ++r1) {
            for (int r2 = -FILTER_SIZE / 2; r2 <= FILTER_SIZE / 2; ++r2) {
                sum += block_shared_mem[threadIdx.y + r1][threadIdx.x + r2];
            }
        }
        IMG_OUT[y * W + x] = sum / (FILTER_SIZE * FILTER_SIZE);;
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
     * Approach 1 (without shared memory)
     */
    /* Launch kernel that computes result and writes it to IMG_OUT */
    for (int loop = 0; loop < 2; ++loop) {
        cudaEventRecord(start);
        img_proc_v1<<<blocks_in_grid, threads_in_block>>>(IMG_IN, IMG_OUT, W, H);
        cudaEventRecord(stop);
        CUDA_CHECK(cudaGetLastError());
        if(loop == 1)
            fprintf(stderr, "%g ms - first approach(without shared memory)\n", get_milliseconds(start, stop));
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
     * Approach 2 (without shared memory)
     */
    /* Allocate memory on GPU for TMP array */
    CUDA_CHECK(cudaMalloc(&TMP, W * H * sizeof(unsigned short)));

    /* Launch kernel that computes result and writes it to IMG_OUT */
    for (int loop = 0; loop < 2; ++loop) {
        cudaEventRecord(start);
        img_proc_v2_step_1<<<blocks_in_grid, threads_in_block>>>(IMG_IN, TMP, W, H);
        img_proc_v2_step_2<<<blocks_in_grid, threads_in_block>>>(TMP, IMG_OUT, W, H);
        cudaEventRecord(stop);
        CUDA_CHECK(cudaGetLastError());
        if (loop == 1)
            fprintf(stderr, "%g ms - second approach(without shared memory)\n", get_milliseconds(start, stop));
    }
    /* Release GPU array */
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

    /*
     * Approach 3 (with shared memory)
     */

    /* Determine grid dimensions */
    int inner_block_size = BLOCK_SIZE - 2 * (FILTER_SIZE / 2);
    dim3 blocks_in_grid_3((W + inner_block_size - 1) / inner_block_size,
                        (H + inner_block_size - 1) / inner_block_size, 1);

    /* Launch kernel that computes result and writes it to IMG_OUT */
    for (int loop = 0; loop < 2; ++loop) {
        cudaEventRecord(start);
        img_proc_v4<<<blocks_in_grid_3, threads_in_block>>>(IMG_IN, IMG_OUT, W, H);
        cudaEventRecord(stop);
        CUDA_CHECK(cudaGetLastError());
        if(loop == 1)
            fprintf(stderr, "%g ms - third approach (approach 1 with shared memory)\n", get_milliseconds(start, stop));
    }
    /* Copy result from GPU memory to main memory */
    CUDA_CHECK(cudaMemcpy(img_out, IMG_OUT, W * H, cudaMemcpyDeviceToHost));

    /* Save image */
    {
        std::vector<unsigned char> image(W * H);
        memcpy(&(image[0]), img_out, W * H);
        unsigned error = lodepng::encode("img_out3.png", image, W, H, LCT_GREY, 8);
        if (error) {
            fprintf(stderr, "png encoder error %d: %s\n", error, lodepng_error_text(error));
            exit(1);
        }
    }

    /*
     * Approach 4 (with shared memory)
     */
    /* Allocate memory on GPU for TMP array */
    CUDA_CHECK(cudaMalloc(&TMP, W * H * sizeof(unsigned short)));

    /* Determine grid dimensions */
    dim3 threads_in_block_4(BLOCK_SIZE, 1, 1); // Just an example; should work fine
    dim3 blocks_in_grid_4_1((W + inner_block_size - 1) / inner_block_size,
                        (H + threads_in_block_4.y - 1) / threads_in_block_4.y, 1);
    dim3 blocks_in_grid_4_2((W + threads_in_block_4.x - 1) / threads_in_block_4.x,
                        (H + inner_block_size - 1) / inner_block_size, 1);

    /* Launch kernel that computes result and writes it to IMG_OUT */
    for (int loop = 0; loop < 2; ++loop) {
        cudaEventRecord(start);
        img_proc_v3_step_1<<<blocks_in_grid_4_1, threads_in_block_4>>>(IMG_IN, TMP, W, H);
        img_proc_v3_step_2<<<blocks_in_grid_4_2, threads_in_block_4>>>(TMP, IMG_OUT, W, H);
        cudaEventRecord(stop);
        CUDA_CHECK(cudaGetLastError());
        if(loop == 1)
            fprintf(stderr, "%g ms - forth approach (approach 2 with shared memory)\n", get_milliseconds(start, stop));
    }
    /* Release GPU array */
    CUDA_CHECK(cudaFree(TMP));

    /* Copy result from GPU memory to main memory */
    CUDA_CHECK(cudaMemcpy(img_out, IMG_OUT, W * H, cudaMemcpyDeviceToHost));

    /* Save image */
    {
        std::vector<unsigned char> image(W * H);
        memcpy(&(image[0]), img_out, W * H);
        unsigned error = lodepng::encode("img_out4.png", image, W, H, LCT_GREY, 8);
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
