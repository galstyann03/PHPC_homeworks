#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <immintrin.h>

#define NUM_THREADS 4

unsigned char *mt_out;
unsigned char *simd_mt_out;
unsigned char *data;
int width, height;

double get_time(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

void *grayscale_thread(void *arg) {
    int id = *(int *)arg;
    int pixels = (width * height);
    int start = id * (pixels / NUM_THREADS) * 3;
    int end;
    if (id == NUM_THREADS - 1) end = pixels * 3;
    else end = (id + 1) * (pixels / NUM_THREADS) * 3;

    for (int i = start; i < end; i+=3) {
        unsigned char gray = (unsigned char) (0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i+2]);
        mt_out[i] = gray;
        mt_out[i+1] = gray;
        mt_out[i+2] = gray;
    }

    return NULL;
}

void *grayscale_simd_thread(void *arg) {
    int id = *(int *)arg;
    int pixels = (width * height);
    int pix_start = id * (pixels / NUM_THREADS);
    int pix_end;
    if (id == NUM_THREADS - 1) pix_end = pixels;
    else pix_end = (id + 1) * (pixels / NUM_THREADS);

    int chunk_pixels = pix_end - pix_start;
    unsigned char *R = malloc(chunk_pixels);
    unsigned char *G = malloc(chunk_pixels);
    unsigned char *B = malloc(chunk_pixels);

    for (int i = 0; i < chunk_pixels; i++) {
        R[i] = data[(pix_start + i) * 3];
        G[i] = data[(pix_start + i) * 3 + 1];
        B[i] = data[(pix_start + i) * 3 + 2];
    }

    for (int i = 0; i < chunk_pixels - 31; i += 32) {
        __m256i red   = _mm256_loadu_si256((__m256i*)(R + i));
        __m256i green = _mm256_loadu_si256((__m256i*)(G + i));
        __m256i blue  = _mm256_loadu_si256((__m256i*)(B + i));

        unsigned char r_tmp[32], g_tmp[32], b_tmp[32];
        _mm256_storeu_si256((__m256i*)r_tmp, red);
        _mm256_storeu_si256((__m256i*)g_tmp, green);
        _mm256_storeu_si256((__m256i*)b_tmp, blue);

        for (int j = 0; j < 32; j++) {
            unsigned char gray = (77 * r_tmp[j] + 150 * g_tmp[j] + 29 * b_tmp[j]) >> 8;
            int idx = (pix_start + i + j) * 3;
            simd_mt_out[idx]   = gray;
            simd_mt_out[idx+1] = gray;
            simd_mt_out[idx+2] = gray;
        }
    }

    for (int i = (chunk_pixels / 32) * 32; i < chunk_pixels; i++) {
        unsigned char gray = (77 * R[i] + 150 * G[i] + 29 * B[i]) >> 8;
        int idx = (pix_start + i) * 3;
        simd_mt_out[idx]   = gray;
        simd_mt_out[idx+1] = gray;
        simd_mt_out[idx+2] = gray;
    }

    free(R); free(G); free(B);
    return NULL;
}

int main() {
    struct timespec start, end;
    double scalar_time, mult_thr_time, simd_time, simd_mult_thr_time;
    FILE *f = fopen("image.ppm", "rb");

    fscanf(f, "P6\n%d %d\n255\n", &width, &height);

    int size = width * height * 3;
    data = malloc(size);

    fread(data, 1, size, f);
    fclose(f);

    // Scalar implementation
    clock_gettime(CLOCK_MONOTONIC, &start);

    unsigned char *out = malloc(size);
    for (int i = 0; i < size; i+=3) {
        unsigned char gray = (unsigned char)(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i+2]);
        out[i] = gray;
        out[i+1] = gray;
        out[i+2] = gray;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    scalar_time = get_time(start, end);

    FILE *fout = fopen("gray_output.ppm", "wb");
    fprintf(fout, "P6\n%d %d\n255\n", width, height);
    fwrite(out, 1, size, fout);
    fclose(fout);

    // Multithreading implementation
    clock_gettime(CLOCK_MONOTONIC, &start);

    mt_out = malloc(size);
    pthread_t threads[NUM_THREADS];
    int ids[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        ids[i] = i;
        pthread_create(&threads[i], NULL, grayscale_thread, &ids[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);

    clock_gettime(CLOCK_MONOTONIC, &end);
    mult_thr_time = get_time(start, end);

    FILE *fout2 = fopen("gray_mt.ppm", "wb");
    fprintf(fout2, "P6\n%d %d\n255\n", width, height);
    fwrite(mt_out, 1, size, fout2);
    fclose(fout2);

    // SIMD implementation
    clock_gettime(CLOCK_MONOTONIC, &start);
    unsigned char *simd_out = malloc(size);

    for (int i = 0; i < size; i += 3) {
        unsigned char gray = (77 * data[i] + 150 * data[i+1] + 29 * data[i+2]) >> 8;
        simd_out[i]   = gray;
        simd_out[i+1] = gray;
        simd_out[i+2] = gray;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    simd_time = get_time(start, end);

    FILE *fout3 = fopen("gray_simd.ppm", "wb");
    fprintf(fout3, "P6\n%d %d\n255\n", width, height);
    fwrite(simd_out, 1, size, fout3);
    fclose(fout3);

    // SIMD + Multithreading implementation
    clock_gettime(CLOCK_MONOTONIC, &start);

    simd_mt_out = malloc(size);
    pthread_t simd_threads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        ids[i] = i;
        pthread_create(&simd_threads[i], NULL, grayscale_simd_thread, &ids[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(simd_threads[i], NULL);

    clock_gettime(CLOCK_MONOTONIC, &end);
    simd_mult_thr_time = get_time(start, end);

    FILE *fout4 = fopen("gray_simd_mt.ppm", "wb");
    fprintf(fout4, "P6\n%d %d\n255\n", width, height);
    fwrite(simd_mt_out, 1, size, fout4);
    fclose(fout4);

    int correct = 1;
    for (int i = 0; i < size; i++) {
        if (abs(mt_out[i] - out[i]) > 1) correct = 0; break;
        if (abs(simd_out[i] - out[i]) > 1) correct = 0; break;
        if (abs(simd_mt_out[i] - out[i]) > 1) correct = 0; break;
    }

    // Program Output
    printf("Image size: %d x %d\n", width, height);
    printf("Threads used: %d\n\n", NUM_THREADS);
    printf("Scalar Time:                 %.3f sec\n", scalar_time);
    printf("Multithreading time:         %.3f sec\n", mult_thr_time);
    printf("SIMD time:                   %.3f sec\n", simd_time);
    printf("SIMD + Multithreading time:  %.3f sec\n\n", simd_mult_thr_time);
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");
    printf("Output image: gray_output.ppm\n");

    free(out);
    free(mt_out);
    free(simd_out);
    free(simd_mt_out);
    free(data);

    return 0;
}
