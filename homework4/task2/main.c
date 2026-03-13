#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <pthread.h>
#include <immintrin.h>

#define SIZE (256 * 1024 * 1024)
#define NUM_THREADS 4

char *buffer;
char *buf1;
char *buf2;
char *buf3;

double get_time(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

void *toUpper(void *arg) {
    int id = *(int *)arg;
    long start = id * (SIZE / NUM_THREADS);
    long end;
    if (id == NUM_THREADS - 1) end = SIZE;
    else end = (id + 1) * (SIZE / NUM_THREADS);

    for (long i = start; i < end; i++)
        if (buf1[i] >= 'a' && buf1[i] <= 'z')
            buf1[i] -= 32;

    return NULL;
}

void *toUpperSimd(void *arg) {
    int id = *(int *)arg;
    long start = id * (SIZE / NUM_THREADS);
    long end;
    if (id == NUM_THREADS - 1) end = SIZE;
    else end = (id + 1) * (SIZE / NUM_THREADS);

    __m256i a = _mm256_set1_epi8('a' - 1);
    __m256i z = _mm256_set1_epi8('z' + 1);
    __m256i target = _mm256_set1_epi8(32);

    for (long i = start; i < end; i+= 32) {
        __m256i chunk = _mm256_loadu_si256((__m256i*)(buf3 + i));
        __m256i greater_a = _mm256_cmpgt_epi8(chunk, a);
        __m256i lower_z = _mm256_cmpgt_epi8(z, chunk);
        __m256i mask = _mm256_and_si256(greater_a, lower_z);
        __m256i sub = _mm256_and_si256(mask, target);
        __m256i result = _mm256_sub_epi8(chunk, sub);
        _mm256_storeu_si256((__m256i*)(buf3 + i), result);
    }

    return NULL;
}


int main() {
    struct timespec start, end;
    double mult_thr_time, simd_time, simd_mult_thr_time;

    buffer = malloc(SIZE);
    char chars[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?";
    int len = sizeof(chars) - 1;

    for (long i = 0; i < SIZE; i++)
        buffer[i] = chars[rand() % len];

    buf1 = malloc(SIZE);
    buf2 = malloc(SIZE);
    buf3 = malloc(SIZE);

    for (long i = 0; i < SIZE; i++) {
        buf1[i] = buffer[i];
        buf2[i] = buffer[i];
        buf3[i] = buffer[i];
    }

    // Multithreading Implementation
    clock_gettime(CLOCK_MONOTONIC, &start);

    pthread_t threads[NUM_THREADS];
    int ids[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        ids[i] = i;
        pthread_create(&threads[i], NULL, toUpper, &ids[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);

    clock_gettime(CLOCK_MONOTONIC, &end);    
    mult_thr_time = get_time(start, end);

    
    // SIMD implementation
    clock_gettime(CLOCK_MONOTONIC, &start);

    __m256i a = _mm256_set1_epi8('a' - 1);
    __m256i z = _mm256_set1_epi8('z' + 1);
    __m256i target = _mm256_set1_epi8(32);

    for (long i = 0; i < SIZE; i+=32) {
        __m256i chunk = _mm256_loadu_si256((__m256i*)(buf2 + i));
        __m256i greater_a = _mm256_cmpgt_epi8(chunk, a);
        __m256i lower_z = _mm256_cmpgt_epi8(z, chunk);
        __m256i mask = _mm256_and_si256(greater_a, lower_z);
        __m256i sub = _mm256_and_si256(mask, target);
        __m256i result = _mm256_sub_epi8(chunk, sub);
        _mm256_storeu_si256((__m256i*)(buf2 + i), result);
    }

    for (long i = (SIZE / 32) * 32; i < SIZE; i++)
        if (buf2[i] >= 'a' && buf2[i] <= 'z')
            buf2[i] -= 32;

    clock_gettime(CLOCK_MONOTONIC, &end);    
    simd_time = get_time(start, end);

    // SIMD + Multithreading implementation
    clock_gettime(CLOCK_MONOTONIC, &start);

    pthread_t simd_threads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        ids[i] = i;
        pthread_create(&simd_threads[i], NULL, toUpperSimd, &ids[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(simd_threads[i], NULL);

    clock_gettime(CLOCK_MONOTONIC, &end);
    simd_mult_thr_time = get_time(start, end);

    // Program Output
    printf("Buffer size: %d MB\n", SIZE / (1024 * 1024));
    printf("Threads used: %d\n\n", NUM_THREADS);
    printf("Multithreading time:         %.3f sec\n", mult_thr_time);
    printf("SIMD time:                   %.3f sec\n", simd_time);
    printf("SIMD + Multithreading time:  %.3f sec\n", simd_mult_thr_time);

    free(buf1);
    free(buf2);
    free(buf3);
    free(buffer);

    return 0;
}
