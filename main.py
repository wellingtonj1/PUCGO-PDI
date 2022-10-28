import csv
import cv2
import numpy as np
import os
import stats

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def sharpen(img):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(img, -1, kernel)

def convolution(img, kernel):
    # get image dimensions
    img_height = img.shape[0]
    img_width = img.shape[1]
    # get kernel dimensions
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]
    # create output image
    output = np.zeros(img.shape)
    # loop over every pixel of the image
    for x in range(img_height):
        for y in range(img_width):
            # element-wise multiplication of the kernel and the image
            output[x, y] = (kernel * img[x: x + kernel_height, y: y + kernel_width]).sum()
    return output

def gaussian_noise(img, mean, var):
    row, col, ch = img.shape
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = img + gauss
    return noisy

def noise_salt_and_pepper(img, prob):
    output = np.zeros(img.shape, np.uint8)
    thres = 1 - prob
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = img[i][j]
    return output

def raileigh_noise(img, sigma):
    row, col, ch = img.shape
    gauss = np.random.normal(0, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = img + gauss
    return noisy

def erlang_noise(img, k, theta):
    row, col, ch = img.shape
    gauss = np.random.gamma(k, theta, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = img + gauss
    return noisy

def exponencial_noise(img, scale):
    row, col, ch = img.shape
    gauss = np.random.exponential(scale, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = img + gauss
    return noisy

def uniform_noise(img, low, high):
    row, col, ch = img.shape
    gauss = np.random.uniform(low, high, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = img + gauss
    return noisy

def noise_reduction_filter(img, kernel_size):
    return cv2.blur(img, (kernel_size, kernel_size))

def aritmetic_median_filter(img, kernel_size):
    # get image dimensions
    img_height = img.shape[0]
    img_width = img.shape[1]
    # get kernel dimensions
    kernel_height = kernel_size
    kernel_width = kernel_size
    # create output image
    output = np.zeros(img.shape)
    # loop over every pixel of the image
    for x in range(img_height):
        for y in range(img_width):
            # element-wise multiplication of the kernel and the image
            output[x, y] = np.median(img[x: x + kernel_height, y: y + kernel_width])
    return output

def harmonic_median_filter(img, kernel_size):
    # get image dimensions
    img_height = img.shape[0]
    img_width = img.shape[1]
    # get kernel dimensions
    kernel_height = kernel_size
    kernel_width = kernel_size
    # create output image
    output = np.zeros(img.shape)
    # loop over every pixel of the image
    for x in range(img_height):
        for y in range(img_width):
            # element-wise multiplication of the kernel and the image
            output[x, y] = stats.hmean(img[x: x + kernel_height, y: y + kernel_width])
    return output

def contraharmonic_median_filter(img, kernel_size, q):
    # get image dimensions
    img_height = img.shape[0]
    img_width = img.shape[1]
    # get kernel dimensions
    kernel_height = kernel_size
    kernel_width = kernel_size
    # create output image
    output = np.zeros(img.shape)
    # loop over every pixel of the image
    for x in range(img_height):
        for y in range(img_width):
            # element-wise multiplication of the kernel and the image
            output[x, y] = stats.contraharmonic_mean(img[x: x + kernel_height, y: y + kernel_width], q)
    return output

def median_filter(img, kernel_size):
    return cv2.medianBlur(img, kernel_size)

def min_max_filter(img, kernel_size):
    # get image dimensions
    img_height = img.shape[0]
    img_width = img.shape[1]
    # get kernel dimensions
    kernel_height = kernel_size
    kernel_width = kernel_size
    # create output image
    output = np.zeros(img.shape)
    # loop over every pixel of the image
    for x in range(img_height):
        for y in range(img_width):
            # element-wise multiplication of the kernel and the image
            output[x, y] = np.max(img[x: x + kernel_height, y: y + kernel_width]) - np.min(img[x: x + kernel_height, y: y + kernel_width])
    return output

def mid_point_filter(img, kernel_size):
    # get image dimensions
    img_height = img.shape[0]
    img_width = img.shape[1]
    # get kernel dimensions
    kernel_height = kernel_size
    kernel_width = kernel_size
    # create output image
    output = np.zeros(img.shape)
    # loop over every pixel of the image
    for x in range(img_height):
        for y in range(img_width):
            # element-wise multiplication of the kernel and the image
            output[x, y] = (np.max(img[x: x + kernel_height, y: y + kernel_width]) + np.min(img[x: x + kernel_height, y: y + kernel_width])) / 2
    return output

def aparated_median_alfa_trimmed_filter(img, kernel_size, d):
    # get image dimensions
    img_height = img.shape[0]
    img_width = img.shape[1]
    # get kernel dimensions
    kernel_height = kernel_size
    kernel_width = kernel_size
    # create output image
    output = np.zeros(img.shape)
    # loop over every pixel of the image
    for x in range(img_height):
        for y in range(img_width):
            # element-wise multiplication of the kernel and the image
            output[x, y] = np.median(img[x: x + kernel_height, y: y + kernel_width]) - d * np.std(img[x: x + kernel_height, y: y + kernel_width])
    return output

def pass_band_filter(img, kernel_size, d0):
    # get image dimensions
    img_height = img.shape[0]
    img_width = img.shape[1]
    # get kernel dimensions
    kernel_height = kernel_size
    kernel_width = kernel_size
    # create output image
    output = np.zeros(img.shape)
    # loop over every pixel of the image
    for x in range(img_height):
        for y in range(img_width):
            # element-wise multiplication of the kernel and the image
            output[x, y] = np.median(img[x: x + kernel_height, y: y + kernel_width]) + d0 * np.std(img[x: x + kernel_height, y: y + kernel_width])
    return output

def periodic_noise(img, low, high):
    row, col, ch = img.shape
    gauss = np.random.uniform(low, high, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = img + gauss
    return noisy

def band_rejection_butterworth_filter(img, kernel_size, d0, n):
    # get image dimensions
    img_height = img.shape[0]
    img_width = img.shape[1]
    # get kernel dimensions
    kernel_height = kernel_size
    kernel_width = kernel_size
    # create output image
    output = np.zeros(img.shape)
    # loop over every pixel of the image
    for x in range(img_height):
        for y in range(img_width):
            # element-wise multiplication of the kernel and the image
            output[x, y] = np.median(img[x: x + kernel_height, y: y + kernel_width]) + d0 * np.std(img[x: x + kernel_height, y: y + kernel_width])
    return output

def band_rejection_gaussian_filter(img, kernel_size, d0):
    # get image dimensions
    img_height = img.shape[0]
    img_width = img.shape[1]
    # get kernel dimensions
    kernel_height = kernel_size
    kernel_width = kernel_size
    # create output image
    output = np.zeros(img.shape)
    # loop over every pixel of the image
    for x in range(img_height):
        for y in range(img_width):
            # element-wise multiplication of the kernel and the image
            output[x, y] = np.median(img[x: x + kernel_height, y: y + kernel_width]) + d0 * np.std(img[x: x + kernel_height, y: y + kernel_width])
    return output

def pass_band_filter(img, kernel_size, d0):
    # get image dimensions
    img_height = img.shape[0]
    img_width = img.shape[1]
    # get kernel dimensions
    kernel_height = kernel_size
    kernel_width = kernel_size
    # create output image
    output = np.zeros(img.shape)
    # loop over every pixel of the image
    for x in range(img_height):
        for y in range(img_width):
            # element-wise multiplication of the kernel and the image
            output[x, y] = np.median(img[x: x + kernel_height, y: y + kernel_width]) + d0 * np.std(img[x: x + kernel_height, y: y + kernel_width])
    return output

def notch_filter(img, kernel_size, d0, n):
    # get image dimensions
    img_height = img.shape[0]
    img_width = img.shape[1]
    # get kernel dimensions
    kernel_height = kernel_size
    kernel_width = kernel_size
    # create output image
    output = np.zeros(img.shape)
    # loop over every pixel of the image
    for x in range(img_height):
        for y in range(img_width):
            # element-wise multiplication of the kernel and the image
            output[x, y] = np.median(img[x: x + kernel_height, y: y + kernel_width]) + d0 * np.std(img[x: x + kernel_height, y: y + kernel_width])
    return output

def wiener_filter(img, kernel_size, d0, n):
    # get image dimensions
    img_height = img.shape[0]
    img_width = img.shape[1]
    # get kernel dimensions
    kernel_height = kernel_size
    kernel_width = kernel_size
    # create output image
    output = np.zeros(img.shape)
    # loop over every pixel of the image
    for x in range(img_height):
        for y in range(img_width):
            # element-wise multiplication of the kernel and the image
            output[x, y] = np.median(img[x: x + kernel_height, y: y + kernel_width]) + d0 * np.std(img[x: x + kernel_height, y: y + kernel_width])
    return output

def main():

    # Read in and grayscale the image
    image = cv2.imread('images/clock.jpg')
    # pass image on wiener filter and show the diference
    # between the original image and the filtered image
    cv2.imshow('Original Image', image)
    cv2.waitKey(0)
    
    

if __name__ == '__main__':
    main()
