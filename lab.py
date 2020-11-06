import tkinter as tk
from skimage.io import imread, imshow, imsave
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rescale
from scipy.signal import convolve2d

im = cv.imread('portrait-Donald-Trump.jpg')


def resize_im(s):
    global im
    im = rescale(im, s)


def refresh():
    global im
    im = cv.imread('portrait-Donald-Trump.jpg')


# cumulative histogram
def cumul_hist():
    cumul_histo = np.zeros((256,))
    c = 0
    for v in range(256):
        c += (im == v).sum()
        cumul_histo[v] = c
    cumul_histo /= cumul_histo.max()
    return cumul_histo


# inversion
def neg_img():
    global im
    array = []
    for i in range(256):
        array.append(255 - i)
    lut = np.array(array)
    new_img = lut[im]
    im = new_img


# auto-level
def auto_level():
    global im
    ch = cumul_hist()
    plt.figure()
    plt.plot(ch)
    plt.savefig("cumulplot.png")

    for v in range(256):
        if ch[v] > 0.05: break
    Tmin = v - 1
    for v in range(256):
        if ch[255 - v] < 0.95: break
    Tmax = (255 - v) + 1

    lut = np.arange(256)
    lut[:Tmin] = 0
    lut[Tmax:] = 255
    lut[Tmin:Tmax] = (255 / (Tmax - Tmin)) * (lut[Tmin:Tmax] - Tmin)
    corrected = lut[im]
    im = corrected


# equalization
def equalization_hist():
    ch = cumul_hist()
    ch = (ch * 255).astype('uint8')
    equalized = ch[im]
    imsave("equalized.png", equalized)


# kernel mean filter
def mean_filter(ksize):
    global im
    kernel = np.ones((ksize, ksize))
    kernel /= kernel.sum()
    meaned = np.zeros(im.shape)
    for dim in range(im.shape[2]):
        meaned[:, :, dim] = convolve2d(im[:, :, dim], kernel, "same")
    im = meaned


# kernel median filter
def median_filter(ksize):
    global im
    output = np.zeros(im.shape)
    print("processing.", end="")
    for i in range(ksize // 2, im.shape[0] - ksize // 2):
        if i % 100 == 0: print(".", end="")
        for j in range(ksize // 2, im.shape[1] - ksize // 2):
            for dim in range(im.shape[2]):
                output[i, j, dim] = np.median(im[i - ksize // 2:i + ksize // 2, j - ksize // 2:j + ksize // 2, dim])
    print("median filter applied succesfully")
    im = output


# GUI
wd = tk.Tk()
B = tk.Button(wd, text="auto-level", command=auto_level())
C = tk.Button(wd, text="equalization", command=equalization_hist())
D = tk.Button(wd, text="refresh", command=refresh())
D.pack()
C.pack()
B.pack()
cv.namedWindow('Image')
while 1:
    wd.update_idletasks()
    wd.update()
    cv.imshow('Image', im)
    cv.waitKey(1)
