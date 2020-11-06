import os.path
from tkinter import *
from tkinter.ttk import Separator
import cv2 as cv
import numpy as np
from skimage.transform import rescale
from scipy.signal import convolve2d

im = cv.imread('portrait-Donald-Trump.jpg')


def save():
    count = 0
    string_name = 'modified.jpg'
    while os.path.isfile(string_name) is True:
        count += 1
        string_name = 'modified%d.jpg' % count
    cv.imwrite('modified%d.jpg' % count, im)
    print('modified%d.jpg' % count)


def resize_im():
    global im
    if w.get() != 0:
        im = ((rescale(im, w.get(), multichannel=True)) * 255).astype('uint8')


def refresh():
    global im
    temp = cv.imread('portrait-Donald-Trump.jpg')
    if w.get() == 0:
        size = 1
    else:
        size = w.get()
    im = ((rescale(temp, size, multichannel=True)) * 255).astype('uint8')


# cumulative histogram
def cumul_hist():
    global im
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
    corrected = lut[im].astype('uint8')
    im = corrected


# equalization
def equalization_hist():
    global im
    ch = cumul_hist()
    ch = (ch * 255).astype('uint8')
    equalized = ch[im]
    im = equalized


# kernel mean filter
def mean_filter():
    try:
        ksize = int(ksize_in.get())
    except:
        ksize = 3
    global im
    kernel = np.ones((ksize, ksize))
    kernel /= kernel.sum()
    meaned = np.zeros(im.shape)
    for dim in range(im.shape[2]):
        meaned[:, :, dim] = convolve2d(im[:, :, dim], kernel, "same")
    im = meaned.astype('uint8')


# kernel median filter
def median_filter():
    try:
        ksize = int(ksize_in.get())
    except:
        ksize = 3
    global im
    output = np.zeros(im.shape)
    print("processing.", end="")
    for i in range(ksize // 2, im.shape[0] - ksize // 2):
        if i % 100 == 0: print(".", end="")
        for j in range(ksize // 2, im.shape[1] - ksize // 2):
            for dim in range(im.shape[2]):
                output[i, j, dim] = np.median(im[i - ksize // 2:i + ksize // 2, j - ksize // 2:j + ksize // 2, dim])
    print("median filter applied succesfully")
    im = output.astype('uint8')


# GUI
wd = Tk()
ksize_label = Label(wd, text="Kernel Size")
ksize_label.pack()
ksize_in = Entry(wd)
ksize_in.pack()
b_mean = Button(wd, text="mean filter", command=mean_filter)
b_mean.pack()
b_median = Button(wd, text="median filter", command=median_filter)
b_median.pack()
sep = Separator(wd)
sep.pack()
B = Button(wd, text="auto-level", command=auto_level)
B.pack()
C = Button(wd, text="equalization", command=equalization_hist)
C.pack()
D = Button(wd, text="refresh", command=refresh)
D.pack()

w = Scale(wd, from_=0, to=2, orient=HORIZONTAL, resolution=0.1)
w.pack()
G = Button(wd, text="Apply rescale", command=resize_im)
G.pack()
S = Button(wd, text="Save", command=save)
S.pack()
cv.namedWindow('Image')

while 1:
    wd.update_idletasks()
    wd.update()
    cv.imshow('Image', im)
    cv.waitKey(20)
