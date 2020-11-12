#!/usr/bin/python

import os.path
from tkinter import *
from tkinter.ttk import Separator
import cv2 as cv
import numpy as np
from skimage.transform import rescale
from scipy.signal import convolve2d

im = cv.imread(str(sys.argv[1]))


def save():
    count = 0
    while os.path.isfile('modified%d.png' % count) is True:
        count += 1
    cv.imwrite('modified%d.png' % count, im)
    print('modified%d.png' % count)


def resize_im():
    global im
    if w.get() != 0:
        im = ((rescale(im, w.get(), multichannel=True)) * 255).astype('uint8')


def refresh():
    global im
    temp = cv.imread('etretat.jpg')
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
    new_img = lut[im].astype('uint8')
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


def luminosity(inc):
    global im
    img_thres = im
    level = 10
    if inc:
        img_thres[im > 255 - level] = 255 - level  # loss of information
        im = im + level
    else:
        img_thres[im < level] = level  # loss of information
        im = im - level


def sat(inc):
    global im
    img = cv.cvtColor(im, cv.COLOR_RGB2HSV)
    h, s, v = cv.split(img)
    img_thresh = s
    level = 20
    if inc:
        img_thresh[s > 255 - level] = 255 - level  # loss of information
        s = s + level
    else:
        img_thresh[s < level] = level  # loss of information
        s = s - level
    img = cv.merge((h, s, v))
    img = cv.cvtColor(img, cv.COLOR_HSV2RGB)
    im = img
    cv.imwrite("test sat.png", im)


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


# kernel max filter
def max_filter():
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
                output[i, j, dim] = np.min(im[i - ksize // 2:i + ksize // 2, j - ksize // 2:j + ksize // 2, dim])
    print("max filter applied succesfully")
    im = output.astype('uint8')


# kernel min filter
def min_filter():
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
                output[i, j, dim] = np.max(im[i - ksize // 2:i + ksize // 2, j - ksize // 2:j + ksize // 2, dim])
    print("min filter applied succesfully")
    im = output.astype('uint8')


# GUI
wd = Tk()
ksize_label = Label(wd, text="Kernel Size (default = 3)", bg='red')
ksize_label.pack()
ksize_in = Entry(wd, bg='red')
ksize_in.pack()
b_mean = Button(wd, text="mean filter", command=mean_filter, bg='red')
b_mean.pack()
b_median = Button(wd, text="median filter", command=median_filter, bg='red')
b_median.pack()
b_max = Button(wd, text="max filter", command=max_filter, bg='red')
b_max.pack()
b_min = Button(wd, text="min filter", command=min_filter, bg='red')
b_min.pack()
sep = Separator(wd)
sep.pack()
B = Button(wd, text="auto-level", command=auto_level, bg='green')
B.pack()
C = Button(wd, text="equalization", command=equalization_hist, bg='green')
C.pack()
inv = Button(wd, text="inversion", command=neg_img, bg='green')
inv.pack()
lump = Button(wd, text="increase luminosity", command=lambda: luminosity(True), bg='green')
lump.pack()
lumm = Button(wd, text="reduce luminosity", command=lambda: luminosity(False), bg='green')
lumm.pack()
satp = Button(wd, text="increase saturation", command=lambda: sat(True), bg='green')
satp.pack()
satm = Button(wd, text="reduce saturation", command=lambda: sat(False), bg='green')
satm.pack()
w = Scale(wd, from_=0, to=2, orient=HORIZONTAL, resolution=0.1)
w.pack()
G = Button(wd, text="Apply rescale", command=resize_im, bg='yellow')
G.pack()
S = Button(wd, text="Save", command=save, bg='yellow')
S.pack()
D = Button(wd, text="Reload", command=refresh, bg='yellow')
D.pack()
cv.namedWindow('Image')

while 1:
    wd.update_idletasks()
    wd.update()
    cv.imshow('Image', im)
    k = cv.waitKey(20) & 0xFF
    if k == ord('q'):  # quitter
        break
