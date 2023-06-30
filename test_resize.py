#!/usr/bin/env python

'''
Wiener deconvolution.

Sample shows how DFT can be used to perform Weiner deconvolution [1]
of an image with user-defined point spread function (PSF)

Usage:
  deconv_cv.py  [--circle]
      [--angle <degrees>]
      [--d <diameter>]
      [--snr <signal/noise ratio in db>]
      [<input image>]

  Use sliders to adjust PSF paramitiers.
  Keys:
    SPACE - switch btw linear/cirular PSF
    ESC   - exit

Examples:
  deconv_cv.py --angle 135 --d 22  ../data/licenseplate_motion.jpg
    (image source: http://www.topazlabs.com/infocus/_images/licenseplate_compare.jpg)

  deconv_cv.py --angle 86 --d 31  ../data/text_motion.jpg
  deconv_cv.py --circle --d 19  ../data/text_defocus.jpg
    (image source: compact digital photo camera, no artificial distortion)


[1] http://en.wikipedia.org/wiki/Wiener_deconvolution
'''

# Python 2/3 compatibility
from __future__ import print_function
import argparse
from configparser import Interpolation

import numpy as np
import cv2

# local module
# from common import nothing


def blur_edge(img, d=31):
    h, w  = img.shape[:2]
    img_pad = cv2.copyMakeBorder(img, d, d, d, d, cv2.BORDER_WRAP)
    img_blur = cv2.GaussianBlur(img_pad, (2*d+1, 2*d+1), -1)[d:-d,d:-d]
    y, x = np.indices((h, w))
    dist = np.dstack([x, w-x-1, y, h-y-1]).min(-1)
    w = np.minimum(np.float32(dist)/d, 1.0)
    return img*w + img_blur*(1-w)

def motion_kernel(angle, d, sz=65):
    kern = np.ones((1, d), np.float32)
    c, s = np.cos(angle), np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    sz2 = sz // 2
    A[:,2] = (sz2, sz2) - np.dot(A[:,:2], ((d-1)*0.5, 0))
    kern = cv2.warpAffine(kern, A, (sz, sz), flags=cv2.INTER_CUBIC)
    return kern

def defocus_kernel(d, sz=65):
    kern = np.zeros((sz, sz), np.uint8)
    cv2.circle(kern, (sz, sz), d, 255, -1, cv2.LINE_AA, shift=1)
    kern = np.float32(kern) / 255.0
    return kern
def update_deblur(_):
    ang = np.deg2rad( cv2.getTrackbarPos('angle', win_deblur) )
    d = cv2.getTrackbarPos('d', win_deblur)
    noise = 10**(-0.1*cv2.getTrackbarPos('SNR (db)', win_deblur))


    if defocus:
        psf = defocus_kernel(d)
    else:
        psf = motion_kernel(ang, d)
    # cv2.imshow('psf', psf)


    psf /= psf.sum()
    psf_pad = np.zeros_like(img_bw)
    kh, kw = psf.shape
    psf_pad[:kh, :kw] = psf
    PSF = cv2.dft(psf_pad, flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows = kh)
    PSF2 = (PSF**2).sum(-1)
    iPSF = PSF / (PSF2 + noise)[...,np.newaxis]

    # RES_BW = cv2.mulSpectrums(IMG_BW, iPSF, 0)
    RES_R = cv2.mulSpectrums(IMG_R, iPSF, 0)
    RES_G = cv2.mulSpectrums(IMG_G, iPSF, 0)
    RES_B = cv2.mulSpectrums(IMG_B, iPSF, 0)


    # res_bw = cv2.idft(RES_BW, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )
    res_r = cv2.idft(RES_R, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )
    res_g = cv2.idft(RES_G, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )
    res_b = cv2.idft(RES_B, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )

    res_rgb = np.zeros_like(img_rgb)
    res_rgb[..., 0] = res_r
    res_rgb[..., 1] = res_g
    res_rgb[..., 2] = res_b

    # res_bw = np.roll(res_bw, -kh//2, 0)
    # res_bw = np.roll(res_bw, -kw//2, 1)
    res_rgb = np.roll(res_rgb, -kh//2, 0)
    res_rgb = np.roll(res_rgb, -kw//2, 1)
    print(ang, d, zp)
    cv2.imshow(win_deblur, res_rgb)

def update_resize(*args):
    zp= args[0]/100
    ori_img = cv2.imread(fn)
    img_rgb = cv2.resize(ori_img, None, fx=zp, fy=zp, interpolation = cv2.INTER_LINEAR)
    cv2.imshow(win_resize, img_rgb)
    return img_rgb

    

def getAngleAndD(p1,p2):
    p1 = np.array(p1)
    p2 = np.array(p2)

    d = np.linalg.norm(p1-p2)
    
    angle = np.arctan2(*np.flip(p1-p2))*180/np.pi

    if angle < 0:
        return 180+angle, d
    return angle,d

def click_event(event, x, y, flags, params):

    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        if len(pts)<2:
            pts.append((x,y))

        
        # displaying the coordinates
        # for i in range(len(pts)):
        if len(pts) > 0:
            cv2.circle(img_inp, pts[0], radius=0, color=(0, 0, 0), thickness=3)
        if len(pts) > 1:
            cv2.line(img=img_inp, pt1=pts[0], pt2=pts[1], color=(0, 0, 0), thickness=2)
        cv2.imshow(win_resize, img_inp)

def back(*args):
    pass

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, default="test_motion2.jpg", help="path of files or directory")
    ap.add_argument("--defocus", action="store_true", help="optical deblur")
    ap.add_argument("--angle", type=int, default=0, help="angle")
    ap.add_argument("--d", type=int, default=0, help="angle")
    ap.add_argument("--snr", type=int, default=25, help="SNR")
    ap.add_argument("--zoom_percent", type=int, default = 100, help="Zoom in or zoom out with percent of origi image")


    return ap.parse_args()

if __name__ == '__main__':
    # print(__doc__)
    import sys, getopt
    args = vars(parse_args())
    # opts, args = getopt.getopt(sys.argv[1:], '', ['circle', 'angle=', 'd=', 'snr='])
    # opts = dict(opts)
    fn = args['path']

    defocus = args['defocus']
    angle = args['angle']
    d = args['d']
    snr = args['snr']
    zp = args['zoom_percent']

    win_deblur = 'demotion-defocus'
    win_resize = 'resize'
    win_tracbar = 'TRACKBAR'

    
    ori_img = cv2.imread(fn, 1)
    if ori_img.shape is None:
        print('Failed to load image:', fn)
        sys.exit(1)


    scaleFactor = 1
    maxScale = 200
    pts = []
    cv2.namedWindow(win_resize, flags=cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("Scale", win_resize, scaleFactor, maxScale, update_resize)
    while True:
        zp = 1 + cv2.getTrackbarPos('Scale', win_resize)/100.0
        scaledImage = cv2.resize(ori_img, None, fx=zp, fy = zp, interpolation = cv2.INTER_LINEAR)
        
        cv2.imshow(win_resize, scaledImage)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("d"):
            break

    img_bw = cv2.cvtColor(scaledImage, cv2.COLOR_BGR2GRAY)
    img_rgb =  scaledImage.copy()

    if not defocus:
        img_inp = img_rgb.copy()
        cv2.namedWindow(win_resize)
        cv2.imshow(win_resize, img_inp)
        #cv2.createButton("Back",back,None,cv2.QT_PUSH_BUTTON,1)
        

        cv2.setMouseCallback(win_resize, click_event)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                pts=[]
                img_inp = img_rgb.copy()
                cv2.imshow(win_resize, img_inp)
                continue
            if key == ord("s"):
                break

    
                

    #Unlink window
    #cv2.setMouseCallback(win_resize, lambda *args : None)


    print(pts)
    assert len(pts) == 2
    angle,d = getAngleAndD(pts[0],pts[1])
    angle = int(angle)
    d = int(d)
   

    img_r = np.zeros_like(img_bw)
    img_g = np.zeros_like(img_bw)
    img_b = np.zeros_like(img_bw)
    
    img_r = img_rgb[..., 0]
    img_g = img_rgb[..., 1]
    img_b = img_rgb[..., 2]

    

    img_rgb = np.float32(img_rgb)/255.0
    img_bw = np.float32(img_bw)/255.0
    img_r = np.float32(img_r)/255.0
    img_g = np.float32(img_g)/255.0
    img_b = np.float32(img_b)/255.0

    # cv2.imshow('input', img_rgb)

    # img_bw = blur_edge(img_bw)
    
    img_r = blur_edge(img_r)
    img_g = blur_edge(img_g)
    img_b = blur_edge(img_b)

    # IMG_BW = cv2.dft(img_bw, flags=cv2.DFT_COMPLEX_OUTPUT)
    IMG_R = cv2.dft(img_r, flags=cv2.DFT_COMPLEX_OUTPUT)
    IMG_G = cv2.dft(img_g, flags=cv2.DFT_COMPLEX_OUTPUT)
    IMG_B = cv2.dft(img_b, flags=cv2.DFT_COMPLEX_OUTPUT)

    # cv2.destroyAllWindows()
    # exit(0)


    # cv2.namedWindow('psf', 0)
    cv2.namedWindow(win_deblur, flags=cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('angle', win_deblur, angle, 180, update_deblur)
    cv2.createTrackbar('d', win_deblur, d, 50, update_deblur)
    cv2.createTrackbar('SNR (db)', win_deblur, snr, 50, update_deblur)
    update_deblur(None)

    while True:
        ch = cv2.waitKey() & 0xFF
        if ch == 27 or ch == ord('q'):
            break
        if ch == 32:
            defocus = not defocus
            update_deblur(None)
