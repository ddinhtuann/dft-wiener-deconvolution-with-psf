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
def update(_):
    ang = np.deg2rad( cv2.getTrackbarPos('angle', win) )
    d = cv2.getTrackbarPos('d', win)
    noise = 10**(-0.1*cv2.getTrackbarPos('SNR (db)', win))

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
    cv2.imshow(win, res_rgb)

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
            cv2.circle(img_ext, pts[0], radius=0, color=(0, 0, 0), thickness=3)
        if len(pts) > 1:
            cv2.line(img=img_ext, pt1=pts[0], pt2=pts[1], color=(0, 0, 0), thickness=2)
        cv2.imshow(win, img_ext)
        if y > button[0] and y < button[1] and x > button[2] and x < button[3]:   
            print('Clicked on Button')

def start_click(event, x, y,flags, params):
    # check if the click is within the dimensions of the button
    if event == cv2.EVENT_LBUTTONDOWN:
        if y > button[0] and y < button[1] and x > button[2] and x < button[3]:   
            print('Clicked on Button')

def back(*args):
    pass


    


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, default="test_motion.jpg", help="path of files or directory")
    ap.add_argument("--defocus", action="store_true", help="optical deblur")
    ap.add_argument("--angle", type=int, default=0, help="angle")
    ap.add_argument("--d", type=int, default=0, help="angle")
    ap.add_argument("--snr", type=int, default=25, help="SNR")


    return ap.parse_args()

if __name__ == '__main__':
    # print(__doc__)
    import sys, getopt
    args = vars(parse_args())
    # opts, args = getopt.getopt(sys.argv[1:], '', ['circle', 'angle=', 'd=', 'snr='])
    # opts = dict(opts)
    fn = args['path']

    win = 'demotion'
    cv2.namedWindow(win, flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)

    pts = []

    img_bw = cv2.imread(fn, 0)
    img_rgb = cv2.imread(fn, 1)
    img_w = img_rgb.shape[1]

    if img_bw is None and img_rgb is None:
        print('Failed to load image:', fn)
        sys.exit(1)

    

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

    defocus = args['defocus']
    angle = args['angle']
    d = args['d']
    snr = args['snr']

    button = [20,60,50,250]

    control_image = np.zeros((80,img_w), np.float32)
    control_image[button[0]:button[1],button[2]:button[3]] = 180
    cv2.putText(control_image, 'Button',(100,50),cv2.FONT_HERSHEY_PLAIN, 2,(0),3)
    
    control_image = cv2.cvtColor(control_image, cv2.COLOR_GRAY2RGB)
    

    if not defocus:
        img_inp = img_rgb.copy()

        img_ext = cv2.vconcat([img_inp, control_image])
        cv2.namedWindow(win)
        #cv2.createButton("Back",back,None,cv2.QT_PUSH_BUTTON,1)
        cv2.imshow(win, img_ext)

        cv2.setMouseCallback(win, click_event)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                pts=[]
                img_inp = img_rgb.copy()
                img_ext = cv2.vconcat([img_inp, control_image])
                cv2.imshow(win, img_ext)
                continue
            if key == ord("s"):
                break

            

            
                        
            

        #Unlink window
        #cv2.setMouseCallback(win, lambda *args : None)


        print(pts)
        assert len(pts) == 2
        angle,d = getAngleAndD(pts[0],pts[1])
        angle = int(angle)
        d = int(d)



    # cv2.destroyAllWindows()
    # exit(0)


    # cv2.namedWindow('psf', 0)
    cv2.namedWindow(win, flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('angle', win, angle, 180, update)
    cv2.createTrackbar('d', win, d, 50, update)
    cv2.createTrackbar('SNR (db)', win, snr, 50, update)
    update(None)

    while True:
        ch = cv2.waitKey(5) & 0xFF
        if ch == 27 or ch == ord('q'):
            break
        if ch == ord(' '):
            defocus = not defocus
            update(None)
