import argparse
import tkinter
import cv2
from tkinter import Label, Tk, Button,ttk, IntVar, DoubleVar, Scale
from PIL import ImageTk,Image
import os
import numpy as np
import sys




def defocus_kernel(d, sz=65):
    kern = np.zeros((sz, sz), np.uint8)
    cv2.circle(kern, (sz, sz), d, 255, -1, cv2.LINE_AA, shift=1)
    kern = np.float32(kern) / 255.0
    return kern

def motion_kernel(angle, d, sz=65):
    kern = np.ones((1, d), np.float32)
    c, s = np.cos(angle), np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    sz2 = sz // 2
    A[:,2] = (sz2, sz2) - np.dot(A[:,:2], ((d-1)*0.5, 0))
    kern = cv2.warpAffine(kern, A, (sz, sz), flags=cv2.INTER_CUBIC)
    return kern

def blur_edge(img, d=31):
    h, w  = img.shape[:2]
    img_pad = cv2.copyMakeBorder(img, d, d, d, d, cv2.BORDER_WRAP)
    img_blur = cv2.GaussianBlur(img_pad, (2*d+1, 2*d+1), -1)[d:-d,d:-d]
    y, x = np.indices((h, w))
    dist = np.dstack([x, w-x-1, y, h-y-1]).min(-1)
    w = np.minimum(np.float32(dist)/d, 1.0)
    return img*w + img_blur*(1-w)

def getAngleAndD(p1,p2):
    p1 = np.array(p1)
    p2 = np.array(p2)

    d = np.linalg.norm(p1-p2)
    
    angle = np.arctan2(*np.flip(p1-p2))*180/np.pi

    if angle < 0:
        return 180+angle, d
    return angle,d

def update_img(_):
	global defocus
	global slider_angle, slider_d, slider_SNR
	
	ang = np.deg2rad(slider_angle.get())
	d = int(slider_d.get())
	noise = 10**(-0.1*slider_SNR.get())

	print(ang, d, noise)


	if defocus:
		psf = defocus_kernel(d)
	else:
		psf = motion_kernel(ang, d)

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
	#res_rgb = (res_rgb*255).astype(np.uint8)
	
	
	to_display(res_rgb, label_img, 150, 20, 256, 256)






def click_callback(event):
	global pts
	global img_inp
	
	if len(pts) <2:
		pts.append((event.x, event.y))

	if len(pts) > 0:
		cv2.circle(img_inp, pts[0], radius=0, color=(0,0,0), thickness= 3)

	if len(pts) > 1:
		cv2.line(img_inp, pts[0], pts[1], (0,0,0), thickness = 2)
	
	to_display(img_inp, label_img, 150, 20, 256, 256)

# def switch(i):
#     label = Label(win, bg="black")
#     img = cv2.imread(list_img[i])
#     #img = change_filter(img)
    

count = 0
#next img
def start():
	global pts
	#global slider_angle, slider_d, slider_SNR
	assert len(pts) == 2
	angle,d = getAngleAndD(pts[0],pts[1])
	angle = int(angle)
	d = int(d)
	snr = 15
	#slider_angle, slider_d, slider_SNR = initialize_trackbar(angle, d, snr)
	initialize_trackbar(angle, d, snr)
	


# previous img
def back():
	global pts
	global img_inp
	pts = []
	img_inp = img_rgb.copy()
	to_display(img_inp, label_img, 150, 20, 256, 256)


def switch():

	global defocus
	defocus = not defocus



def initialize_trackbar(ang, d, snr):

	global slider_angle, slider_d, slider_SNR

	label_angle =  tkinter.Label(win, text = 'Angle')
	label_angle.grid(
    column=0,
    row=0,
    sticky='w'
)

	slider_angle = tkinter.Scale(win, from_= 0, to=180, orient= 'horizontal',
								command= update_img,
								showvalue = 1)
	slider_angle.grid(
    column=5,
    row=0,
    sticky='we'
)
	slider_angle.set(ang)

	d_label =  ttk.Label(win, text = 'd')
	d_label.grid(
    column=0,
    row=5,
    sticky='w'
)
	#d_label.pack()

	slider_d = tkinter.Scale(win, from_= 0, to=50, orient= 'horizontal', 
							command = update_img,
							variable= IntVar(),
							showvalue= 1
							
							#tickinterval = 1
							)
	
	slider_d.grid(
    column=5,
    row=5,
    sticky='nsew'
)
	slider_d.set(d)
	
	SNR_label =  ttk.Label(win, text = 'SNR')
	SNR_label.grid(
    column=0,
    row=10,
    sticky='w'
)


	slider_SNR = tkinter.Scale(win, from_= 15, to=45, orient= 'horizontal',
							command=update_img, showvalue =1)
	slider_SNR.set(snr)
	slider_SNR.grid(
    column=5,
    row=10,
    sticky='we'
)

	#return slider_angle, slider_d, slider_SNR
	#return slider_d



def to_display(img, label, x, y, w, h):


	#img = cv2.cvtColor((img * 255 / np.max(img)).astype('uint8'), cv2.COLOR_RGB2BGR)
	img = (img * 255.0 / np.max(img)).astype('uint8')
	#cv2.imwrite("output_test.jpg", img)
	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	
	#image = Image.fromarray((img*255).astype(np.uint8))
	image = Image.fromarray(img)
	image = image.resize((w, h), Image.ANTIALIAS)
	

	pic = ImageTk.PhotoImage(image)
	label.configure(image=pic)
	label.image = pic
	label.place(x=x, y=y)

def parse_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("--path", type = str, default= "test_motion.jpg", help = "path of image")
	return ap.parse_args()
	


if __name__ == "__main__":
	defocus = False
	pts = []
	#path = "F:\dft-wiener-deconvolution-with-psf/test_motion.jpg"

	args = vars(parse_args())
	path = args['path']

	win = Tk()
	win.geometry("600x600")
	right = Button(win,text="▶",bg="gray",fg="white",command=start).place(x=330,y=500,width = 40)
	left = Button(win,text="◀",bg="gray",fg="white",command=back).place(x=230,y=500,width = 40)
	switche = Button(win,text="switch",bg="gray",fg="white",command=switch).place(x=430,y=500,width = 40)

	#filters ="RGB","GRAY","HSV","FHSV","HLS"
	#com_box = ttk.Combobox(win,values = filters)
	#com_box.current(0)
	#com_box.place(x=50,y=500,width = 100)
	


	label_img = Label(win, bg="black")
	
	
	img_bw = cv2.imread(path, 0)
	img_rgb = cv2.imread(path, 1)
	
	
	
	if img_bw is None and img_rgb is None:
		print('Failed to load image:', path)
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

	#slider_angle, slider_d, slider_SNR = initialize_trackbar()
	img_inp = img_rgb.copy()
	to_display(img_inp, label_img, 150, 20, 256, 256)
	label_img.bind("<Button-1>", click_callback)
	#cv2.setMouseCallback(win, click_event)

	win.mainloop()