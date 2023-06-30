# importing the module
import cv2
import numpy as np

pts = []
# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):

    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        pts.append([x,y])

        
        # displaying the coordinates
        # for i in range(len(pts)):
        if len(pts) > 0:
            cv2.circle(img, pts[0], radius=0, color=(0, 0, 0), thickness=3)
        if len(pts) > 1:
            cv2.line(img=img, pt1=pts[0], pt2=pts[1], color=(0, 0, 0), thickness=2)
        cv2.imshow('image', img)




# driver function
if __name__=="__main__":

    # reading the image
    img = cv2.imread('image_2022_05_11T03_26_46_071Z.png')

    # displaying the image
    cv2.imshow('image', img)

    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord("s"):
            print(pts)
            break
    cv2.destroyAllWindows()
