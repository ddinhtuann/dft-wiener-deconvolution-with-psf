import numpy as np
import cv2




def getAngle(p1,p2):
    
    angle = np.arctan2(*np.flip(p1-p2))*180/np.pi

    if angle < 0:
        return 180+angle
    return angle



if __name__ == '__main__':
    
    im = cv2.imread("image_2022_05_11T03_26_46_071Z.png")

    # r = cv2.selectROI(im)
    pts = [[43, 87], [63, 53]]
    pts = np.array(pts)


    print(np.subtract(pts,axis=1))

    # d = np.linalg.norm(p1-p2)

    # # print(p1-p2)
    # angle = getAngle(p1,p2)
    # print(d,angle)


