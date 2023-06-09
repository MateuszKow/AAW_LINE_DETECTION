import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import copy
import time

def edge_detection(img):
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binarized=(img>100).astype(np.uint8)*255
    eroded_1= cv2.erode(binarized, np.ones((3,3), np.uint8), iterations=1)
    eroded_2= cv2.erode(eroded_1, np.ones((3,3), np.uint8), iterations=1)
    final= eroded_1-eroded_2
    return final

def hough_transform(img_bin, resolution_theta=1):
    height, width = img_bin.shape
    thetas = np.deg2rad(np.arange(-90, 90,resolution_theta))
    rho_list=[]
    for y in range(height):
        for x in range(width):
            if img_bin[y,x]==255:
                rho = []
                for theta in thetas:
                    rho.append(int(round(x * np.cos(theta) + y * np.sin(theta)))+height+width)
                rho_list.append((x,y,rho))
    thetas=thetas+90
    return rho_list,thetas,height,width

def accumulator(rho_list,thetas,height,width):
    acc = np.zeros((2*(height+width), len(thetas))).astype(np.uint8)
    acc2 = np.zeros(2*180*(width+height)).astype(np.uint8)
    for i in rho_list:
        elem=i[2]
        for theta in range(len(elem)):
            acc[elem[theta],theta]+=1
            acc2[elem[theta]*180+theta]+=1
    acc2=acc2.reshape(2*(width+height),180)
    return acc
    
def choose_maxima(acc):
    (rho_range,theta_range)=acc.shape
    maxima=[]
    for rho in range(rho_range):
        for theta in range(theta_range):
            if acc[rho][theta]>180:
                maxima.append((rho,theta))
    return maxima

def visualize(img_bin,maxima):
    img_bin = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)
    height, width, _ = img_bin.shape
    for i in maxima:
        theta=np.deg2rad(i[1]-90)
        rho=i[0]-height-width
        x0 = math.cos(theta) * rho
        y0 = math.sin(theta) * rho
        pt1 = (int(x0 + 1000*(-math.sin(theta))), int(y0 + 1000*(math.cos(theta))))
        pt2 = (int(x0 - 1000*(-math.sin(theta))), int(y0 - 1000*(math.cos(theta))))
        cv2.line(img_bin, pt1, pt2, (255,0,255), 1, cv2.LINE_AA)
    
    # cv2.imwrite('images/wynik.bmp',img_bin)


    # plt.figure(figsize = [20, 20])
    # plt.imshow(img_bin)
    # plt.title('img_bin')
    # plt.show()



def line_detect(img):
    img_bin = edge_detection(img)
    rho_list,thetas,height,width = hough_transform(img_bin)
    acc=accumulator(rho_list,thetas,height,width)
    maxima=choose_maxima(acc)
    visualize(img_bin,maxima)

start= time.time()
line_detect('images/53_3.bmp')
end= time.time()
print((end-start)*1000)