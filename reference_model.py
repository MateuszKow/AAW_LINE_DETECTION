import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

import time



def edge_detection(img):
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binarized=(img>50).astype(np.uint8)*255
    # plt.figure(figsize = [20, 20])
    # plt.imshow(binarized, cmap='gray')
    # plt.title('binarized')
    # plt.show()
    eroded_1= cv2.erode(binarized, np.ones((3,3), np.uint8), iterations=1)
    eroded_2= cv2.erode(eroded_1, np.ones((3,3), np.uint8), iterations=1)
    final= eroded_1-eroded_2
    # plt.figure(figsize = [20, 20])
    # plt.imshow(final, cmap='gray')
    # plt.title('binarized')
    # plt.show()
    return final

def hough_transform(img_bin, resolution_theta=1):
    # Rho and Theta ranges:
    height, width = img_bin.shape
    thetas = np.deg2rad(np.arange(-90, 90,resolution_theta))
    # print(thetas)
    rho_list=[]
    for x in range(height):
        for y in range(width):
            if img_bin[x,y]==255:
                rho = []
                for theta in thetas:
                    rho.append(x * np.cos(theta) + y * np.sin(theta))
                rho_list.append((x,y,rho))
    return rho_list,thetas

def line_detect(img):
    start= time.time()
    # pobranie obrazka wejściowego
    source_image = cv2.imread(img)
    source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
    
    # użycie algoytmu detekcji krawędzi Canny
    start_canny = time.time()
    # distance = cv2.Canny(source_image, 50, 200, None, 3)
    distance = edge_detection(img)
    end_canny = time.time()
    
    # kopiowanie krawędzi do obrazka wyświetlającego wynik w formacie BGR
    distance_2 = cv2.cvtColor(distance, cv2.COLOR_GRAY2BGR)
    
    # 
    start_Hough = time.time()
    lines = cv2.HoughLines(distance, 1, np.pi / 180, 50, None, 0, 0)
    

    end_Hough = time.time()
    if lines is not None:
        for i in range(0, len(lines)):
            
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            print(rho, theta)
            

            x0 = math.cos(theta) * rho
            y0 = math.sin(theta) * rho
            
            pt1 = (int(x0 + 1000*(-math.sin(theta))), int(y0 + 1000*(math.cos(theta))))
            pt2 = (int(x0 - 1000*(-math.sin(theta))), int(y0 - 1000*(math.cos(theta))))
            
            cv2.line(distance_2, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)

    end= time.time()
    print('Czas wykonania algorytmu Canny [ms]: ', (end_canny - start_canny)*1000)
    print('Czas wykonania algorytmu Hough [ms]: ', (end_Hough - start_Hough)*1000)
    print('Czas wykonania całego algorytmu [ms]: ', (end - start)*1000)

    # plt.figure(figsize = [20, 20])
    # plt.imshow(source_image)
    # plt.title('oryginalny obrazek')
    # plt.show()
    
    plt.figure(figsize = [20, 20])
    plt.imshow(distance_2)
    plt.title('Standard Hough Line Transform')
    plt.show()
    
    return 0

def display(rho_list,thetas):
    for i in range(0, len(rho_list), 500):
        plt.plot(thetas,rho_list[i][2])
        print(rho_list[i][0:2])
        print(rho_list[i][2])
    plt.show()


def biale_krawedzie(img):  
    eroded = edge_detection(img)
    plt.figure(figsize = [20, 20])
    plt.imshow(eroded, cmap='gray')
    plt.title('binarized')
    plt.show()
    j=0
    for x in range(eroded.shape[0]):
        for y in range(eroded.shape[1]):
            if eroded[x,y]==255:
                j+=1
            else:
                pass
    print(j)

def hough():
    img = 'images/53.jpg'
    image = edge_detection(img)
    plt.figure(figsize = [20, 20])
    plt.imshow(image, cmap='gray')
    plt.title('binarized')
    plt.show()
    lista=[]
    A=200
    B=200
    for theta in range(-90,91):
        rho = A * np.cos(theta*math.pi/180) + B * np.sin(theta*math.pi/180)
        lista.append(int(round(rho)))
    # for x in range(image.shape[0]):
    #     for y in range(image.shape[1]):
    #         if x==200 and y==280:
    #             if image[x,y]==255:
    #                 for theta in range(-90,91):
    #                     rho = A * np.cos(theta*math.pi/180) + B * np.sin(theta*math.pi/180)
    #                     print(rho)
    #                     lista.append((rho))
    plt.plot(lista)
    plt.show()
    print(lista)

# img = 'images/53_3.bmp'
# img = 'from_camera.bmp'
# line_detect(img)
# edge_detection(img)
# image = edge_detection(img)
# rho_list,thetas=hough_transform(image)

# display(rho_list,thetas)
# line_detect('test.bmp')
line_detect('SobelFilterImage_Input.bmp')


# def check_hough():
#     img = 'images/53.jpg'
#     image = edge_detection(img)
#     source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)

# biale_krawedzie(img)
# hough()
