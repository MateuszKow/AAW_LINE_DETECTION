import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

import time


def line_detect(img):
    start= time.time()
    # pobranie obrazka wejściowego
    source_image = cv2.imread(img)
    source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
    
    # użycie algoytmu detekcji krawędzi Canny
    start_canny = time.time()
    distance = cv2.Canny(source_image, 50, 200, None, 3)
    canny = time.time()
    
    # kopiowanie krawędzi do obrazka wyświetlającego wynik w formacie BGR
    distance_2 = cv2.cvtColor(distance, cv2.COLOR_GRAY2BGR)
    
    # 
    start_Hough = time.time()
    lines = cv2.HoughLines(distance, 1, np.pi / 180, 250, None, 0, 0)
    
    if lines is not None:
        for i in range(0, len(lines)):
            
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            
            x0 = math.cos(theta) * rho
            y0 = math.sin(theta) * rho
            
            pt1 = (int(x0 + 1000*(-math.sin(theta))), int(y0 + 1000*(math.cos(theta))))
            pt2 = (int(x0 - 1000*(-math.sin(theta))), int(y0 - 1000*(math.cos(theta))))
            
            cv2.line(distance_2, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)
    end_Hough = time.time()
    end= time.time()
    print('Czas wykonania algorytmu Canny [ms]: ', (canny - start_canny)*1000)
    print('Czas wykonania algorytmu Hough [ms]: ', (end_Hough - start_Hough)*1000)
    print('Czas wykonania całego algorytmu [ms]: ', (end - start)*1000)

    plt.figure(figsize = [20, 20])
    plt.imshow(source_image)
    plt.title('oryginalny obrazek')
    plt.show()
    
    plt.figure(figsize = [20, 20])
    plt.imshow(distance_2)
    plt.title('Standard Hough Line Transform')
    plt.show()
    
    return 0
    
img = 'images/78.jpg'

line_detect(img)