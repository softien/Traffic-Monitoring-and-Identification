# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 12:06:05 2019

@author: softien
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 11:47:19 2019

@author: softien
"""


def getCoord(img):
    import cv2
    import numpy as np
    
    global refPoint, Flag
    Flag = 0
     
    #image = cv2.imread('images/example_01.jpg')
       
    def onMouse(event, x, y, flags, param):
        global Flag, refPoint
        if event == cv2.EVENT_LBUTTONDOWN:
           # draw circle here (etc...)
           print('x = %d, y = %d'%(x, y))
           refPoint = [x, y]
           Flag = 1
    
    while (Flag == 0):
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", onMouse)
        cv2.imshow("image", img)
        cv2.waitKey(30)
        
    cv2.destroyAllWindows()
    return refPoint