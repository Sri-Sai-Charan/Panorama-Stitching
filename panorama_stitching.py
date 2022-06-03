#!/usr/bin/python3
import numpy as np
import cv2 as cv


def main():
    img_a=cv.imread('Code/Media/Q2/Q2imageA.png')
    img_b=cv.imread('Code/Media/Q2/Q2imageB.png')
    
        
    img_a_gray =cv.cvtColor(img_a,cv.COLOR_BGR2GRAY)
    img_b_gray =cv.cvtColor(img_b,cv.COLOR_BGR2GRAY)
    
    sift = cv.SIFT_create()
    key_points1, descriptor_1 = sift.detectAndCompute(img_b_gray,None)
    key_points2, descriptor_2 = sift.detectAndCompute(img_a_gray,None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(descriptor_1,descriptor_2, k=2)
    
    good = []
    for m in matches:
        if (m[0].distance < 0.5*m[1].distance):
            good.append(m)
    matches = np.asarray(good)

    if (len(matches[:,0]) >= 4):
        src = np.float32([ key_points1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ key_points2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        H, masked = cv.findHomography(src, dst, cv.RANSAC, 5.0)
    else:
        raise AssertionError('Can t find enough keypoints.')
    dst = cv.warpPerspective(img_b,H,((img_b.shape[1] + img_a.shape[1]), img_a.shape[0])) 
    
    dst[0:img_b.shape[0], 0:img_b.shape[1]] = img_b
    dst[0:img_a.shape[0], 0:img_a.shape[1]] = img_a
    final_output = dst[0:799,0:626,:]

    cv.namedWindow('Question 2 Results',cv.WINDOW_NORMAL)
    cv.imshow('Question 2 Results',final_output)
    cv.waitKey(0)


if __name__ == '__main__':
    main()