import cv2
import numpy as np

im_rgb = cv2.imread('/home/yhch/Pictures/mario.png')

#cv2.imshow('im',im_rgb)
im_gray = cv2.cvtColor(im_rgb,cv2.COLOR_BGR2GRAY)

template = cv2.imread('/home/yhch/Pictures/mario_coin.png',0)
w,h = template.shape[::-1]

res = cv2.matchTemplate(im_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.9
loc = np.where(res>=threshold)
print(loc[1][0])


for pt in zip(*loc[::-1]):
    #print(pt)
    cv2.rectangle(im_rgb,pt,(pt[0]+w,pt[1]+h),(0,0,255),1)

cv2.imshow('res.png',im_rgb)
cv2.waitKey(0)