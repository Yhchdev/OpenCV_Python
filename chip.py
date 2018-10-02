import cv2
import numpy as np

# 读取名称为 p20.png 的图片，并转成黑白
img = cv2.imread("/home/yhch/Pictures/P20.png",1)
gray = cv2.imread("/home/yhch/Pictures/P20.png",0)
cv2.imshow('pic',gray)

# 读取需要检测的芯片图片（黑白）
img_template = cv2.imread("/home/yhch/Pictures/P20_temp.png",0)
# 得到芯片图片的高和宽
w, h = img_template.shape[::-1]
print(w,h)

#img_template = abs(255 - img_template)
#res = cv2.matchTemplate(gray,img_template,cv2.TM_CCOEFF_NORMED)


# 模板匹配操作
res = cv2.matchTemplate(gray,img_template,cv2.TM_SQDIFF)

# 得到最大和最小值得位置

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = min_loc #左上角的位置
bottom_right = (top_left[0] + w, top_left[1] + h) #右下角的位置



#x, y = np.unravel_index(res.argmax(), res.shape)


#print(x,y)
# 展示圈出来的区域
#cv2.rectangle(img, (y, x), (y+w, x + h), (0, 0, 255), 2)

# 在原图上画矩形
cv2.rectangle(img,top_left, bottom_right, (0,0,255), 2)



# 显示原图和处理后的图像
cv2.imshow("img_template",img_template)
cv2.imshow("processed",img)

cv2.waitKey(0)
