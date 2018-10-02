# OpenCV_Python

#### 1.Template Matching(模板匹配)

模板匹配是一种在较大图像中搜索和查找模板图像位置的方法。OpenCV提供了一个函数cv2.matchTemplate（）。它只是在输入图像上滑动模板图像（如在2D卷积中），并比较模板图像下的输入图像的模板和补丁。在OpenCV中实现了几种比较方法。它返回一个灰度图像，其中每个像素表示该像素的邻域与模板匹配的程度。

假设输入图像的大小（WxH）且模板图像的大小（wxh），则输出图像的大小为（W-w + 1，H-h + 1）。获得结果后，可以使用cv2.minMaxLoc（）函数查找最大/最小值的位置。将其作为矩形的左上角，并将（w，h）作为矩形的宽度和高度。那个矩形是你的模板区域匹配后得到的区域。

#### 2.匹配实例

找出一张电路的图像中，指定的芯片，并标记出来

    import cv2
    import numpy as np

    # 读取名称为 p20.png 的图片，并转成黑白
    img = cv2.imread("/home/yhch/Pictures/P20.png",1）
    gray = cv2.imread("/home/yhch/Pictures/P20.png",0)
    cv2.imshow('pic',gray)

    # 读取需要检测的模板图片（黑白）
    img_template = cv2.imread("/home/yhch/Pictures/P20_temp.png",0)
    # 得到图片的高和宽
    w, h = img_template.shape[::-1]
    print(w,h)


    # 模板匹配操作
    res = cv2.matchTemplate(gray,img_template,cv2.TM_SQDIFF)

    # 得到最大和最小值得位置

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = min_loc #左上角的位置
    bottom_right = (top_left[0] + w, top_left[1] + h) #右下角的位

    # 在原图上画矩形
    cv2.rectangle(img,top_left, bottom_right, (0,0,255), 2)

    # 显示原图和处理后的图像
    cv2.imshow("img_template",img_template)
    cv2.imshow("processed",img)
    cv2.waitKey(0
    
    
##### 运行效果

![img](http://pbn3uskcn.bkt.clouddn.com/yzm2_06.png)


#### 3. API
* 用法

    
    cv2.matchTemplata(img_big,img_temp,cv2.method)
    
    img_big:在该图上查找图像
    img_temp:待查找的图像，模板图像
    method: 模板匹配的方法
    

* 关于参数 method：

method | introduce
-------|----------
CV_TM_SQDIFF 平方差匹配法|该方法采用平方差来进行匹配；最好的匹配值为0；匹配越差，匹配值越大
CV_TM_CCORR 相关匹配法|该方法采用乘法操作；数值越大表明匹配程度越好。
CV_TM_CCOEFF 相关系数匹配法|1表示完美的匹配；-1表示最差的匹配。
CV_TM_SQDIFF_NORMED |归一化平方差匹配法　　　　　　
CV_TM_CCORR_NORMED |归一化相关匹配法　　　　　　
CV_TM_CCOEFF_NORMED |归一化相关系数匹配法





#### 4.与多个对象匹配的模板

在上一实例，搜索了芯片的图像，该图像仅在图像中出现一次。如果正在搜索的图像中有多个对象出现，cv2.minMaxLoc（）就不会为提供模板图像所有位置。在这种情况下，可以使用阈值来匹配多个对象。在这个例子中，使用了游戏Mario的截图，会在其中找到硬币并标记出来。



    import cv2
    import numpy as np

    im_rgb = cv2.imread('/home/yhch/Pictures/mario.png')

    cv2.imshow('im',im_rgb)
    im_gray = cv2.cvtColor(im_rgb,cv2.COLOR_BGR2GRAY)

    template = cv2.imread('/home/yhch/Pictures/mario_coin.png',0)
    w,h = template.shape[::-1]

    res = cv2.matchTemplate(im_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(res>=threshold)


    for pt in zip(*loc[::-1]):
        cv2.rectangle(im_rgb,pt,(pt[0]+w,pt[1]+h),(0,0,255),1)

    cv2.imshow('res.png',im_rgb)
    cv2.waitKey(0
    
    
##### 运行效果    
![img](http://pbn3uskcn.bkt.clouddn.com/yzm2_07.png)


##### 说明

* python3数组的倒序 a[::-1]
    
    >>> list = [1,2,3,4,5,6]
    >>>print(list[::-1])
    [6, 5, 4, 3, 2, 1]
    >>>arry = ([1,2,3],[4,5,6])
    print(arry[::-1])
    ([4, 5, 6], [1, 2, 3])
    
    
    



* python3 zip() 函数

zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象，这样做的好处是节约了不少的内存。

我们可以使用 list() 转换来输出列表。

如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 * 号操作符，可以将元组解压为列表。
        
    >>> a = [1,2,3]
    >>> b = [4,5,6]
    >>> c = ['yhch','apple']
    >>> obj_ziped = zip(a,b,c) #返回值为一个对象
    >>> print(obj_ziped)
        <zip object at 0x7f35ac530b88> # 对象

    >>> for i in obj_ziped:
        ...     print(i)
        ... 
        (1, 4, 'yhch')
        (2, 5, 'apple')
    >>> 


* 不同阀值下匹配出的数

识别网易易盾滑动验证码

threshould = 0.4

103个，好多位置并不准确，当然有一个最准确的位置就在其中

![img](http://pbn3uskcn.bkt.clouddn.com/yzm2_08.png)

threshould = 0.57

只有1个，并且准确的找出位置
![img](http://pbn3uskcn.bkt.clouddn.com/yzm2_09.png)

threshould = 0.6

一个都没有
![img](http://pbn3uskcn.bkt.clouddn.com/yzm02_10.png)


#### 网易易盾验证码的识别

写这篇教程不是空穴来风，是我在学习爬虫的过程中，遇到滑动验证码的识别，遇到了问题
可以看到，已经能够识别出准确的位置，配和selenium滑动滑块，就能破解滑动验证码了。
但在实际过程中，是不知道会出现什么画面的验证码，不同的图像，颜色，透明度是不一样，阀值也就不一样。而只有找到准确的阀值才能得到准确的位置。

##### 如何动态的分析不同图片的阀值

* 从上面的实例可以发现阀值越小，结果就越多，阀值越大，结果越少,甚至没有结果。阀值介于[0,1],因此通过循环用二分法去试一试,当结果有且只有一个的时候，得到的threshould便是我们想要的，再通过threshold获取位置信息

    
    * 阈值始终为区间左端和右端的均值，即 threshhold = (R+L)/2；
    * 如果当前阈值查找结果数量大于1，则说明阈值太小，需要往右端靠近，即左端就增大，即L += (R - L) / 2；
    * 如果结果数量为0，则说明阈值太大，右端应该减小，即R -= (R - L) / 2；
    * 当结果数量为1时，说明阈值刚
    

##### 代码实现


    img = cv2.imread("/home/yhch/Pictures/target.jpg",1)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('/home/yhch/Pictures/template.png', 0)

    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

    L = 0
    R = 1
    count = 0
    while count < 20:
        threshold = (L+R)/2
        count += 1
        loc = np.where(res >= threshold)
        if len(loc[0]) > 1:
            L += (R-L) /2
        elif len(loc[0]) == 1:
            pt = loc[::-1]
            print('目标区域的左上角坐标:',pt[0],pt[1])
            print('次数:',count)
            print('阀值',threshold)
            break
        elif len(loc[0]) < 1:
            R -= (R-L) / 2


    cv2.rectangle(img,pt,(pt[0]+w,pt[1]+h),(34,139,34),2)

    cv2.imshow("img_template",template)
    cv2.imshow("processed",img)

    cv2.waitKey(0)
    
![img](http://pbn3uskcn.bkt.clouddn.com/yzm02_16.png)



#### 小结
通过OPencv的模板识别功能，并且用二分法对针实际场景进行二次开发，后面会利用这里的知识点，对网易易盾滑动验证码进行破解。代码以上传到github。

