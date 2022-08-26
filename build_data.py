import cv2
import os
import numpy as np
import copy

# root = '.\\datasets\\voc_mine\\JPEGImages'
# img_list = os.listdir(root)
# for i in img_list:
#     img = cv2.imread(os.path.join(root, i))
#     img_name = i.split('.')[0]
#     cv2.imwrite(os.path.join(root, img_name+'.jpg'), img)

points = []
def on_mouse(event,x,y,flags,save_name):
    global points, img,Cur_point,Start_point
    copyImg = copy.deepcopy(img)
    h,w = img.shape[:2]
    mask_img = np.zeros([h+2,w+2],dtype=np.uint8)
    if  event == cv2.EVENT_LBUTTONDOWN:
        Start_point = [x,y]
        points.append(Start_point)
        cv2.circle(img,tuple(Start_point),1,(255,255,255),0)
        cv2.imshow("",img)
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        Cur_point = [x,y]
        # print(points)
        cv2.line(img,tuple(points[-1]),tuple(Cur_point),(255,255,255))
        cv2.imshow("",img)
        points.append(Cur_point)
    elif event == cv2.EVENT_LBUTTONUP:
        Cur_point=Start_point
        cv2.line(img,tuple(points[-1]),tuple(Cur_point),(255,255,255))
        cv2.circle(img,tuple(Cur_point),1,(255,255,255))
        ret, image, mask, rect  = cv2.floodFill(img,mask_img,(x,y),(255,255,255),cv2.FLOODFILL_FIXED_RANGE)
        cv2.imwrite("maskImage.jpg",img)
        print(np.shape(image))
        segImg = np.zeros((h,w,3),np.uint8)
        src =cv2.bitwise_and(img,image)
        cv2.imwrite("segImg.jpg",src)
        cv2.waitKey(0)
        img = cv2.imread('segImg.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)  # opencv里面画轮廓是根据白色像素来画的，所以反转一下。
        ret, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        im, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
        cv2.drawContours(copyImg, contours, -1, (0, 0, 255), 3)
        cv2.imshow('RoiImg', copyImg)  # 只显示零件外轮廓
        cv2.waitKey(0)
        cimg = np.zeros_like(img)
        cimg[:, :, :] = 255
        cv2.drawContours(cimg, contours, 1, color=(0, 0, 0), thickness=-1)
        cv2.imshow('maskImg', cimg)  # 将零件区域像素值设为(0, 0, 0)
        cv2.imwrite(os.path.join('.\\datasets\\voc_mine\\SegmentationClass', save_name+'.png'), cimg)
        cv2.waitKey(0)
        # final = cv2.bitwise_or(copyImg, cimg)
        # cv2.imshow('finalImg', final)  # 执行或操作后生成想要的图片
        # cv2.waitKey(0)


root = '.\\datasets\\voc_mine\\JPEGImages'
img_list = os.listdir(root)
for i in img_list:
    ii = int(i.split('.')[0])
    if ii < 23:
        continue
    img = cv2.imread(os.path.join(root, i))
    cv2.namedWindow("")
    cv2.setMouseCallback("", on_mouse, i.split('.')[0])
    cv2.imshow("", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()