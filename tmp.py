# import cv2, random
# 
# from utils import *
# 
# 
# img = np.zeros((1080, 1920, 3), np.uint8)
# triangle = np.array([(0, 122), (147, 0), (172, 30), (25, 152)])
# 
# cv2.fillConvexPoly(img, triangle, (255, 255, 255))
# 
# 
# 
# txts = open("ppocr_keys_v1.txt","r",encoding="utf-8").read().split("\n")
# 
# t = my_get_text(txts)
# print(t)
# 
# img_draw = get_text_img(text=t)
# rot, coor = my_rotate_PIL(img_draw, angle=40)
# print(coor)
# background = Image.open("1.png")
# foreground = rot
# 
# start_point = (500,600)
# box_list = [(start_point[0]+point[0],start_point[1]+point[1]) for point in coor]
# 
# background.paste(foreground,start_point, foreground)
# # background.show()
# 
# img = cv2.cvtColor(np.asarray(background), cv2.COLOR_RGBA2BGR)
# my_line_4point(img,box_list)
# cv2.imshow("img",img)
# cv2.waitKey(0)


import numpy as np
import cv2

def mask_bite(mask,arr):
    img = np.zeros(mask.shape, np.uint8)
    triangle = np.array(arr)
    cv2.fillConvexPoly(img, triangle, (100, 100))
    return True, img

if __name__ == '__main__':

    mask = np.zeros((1080,960), np.uint8)
    arr = [[(500, 728), (652, 600), (677, 630)]]

    flag,mask2=mask_bite(mask,arr)

    cv2.imshow("1",mask)
    cv2.imshow("2",mask2)
    cv2.waitKey(0)
