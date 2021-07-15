from PIL import Image, ImageFont, ImageDraw
import cv2
import numpy as np
from utils import *





if __name__ == '__main__':
    img_draw, box, cut_img = get_text_img(
        font_size=40,
        font_type_path="STFANGSO.TTF",
        text="测试ing中\n分我阿比我让你",
        back_color=(255, 255, 255,0),
        font_color=(200, 0, 0),pad_size=10)

    w,h,c=cut_img.shape
    img = cv2.imread("1.png")
    start_point = (40,100)
    tmp_img = img[start_point[0]:start_point[0]+w,start_point[1]:start_point[1]+h]
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # tmp_img[cut_img<255] = cv2.addWeighted(tmp_img[cut_img<255],0.5,cut_img[cut_img<255],0.5,gamma=0)
    # print(np.where(cut_img<255))
    # a = np.where(cut_img<255)
    # tmp_img[cut_img==0]=cut_img[cut_img==0]
    # tmp_img[cut_img<255]=cut_img[cut_img<255]
    # cv2.imshow("img",img)
    # cv2.waitKey(0)
