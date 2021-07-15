import cv2
import numpy as np
from PIL import Image,ImageFont,ImageDraw
from utils import get_min_box

def get_text_img_tmp(img_path="1.png",
        font_size=40,
        font_type_path="STFANGSO.TTF",
        text="测试ing中",
        back_color=(255, 255, 255, 0),
        font_color=(255, 0, 0, 255),
        pad_size=10):
    """ 画两倍框，再裁剪 """
    text = text.replace("\n","")
    text_len = len(text)
    font = ImageFont.truetype(font_type_path, font_size, encoding="utf-8")
    img_draw = Image.new('RGBA', ((text_len * font_size + pad_size) * 2, (font_size + pad_size) * 2), back_color)
    draw = ImageDraw.Draw(img_draw)
    draw.text((int(font_size * 0.1)+pad_size, int(font_size * 0.1)+pad_size), text, fill=font_color, font=font)
    img = cv2.cvtColor(np.asarray(img_draw), cv2.COLOR_RGBA2BGRA)
    box = get_min_box(img)
    return img, box


img_draw, box = get_text_img_tmp()
# w, h, c = cut_img.shape
# img = cv2.imread("1.png")
# start_point = (40, 100)
# tmp_img = img[start_point[0]:start_point[0] + w, start_point[1]:start_point[1] + h]
# image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

background = Image.open("1.png")
foreground = Image.open("tmp.png")#Image.fromarray(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB))
foreground = img_draw.resize((1800,1200))
background.paste(foreground, (40, 80), foreground)

background.show()
