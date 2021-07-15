import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw

def get_min_box(img):

    if len(img.shape)==3:
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img = 1 - img / 255.0

    cols = np.sum(img, axis=1)
    col = np.where(cols > 0)
    col = col[0].tolist()

    rows = np.sum(img, axis=0)
    row = np.where(rows > 0)
    row = row[0].tolist()

    x_min = max(0, row[0]-1)
    y_min = max(0, col[0]-1)
    x_max = min(img.shape[1], row[-1]+1)
    y_max = min(img.shape[0], col[-1]+1)

    box = [(x_min,y_min),(x_max,y_max)]
    return box


def get_text_img(
        font_size=40,
        font_type_path="STFANGSO.TTF",
        text="测试ing中",
        back_color=(255, 255, 255,0),
        font_color=(255, 0, 0),
        pad_size=0):
    """ 画两倍框，再裁剪 """
    text = text.replace("\n","")
    text_len = len(text)
    font = ImageFont.truetype(font_type_path, font_size, encoding="utf-8")
    img_draw = Image.new('RGBA', (text_len * font_size * 2, font_size * 2), back_color)
    draw = ImageDraw.Draw(img_draw)
    draw.text((int(font_size * 0.1), int(font_size * 0.1)), text, fill=font_color, font=font)
    # img_draw.show()
    img_draw.save("tmp.png")
    img_draw = cv2.cvtColor(np.asarray(img_draw), cv2.COLOR_RGB2BGR)
    box = get_min_box(img_draw)
    cut_img = img_draw[box[0][1]:box[1][1], box[0][0]:box[1][0]]
    if pad_size>0:
        cut_img = np.pad(cut_img, [(pad_size, pad_size), (pad_size, pad_size), (0, 0)], mode='constant', constant_values=255)


    return img_draw, box, cut_img


def get_text_img_bg(
        img,
        font_size=40,
        font_type_path="STFANGSO.TTF",
        text="测试ing中",
        back_color=(255, 255, 255),
        font_color=(255, 0, 0),
        pad_size=0):

    w,h,c = img.shape
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    font_size = int(h/2)


    return img




