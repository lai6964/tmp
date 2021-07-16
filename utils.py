import numpy as np
import cv2, math
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


def my_rotate_PIL(img_draw, angle=-30.0):
    angle = angle%360.0 ####逆时针多少度
    w, h = img_draw.size
    angle_cv2 = math.radians(angle)#angle
    rotn_center = (w / 2.0, h / 2.0)
    angle_cos = math.cos(angle_cv2)
    angle_sin = math.sin(angle_cv2)

    coor_w1 = abs(+ rotn_center[0] * angle_cos - rotn_center[1] * angle_sin)
    coor_w2 = abs(+ rotn_center[0] * angle_cos + rotn_center[1] * angle_sin)
    coor_h1 = abs(+ rotn_center[0] * angle_sin + rotn_center[1] * angle_cos)
    coor_h2 = abs(+ rotn_center[0] * angle_sin - rotn_center[1] * angle_cos)

    rot = img_draw.rotate(angle, expand=1)
    w, h = rot.size
    w = w / 2
    h = h / 2
    coor = [0, 1, 2, 3]
    coor[0] = (int(w - coor_w1), int(h + coor_h1))
    coor[1] = (int(w + coor_w1), int(h + coor_h2))
    coor[2] = (int(w + coor_w2), int(h - coor_h1))
    coor[3] = (int(w - coor_w2), int(h - coor_h2))
    print(coor)
    return rot, coor


if __name__ == '__main__':
    img_draw = Image.open("tmp.png")
    rot, coor = my_rotate_PIL(img_draw)

    img1 = cv2.cvtColor(np.asarray(rot), cv2.COLOR_RGBA2RGB)
    cv2.line(img1,coor[0],coor[1],(0,255,0),3)
    cv2.line(img1,coor[1],coor[2],(0,255,0),3)
    cv2.line(img1,coor[2],coor[3],(0,255,0),3)
    cv2.line(img1,coor[0],coor[3],(0,255,0),3)
    cv2.imshow("img",img1)
    cv2.waitKey(0)
