import cv2
import numpy as np
from PIL import Image,ImageFont,ImageDraw
from utils import get_min_box

def get_text_img_tmp(
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
    img = cv2.cvtColor(np.asarray(img_draw), cv2.COLOR_RGB2BGR)
    box = get_min_box(img)
    cut = img_draw.crop((box[0][0],box[0][1],box[1][0],box[1][1]))
    cut.save("tmp.png")
    return cut


img_draw = get_text_img_tmp(back_color=(255, 255, 255, 0))
img_draw = Image.open("tmp.png")
# rot.show()
from utils import my_rotate_PIL
rot, coor = my_rotate_PIL(img_draw, angle=40)

# img1 = cv2.cvtColor(np.asarray(rot), cv2.COLOR_RGBA2RGB)
# cv2.line(img1,coor[0],coor[1],(0,255,0),3)
# cv2.line(img1,coor[1],coor[2],(0,255,0),3)
# cv2.line(img1,coor[2],coor[3],(0,255,0),3)
# cv2.line(img1,coor[0],coor[3],(0,255,0),3)
# cv2.imshow("img",img1)
# cv2.waitKey(0)



background = Image.open("1.png")
foreground = rot#.resize((180,120))
background.paste(foreground, (40, 80), foreground)
background.show()
# img = cv2.cvtColor(np.asarray(rot), cv2.COLOR_RGBA2BGRA)
# cv2.imshow("img",img)
# cv2.waitKey(0)
# print(1)
