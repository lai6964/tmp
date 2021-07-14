import cv2
import numpy as np
from PIL import Image,ImageFont,ImageDraw

fontsize = 50
font = ImageFont.truetype("STFANGSO.TTF",fontsize,encoding="utf-8")


img = cv2.imread("1.jpg")
w,h,_ = img.shape
img_draw = Image.new('RGB',(w,h),(255,255,255))

draw = ImageDraw.Draw(img_draw)
draw.text((40,100),"测试中ing",fill=(255,0,0),font=font)
img_draw.show()
img_draw = cv2.cvtColor(np.asarray(img_draw),cv2.COLOR_RGB2BGR)
img[img_draw<255] = img_draw

print(1)
