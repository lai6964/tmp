import cv2
import numpy as np
from PIL import Image,ImageFont,ImageDraw

img = cv2.imread("tmp.png",cv2.IMREAD_UNCHANGED)

cv2.imshow("img",img)
cv2.waitKey(0)
print(1)
