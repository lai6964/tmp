import random
import cv2
import numpy as np
from matplotlib import pyplot as plt

def my_perspective_point(point,M):
    point.append(1)
    point = np.array(point, dtype=np.float32)
    c = M @ point  # c = np.matmul(M,point)  一个意思
    c = (c / c[2])
    c = c[:2]
    c = c.tolist()
    c = [int(t) for t in c]
    return c
def my_perspective(img,bboxes_old,bboxes_new):
    min_x = min([x for x,y in bboxes_new])
    max_x = max([x for x,y in bboxes_new])
    min_y = min([y for x,y in bboxes_new])
    max_y = max([y for x,y in bboxes_new])
    bboxes_new = [[x-min_x,y-min_y] for x,y in bboxes_new]
    pts1 = np.float32(bboxes_old)
    pts2 = np.float32(bboxes_new)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (max_x-min_x, max_y-min_y))
    points = [my_perspective_point(box,M) for box in bboxes_old]
    return dst, points




if __name__ == '__main__':
    img = cv2.imread('tmp.png', cv2.IMREAD_UNCHANGED)
    h, w, ch = img.shape
    bboxes_old = [[0, 0], [w, 0], [w, h], [0, h]]
    bboxes_new = [[x+random.randint(0,int(w/2)),y+random.randint(0,int(h/2))] for x,y in bboxes_old]
    dst, points = my_perspective(img,bboxes_old,bboxes_new)
    plt.imshow(dst), plt.title('Output')
    plt.show()
    cv2.imshow("d",dst)
    cv2.waitKey(0)
    cv2.imwrite("tmp_per.png",dst)
