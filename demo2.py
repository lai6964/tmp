import glob
import cv2, os
import numpy as np
import random

mode="train11"

if not os.path.exists(mode):
    os.mkdir(mode)


def my_perspective_img(img,bboxes_translate):
    h, w, _ = img.shape
    bboxes_old = [[0, 0], [w, 0], [w, h], [0, h]]
    bboxes_new = [[old[0]+trans[0],old[1]+trans[1]] for old,trans in zip(bboxes_old,bboxes_translate)]
    min_x = min([x for x,y in bboxes_new])
    max_x = max([x for x,y in bboxes_new])
    min_y = min([y for x,y in bboxes_new])
    max_y = max([y for x,y in bboxes_new])
    bboxes_new = [[x-min_x,y-min_y] for x,y in bboxes_new]
    pts1 = np.float32(bboxes_old)
    pts2 = np.float32(bboxes_new)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (max_x-min_x, max_y-min_y), borderValue=(255,255,255))
    return dst, M


dir = "/media/lx/98489B6D489B4940/mydata/formulas/outimg"
names = os.listdir(dir)
for name in names:
    path = os.path.join(dir,name)
    print(name)
    for idx in range(100):
    # while(1):
        img = cv2.imread(path)
        h,w,c=img.shape
        new_h = int(h*random.uniform(0.5,1.5))
        new_w = int(w*random.uniform(0.5,1.5))
        img = cv2.resize(img,(new_w,new_h))

        #### 透视变换
        rot_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGBA2BGR)
        bboxes_translate = [[random.randint(-int(rot_img.shape[1] / 30), int(rot_img.shape[1] / 30)),
                             random.randint(-int(rot_img.shape[0] / 10), int(rot_img.shape[0] / 10))] for i in range(4)]
        img, perspect_M = my_perspective_img(rot_img, bboxes_translate)


        h,w,c=img.shape
        h=int(h/10)
        w=int(w/10)
        pad_left = random.randint(0,w)
        pad_right = random.randint(0,w)
        pad_up = random.randint(0,h)
        pad_down = random.randint(0,h)
        cut = np.pad(img,[(pad_up,pad_down),(pad_left,pad_right),(0,0)],constant_values=255)

        if random.random()>=0.9:
            seg1 = random.randint(0,pad_up)
            seg2 = random.randint(0,pad_up)
            cut[min(seg1,seg2):max(seg1,seg2),:]=0
        if random.random()>=0.9:
            seg1 = random.randint(0,pad_left)
            seg2 = random.randint(0,pad_left)
            cut[:,min(seg1,seg2):max(seg1,seg2)]=0
        cuth, cutw, cutc = cut.shape
        if random.random()>=0.9:
            seg1 = random.randint(0,pad_down)
            seg2 = random.randint(0,pad_down)
            cut[cuth-max(seg1,seg2):cuth-min(seg1,seg2),:]=0
        if random.random()>=0.9:
            seg1 = random.randint(0,pad_right)
            seg2 = random.randint(0,pad_right)
            cut[:,cutw-max(seg1,seg2):cutw-min(seg1,seg2)]=0

        cv2.imwrite(mode+"/"+str(idx)+"_"+name,cut)
        #print(mode+"/"+str(idx)+"_"+name)
        #cv2.imshow("c",cut)
        #cv2.waitKey(1)
