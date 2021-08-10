import cv2, os
import numpy as np
import random



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


def rotate_bound_white_bg(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    # -angle位置参数为角度参数负值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数），建议0.75
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    # borderValue 缺失背景填充色彩，此处为白色，可自定义
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))
    # borderValue 缺省，默认是黑色（0, 0 , 0）
    # return cv2.warpAffine(image, M, (nW, nH))



mode="train"
if not os.path.exists("/media/lx/98489B6D489B4940/mydata/formula/"+mode):
    os.mkdir("/media/lx/98489B6D489B4940/mydata/formula/"+mode)
# if 1:
for ttttttt in range(10,20):
    mode = "train/"+str(ttttttt)
    dir = "/media/lx/98489B6D489B4940/mydata/formula/outimg"
    if not os.path.exists("/media/lx/98489B6D489B4940/mydata/formula/"+mode):
        os.mkdir("/media/lx/98489B6D489B4940/mydata/formula/"+mode)
    names = os.listdir(dir)
    # for name in names:
    #     path = os.path.join(dir,name)
    for tmp_index in range(1,3181):
        name = str(tmp_index)+".png"
        path = os.path.join(dir,name)
        print(name)
        # for idx in range(1):
        if 1:
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

            rotate_angle = random.randint(-5,5)
            img = rotate_bound_white_bg(img,rotate_angle)

            h,w,c=img.shape
            h=int(h/3)
            w=int(w/10)
            pad_left = random.randint(0,w)
            pad_right = random.randint(0,w)
            pad_up = random.randint(0,h)
            pad_down = random.randint(0,h)
            cut = np.pad(img,[(pad_up,pad_down),(pad_left,pad_right),(0,0)],constant_values=255)

            if random.random()>=0.9:
                if random.random() >= 0.5:
                    if random.random() >= 0.5:
                        seg1 = 0
                    else:
                        seg1 = pad_up - 1
                else:
                    seg1 = random.randint(0,pad_up)
                seg2 = random.randint(0,pad_up)
                cut[min(seg1,seg2):max(seg1,seg2),:]=0
            if random.random()>=0.9:
                if random.random() >= 0.5:
                    if random.random() >= 0.5:
                        seg1 = 0
                    else:
                        seg1 = pad_left - 1
                else:
                    seg1 = random.randint(0,pad_left)
                seg2 = random.randint(0,pad_left)
                cut[:,min(seg1,seg2):max(seg1,seg2)]=0
            cuth, cutw, cutc = cut.shape
            if random.random()>=0.9:
                if random.random() >= 0.5:
                    if random.random() >= 0.5:
                        seg1 = 0
                    else:
                        seg1 = pad_down - 1
                else:
                    seg1 = random.randint(0,pad_down)
                seg2 = random.randint(0,pad_down)
                cut[cuth-max(seg1,seg2):cuth-min(seg1,seg2),:]=0
            if random.random()>=0.9:
                if random.random() >= 0.5:
                    if random.random() >= 0.5:
                        seg1 = 0
                    else:
                        seg1 = pad_right - 1
                else:
                    seg1 = random.randint(0,pad_right)
                seg2 = random.randint(0,pad_right)
                cut[:,cutw-max(seg1,seg2):cutw-min(seg1,seg2)]=0

            cv2.imwrite("/media/lx/98489B6D489B4940/mydata/formula/"+mode+"/"+name,cut)
            # cv2.imwrite("/media/lx/98489B6D489B4940/mydata/formula/"+mode+"/"+str(idx)+"_"+name,cut)
            #print(mode+"/"+str(idx)+"_"+name)
            #cv2.imshow("c",cut)
            #cv2.waitKey(1)
