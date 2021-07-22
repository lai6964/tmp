import numpy as np
import cv2, math, random
from PIL import Image, ImageFont, ImageDraw
from shapely.geometry import Polygon
from skimage.transform import warp, AffineTransform
from skimage.exposure import equalize_adapthist

def get_fontcolor(bg_img):
    image = np.asarray(bg_img)
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)

    bg = lab_image[:, :, 0]
    l_mean = np.mean(bg)

    new_l = random.randint(0, 127 - 80) if l_mean > 127 else random.randint(127 + 80, 255)
    new_a = random.randint(0, 255)
    new_b = random.randint(0, 255)

    lab_rgb = np.asarray([[[new_l, new_a, new_b]]], np.uint8)
    rbg = cv2.cvtColor(lab_rgb, cv2.COLOR_Lab2RGB)

    r = rbg[0, 0, 0]
    g = rbg[0, 0, 1]
    b = rbg[0, 0, 2]

    return (r, g, b, 255)

def is_no_intersection(img,bboxes_list,coor):
    w, h = img.size
    tmp_w = max([c[0] for c in coor]) - min([c[0] for c in coor])
    tmp_h = max([c[1] for c in coor]) - min([c[1] for c in coor])
    start_point_x = random.randint(0,w-tmp_w)
    start_point_y = random.randint(0,h-tmp_h)
    bboxes = [(start_point_x+c[0],start_point_y+c[1]) for c in coor]
    flag = True
    for last_bboxes in bboxes_list:
        if intersection(bboxes,last_bboxes)>0:
            flag=False
            break
    return flag, (start_point_x,start_point_y)

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


def get_text_img_ori(
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

def get_text_img(
        font_size=40,####字号
        font_type_path="STFANGSO.TTF",####字体
        text="测试ing中",
        back_color=(255, 255, 255, 0),
        font_color=(255, 0, 0, 255),####字色
        font_wide=0,####字粗
        pad_size=10):
    """ 画两倍框，再裁剪 """
    text = text.replace("\n","")
    text_len = len(text)
    font = ImageFont.truetype(font_type_path, font_size, encoding="utf-8")
    img_draw = Image.new('RGBA', ((text_len * font_size + pad_size) * 2, (font_size + pad_size) * 2), back_color)
    draw = ImageDraw.Draw(img_draw)
    draw.text((int(font_size * 0.1)+pad_size, int(font_size * 0.1)+pad_size), text, fill=font_color, font=font, stroke_width=font_wide)
    img = cv2.cvtColor(np.asarray(img_draw), cv2.COLOR_RGB2BGR)
    box = get_min_box(img)
    cut = img_draw.crop((box[0][0],box[0][1],box[1][0],box[1][1]))
    return cut

def my_rotate_PIL(img_draw, angle=30.0):
    angle = angle%360.0 ####逆时针多少度
    w, h = img_draw.size
    angle_cv2 = -math.radians(angle)#angle
    rotn_center = (w / 2.0, h / 2.0)
    angle_cos = math.cos(angle_cv2)
    angle_sin = math.sin(angle_cv2)

    coor_w3 = (- rotn_center[0] * angle_cos - rotn_center[1] * angle_sin)
    coor_h3 = (- rotn_center[0] * angle_sin + rotn_center[1] * angle_cos)

    coor_w2 = (+ rotn_center[0] * angle_cos - rotn_center[1] * angle_sin)
    coor_h2 = (+ rotn_center[0] * angle_sin + rotn_center[1] * angle_cos)

    coor_w1 = (+ rotn_center[0] * angle_cos + rotn_center[1] * angle_sin)
    coor_h1 = (+ rotn_center[0] * angle_sin - rotn_center[1] * angle_cos)

    coor_w0 = (- rotn_center[0] * angle_cos + rotn_center[1] * angle_sin)
    coor_h0 = (- rotn_center[0] * angle_sin - rotn_center[1] * angle_cos)

    rot = img_draw.rotate(angle, expand=1)
    w, h = rot.size
    w = w / 2
    h = h / 2
    coor = [0, 1, 2, 3]
    coor[0] = (int(w + coor_w0), int(h + coor_h0))
    coor[1] = (int(w + coor_w1), int(h + coor_h1))
    coor[2] = (int(w + coor_w2), int(h + coor_h2))
    coor[3] = (int(w + coor_w3), int(h + coor_h3))

    return rot, coor

def my_rotate_PIL2(img_draw, coor=None, angle=30.0):
    """绕矩形图像img_draw中心旋转"""
    angle = angle%360.0 ####逆时针多少度
    angle_cv2 = -math.radians(angle)#angle
    angle_cos = math.cos(angle_cv2)
    angle_sin = math.sin(angle_cv2)
    w, h = img_draw.size
    if coor is None:
        coor = [[0, 0], [w, 0], [w, h], [0, h]]

    rotn_center = (w / 2.0, h / 2.0)#np.sum(coor, axis=0)/4.0

    coor_w3 = (coor[3][0] - rotn_center[0]) * angle_cos - (coor[3][1] - rotn_center[1]) * angle_sin
    coor_h3 = (coor[3][0] - rotn_center[0]) * angle_sin + (coor[3][1] - rotn_center[1]) * angle_cos

    coor_w2 = (coor[2][0] - rotn_center[0]) * angle_cos - (coor[2][1] - rotn_center[1]) * angle_sin
    coor_h2 = (coor[2][0] - rotn_center[0]) * angle_sin + (coor[2][1] - rotn_center[1]) * angle_cos

    coor_w1 = (coor[1][0] - rotn_center[0]) * angle_cos - (coor[1][1] - rotn_center[1]) * angle_sin
    coor_h1 = (coor[1][0] - rotn_center[0]) * angle_sin + (coor[1][1] - rotn_center[1]) * angle_cos

    coor_w0 = (coor[0][0] - rotn_center[0]) * angle_cos - (coor[0][1] - rotn_center[1]) * angle_sin
    coor_h0 = (coor[0][0] - rotn_center[0]) * angle_sin + (coor[0][1] - rotn_center[1]) * angle_cos

    rot = img_draw.rotate(angle, expand=1)
    w, h = rot.size
    w = w / 2
    h = h / 2
    coor = [0, 1, 2, 3]
    coor[0] = (int(w + coor_w0), int(h + coor_h0))
    coor[1] = (int(w + coor_w1), int(h + coor_h1))
    coor[2] = (int(w + coor_w2), int(h + coor_h2))
    coor[3] = (int(w + coor_w3), int(h + coor_h3))

    return rot, coor

def my_line_4point(img,coor,color=(0,255,0),line_thick=3):
    cv2.line(img,coor[0],coor[1],color,line_thick)
    cv2.line(img,coor[1],coor[2],color,line_thick)
    cv2.line(img,coor[2],coor[3],color,line_thick)
    cv2.line(img,coor[0],coor[3],color,line_thick)
    return img

def my_get_text(txts, num=5):
    txt = ""
    length = len(txts)-1
    for i in range(num):
        index = random.randint(0,length)
        txt = txt + txts[index]
    return txt

def intersection(g, p):
    g=np.asarray(g)
    p=np.asarray(p)
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter/union

def augment_affine(
        image,
        bg_image,
        bboxes=None,  # list of tuples
        random_seed=0,
        range_scale=(0.8, 1.2),  # percentage
        range_translation=(-100, 100),  # in pixels
        range_rotation=(-45, 45),  # in degrees
        range_sheer=(-45, 45)  # in degrees
):

    # convert bboxes to x,y coordinates
    if bboxes is not None:
        ls_bboxes_coord = []
        for box in bboxes:
            tmp = [[x,y] for (x,y) in box]
            ls_bboxes_coord.append(tmp)

    # ------------------------------------------------------- get random values
#     # set random seed
#     np.random.seed(random_seed)
    # degrees to radians
    range_rotation = np.radians(range_rotation)
    range_sheer = np.radians(range_sheer)

    # get random values
    param_scale = np.random.uniform(low=range_scale[0], high=range_scale[1])
    param_trans_1 = np.random.randint(low=range_translation[0], high=range_translation[1])
    param_trans_2 = np.random.randint(low=range_translation[0], high=range_translation[1])
    param_rot = np.random.uniform(low=range_rotation[0], high=range_rotation[1])
    param_sheer = np.random.uniform(low=range_sheer[0], high=range_sheer[1])

    # -------------------------------------------- process all image variations
    # configure an affine transform based on the random values
    tform = AffineTransform(
        scale=(param_scale, param_scale),
        rotation=param_rot,
        shear=param_sheer,
        translation=(param_trans_1, param_trans_2)
    )

    image_transformed = warp(  # warp image (pixel range -> float [0,1])
        image,
        tform.inverse,
        mode='constant'
    )
    # convert range back to [0,255]
    image_transformed *= 255
    image_transformed = image_transformed.astype(np.uint8)


    # ------------- transform bboxes to the new coordinates of the warped image
    flag_truncated = False
    if bboxes is not None:
        ls_bboxes_coord_new = []
        for ls_bboxes_coord_tmp in ls_bboxes_coord:
            if flag_truncated == True:
                break
            tmp_box = []
            for j in range(4):
                vector = np.array([ls_bboxes_coord_tmp[j][0], ls_bboxes_coord_tmp[j][1], 1])
                new_coord = np.matmul(tform.params, vector)
                x = int(round(new_coord[0]))
                y = int(round(new_coord[1]))
                tmp_box.append((x, y))
            ls_bboxes_coord_new.append(tmp_box)

    #### remain bboxes in image without outliers
    h, w, _ = image_transformed.shape
    mask = np.zeros((h, w))
    remain_boxes = []
    for box in ls_bboxes_coord_new:
        box = [(x, y) for x, y in box]
        flag = True
        for x, y in box:
            if x < 0 or x > w or y < 0 or y > h:
                flag = False
                break
        if flag == True:
            cv2.fillConvexPoly(mask, np.array(box), color=255)
            remain_boxes.append(box)

    bg_image_transformed = warp(bg_image, tform.inverse, mode='symmetric')
    bg_image_transformed *= 255
    bg_image_transformed = bg_image_transformed.astype(np.uint8)
    bg_image_transformed[mask == 255] = image_transformed[mask == 255]
    # cv2.imshow("mask",mask)
    # cv2.imshow("out",bg_image_transformed)

    return bg_image_transformed, remain_boxes, image_transformed

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

def my_perspective_point(point,M):
    point = [x for x in point]
    point.append(1)
    point = np.array(point, dtype=np.float32)
    c = M @ point  # c = np.matmul(M,point)  一个意思
    c = (c / c[2])
    c = c[:2]
    c = c.tolist()
    c = [int(t) for t in c]
    return c

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
    dst = cv2.warpPerspective(img, M, (max_x-min_x, max_y-min_y))
    return dst, M

if __name__ == '__main__':
    # img_draw = Image.open("tmp.png")
    # rot, coor = my_rotate_PIL(img_draw)
    #
    # img1 = cv2.cvtColor(np.asarray(rot), cv2.COLOR_RGBA2RGB)
    # cv2.line(img1,coor[0],coor[1],(0,255,0),3)
    # cv2.line(img1,coor[1],coor[2],(0,255,0),3)
    # cv2.line(img1,coor[2],coor[3],(0,255,0),3)
    # cv2.line(img1,coor[0],coor[3],(0,255,0),3)
    # cv2.imshow("img",img1)
    # cv2.waitKey(0)
    lines = [[(227, 186), (427, 197), (425, 236), (225, 225)],
[(231, 170), (432, 156), (435, 195), (234, 209)],
[(265, 224), (466, 217), (467, 256), (266, 263)],
[(176, 166), (413, 124), (420, 162), (183, 204)],
[(53, 216), (293, 241), (289, 281), (49, 256)]]
    print(intersection(lines[0],lines[1]))
