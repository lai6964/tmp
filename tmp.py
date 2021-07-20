from utils import *


def create_new_data(img_path, txts, font_type_path_list):
    img = Image.open(img_path)
    w, h = img.size
    bg = cv2.imread(img_path)
    
    bboxes_list = []
    add_number=0
    for i in range(100):
        angle = random.randint(-10,10)
        txt_length = random.randint(1,10)
        font_size = random.randint(int(min(h,w)/50),int(min(h,w)/15))
        font_wide = random.randint(0,int(font_size/50))
        font_type_path = random.choice(font_type_path_list)
        if random.random()<0.5:
            font_color = get_fontcolor(img)
        else:
            font_color = (0, 0, 0, 255)

        t = my_get_text(txts,num=txt_length)
        img_draw = get_text_img(text=t,font_size=font_size,font_type_path=font_type_path,font_color=font_color,font_wide=font_wide)
        rot_img, coor = my_rotate_PIL(img_draw, angle=angle)

        flag, start_point = is_no_intersection(img,bboxes_list,coor)
        if flag:
            bboxes = [(start_point[0]+c[0],start_point[1]+c[1]) for c in coor]
            bboxes_list.append(bboxes)
            img.paste(rot_img,start_point, rot_img)
            add_number = add_number+1
            if add_number>40:
                break
    # print(add_number)
    # print(bboxes_list)
    # # img.show()
    img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGBA2BGR)

    range_scale = (0.5, 1.5)  # percentage
    range_translation = (-int(min(img.shape[:2])*0.3), int(min(img.shape[:2])*0.3))  # in pixels
    range_rotation = (-10, 10)  # in degrees
    range_sheer = (-30, 30)  # in degrees
    img, boxes = augment_affine(
        image=img,
        bg_image=bg,
        bboxes=bboxes_list,
        random_seed=0,
        range_scale=range_scale,#(0.5, 1.5),  # percentage
        range_translation=range_translation,#(-30, 30),  # in pixels
        range_rotation=range_rotation,#(-10, 10),  # in degrees
        range_sheer=range_sheer#(-30, 30)  # in degrees
    )
    return img, boxes

if __name__ == '__main__':
    img_path = "1.png"
    # txts = open("sentence_re.txt","r",encoding="utf-8").read().replace("\n","")
    txts = open("ppocr_keys_v1.txt", "r", encoding="utf-8").read().replace("\n", "")
    font_type_path_list = ["STFANGSO.TTF"]

    img, boxes = create_new_data(img_path,txts,font_type_path_list)

    # print(boxes)
    # print(len(boxes))
    # for box in boxes:
    #     img = my_line_4point(img,box)
    #     cv2.imshow("img", img)
    #     cv2.waitKey(0)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    
