import imageio
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.transform import warp, AffineTransform
from skimage.exposure import equalize_adapthist
import cv2


def flip_image(image, direction='lr'):
    # direction='lr' -> flip left right
    # direction='ud' -> flip up down

    image_flipped = image.copy()

    if direction == 'ud':
        image_flipped = np.flipud(image_flipped)
    else:
        image_flipped = np.fliplr(image_flipped)

    return image_flipped



def augment_affine(
        image,
        bboxes=None,  # list of tuples
        how_many=1,
        random_seed=0,
        range_scale=(0.8, 1.2),  # percentage
        range_translation=(-100, 100),  # in pixels
        range_rotation=(-45, 45),  # in degrees
        range_sheer=(-45, 45),  # in degrees
        flip_lr=None,
        flip_ud=None,
        enhance=False
):


    if enhance is True:
        image = equalize_adapthist(
            image,
            kernel_size=None,
            clip_limit=0.01,
            nbins=256
        )

    # convert bboxes to x,y coordinates
    if bboxes is not None:
        ls_bboxes_coord = []
        for box in bboxes:
            tmp = [[x,y] for (x,y) in box]
            ls_bboxes_coord.append(tmp)

    # ------------------------------------------------------- get random values

    # set random seed
    np.random.seed(random_seed)

    # degrees to radians
    range_rotation = np.radians(range_rotation)
    range_sheer = np.radians(range_sheer)

    # get random values
    param_scale = np.random.uniform(
        low=range_scale[0],
        high=range_scale[1],
        size=how_many
    )
    param_trans = np.random.uniform(
        low=range_translation[0],
        high=range_translation[1],
        size=(how_many, 2)
    ).astype(int)
    param_rot = np.random.uniform(
        low=range_rotation[0],
        high=range_rotation[1],
        size=how_many
    )
    param_sheer = np.random.uniform(
        low=range_sheer[0],
        high=range_sheer[1],
        size=how_many
    )

    # -------------------------------------------- process all image variations



    # for all images
    out_imgs = []
    out_boxes = []
    for i in range(how_many):
        # configure an affine transform based on the random values
        tform = AffineTransform(
            scale=(param_scale[i], param_scale[i]),
            rotation=param_rot[i],
            shear=param_sheer[i],
            translation=(param_trans[i, 0], param_trans[i, 1])
        )

        image_transformed = warp(  # warp image (pixel range -> float [0,1])
            image,
            tform.inverse,
            mode='edge'
        )

        # convert range back to [0,255]
        image_transformed *= 255
        image_transformed = image_transformed.astype(np.uint8)



        # ------------- transform bboxes to the new coordinates of the warped image

        flag_truncated = False
        if bboxes is not None:
            im_width = image_transformed.shape[1]
            im_height = image_transformed.shape[0]
            ls_bboxes_coord_new=[]
            for ls_bboxes_coord_tmp in ls_bboxes_coord:
                if flag_truncated==True:
                    break
                tmp_box = []
                for j in range(4):
                    vector = np.array([ls_bboxes_coord_tmp[j][0],ls_bboxes_coord_tmp[j][1],1])
                    new_coord = np.matmul(tform.params, vector)
                    x = int(round(new_coord[0]))
                    y = int(round(new_coord[1]))
                    tmp_box.append((x,y))
                    if x<0 or x>im_width or y<0 or y>im_height:
                        flag_truncated = True
                        break
                ls_bboxes_coord_new.append(tmp_box)



        if flag_truncated == True:
            continue


        out_imgs.append(image_transformed)
        out_boxes.append(ls_bboxes_coord_new)

    return out_imgs, out_boxes

if __name__ == '__main__':

    filename = "2.png"
    # ls_bboxes = [[50,60,210,245]]
    ls_bboxes = [[(50, 81), (290, 60), (294, 99), (54, 120)]]


    img = cv2.imread(filename)
    # getting augmented images
    imgs, boxes = augment_affine(
        image=img,
        # bboxes = None,
        bboxes=ls_bboxes,
        how_many=10,
        random_seed=0,
        range_scale=(0.5, 1.5),  # percentage
        range_translation=(-30, 30),  # in pixels
        range_rotation=(-10, 10),  # in degrees
        range_sheer=(-30, 30),  # in degrees
        flip_lr=None,#'random',
        flip_ud=None,
        enhance=True,
    )
    # print(image_augm)

    from utils import my_line_4point
    import cv2
    for img,box in zip(imgs,boxes):
        for b in box:
            b = [(x,y) for x,y in b]
            img = my_line_4point(img,b,color=(255,0,0))
        cv2.imshow("img",img)
        cv2.waitKey(0)
