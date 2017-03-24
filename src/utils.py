import scipy.misc, numpy as np, os, sys
import cv2


def save_img(out_path, img):
    cv2.imwrite(out_path, img)


def scale_img(style_path, style_scale):
    scale = float(style_scale)
    o0, o1, o2 = scipy.misc.imread(style_path, mode='RGB').shape
    scale = float(style_scale)
    new_shape = (int(o0 * scale), int(o1 * scale), o2)
    style_target = get_img(style_path, img_size=new_shape)
    return style_target


def get_img(src, img_size=None):
    img = cv2.imread(src) # misc.imresize(, (256, 256, 3))
    if not (len(img.shape) == 3 and img.shape[2] == 3):
       img = np.dstack((img,img,img))
    if img_size:
       img = cv2.resize(img, img_size[:2])
    return img


def exists(p, msg):
    assert os.path.exists(p), msg


def list_files(in_path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        files.extend(filenames)
        break

    return files

