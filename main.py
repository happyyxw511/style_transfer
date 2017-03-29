
import src.utils as utils
import src.loss_calculation as loss_calculation
import src.train as train
import os

style_image_path = 'data/style/wave.jpg'
content_image_path = 'data/content_images/'
vgg_model_path = 'data/imagenet-vgg-verydeep-19.mat'
content_weight = 7.5e0
style_weight = 1e2
tv_weight = 2e2
style_image = utils.get_img(style_image_path)

style_loss = loss_calculation.style_and_content_loss(style_image,
                                                     vgg_model_path,
                                                     content_weight,
                                                     style_weight,
                                                     tv_weight)
def _get_files(img_dir):
    files = utils.list_files(img_dir)
    return [os.path.join(img_dir,x) for x in files]

saver_path = 'saver/'
try:
    os.mkdir(saver_path)
except Exception:
    pass

content_image_paths = _get_files(content_image_path)
train.optimize(style_loss, content_image_paths, shoud_print=True)
