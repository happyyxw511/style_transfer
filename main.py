import tensorflow as tf
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

def _get_files(img_dir):
    files = utils.list_files(img_dir)
    return [os.path.join(img_dir,x) for x in files]

saver_path = 'saver/'
try:
    os.mkdir(saver_path)
except Exception:
    pass

run_config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
run_config.gpu_options.allow_growth=True
with tf.Session(config=run_config) as sess:
    style_loss = loss_calculation.style_and_content_loss(
        sess,
        style_image,
        vgg_model_path,
        content_weight,
        style_weight,
        tv_weight)
    content_image_paths = _get_files(content_image_path)
    train.optimize(sess, style_loss, content_image_paths, shoud_print=True)
