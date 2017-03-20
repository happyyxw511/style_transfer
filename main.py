import tensorflow as tf
import numpy as np
import src.vgg as vgg
import src.utils as utils

STYLE_LAYER_NAMES = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

style_image_path = 'data/style/wave.jpg'
vgg_model_path = 'data/imagenet-vgg-verydeep-19.mat'
style_image = utils.get_img(style_image_path)
# Test the VGG Net
image_shape = (1, ) + style_image.shape
with tf.Session() as sess:
    image = tf.placeholder(tf.float32, shape=image_shape, name='image')
    preprocessed_image = vgg.preprocess(image)
    net = vgg.net(vgg_model_path, preprocessed_image)
    # numpy array in the feed_dict are transformed to tensor by tensorflow
    style_pre = np.array([style_image])

    for layer_name in STYLE_LAYER_NAMES:
        print layer_name
        feature = net[layer_name].eval(feed_dict={image: style_pre})
