import tensorflow as tf
import numpy as np
import src.vgg as vgg

STYLE_LAYER_NAMES = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')


def style_and_content_loss(content_images,
                           style_image,
                           vgg_model_path,
                           content_weight,
                           style_weight,
                           batch_size=4):
    # Test the VGG Net
    style_features = {}
    image_shape = (1,) + style_image.shape
    batch_shape = (batch_size, 256, 256, 3)
    with tf.Session() as sess:
        image = tf.placeholder(tf.float32, shape=image_shape, name='image')
        preprocessed_image = vgg.preprocess(image)
        net = vgg.net(vgg_model_path, preprocessed_image)
        # numpy array in the feed_dict are transformed to tensor by tensorflow
        style_pre = np.array([style_image])

        # precompute the features for the style image
        for layer_name in STYLE_LAYER_NAMES:
            feature = net[layer_name].eval(feed_dict={image: style_pre})
            columnlized_feature = np.reshape(feature, [-1, feature.shape[3]])
            gram = np.matmul(columnlized_feature.T, columnlized_feature) / columnlized_feature.size
            style_features[layer_name] = gram
