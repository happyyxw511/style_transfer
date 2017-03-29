import tensorflow as tf
import numpy as np
import src.vgg as vgg
import src.style_transfer_conv_net as style_transfer_conv_net

STYLE_LAYER_NAMES = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'


def style_and_content_loss(style_image,
                           vgg_model_path,
                           content_weight,
                           style_weight,
                           tv_weight,
                           batch_size=4):
    # Test the VGG Net
    style_features = {}
    image_shape = (1,) + style_image.shape
    batch_shape = (batch_size, 256, 256, 3)

    image = tf.placeholder(tf.float32, shape=image_shape, name='image')
    preprocessed_image = vgg.preprocess(image)
    net = vgg.net(vgg_model_path, preprocessed_image)
    # numpy array in the feed_dict are transformed to tensor by tensorflow
    style_pre = np.array([style_image])
    with tf.Session() as sess:
        # precompute the features for the style image
        for layer_name in STYLE_LAYER_NAMES:
            feature = net[layer_name].eval(feed_dict={image: style_pre})
            columnlized_feature = np.reshape(feature, [-1, feature.shape[3]])
            gram = np.matmul(columnlized_feature.T, columnlized_feature) / columnlized_feature.size
            style_features[layer_name] = gram


        X_content_images = tf.placeholder(tf.float32, shape=batch_shape, name='X_content_images')
        preprocessed_content_image = vgg.preprocess(X_content_images)
        content_net = vgg.net(vgg_model_path, preprocessed_content_image)
        content_feature = content_net[CONTENT_LAYER]

        # the generated image
        preds = style_transfer_conv_net.net(X_content_images/255.)
        preds_pre = vgg.preprocess(preds)
        preds_net = vgg.net(vgg_model_path, preds_pre)
        content_feature_size = _tensor_size(content_feature)*batch_size
        content_loss = tf.nn.l2_loss(preds_net[CONTENT_LAYER] - content_feature)/content_feature_size

        style_loss = 0
        for layer_name in STYLE_LAYER_NAMES:
            layer = preds_net[layer_name]
            bs, height, width, num_filters = map(lambda i: i.value, layer.get_shape())
            size = height * width * num_filters
            feature = tf.reshape(layer, (bs, -1, num_filters))
            feature_T = tf.transpose(feature, [0, 2, 1])
            style_gram = style_features[layer_name]
            layer_gram = tf.matmul(feature_T, feature)/size
            style_loss += tf.nn.l2_loss(layer_gram - style_gram)/style_gram.size
        style_loss = style_loss / batch_size

        # total variation denoising
        tv_y_size = _tensor_size(preds[:,1:,:,:])
        tv_x_size = _tensor_size(preds[:,:,1:,:])
        y_tv = tf.nn.l2_loss(preds[:,1:,:,:] - preds[:,:batch_shape[1]-1,:,:])
        x_tv = tf.nn.l2_loss(preds[:,:,1:,:] - preds[:,:,:batch_shape[2]-1,:])
        tv_loss = (x_tv/tv_x_size + y_tv/tv_y_size)/batch_size

        return style_loss*style_weight + content_loss*content_weight + tv_weight*tv_loss


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, [d.value for d in tensor.get_shape()[1:]], 1)
