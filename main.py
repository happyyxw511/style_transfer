import tensorflow as tf
import numpy as np
import src.vgg as vgg
import src.utils as utils
import src.loss_calculation as loss_calculation

style_image_path = 'data/style/wave.jpg'
vgg_model_path = 'data/imagenet-vgg-verydeep-19.mat'
style_image = utils.get_img(style_image_path)

style_loss = loss_calculation.style_image_loss(style_image, vgg_model_path)
