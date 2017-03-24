import os
import numpy as np
import src.utils as utils
import tensorflow as tf
import src.style_transfer_conv_net as style_transfer_conv_net

content_image_path = 'data/content_images/'
generated_image_path = 'data/generated_images/'
try:
    os.mkdir(content_image_path)
    os.mkdir(generated_image_path)
except Exception:
    pass

model_checkpoint_path = 'saver/fns.ckpt'
batch_size = 4

def _get_files(img_dir):
    files = utils.list_files(img_dir)
    return files

with tf.Session() as sess:
    batch_shape = (batch_size, 256, 256, 3)
    X_content_images = tf.placeholder(tf.float32, shape=batch_shape, name='X_content_images')
    preds = style_transfer_conv_net.net(X_content_images/255.)
    saver = tf.train.Saver()
    saver.restore(sess, model_checkpoint_path)
    content_images = _get_files(content_image_path)
    num_images = 50
    num_iter = int(num_images/batch_size)
    for i in xrange(num_iter):
        selected_files = content_images[i*batch_size: (i+1)*batch_size]
        X = np.zeros(batch_shape, dtype=np.float32)
        output_image_paths = []
        index = 0
        for index, selected_file in enumerate(selected_files):
            full_image_path = os.path.join(content_image_path, selected_file)
            output_image_paths.append(os.path.join(generated_image_path, selected_file))
            X[index] = utils.get_img(full_image_path, img_size=(256, 256, 3))

        generated_images = sess.run(preds, feed_dict={'X_content_images:0': X})
        for index, output_image_path in enumerate(output_image_paths):
            utils.save_img(output_image_path, generated_images[index])


