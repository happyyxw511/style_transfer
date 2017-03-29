import tensorflow as tf
import numpy as np
import src.utils as utils

def optimize(loss,
             content_image_paths,
             batch_size=4,
             learning_rate=1e-3,
             epochs=20,
             shoud_print=False,
             save_path='saver/fns.ckpt'):
    batch_shape = (batch_size,256,256,3)
    with tf.Session() as sess:
        optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)
        sess.run(tf.global_variables_initializer())
        for epoch in xrange(epochs):
            num_examples = len(content_image_paths)
            np.random.shuffle(content_image_paths)
            print content_image_paths[:10]
            iterations = 0
            infered_losses = []
            # the input should have been randomized
            while iterations * batch_size < num_examples:
                curr = iterations*batch_size
                step = curr + batch_size
                X_batch = np.zeros(batch_shape, dtype=np.float32)
                for j, path in enumerate(content_image_paths[curr: step]):
                    X_batch[j] = utils.get_img(path, img_size=(256, 256, 3)).astype(np.float32)
                iterations += 1
                optimizer.run(feed_dict={'X_content_images:0': X_batch})
                saver = tf.train.Saver()
                saver.save(sess, save_path)
                if shoud_print:
                    inferred_loss = sess.run([loss], feed_dict={'X_content_images:0': X_batch})
                    infered_losses.append(inferred_loss)
                    if np.mod(iterations, 10) == 0:
                        print np.mean(inferred_loss)
                        infered_losses = []
