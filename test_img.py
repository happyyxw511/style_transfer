import src.utils as utils
import os

content_image_path = 'data/content_images/'
resized_content_image_path = 'data/resized_content_images/'


def _get_files(img_dir):
    files = utils.list_files(img_dir)
    return files

content_image_paths = _get_files(content_image_path)

for filename in content_image_paths:
    fullpath = os.path.join(content_image_path, filename)
    print fullpath
    image = utils.get_img(fullpath, img_size=(256, 256, 3))
    output_path = os.path.join(resized_content_image_path, filename)
    print output_path
    utils.save_img(output_path, image)
