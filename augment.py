from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os

image_path = "C:/Users/joshu/Python Projects/Machine Learning Projects/Web Image Classifier/images/sonic/"
datagen = ImageDataGenerator(rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, fill_mode = "nearest")
def generate_images(x):
    i = 0
    for batch in datagen.flow(x, batch_size = 20, save_to_dir = "augmented", save_prefix = "sonic", save_format = "jpg"):
        i += 1
        if i > 20:
            break

for image in os.listdir(image_path):
    img = load_img(image_path + image, color_mode = "rgb")
    img_arr = img_to_array(img)
    img_arr = img_arr.reshape((1,) + img_arr.shape)
    generate_images(img_arr)