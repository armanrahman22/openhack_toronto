import numpy as np
from skimage import data, io, exposure, img_as_float
from skimage.transform import resize, rescale
import matplotlib.pyplot as plt
import os
import glob


def process_files(path):
    npy_files = glob.glob(path + '\\**\\*.npy', recursive=True)
    all_files = glob.glob(path + '\\**\\*.*', recursive=True)
    all_image_files = set(all_files) - set(npy_files)
    print(len(all_image_files))
    x = []
    y = []
    mod = len(all_image_files)//10
    for index, image_file in enumerate(all_image_files):
        if index % mod == 0:
            print('processing file {0} of {1}'.format(index, len(all_image_files)))
        vector, label = preprocess_file(image_file)
        x.append(vector)
        y.append(label)
    np.save(os.path.join(path,"x_image_arrays"), x)
    np.save(os.path.join(path, "y_image_labels"), y)


def get_output_file(path, filename):
    os.join(path,filename)
    return '\\'.join(file_path_parts) + image_name_base + '.npy'


def get_label(image_file):
    file_path_parts = image_file.split('\\')
    file_path_parts.pop()
    return file_path_parts.pop()


def preprocess_file(image_file):
    original_image = io.imread(image_file)
    padded_image = pad_and_resize(original_image, 128)
    equalized_image = equalize_image(padded_image)
    label = get_label(image_file)
    return equalized_image, label


def pad_and_resize(image, desired_size):
    rows, columns, channels = image.shape
    delta_y = abs(rows - max(columns, rows))
    delta_x = abs(columns - max(columns, rows))
    top, bottom = delta_y//2, delta_y - (delta_y//2)
    left, right = delta_x//2, delta_x - (delta_x//2)
    image = np.pad(image, ((top,bottom),(left,right),(0,0)), 'constant', constant_values=(255))
    if image.shape[0] < desired_size:
        delta_min = desired_size - max(rows, columns)
        image = np.pad(image, ((delta_min//2, -(-delta_min//2)), (delta_min//2, -(-delta_min//2)), (0,0)), 'constant', constant_values=(255))
    elif image.shape[0] > desired_size:
        image = resize(image, (desired_size, desired_size))
    return image
        


def equalize_image(image):
    image = exposure.rescale_intensity(image, in_range=(0, 255))
    return exposure.equalize_adapthist(image)


def plot_img_and_hist(image, axes):
    image = img_as_float(image)
    ax_img, ax_hist = axes

    # Display image
    ax_img.imshow(image)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=255, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    return ax_img, ax_hist


if __name__ == "__main__":
    process_files(r'C:\Users\Arman\Downloads\openhack_toronto\gear_images')