# For converting .jpg to numpy array
import numpy as np
from PIL import Image
import os
import polars as pl

def image_size(image_path):
    """ Return image size in pixel"""
    with Image.open(image_path) as image:
        return image.size

def load_and_process_image(image_path, target_size):
    """ Reading, converting to grayscale, normalize and flatten to a feature vector
    Args: 
        image_path (str): Path to image.
        target_size (tuple): Expected (width, height) in pixel

    returns:
        np.ndarray: Flattened and normalized feature vector of the image

    Raises:
        ValueError: If the size of the image doesnt match target_size.
    """
    with Image.open(image_path) as image:
        image_gray = image.convert('L')
        image_array = np.array(image_gray)
    if image_array.shape != (target_size[1], target_size[0]):
        raise ValueError(f"Wrong size of {image_path}: {image_array.shape}, expecting {target_size[::-1]}")
    image_normalized = image_array.astype(np.float32) / 255.0
    feature_vector = image_normalized.flatten()
    return feature_vector

def preprocessing_image_batch(image_folder, filenames, target_size):
    """ 
    Converting a list of images to a feature matrix, using load_and_process_image.
    Iterating the image folder in batches of 1000, to convert each image to to a flattened and normalized feature vector

    Args:
        image_folder (str): Path to images
        filenames (list): Names of the files to be converted
        target_size (tuple): width and height of image

    Returns:
        np.ndarray: Feature matrix where each row corresponds to an image 
    
    """
    n_images = len(filenames)
    n_features = target_size[0] * target_size[1]

    X = np.zeros((n_images, n_features), dtype=np.float32)

    for i, filename in enumerate(filenames):
        image_path = os.path.join(image_folder, filename)
        features = load_and_process_image(image_path, target_size)
        X[i, :] = features

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{n_images} images")
    return X

def main():
    df =  pl.read_parquet('../data/processed/labels/df_full_filtered_with_title.parquet')
    filenames = df['file_name'].to_list()[:1000]
    image_folder = '../data/Pub_Lay_Net/documents'
    first_image_path = os.path.join(image_folder, filenames[0])
    target_size = image_size(first_image_path)
    print(f"Target size: {target_size}")
    X_raw = preprocessing_image_batch(image_folder, filenames, target_size)
    np.save('../data/processed/pln_X_features_raw_original_size.npy', X_raw)
    print("Feature matrix saved")

if __name__ == '__main__':
    main()