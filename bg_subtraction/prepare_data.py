import os
import cv2
import numpy as np
from tqdm import tqdm


def load_inputs(img_list):
    n_imgs = len(img_list)
    data = np.zeros((n_imgs, 1600))

    for (i, img_path) in tqdm(enumerate(img_list), total=n_imgs, desc="Loading inputs"):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (40, 40))
        data[i, :] = img.flatten().reshape(1, -1)

    bg = np.median(data, axis=0, keepdims=True)
    bg = np.broadcast_to(bg, (n_imgs, bg.shape[1]))

    return np.hstack((data, bg)) / 255.


def load_labels(img_list):
    n_imgs = len(img_list)
    data = np.zeros((n_imgs, 1600, 2))

    for (i, img_path) in tqdm(enumerate(img_list, total=n_imgs, desc="Loading labels")):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (40, 40)).flatten()
        fg_indices = np.argwhere(img == 255).flatten()
        bg_indices = np.argwhere(img == 0).flatten()
        label = np.zeros((1600, 2))
        label[bg_indices, 0] = 1
        label[fg_indices, 1] = 1
        data[i, :, :] = label
    return data


if __name__ == "__main__":
    data_dir = "../data/dataset2014/dataset"
    save_path = "../data"
    inputs_fname = "input_imgs"
    labels_fname = "labels"

    input_imgs = []
    label_imgs = []
    for category in sorted(os.walk(data_dir).next()[1]):
        for video in sorted(os.walk(os.path.join(data_dir, category)).next()[1]):
            video_dir = os.path.join(data_dir, category, video)
            input_dir = os.path.join(video_dir, "input")
            label_dir = os.path.join(video_dir, "groundtruth")
            if len(os.listdir(input_dir)) != len(os.listdir(label_dir)):
                print(video_dir)
            for input_img in sorted(os.listdir(input_dir)):
                input_imgs.append(os.path.join(input_dir, input_img))
            for label_img in sorted(os.listdir(label_dir)):
                label_imgs.append(os.path.join(label_dir, label_img))
    inputs = load_inputs(input_imgs)
    labels = load_labels(label_imgs)
    np.save(os.path.join(save_path, inputs_fname), inputs)
    np.save(os.path.join(save_path, labels_fname), labels)
