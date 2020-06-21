import os.path as path

import cv2
import numpy as np
from tqdm import tqdm
import argparse


def load_data_npz(npz_path):
    d = np.load(npz_path)
    return d["image"], d["pose"]


def main(args):
    image = []
    pose = []
    print("Loading data...")
    for db in args.dbs:
        image_temp, pose_temp = load_data_npz(db)
        image.append(image_temp)
        pose.append(pose_temp)
        print("Loaded batch...")
    image = np.concatenate(image, 0)
    pose = np.concatenate(pose, 0)

    x_data = []
    y_data = []
    for i in range(0, pose.shape[0]):
        temp_pose = pose[i, :]
        if np.max(temp_pose) <= 99.0 and np.min(temp_pose) >= -99.0:
            x_data.append(image[i, :, :, :])
            y_data.append(pose[i, :])
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    pbar = tqdm(total=x_data.shape[0])
    names = []
    for i, (image, pose) in enumerate(zip(x_data, y_data)):
        im_name = "/image_{:0>5}.png".format(i)
        if not path.exists(args.path + im_name):  cv2.imwrite(args.path  + im_name, image)

        names.append("{}:{:.6}:{:.6}:{:.6}".format(im_name,
                                                   float(pose[0]),
                                                   float(pose[1]),
                                                   float(pose[2])))
        pbar.update(1)
    pbar.close()
    open(args.output, "w+").write("\n".join(names))

# Authored by https://github.com/shamangary/FSA-Net
def get_args():
    parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
                                                 "and creates database for training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', type=str)
    parser.add_argument('--dbs', nargs='+')
    parser.add_argument("--output", type=str,
                        help="Path to csv")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    main(args)