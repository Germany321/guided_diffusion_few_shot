import os
import cv2
import argparse
import ipdb
import numpy as np

def img2npz(root_path, out_path):
    datanames = os.listdir(root_path)
    all_imgs = []
    for dataname in datanames:
        if os.path.splitext(dataname)[-1] == ".jpg":
            img = cv2.imread(os.path.join(root_path, dataname))
            img = cv2.resize(img, (256, 256))
            img = np.expand_dims(img, axis=0)
            all_imgs.append(img)
    arr = np.concatenate(all_imgs)
    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = os.path.join(out_path, f"samples_{shape_str}.npz")
    np.savez(out_path, arr)

if __name__ == "__main__":
    root_path = "/home/linghuxiongkun/workspace/guided-diffusion/datasets/cat500"
    out_path = "/home/linghuxiongkun/workspace/guided-diffusion/datasets/cat500/npzfiles"
    img2npz(root_path, out_path)