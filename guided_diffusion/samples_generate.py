import cv2
import os
import argparse
import ipdb
import numpy as np
from image_writer import np2jpg, imgs_concat

def getListFiles(path):   # 获取路径下所有npz文件的目录列表
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default="", help="root path of files")
    parser.add_argument('--method', type=str, choices=["samples_gen","imgs_concat"], 
                        help="samples_gen: generate samples from npz; imgs_concat: concat imgs from jpg etc.")
    parser.add_argument("--max_cnt", type=int, help="the number of imgs to be concatenated")
    parser.add_argument("--M", type=int, default=3, help="M x N matrix to concatenate the imgs")
    parser.add_argument("--N", type=int, default=3, help="")
    args = parser.parse_args()
    if args.method == "samples_gen":
        ret = getListFiles(args.root_path)
        listSpicalpath = set()
        for i in ret:
            if os.path.splitext(i)[1] == ".npz":
                ss = os.path.splitext(i)[0].split('/')
                save_path = "/".join(ss[:-1])
                print(save_path)
                imgs = np.load(i)['arr_0']
                np2jpg(imgs, save_path)
    else:
        ret = getListFiles(args.root_path)
        listSpicalpath = set()
        imgs = []
        for i in ret:
            if os.path.splitext(i)[1] == ".jpg" or ".png":
                ss = os.path.splitext(i)[0].split('/')
                save_path = "/".join(ss[:-1])
                img = cv2.imread(i)
                imgs.append(img)
                # img = cv2.imread()
        print(save_path)
        imgs_concat(imgs, save_path, args.M, args.N)