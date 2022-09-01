import cv2
import os
import ipdb
import numpy as np
def np2jpg(imgs, output_dir, K=6):
    # ipdb.set_trace()
    # for i in range(imgs.shape[0]):
    #     output_path = os.path.join(output_dir, "samples{}.jpg".format(i))
    #     cv2.imwrite(output_path, imgs[i, :,:,:])
    m, n = K, K
    for i in range(m):
        img_list = []
        for j in range(n):
            img_list.append(imgs[i * m + j])
            img_raw = np.concatenate(img_list, axis=0)
        if i == 0:
            img_show = img_raw
        else:
            img_show = np.concatenate([img_show, img_raw], axis=1)
    img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
    output_path = os.path.join(output_dir, "samples_{}x{}.jpg".format(K,K))
    cv2.imwrite(output_path, img_show)

def imgs_concat(imgs, output_dir, M, N):
    print("number of images: {}".format(len(imgs)))
    for i in range(M):
        img_list = []
        for j in range(N):
            img_list.append(imgs[i*N + j])
            img_raw = np.concatenate(img_list, axis=0)
        if i == 0:
            img_show = img_raw
        else:
            img_show = np.concatenate([img_show, img_raw], axis=1)
    # img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
    output_path = os.path.join(output_dir, "samples_{}x{}.jpg".format(M,N))
    cv2.imwrite(output_path, img_show)
