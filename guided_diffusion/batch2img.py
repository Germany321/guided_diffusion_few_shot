import cv2
import numpy as np
def batch2img(batch, output_dir, i):
    img = np.array(batch[0])
    img = np.transpose(img, [1,2,0])
    img = (img+1)*127.5
    cv2.imwrite(output_dir+'/test_{}.jpg'.format(i), img)