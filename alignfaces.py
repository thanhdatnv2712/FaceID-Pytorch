import os
import cv2
import numpy as np
from skimage import transform as trans
from sklearn import preprocessing

src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041] ], dtype=np.float32)
src[:,0] += 8.0

def align_face(img_raw, landmark5):
    dst = np.reshape(landmark5, (5, 2))
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2, :]
    warped = cv2.warpAffine(img_raw, M, (96, 112), borderValue = 0.0)
    return warped

# embedding = torch.cat(embs).mean(0,keepdim=True)