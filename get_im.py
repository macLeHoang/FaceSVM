import cv2
import glob
import numpy as np
import random
import os

# w = 4
# h = 4
# sample = np.zeros((128*h, 128*w, 3))

# ims = glob.glob(r"database\*\norm\*.jpg", recursive=True)
# ims = sorted(ims, key=lambda x: int(x.split(os.sep)[-3][4:]))[20:]

# random.shuffle(ims)
# ims = sorted(ims[:w*h], key=lambda x: int(x.split(os.sep)[-3][4:]))

# ims = ims[:h*w]

# for idx, im in enumerate(ims):
#     i = cv2.imread(im)
#     i = cv2.resize(i, (128, 128))

#     y = int(idx // w)
#     x = int(idx % w)
#     sample[y*128: (y+1)*128, x*128: (x+1)*128, :] = i

# cv2.imwrite("sample.jpg", np.uint8(sample))


if __name__ == "__main__":
    path = r"lfw-align-128\lfw-align-128"
    max_pairs = 8
    lbs = np.load("pair.npy")
    with open("lfw_test_pair.txt", 'r') as fd:
            pairs = fd.readlines()

    good_pairs = []
    bad_pairs = []
    for i, pair in enumerate(pairs):
        if lbs[i]:
            good_pairs.append(pair)
        else:
            bad_pairs.append(pair)

    random.shuffle(good_pairs)
    random.shuffle(bad_pairs)

    o = 10
    oo = 1
    p = np.zeros((128*max_pairs, 128*8 + o + oo*2, 3))

    for ii in range(max_pairs*2):
        good_pair = good_pairs[ii]
        good_splits = good_pair.split()
        im1_good = cv2.resize(cv2.imread(os.path.join(path, good_splits[0])), (128, 128))
        im2_good = cv2.resize(cv2.imread(os.path.join(path, good_splits[1])), (128, 128))

        bad_pair = bad_pairs[ii]
        bad_splits = bad_pair.split()
        im1_bad = cv2.resize(cv2.imread(os.path.join(path, bad_splits[0])), (128, 128))
        im2_bad = cv2.resize(cv2.imread(os.path.join(path, bad_splits[1])), (128, 128))

        if ii % 2 == 0:
            offset = 0
        else:
            offset = 256

        i = ii // 2
        x = ii % 2


        p[i*128 : (i+1)*128, 
          oo + 0 + offset : oo + 128 + offset, :] = im1_good
        p[i*128 : (i+1)*128, 
          128 + offset : 256 + offset, :] = im2_good

        p[i*128 : (i+1)*128,
          oo + o + 512 + offset : oo + o + 512 + 128 + offset, :] = im1_bad
        p[i*128 : (i+1)*128, 
          o + 512 + 128 + offset : o + 512 + 256 + offset, :] = im2_bad
    
    cv2.imwrite("pairs.jpg", np.uint8(p))




    
          