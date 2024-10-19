import torch
from skimage import img_as_ubyte
import numpy as np
import cv2

import face_alignment
from utils.matlab_cp2tform import get_similarity_transform_for_cv2


def alignment(src_img, src_pts, crop_size=(128, 128)):
        ref_pts_96_112 = [ 
            [30.2946, 51.6963],
            [65.5318, 51.5014], 
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]
        ]

        ref_pts = []
        for pts in ref_pts_96_112:
            x, y = pts
            ref_pts.append([
                x / 96  * crop_size[0], 
                y / 112 * crop_size[1]
            ])

        src_pts = np.array(src_pts).reshape(5, 2)
        s = np.array(src_pts).astype(np.float32)
        r = np.array(ref_pts).astype(np.float32)

        tfm = get_similarity_transform_for_cv2(s, r)
        face_img = cv2.warpAffine(src_img, tfm, crop_size)
        return face_img


def alligning_face(input, fa):
    """
    Parameters:
    -----------
        input, ndarray,
            if ndarray, input is in RGB format
    """
    le_eye_pos = [36, 37, 38, 39, 40, 41]
    r_eye_pos  = [42, 43, 44, 45, 47, 46]

    preds = fa.get_landmarks_from_image(input)
    lmks = preds[0]
    le_eye_x, le_eye_y = 0.0, 0.0
    r_eye_x, r_eye_y = 0.0, 0.0
    for l_p, r_p in zip(le_eye_pos, r_eye_pos):
        le_eye_x += lmks[l_p][0]
        le_eye_y += lmks[l_p][1]
        r_eye_x  += lmks[r_p][0]
        r_eye_y  += lmks[r_p][1]
    
    le_eye_x = int(le_eye_x / len(le_eye_pos))
    le_eye_y = int(le_eye_y / len(le_eye_pos))
    r_eye_x  = int(r_eye_x  / len(r_eye_pos))
    r_eye_y  = int(r_eye_y  / len(r_eye_pos))
    nose     = (int(lmks[30][0]), int(lmks[30][1]))
    left_mo  = (int(lmks[60][0]), int(lmks[60][1]))
    ri_mo    = (int(lmks[64][0]), int(lmks[64][1]))
    final_lmks = [
        (le_eye_x, le_eye_y), 
        (r_eye_x, r_eye_y), 
        nose, 
        left_mo,
        ri_mo
    ]
    
    landmark = []
    for lmk in final_lmks:
        landmark.append(lmk[0])
        landmark.append(lmk[1])

    img = img_as_ubyte(input) # RGB format
    cropped_align = alignment(img, landmark)
    return cropped_align


def preprocess(im):
    pass

if __name__ == "__main__":

    # face alignment
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, 
        flip_input=False,
        device=device
    )

    im = None
    align_im = alligning_face(im, fa)

