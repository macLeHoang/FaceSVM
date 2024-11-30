import torch
from skimage import img_as_ubyte
import numpy as np
from numpy.linalg import norm
import cv2

import face_alignment
from utils.matlab_cp2tform import get_similarity_transform_for_cv2

from deepface.DeepFace import extract_faces
import glob
import os
from tqdm import tqdm

from models import resnet_face18

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def alignment(src_img, src_pts, crop_size=(128, 128)):
        ref_pts = ref_pts_96_112 = [ 
            [30.2946, 51.6963],
            [65.5318, 51.5014], 
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]
        ]

        # ref_pts = []
        # for pts in ref_pts_96_112:
        #     x, y = pts
        #     ref_pts.append([
        #         x / 96  * crop_size[0], 
        #         y / 112 * crop_size[1]
        #     ])

        src_pts = np.array(src_pts).reshape(5, 2)
        s = np.array(src_pts).astype(np.float32)
        r = np.array(ref_pts).astype(np.float32)

        tfm = get_similarity_transform_for_cv2(s, r)
        face_img = cv2.warpAffine(src_img, tfm, crop_size)
        return face_img[..., ::-1]


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


def get_face(im):
    result = extract_faces(im, detector_backend="retinaface")
    facial_area = result[0]["facial_area"]
    x = facial_area['x']
    y = facial_area['y']
    w = facial_area['w']
    h = facial_area['h']
    face = im[y:y+h, x:x+w]

    return face

def preprocess(image):
    image = cv2.resize(image, (128, 128), cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[..., None]
    image = image.transpose((2, 0, 1))
    image = image[None]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5

    return image

if __name__ == "__main__":
    save_name = "database\\name4" # name to save in database
    vid_name = r"test_vid\6001519687621.mp4" # video path
    name = (vid_name.split(os.sep)[-1]).split('.')[0]

    if not os.path.isdir(save_name):
        os.mkdir(save_name)
        os.mkdir(os.path.join(save_name, "data"))
        os.mkdir(os.path.join(save_name, "norm"))
        os.mkdir(os.path.join(save_name, "align"))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # face alignment
    # 
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, 
        flip_input=False,
        device=device
    )

    cap = cv2.VideoCapture(vid_name)
    
    i = 0
    c = 0
    while True:
        suc, frame = cap.read()
        if not suc:
            break

        if i % 20 == 0:
            cv2.imwrite(os.path.join(save_name, "data", f"{name}_{c:06d}.jpg"), frame)
            c+=1
        i+=1

    folder = "data"
    list_images = glob.glob(os.path.join(save_name, "data", "*.jpg"))

    for i, im in tqdm(enumerate(list_images)):
        im1 = cv2.imread(im)
        face1 = get_face(im1)
        cv2.imwrite(os.path.join(save_name, f"norm\\{i:06d}.jpg"), np.uint8(face1))
        face1 = alligning_face(face1[..., ::-1], fa)
        cv2.imwrite(os.path.join(save_name, f"align\\{i:06d}.jpg"), np.uint8(face1))
    

    # get feature vector
    path_im = os.path.join(save_name, "data")
    # load model
    model = resnet_face18(False)
    checkpoint = torch.load("resnet18_110.pth", map_location ="cpu",
                            weights_only=True)
    state_dict = checkpoint#["model"]
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k,
                        v in state_dict.items()}
            
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    data = sorted(glob.glob(os.path.join(os.path.join(path_im, "*.jpg"))))
    image = [cv2.imread(d) for d in data]
    image = [preprocess(i) for i in image]
    ims = np.concatenate(image, axis=0)
    ims = torch.from_numpy(ims).to(device)

    with torch.no_grad():
        feat = model(ims)
    feat = feat.cpu().numpy()

    save_feat = os.path.split(path_im)[0]
    np.save(os.path.join(save_feat, "feat.npy"), feat)




    

    

