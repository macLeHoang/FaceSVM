import torch
import numpy as np
from numpy.linalg import norm
import cv2
from models import resnet_face18
import os
import glob



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
    # query image | define path to query image here
    face = cv2.imread(r"test_vid\name1\5988457492112_000004.jpg") 
    face1 = face.copy()

    save_demo = "demo"
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    prototxtPath = r"face_detector/deploy.prototxt"
    weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    
    
    (h, w) = face1.shape[:2]
    blob = cv2.dnn.blobFromImage(face1, 1.0, (224, 224),
        (104.0, 177.0, 123.0))
    faceNet.setInput(blob) # pass the blob through the network and obtain the face detections
    detections = faceNet.forward()
    locs = []
    # loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            locs.append((startX, startY, endX, endY))
    
    if not len(locs):
        cv2.putText(face, f"No Face in image", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
            1, (0,255,0), 2, cv2.LINE_AA)
    else:
        locs_max = max(locs, key=lambda x: (x[2]-x[0]) * ([x[3]-x[1]]))
        (startX, startY, endX, endY) = locs_max
        face1 = face1[startY:endY, startX:endX]

        cv2.imwrite(os.path.join(save_demo, "query.jpg"), face1)
        pre1 = preprocess(face1)
        ims = torch.from_numpy(pre1).to(device)
        with torch.no_grad():
            feat = model(ims)
        feat = feat.cpu().numpy() # query im feature

        database = "database"
        names = os.listdir(database)

        most_sim = None
        most_sim_conf = -1
        idx = None

        for name in names:
            dfeat = np.load(os.path.join(database, name, "feat.npy"))
            cosine = (feat @ dfeat.T) / (norm(feat, axis=-1) * norm(dfeat, axis=-1))

            indexes = np.argsort(cosine)[0, -5:] # get maxmimum most k similar images
            cosine = cosine[0, indexes]
            print(cosine, cosine.mean(), indexes)

            conf = (cosine.mean() + 1) / 2
            # conf = np.arccos(cosine.mean()) / np.pi

            if conf > 0.8 and conf > most_sim_conf:
                most_sim = name
                most_sim_conf = conf
                idx = indexes
        
        print(most_sim, most_sim_conf)
        if most_sim is not None:
            cv2.putText(face, f"{most_sim} - {most_sim_conf}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0,255,0), 2, cv2.LINE_AA)
        else:
            cv2.putText(face, f"Stranger", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0,255,0), 2, cv2.LINE_AA)
    
    cv2.imshow("face", face)
    cv2.waitKey(0)

    idx = idx.astype("int").tolist()
    ims = sorted(glob.glob(os.path.join(database, most_sim, "norm", "*.jpg")))

    empty = np.zeros((128, 128*len(idx), 3))
    for ix, index in enumerate(idx):
        i = cv2.imread(ims[index])
        i = cv2.resize(i, (128, 128))

        empty[:, ix*128:(ix+1)*128] = i

    cv2.imwrite("im.jpg", np.uint8(empty))        
        
    







    

    

