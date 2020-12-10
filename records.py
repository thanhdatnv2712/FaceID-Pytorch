import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from datetime import datetime
import torchvision.transforms as transforms

from alignfaces import align_face
from models.mtcnn import MTCNN
from models.mobilefacenet import MobileFacenet

device = "cuda" if torch.cuda.is_available() else "cpu"
dt = datetime.now().isoformat(' ', 'seconds')
print ("[INFO {}] Running on device: {}".format(dt, device))
# MTCNN model to face detection
model = MTCNN(device= device)

embeddings = MobileFacenet()
state_dict= torch.load("./weights/068.ckpt", map_location="cpu")
embeddings.load_state_dict(state_dict["net_state_dict"])
embeddings.eval()
embeddings.to(device)

test_trainsforms= transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def main(name):
    vid = cv2.VideoCapture(0)
    path = "./weights/face_bank.pth"
    namepath= "./weights/name_bank.txt"
    namebank= []
    facebank= None
    if os.path.exists(namepath):
        with open(namepath) as f:
            namebank= f.read().strip('\n').split('\n')
    if os.path.exists(path):
        facebank= torch.load(path, map_location="cpu")
    big_lmd= None
    big_bbox= None
    embeds= None
    area= -1
    while True:
        ret, image = vid.read() 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, probs, lmds = model.detect([img], True)
        boxes = boxes.reshape(-1, boxes.shape[-1])
        probs = probs.reshape(probs.shape[-1], 1)
        lmds = lmds.reshape(lmds.shape[1:])
        for box, prob, lmd in zip(boxes, probs, lmds):
            if prob < 0.6:
                continue
            w, h= box[2] - box[0], box[3] - box[1]
            if w * h > area:
                area= w * h
                big_lmd = lmd
                big_bbox = box
            lmds= []
            for v in lmd:
                tmp= list(v)
                lmds.append(tmp)
        wraper= align_face(image, big_lmd)
        _wraper= test_trainsforms(wraper).unsqueeze(0).to(device)
        with torch.no_grad():
            embeds= embeddings(_wraper).cpu()
        cv2.imshow("camera", wraper)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    if embeds is not None and facebank is not None:
        facebank= torch.stack((facebank, embeds[0]), dim=0)
    elif embeds is not None:
        facebank= embeds[0]

    namebank.append(name)
    list2str= "\n".join(map(str, namebank))
    with open(namepath, 'w') as f:
        f.write(list2str)
    print (len(namebank))
    print (facebank.shape)
    vid.release()
    cv2.destroyAllWindows()
    torch.save(facebank, path)

if __name__ == "__main__":
    num= len(sys.argv)
    if num < 2:
        print ("Enter the name Records...")
        exit()
    main(sys.argv[1])