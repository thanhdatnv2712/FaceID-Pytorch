import cv2
import torch
import torch.nn as nn
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

cos = nn.CosineSimilarity(dim=0, eps=1e-6)

test_trainsforms= transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def main():
    namebank= []
    with open("weights/name_bank.txt") as f:
        namebank= f.read().strip('\n').split('\n')
    facebank = torch.load("weights/face_bank.pth", map_location="cpu")

    vid = cv2.VideoCapture(0)
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
            lmds= []
            for v in lmd:
                tmp= list(v)
                lmds.append(tmp)
            wraper= align_face(image, lmd)
            _wraper= test_trainsforms(wraper).unsqueeze(0).to(device)
            with torch.no_grad():
                embeds= embeddings(_wraper).cpu()
            verify= "Unknown"
            _score= -1
            for i in range(facebank.shape[0]):
                face= facebank[i]
                score= cos(face, embeds[0])
                if score > _score:
                    _score= score
                    verify= namebank[i]
            image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            image = cv2.putText(image, verify, (int(box[0]), int(box[1]+ 12)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("camera", image)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()