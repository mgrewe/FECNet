### imports
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
from models.FECNet import FECNet
from models.mtcnn import MTCNN
#from models.utils.detect_face import extract_face
from torchvision.transforms import ToPILImage
from PIL import Image
import cv2


def extract_face(img, box, image_size=160, margin=0, save_path=None):
    """Extract face + margin from PIL Image given bounding box.
    
    Arguments:
        img {PIL.Image} -- A PIL Image.
        box {numpy.ndarray} -- Four-element bounding box.
        image_size {int} -- Output image size in pixels. The image will be square.
        margin {int} -- Margin to add to bounding box, in terms of pixels in the final image. 
            Note that the application of the margin differs slightly from the davidsandberg/facenet
            repo, which applies the margin to the original image before resizing, making the margin
            dependent on the original image size.
        save_path {str} -- Save path for extracted face image. (default: {None})
    
    Returns:
        torch.tensor -- tensor representing the extracted face.
    """
    margin = [
        margin * (box[2] - box[0]) / (image_size - margin),
        margin * (box[3] - box[1]) / (image_size - margin),
    ]
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, img.size[0])),
        int(min(box[3] + margin[1] / 2, img.size[1])),
    ]

    face = img.crop(box).resize((image_size, image_size), 2)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) + "/", exist_ok=True)
        save_args = {"compress_level": 0} if ".png" in save_path else {}
        face.save(save_path, **save_args)

    #face = F.to_tensor(np.float32(face))
    face = np.float32(face)

    return face

if __name__ == '__main__':

    batch_size=1


    # set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    mtcnn = MTCNN()
    fecnet = FECNet(pretrained=True)
    Num_Param = sum(p.numel() for p in fecnet.parameters() if p.requires_grad)
    print("Number of Trainable Parameters= %d" % (Num_Param))

    img = Image.open('/nfs/visual/bzfgrewe/projects/latent_spaces/FECNet/data/own/test.jpg')
    bbox, confidenxe, lm = mtcnn.detect(img,True)
    print(bbox)

    cropbox = np.maximum(bbox,0)
    X = extract_face(img, image_size=224, box=cropbox[0])

    print(X.shape)
    X = np.array([X,X,X]).astype(np.float32).reshape(-1, 3, 224, 224)
    print(X.shape)

    targets = torch.FloatTensor(X).view( 3, 3, 224, 224).cpu()

    #F.To
    #ext.ToPILImage()
    
    #image = ToPILImage(ext)
    #ext.save('/nfs/visual/bzfgrewe/projects/latent_spaces/FECNet/data/own/test_rect.jpg')

    latent_code = fecnet.forward(x=targets)

    print(latent_code)

    cv2.imshow('jkhd',cv2.np.array(ext))
    cv2.waitKey()




