### imports
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import pathlib as pl
from models.FECNet import FECNet
from models.mtcnn import MTCNN
#from models.utils.detect_face import extract_face
from torchvision.transforms import ToPILImage
from PIL import Image


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

    parser = argparse.ArgumentParser(description='PyTorch FECNet inference')
    parser.add_argument('dir')
    parser.add_argument('csv')
    args = parser.parse_args()
    cdir = pl.Path(args.dir)

    os.makedirs(cdir / 'detected', exist_ok=True)
    output = open(args.csv,'w')

    mtcnn = MTCNN(image_size=224)
    fecnet = FECNet(pretrained=True)

    import sys
    # faces = {}
    for file in [e for e in cdir.iterdir() if e.is_file()]:
        #if (file.parent / 'detected' / Path(file.name)).is_file():
        print(file)
        img = Image.open(file)
        bbox, prob, lm = mtcnn.detect(img,True)
        cropbox = np.maximum(bbox,0)
        X = extract_face(img, image_size=224, box=cropbox[0], save_path=str(cdir / 'detected' / file.name))

        X = np.array([X,X,X]).astype(np.float32).reshape(-1, 3, 224, 224)
        targets = torch.FloatTensor(X).view( 3, 3, 224, 224).cpu()
        latent_code = fecnet.forward(x=targets)[0].detach().numpy()
        print(prob)
        line = ','.join([str(file.name),str(prob[0])])
        line += ',' + ','.join(['%.5f' % num for num in latent_code])
        output.write(line + os.linesep)


    output.close()





