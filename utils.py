import numpy as np
from matplotlib import pyplot as plt
import os


def showImage(images,epoch=-99):

    images = images.cpu().numpy()
    images = images/2 + 0.5
    plt.imshow(np.transpose(images,axes = (1,2,0)))
    plt.axis('off')
    if epoch!=-99:
        plt.savefig("outs/e" + str(epoch) + ".png")


def check_folders(folders=["outs","checkpoints"]):
  #print("Moving to", path)#todo: log
  for folder in folders:
    if not os.path.exists(folder):
      print("Creating", folder)
      os.mkdir(folder)



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)