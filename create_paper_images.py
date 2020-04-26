from PIL import Image
import numpy as np
import os

source = "./Latex_paper/images"

def fig1():
    img0 = Image.open(os.path.join(source,"tar-0.png"))
    img1 = Image.open(os.path.join(source,"tar-1.png"))
    img2 = Image.open(os.path.join(source,"tar-2.png"))
    img = np.concatenate((np.array(img0),np.array(img1),np.array(img2)),1)
    Image.fromarray(img).save(os.path.join(source,"img_fig1.png"))

def fig2():
    img0 = Image.open(os.path.join(source,"elip-0.png"))
    img1 = Image.open(os.path.join(source,"elip-1.png"))
    img2 = Image.open(os.path.join(source,"elip-2.png"))
    img = np.concatenate((np.array(img0),np.array(img1),np.array(img2)),1)
    Image.fromarray(img).save(os.path.join(source,"img_fig2.png"))

def fig3():
    img0 = Image.open(os.path.join(source,"blur-0.png"))
    img1 = Image.open(os.path.join(source,"blur-1.png"))
    img2 = Image.open(os.path.join(source,"blur-2.png"))
    img = np.concatenate((np.array(img0),np.array(img1),np.array(img2)),1)
    Image.fromarray(img).save(os.path.join(source,"img_fig3.png"))

fig1()
fig2()
fig3()
