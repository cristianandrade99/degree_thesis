import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import shutil

def imshow(im):

    h = im.shape[0]
    w = im.shape[1]
    c = im.shape[2]

    im_p = im
    t1 = "("+str(h)+","+str(w)+","+str(c)+")"
    t2 = "("+str(np.min(im).round(2))+","+str(np.max(im).round(2))+")"

    cmap = "gray"

    if(c==1):
        im_p = im.reshape((im.shape[0],im.shape[1]))
        cmap = "gray"

    plt.imshow(im_p,cmap=cmap)
    plt.title(t1+" "+t2)
    plt.show()

def max_conv_deconv_layers(hw):
    max_deconv_lays = int(np.log2(hw))
    print("for H,W = "+str(hw)+", # deconv layers and # conv layers <= "+str(max_deconv_lays))
    l = [ int(hw/np.power(2,i+1)) for i in range(max_deconv_lays) ]
    n_ls = [ i+1 for i in range(max_deconv_lays) ]
    print("(# conv layers, heigh,width of last enc volume)")
    print("(# deconv layers, heigh,width of first dec volume)")
    print(list(zip(n_ls,l)),"\n")

    return max_deconv_lays

def delete_All(directory):
    fileList = glob.glob(directory+"*")

    print("Are you sure you want to delete this?","\n")
    for f in fileList:
        print(f)

    ans = input("Type y or n: ")

    if ans == "y":
        for f in fileList:
            try:
                pass
                shutil.rmtree(f)
            except:
                pass
                os.remove(f)
        print("\n","Deleted Files")
