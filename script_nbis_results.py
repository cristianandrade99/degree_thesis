import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import subprocess
import shutil
import time
import wsq
import os

dir_sources = "./Output_valid"
dir_nbis_modules = "./NBIS_modules"
names_cats = ["to_enh","enhan","tar"]

dir_file_mindtct = os.path.join(dir_nbis_modules,"mindtct")
dir_file_bozorth3 = os.path.join(dir_nbis_modules,"bozorth3")
dir_file_nfiq = os.path.join(dir_nbis_modules,"nfiq")

wsq_ext = "wsq"
xyt_ext = "xyt"

images_folder_name = "images"

class Source():
    def __init__(self,dir_source):
        self.dir_source = dir_source

        self.dir_images_files = os.path.join(dir_source,images_folder_name)
        self.dir_wsq_files = os.path.join(dir_source,"wsq")
        self.dir_xyt_files = os.path.join(dir_source,"xyt")

        self.dir_input_xyt_pairs_file = os.path.join(self.dir_source,"xyt_pairs.lst")
        self.dir_output_scores_file = os.path.join(self.dir_source,"scores.txt")

        self.num_images = int(len(os.listdir(self.dir_images_files))/3)

    def call_init_source_methods(self):
        self.remove_files()
        self.obtain_ext()
        self.create_folders()
        self.create_wsq_files()
        self.create_xyt_files()
        self.remove_mindctc_leftover_files()
        self.create_xyt_compare_file()
        self.create_output_score_file()
        self.create_roc_images()
        self.create_cmc_images()
        self.obtain_qualities()

    def remove_files(self):

        for element in os.listdir(self.dir_source):
            dir_act_element = os.path.join(self.dir_source,element)

            if os.path.isfile(dir_act_element):
                os.remove(dir_act_element)
            else:
                if images_folder_name not in dir_act_element:
                    shutil.rmtree(dir_act_element)

        '''
        for root,folders,files in os.walk(self.dir_source):
            for file in files:
                ruta = os.path.join(root,file)
                if os.path.isfile(ruta) and images_folder_name not in ruta:
                    os.remove(ruta)
        '''

    def obtain_ext(self):
        files = os.listdir(self.dir_images_files)
        self.ext = os.path.splitext(os.path.join(self.dir_images_files,files[0]))[1][1:]

    def create_folders(self):
        if not os.path.exists(self.dir_wsq_files): os.mkdir(self.dir_wsq_files)
        if not os.path.exists(self.dir_xyt_files): os.mkdir(self.dir_xyt_files)

    def create_wsq_files(self):
        self.dirs_wsq_files = []

        for cat_name in names_cats:
            for i in range(self.num_images):

                file_name = "{}-{}".format(cat_name,i)
                dir_image = os.path.join(self.dir_images_files,"{}.{}".format(file_name,self.ext))
                dir_wsq_file = os.path.join(self.dir_wsq_files,"{}.{}".format(file_name,wsq_ext))

                self.dirs_wsq_files.append(dir_wsq_file)
                if not os.path.exists(dir_wsq_file):
                    img = Image.open(dir_image)
                    img.save(dir_wsq_file)

    def create_xyt_files(self):
        self.dirs_xyt_files = []

        for cat_name in names_cats:
            for i in range(self.num_images):

                file_name = "{}-{}".format(cat_name,i)
                dir_xyt_file = os.path.join(self.dir_xyt_files,file_name)
                dir_xyt_file_ext = "{}.{}".format(dir_xyt_file,xyt_ext)
                self.dirs_xyt_files.append(dir_xyt_file_ext)

                dir_wsq_file_ext = os.path.join(self.dir_wsq_files,"{}.{}".format(file_name,wsq_ext))
                if not os.path.exists(dir_xyt_file_ext):
                    subprocess.run([dir_file_mindtct,dir_wsq_file_ext,dir_xyt_file])

    def remove_mindctc_leftover_files(self):
        for root,folders,files in os.walk(self.dir_xyt_files):
            for file in files:
                if xyt_ext not in file:
                    os.remove(os.path.join(root,file))

    def create_xyt_compare_file(self):
        xyt_compare_list_file = open(self.dir_input_xyt_pairs_file,"w")

        for i in range(self.num_images):
            for j in range(self.num_images):
                xyt_compare_list_file.write("{}\n{}\n".format(self.dirs_xyt_files[j],self.dirs_xyt_files[i+self.num_images*2]))

        for i in range(self.num_images):
            for j in range(self.num_images):
                xyt_compare_list_file.write("{}\n{}\n".format(self.dirs_xyt_files[j+self.num_images],self.dirs_xyt_files[i+self.num_images*2]))

        xyt_compare_list_file.close()

    def create_output_score_file(self):
        os.system("{} {} {} {} {}".format(dir_file_bozorth3,
                                            "-A outfmt=pgs",
                                            "-A maxfiles=10000000",
                                            "-o {}".format(self.dir_output_scores_file),
                                            "-M {}".format(self.dir_input_xyt_pairs_file)))

    def create_roc_images(self):

        matriz_to_enh,matriz_enhan = self.create_score_matrices()

        thrs_to_enh = list(np.arange(0,np.max(matriz_to_enh),5))
        thrs_enhan = list(np.arange(0,np.max(matriz_enhan),5))
        num_thrs_to_enh = len(thrs_to_enh)
        num_thrs_enhan = len(thrs_enhan)

        TPR_to_enh = np.zeros(num_thrs_to_enh)
        FPR_to_enh = np.zeros(num_thrs_to_enh)
        TPR_enhan = np.zeros(num_thrs_enhan)
        FPR_enhan = np.zeros(num_thrs_enhan)

        for i in range(self.num_images):

            indexs = list(np.arange(self.num_images))
            indexs.remove(i)
            rand1 = indexs[int(np.random.rand()*(self.num_images-1))]
            rand2 = indexs[int(np.random.rand()*(self.num_images-1))]

            for k in range(num_thrs_to_enh):

                if matriz_to_enh[i,i]>=thrs_to_enh[k]:
                    TPR_to_enh[k]+=1

                if matriz_to_enh[i,rand1]>=thrs_to_enh[k]:
                    FPR_to_enh[k]+=1

            for k in range(num_thrs_enhan):

                if matriz_enhan[i,i]>=thrs_enhan[k]:
                    TPR_enhan[k]+=1

                if matriz_enhan[i,rand2]>=thrs_enhan[k]:
                    FPR_enhan[k]+=1

        TPR_to_enh/=self.num_images
        FPR_to_enh/=self.num_images
        TPR_enhan/=self.num_images
        FPR_enhan/=self.num_images

        fig = plt.figure()
        plt.plot(FPR_to_enh,TPR_to_enh,ls="-",c="b",marker="^",mfc="r",mec="r")
        plt.plot(FPR_enhan,TPR_enhan,ls="-",c="b",marker="^",mfc="g",mec="g")
        plt.xlabel("FPR",fontsize=20)
        plt.ylabel("TPR",fontsize=20)
        plt.title("ROC Comparison",fontsize=20)
        plt.legend(["to enhance","enhanced"])
        fig.savefig(os.path.join(self.dir_source,"roc_comparison.jpg"))

    def create_cmc_images(self):

        matriz_to_enh,matriz_enhan = self.create_score_matrices()

        matriz_sort_to_enh = np.flip(np.sort(matriz_to_enh),1)
        matriz_sort_enhan = np.flip(np.sort(matriz_enhan),1)

        ranks_to_enh = np.zeros((self.num_images,1))
        ranks_enhan= np.zeros((self.num_images,1))

        for r in range(self.num_images):
            ranks_to_enh[r] = list(matriz_sort_to_enh[r,:]).index(matriz_to_enh[r,r])+1
            ranks_enhan[r] = list(matriz_sort_enhan[r,:]).index(matriz_enhan[r,r])+1

        cmc_to_enh = np.zeros((self.num_images,1))
        cmc_enhan = np.zeros((self.num_images,1))

        for r in range(self.num_images):
            val_to_enh = list(ranks_to_enh).count(r+1)/self.num_images
            cmc_to_enh[r] = val_to_enh if r==0 else val_to_enh + cmc_to_enh[r-1]

            val_enhan = list(ranks_enhan).count(r+1)/self.num_images
            cmc_enhan[r] = val_enhan if r==0 else val_enhan + cmc_enhan[r-1]

        fig = plt.figure()
        plt.plot(cmc_to_enh,ls="-",c="b",marker="^",mfc="r",mec="r")
        plt.plot(cmc_enhan,ls="-",c="b",marker="^",mfc="g",mec="g")
        plt.xlabel("k",fontsize=20)
        plt.ylabel("Accuracy ",fontsize=20)
        plt.title("CMC Comparison",fontsize=20)
        plt.legend(["to enhance","enhanced"])
        fig.savefig(os.path.join(self.dir_source,"cmc_comparison.jpg"))

    def create_score_matrices(self):

        scores_file = open(self.dir_output_scores_file)
        lines = scores_file.readlines()

        matriz_to_enh = np.zeros((self.num_images,self.num_images))
        matriz_enhan = np.zeros((self.num_images,self.num_images))

        num_lines = 2*self.num_images**2
        for i in range( num_lines ):

            fields = lines[i].split()
            indx = int(fields[0][-5])
            indx_tar = int(fields[1][-5])

            matriz_act = matriz_to_enh if i < num_lines/2 else matriz_enhan
            matriz_act[indx,indx_tar] = int(fields[2])

        scores_file.close()

        return matriz_to_enh,matriz_enhan

    def obtain_qualities(self):
        self.qualities = np.zeros((self.num_images,len(names_cats)))

        for j in range(len(names_cats)):
            for i in range(self.num_images):
                index = j*self.num_images+i
                process = subprocess.run([dir_file_nfiq,self.dirs_wsq_files[index]],stdout=subprocess.PIPE,universal_newlines=True)
                self.qualities[i,j] = int(process.stdout[0])

        self.quality_means = np.mean(self.qualities,0).round(2)

def create_sources(dir_sources):
    sources_dicc = {}

    for source_name in os.listdir(dir_sources):
        act_path = os.path.join(dir_sources,source_name)

        if not os.path.isfile(act_path):
            act_source = Source(os.path.join(dir_sources,source_name))
            act_source.call_init_source_methods()
            sources_dicc[source_name] = act_source

    return sources_dicc

def create_quality_comparison(sources_dicc):
    labels = [ key for key in sources_dicc ]

    to_enh_means = [ sources_dicc[key].quality_means[0] for key in sources_dicc ]
    enh_means = [ sources_dicc[key].quality_means[1] for key in sources_dicc ]
    tar_means = [ sources_dicc[key].quality_means[2] for key in sources_dicc ]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, to_enh_means, width, label=names_cats[0])
    rects2 = ax.bar(x, enh_means, width, label=names_cats[1])
    rects3 = ax.bar(x + width, tar_means, width, label=names_cats[2])

    ax.set_ylabel('Mean Quality')
    ax.set_title('Mean Quality by to Enhance, Enhanced and Target')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),textcoords="offset points",ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()
    fig.savefig(os.path.join(dir_sources,"mean_qualities.jpg"))

def verif(sources_dicc):

    for source_key in sources_dicc:
        if source_key == "source_2":
            act_source = sources_dicc[source_key]

            for wsq in act_source.dirs_wsq_files:
                print(wsq)
            print()
            for xyt in act_source.dirs_xyt_files:
                print(xyt)
            print()

sources_dicc = create_sources(dir_sources)
#verif(sources_dicc)
create_quality_comparison(sources_dicc)
