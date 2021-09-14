import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pylab as pl
from sklearn.metrics import confusion_matrix,accuracy_score
from scipy.cluster.vq import kmeans,vq
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import joblib
import csv
from skimage import color, data, restoration, morphology

train_path="E:/CS/Research/FinalDataSet2/Training"
class_names=os.listdir(train_path)
class_names.sort()

image_paths=[]
image_classes=[]

def img_list(path):
    return (os.path.join(path,f) for f in os.listdir(path))

for training_name in class_names:
    dir_=os.path.join(train_path,training_name)
    class_path=img_list(dir_)
    image_paths+=class_path
    image_paths.sort()


image_classes_0=[1]*200
image_classes_1=[2]*200

image_classes=image_classes_0+image_classes_1

D=[]

for i in range(len(image_paths)):
    D.append((image_paths[i],image_classes[i]))

dataset = D
train = dataset[:400]
image_paths, y_train = zip(*train)

des_list=[]

for image_pat in image_paths:
    im=cv2.imread(image_pat)
    height, width = im.shape[:2]


    
    #Create a mask holder
    mask = np.zeros(im.shape[:2],np.uint8)

    #Grab Cut the object
    bgdModel = np.zeros((1,65),np.float64)   
    fgdModel = np.zeros((1,65),np.float64)   

    #Hard Coding the Rect The object must lie within this rect.
    rect = (10,10,width-30,height-30)
    cv2.grabCut(im,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img1 = im*mask[:,:,np.newaxis]


    #Get the background    
    background = im - img1

    #Change all pixels in the background that are not black to white
    background[np.where((background > [0,0,0]).all(axis = 2))] = [255,255,255]

    #Add the background and the image
    OImage1 = background + img1
    OImage1 = cv2.cvtColor(OImage1, cv2.COLOR_BGR2GRAY)
    BWImageO = cv2.adaptiveThreshold(OImage1,255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,19,2)



    #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #sift = cv2.xfeatures2d.SIFT_create()
    #keypoints, descriptor = sift.detectAndCompute(im,None) 
    #des_list.append((image_pat,descriptor))

    orb=cv2.ORB_create() 
    keypoints, descriptor = orb.detectAndCompute(BWImageO, None)
    des_list.append((image_pat,descriptor))


descriptors=des_list[0][1]
for image_path,descriptor in des_list[1:]:
    descriptors=np.vstack((descriptors,descriptor))

descriptors_float=descriptors.astype(float)

k=150
voc,variance=kmeans(descriptors_float,k,1)
im_features=np.zeros((len(image_paths),k),"float32")
for i in range(len(image_paths)):
    words,distance=vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w]+=1

stdslr=StandardScaler().fit(im_features)
im_features=stdslr.transform(im_features)

C_2d_range = [1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4]
gamma_2d_range = [0.01,0.1, 1, 10,100]

classifiers = []
for C in C_2d_range:
    for gamma in gamma_2d_range:
        svm = SVC(C=C ,max_iter=8000, kernel = 'linear',gamma = gamma)
        clf = OneVsRestClassifier(svm)
        clf.fit(im_features, np.array(y_train))
        classifiers.append(clf)

joblib.dump((classifiers,y_train, stdslr, k, voc), 'E:/CS/Research/bof5.pkl', compress = 3)