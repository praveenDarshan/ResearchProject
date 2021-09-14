import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pylab as pl
from sklearn.metrics import confusion_matrix,accuracy_score
from scipy.cluster.vq import kmeans,vq
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
from numpy import loadtxt

classifiers,y_train, stdslr, k, voc = joblib.load("E:/CS/Research/bof5.pkl")

test_path="E:/CS/Research/FinalDataSet2/Training"
testClass_names=os.listdir(test_path)
testClass_names.sort()

testImage_paths=[]
testImage_classes=[]

def testImg_list(path):
    return (os.path.join(path,f) for f in os.listdir(path))

for testing_name in testClass_names:
    dir_=os.path.join(test_path,testing_name)
    class_path=testImg_list(dir_)
    testImage_paths+=class_path
    testImage_paths.sort()
    testImage_paths+=class_path
    testImage_paths.sort()

image_classes_6=[1]*200
image_classes_7=[2]*200


testImage_classes=image_classes_6+image_classes_7

T=[]

for i in range(len(testImage_paths)):
    T.append((testImage_paths[i],testImage_classes[i]))

testDataset = T
test = testDataset[:400]
image_paths_test, y_test = zip(*test)

des_list_test=[]


for image_pat in image_paths_test:
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
    des_list_test.append((image_pat,descriptor))



##############

# for image_pat in image_paths_test:
#     im=cv2.imread(image_pat)
#     im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#     sift = cv2.xfeatures2d.SIFT_create()
#     keypoints, descriptor_test = sift.detectAndCompute(im,None)
#     des_list_test.append((image_pat,descriptor_test))

######################

test_features=np.zeros((len(image_paths_test),k),"float32")
for i in range(len(image_paths_test)):
    words,distance=vq(des_list_test[i][1],voc)
    for w in words:
        test_features[i][w]+=1


test_features=stdslr.transform(test_features)

true_classes=[]
for i in y_test:
    if i==1:
        true_classes.append("Non-healthy")
    elif i==2:
        true_classes.append("healthy")


for  clf in classifiers:
    predict_classes=[]
    for i in clf.predict(test_features):
        if i==1:
            predict_classes.append("Non-healthy")
        elif i==2:
            predict_classes.append("healthy")

  
     

accuracy = accuracy_score(true_classes,predict_classes)


print(accuracy)

