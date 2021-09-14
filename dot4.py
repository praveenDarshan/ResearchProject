import cv2
import numpy as np
import math

obj = ([0, 2])
objnum = ([10,10])
success = 0
zero = 0
ones = 0

for j in range(0,1):  
    for k in range(1, objnum[j]):
        
        OImage1 = cv2.imread(f"E:/CS/Research/FinalDataset2/Training/{j}/{k}.jpg")

        height, width = OImage1.shape[:2]
#back graound removing start
#Create a mask holder
        mask = np.zeros(OImage1.shape[:2],np.uint8)

#Grab Cut the object
        bgdModel = np.zeros((1,65),np.float64)   
        fgdModel = np.zeros((1,65),np.float64)   

#Hard Coding the Rect The object must lie within this rect.
        rect = (10,10,width-30,height-30)
        cv2.grabCut(OImage1,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img1 = OImage1*mask[:,:,np.newaxis]

#Get the background    
        background = OImage1 - img1

#Change all pixels in the background that are not black to white
        background[np.where((background > [0,0,0]).all(axis = 2))] = [255,255,255]

#Add the background and the image
        final = background + img1

#To be done - Smoothening the edges

        cv2.imshow('image', final )

#back graound removing finished        




#colordetection start

        hsv = cv2.cvtColor(final, cv2.COLOR_BGR2HSV)

#by using this code we can find the brown spot if a leaf

        lower_color = np.array([0,100,100])
        upper_color =   np.array([20,255,255])



#mask = cv2.inRange(hsv, lower_range, upper_range)
        mask = cv2.inRange(hsv, lower_color, upper_color)

        res = cv2.bitwise_and(final, final, mask = mask) 

        #cv2.imshow("image",final)
        #cv2.imshow("mask", final)
        #cv2.imshow('res',final)



#colordetection  finished



        OImage = cv2.resize(OImage1, (700, 400))
        OImage = cv2.cvtColor(OImage, cv2.COLOR_BGR2GRAY)

        OImage1=res 

        hsv = cv2.cvtColor(OImage1, cv2.COLOR_BGR2HSV)

        ## mask of green (36,25,25) ~ (86, 255,255)
        # mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
        mask = cv2.inRange(hsv, (36, 25, 25), (70, 255,255))

        ## slice the green
        imask = mask>0
        green = np.zeros_like(OImage1, np.uint8)
        green[imask] = OImage1[imask]

       
        thresh, BWImage = cv2.threshold(res, 60, 180, cv2.THRESH_BINARY_INV)

        #cv2.imshow('thresh', BWImage) # added by me for testing 

        h, w = BWImage.shape    


        #BWImage = ~(BWImage[80:250,15:60])
        OImage1 = cv2.resize(res, (700, 400))
        #OImage1 = (OImage1[80:250,15:60])
        h, w = BWImage.shape 

        imFilled = BWImage.copy()
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(imFilled, mask, (0,0), 255)

        BWImagecanny = cv2.Canny(BWImage,h,w)
        im_out = cv2.subtract(imFilled,BWImagecanny )
        kernel = np.ones((5,5), np.uint8)

        opening = cv2.morphologyEx(~(im_out), cv2.MORPH_OPEN, kernel)
        opening = ~(opening)

        cnts = cv2.findContours(opening, cv2.RETR_LIST, 
                            cv2.CHAIN_APPROX_SIMPLE)[-2] 
        #s1 = 100
       # s2 = 300
        xcnts = [] 
# for cnt in cnts: 
#     if s1<cv2.contourArea(cnt) <s2: 
        
#         xcnts.append(cnt) 

        cv2.drawContours(BWImage, cnts, -1, (0, 255, 0), 3)
  
        cv2.imshow('Contours', BWImage)
        cv2.waitKey(0)

        
        print(j,k)
        print("\nDots number: {}".format(len(cnts)))


        x = len(cnts)


        if(x>=10):
            n = 0
          
            zero  += 1
           
            
            
            
        else:
            n = 1
           
            ones  += 1
          
            




        print("success rate =", n)    

  

print("zeros",zero)
print("ones",ones)
      
        


# cv2.waitKey(0)