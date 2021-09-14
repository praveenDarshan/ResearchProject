import cv2
import numpy as np 
from matplotlib import pyplot as plt 


obj = ([0, 2])
objnum = ([10,10])
success = 0
zero = 0
ones = 0

for j in range(1,2):  
    for k in range(1, objnum[j]):
        
        img = cv2.imread(f"E:/CS/Research/FinalDataset2/Testing/{j}/{k}.jpg")

       
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)






#by using this code we can find the brown spot if a leaf

        lower_color = np.array([0,100,100])
        upper_color = np.array([20,255,255])



#mask = cv2.inRange(hsv, lower_range, upper_range)
        mask = cv2.inRange(hsv, lower_color, upper_color)

        res = cv2.bitwise_and(img, img, mask = mask) 

        #cv2.imshow("image",img)
       

        #cv2.imshow("mask", img)
        #cv2.imshow('res',res)
    
        cv2.waitKey(0)
        cv2.destroyAllWindows()    








       # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

## mask of green (36,25,25) ~ (86, 255,255)
# mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
        #mask = cv2.inRange(hsv, (36, 25, 25), (70, 255,255))

## slice the green
        imask = mask>0
        green = np.zeros_like(res, np.uint8)
        green[imask] = res[imask]



## save 
        convetedName = str(k)
        cv2.imwrite(convetedName.png, green)
        #cv2.imshow('green17.png',res)




#opening = cv2.morphologyEx(~(im_out), cv2.MORPH_OPEN, kernel)
#opening = ~(opening)

        thresh, BWImage = cv2.threshold(green, 25, 255, cv2.THRESH_BINARY) #inv = inverse  thresh_binary = black or white out put
        #cv2.imshow('green17.png',BWImage)
        cv2.waitKey(0)



        w = BWImage.shape [0]
        h = BWImage.shape [1]

##must crop the leaf 

#imFilled = BWImage.copy()
#mask = np.zeros((h+2, w+2), np.uint8)
#cv2.floodFill(imFilled, mask, (0,0), 255) 

        BWImagecanny = cv2.Canny(BWImage,h,w)
#im_out = cv2.subtract(imFilled,BWImagecanny )



        cnts = cv2.findContours(BWImagecanny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2] 
        s1 = 5      #range 
        s2 = 400
        xcnts = [] 
# for cnt in cnts: 
#     if s1<cv2.contourArea(cnt) <s2: 
        
#         xcnts.append(cnt) 

        cv2.drawContours(BWImage, cnts, -1, (0, 255, 0), 3)
  
        #cv2.imshow('Contours', BWImage)
        cv2.waitKey(0)

        print("\nDots number: {}".format(len(cnts)))

        x = len(cnts)
        if(x>=10):
            n = 0
            zero  += 1

        else:
            n = 1
            ones  += 1

        print(j,k)
        print(n)

    print("(non helthy)zeros=",zero)
    print("(helthy)ones=",ones)



# cv2.waitKey(0)