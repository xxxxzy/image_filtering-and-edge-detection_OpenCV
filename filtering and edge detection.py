import cv2
import numpy as np
from matplotlib import pyplot as plt


lenna=cv2.imread('.../lenna.jpg') #print the path of image
img = 0.2125*lenna[:,:,0] + 0.7154*lenna[:,:,1] + 0.0721*lenna[:,:,2]


#cv2.imshow('result',lenna)
#cv2.waitKey(1)


def Gaussianfilter(img,size,sigma):
    
    out=cv2.GaussianBlur(img,(size,size),sigma) #function of Gaussian filter.
    
    """      
    cv2.imwrite('Gaussian.jpg',out)

    cv2.imshow('Gaussian',out.astype(np.uint8))
    cv2.waitKey(1)
    
    cv2.destroyAllWindows()
    """
    
    plt.figure(figsize=(9,9))
    plt.axis('off')
    plt.imshow(out, cmap='Greys_r')
    plt.show()
    
    return

Gaussianfilter(img,13,1) #use the 13*13 kernel matrix
Gaussianfilter(img,13,2)
Gaussianfilter(img,13,4)
Gaussianfilter(img,13,8)



def Gradient(img,size,sigma):
    
    out=cv2.GaussianBlur(img,(size,size),sigma) #use the gaussian filter to smooth the image first
    
    gx = 0.5*(out[1:,:] - out[:-1,:])
    gy = 0.5*(out[:,1:] - out[:,:-1])
    
    ng = np.sqrt(gx[:,1:]**2 + gy[1:,:]**2)
    
    """
    cv2.imwrite('edge.jpg',ng)
    
    cv2.imshow('edge',ng.astype(np.uint8))
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    """
    
    plt.figure(figsize=(9,9))
    plt.axis('off')
    plt.imshow(ng, cmap='Greys_r')
    plt.show()


    return

Gradient(img,13,1)
Gradient(img,13,2)
Gradient(img,13,3)
Gradient(img,13,8)

    

def GoL(img,size,sigma):
    
    out=cv2.GaussianBlur(img,(size,size),sigma) #use the gaussian filter to smooth the image first
    
    laplacian = cv2.Laplacian(out,cv2.CV_64F) #then use the laplcian filter
    laplacian1 = laplacian/laplacian.max()
    
    """
    cv2.imshow('GoL',laplacian1)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    """

    plt.figure(figsize=(9,9))    
    plt.axis('off')
    plt.imshow(laplacian1, cmap='Greys_r')
    plt.show()
    return

GoL(img,13,1)
GoL(img,13,2)
GoL(img,13,4)
GoL(img,13,8)

    

def canny(img,size,sigma):
    
    out = cv2.GaussianBlur(img,(size,size),sigma) #use the gaussian filter to smooth the image first
     
    canny = cv2.Canny(out,40,80) #use the threhold with 40 and 80
    
    """   
    cv2.imshow('canny',canny)
    cv2.waitKey(0)
        
    cv2.destroyAllWindows()
    """
    
    plt.figure(figsize=(9,9))    
    plt.axis('off')    
    plt.imshow(canny, cmap='Greys_r')
    plt.show()    
    
    return

canny(lenna,13,1)
canny(lenna,13,2)
canny(lenna,13,4)
canny(lenna,13,8)