# -*- coding: utf-8 -*-

#for debugging purposes, import images locally

from extract_patch import unpickle
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import numpy as np
import scipy as sp

from copy import deepcopy
import cv2

dict=unpickle('../dataset/cifar-10-batches-py/data_batch_1')
images=dict['data']



def shuffle(img):
    img = img.reshape(3,32,32).transpose(1,2,0)
    return img
    
def un_shuffle(img):
#    img = img.flatten(order="C")
#    img = img.reshape(3072)
    loc1 = img[:,:,0].flatten()
    loc2 = img[:,:,1].flatten()
    loc3 = img[:,:,2].flatten()
    img = np.hstack((loc1,loc2,loc3))
    return img
    
def whitening(img,image):
    img=images
    image=deepcopy(image)
    moy = np.zeros((img.shape[1],img.shape[1]))
    
#    compute eigenvalues and eigenvectors of the mean XX.T matric on the whole dataset
    for i in range(1,img.shape[0]):
        moy += np.multiply(img[i,:], img[i,:].T)
        
    moy = moy/img.shape[0]
    w,v = np.linalg.eig(moy)    
  
#   whiten the image
    mat_list = [v,sp.linalg.sqrtm(np.diag(1/w)),v.T]

    whitened_image = np.zeros(image.shape)
    A_tilde = reduce(np.dot,mat_list)
    whitened_image = np.dot(A_tilde,image)
        
    return  whitened_image

    
    
def normalization(img):
#    substract of the mean and divide by the variance of the pixels (for each channel?)
    image=deepcopy(img)
    mean1 = np.mean(image[:,:,0])
    mean2 = np.mean(image[:,:,1])
    mean3 = np.mean(image[:,:,2])
    
    var1 = np.var(image[:,:,0])
    var2 = np.var(image[:,:,1])
    var3 = np.var(image[:,:,2])

#substract and divide
    image[:,:,0] = np.divide((image[:,:,0]-mean1),np.floor(var1/float(255)))
    image[:,:,1] = np.divide((image[:,:,1]-mean2),np.floor(var2/float(255)))
    image[:,:,2] = np.divide((image[:,:,2]-mean3),np.floor(var3/float(255)))

        
    return image
    
    
   
def sharpening (img, param):
#    https://www.packtpub.com/mapt/book/Application+Development/9781785283932/2/ch02lvl1sec22/Sharpening

    kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    kernel_sharpen_2 = np.array([[1,1,1], [1,-7,1], [1,1,1]])
    kernel_sharpen_3 = np.array([[-1,-1,-1,-1,-1],
                             [-1,2,2,2,-1],
                             [-1,2,8,2,-1],
                             [-1,2,2,2,-1],
                             [-1,-1,-1,-1,-1]]) / 8.0    
    if(param==1):
#        normal sharpening
        img = cv2.filter2D(img, -1, kernel_sharpen_1)
#       excessive sharpening 
    if(param==2):
        img = cv2.filter2D(img, -1, kernel_sharpen_2)
#        sharpening with edge enhancement
    if(param==3):
        img = cv2.filter2D(img, -1, kernel_sharpen_3)
        
    return img
    
def contrast_shifting(img):        
    norm_image = cv2.normalize(img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return norm_image 
    
def grayscale(img):
#    http://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
#    img = np.dot(img[:,:,:3],[0.299, 0.587, 0.114])
    img = np.mean(img,-1)

    return img        
        
def blurring_smoothing (img,size_kernel):    
#    if(1==1):  img = cv2.GaussianBlur(img,(size_kernel,size_kernel),0)
    
#    img = cv2.medianBlur(img,size_kernel)
    
#     semble être le meilleur
    img = cv2.bilateralFilter(img,size_kernel,75,75)
    return img
    
    
    
    
    
    
    
                            #######tests#######
#==============================================================================
# vect_img0 = images[0,:]
# vect_img1 = images[1,:]
# vect_img2 = images[2,:]
# vect_img3 = images[3,:]
# 
# img = shuffle(vect_img2)
#==============================================================================


#==============================================================================
# ret = shuffle(vect_img0)
# 
# plt.figure()
# plt.imshow(ret)
# 
# ret = un_shuffle(ret)
# plt.figure()
# plt.imshow(shuffle(ret))
# 
# 
#==============================================================================

#test blurring


#==============================================================================
# plt.figure() 
# blurred = blurring_smoothing(img,3) 
# 
# plt.subplot(1,2,1)
#  
# plt.imshow(img)
#  
# plt.subplot(1,2,2)
#  
# plt.imshow(blurred)
# 
# plt.savefig('../blurring.png', bbox_inches='tight')
# 
# plt.show()
# 
#==============================================================================

#==============================================================================
# plt.figure() 
# blurred = blurring_smoothing(img,3) 
# 
#  
# plt.imshow(img)
# plt.savefig('../blurring1.png', bbox_inches='tight')
# 
# plt.close()
#  
# plt.figure()
# plt.imshow(blurred)
# plt.savefig('../blurring2.png', bbox_inches='tight')
# 
# plt.show()
# plt.close
#==============================================================================

#==============================================================================
# #test grey
# plt.figure()
# gray = grayscale(img)
# plt.imshow(gray)
# 
#==============================================================================


#test normalisation contraste

#==============================================================================
# plt.figure()
# shifted = contrast_shifting(img)
# 
# plt.subplot(1,2,1)
# 
# plt.imshow(img)
# 
# plt.subplot(1,2,2)
# 
# plt.imshow(shifted)
# 
# plt.savefig('../shifted_contrast.png', bbox_inches='tight')
# 
# plt.show()
#==============================================================================


#==============================================================================
# plt.figure() 
# shifted = contrast_shifting(img)
# 
#  
# plt.imshow(img)
# plt.savefig('../contrast1.png', bbox_inches='tight')
# 
# plt.close()
#  
# plt.figure()
# plt.imshow(shifted)
# plt.savefig('../contrast2.png', bbox_inches='tight')
# 
# plt.show()
# plt.close
#==============================================================================


        #test sharpening

#==============================================================================
# sharpened = sharpening(img,1)
# 
# plt.figure()
#  
# plt.subplot(1,2,1)
#  
# plt.imshow(img)
#  
# plt.subplot(1,2,2)
#  
# plt.imshow(sharpened)
# 
# plt.savefig('../sharpening.png', bbox_inches='tight')
#  
# plt.show()
# plt.close()
#==============================================================================
#==============================================================================
# plt.figure() 
# sharpened = sharpening(img,1)
# 
#  
# plt.imshow(img)
# plt.savefig('../sharpening1.png', bbox_inches='tight')
# 
# plt.close()
#  
# plt.figure()
# plt.imshow(sharpened)
# plt.savefig('../sharpening2.png', bbox_inches='tight')
# 
# plt.show()
# plt.close
# 
#==============================================================================


        #test whitening
#whitened = whitening(img) 



#==============================================================================
# plt.figure()
#   
# plt.subplot(1,2,1)
#   
# plt.imshow(shuffle(vect_img2))
#   
# plt.subplot(1,2,2)
# 
# plt.imshow(whitening(shuffle(vect_img2)))
#   
# plt.show()
#==============================================================================

