import numpy as np
from PIL import Image
from sklearn.feature_extraction.image import extract_patches_2d
from skimage.util.shape import view_as_windows
from kmeans import kmeanTriangle, kmeanHard
import math


#fonction qui transforme une image (32,32,3) en un vecteur de features

def transform_representation(image, w, s, centroids, affect='hard'):
    #image is in (32,32,3) order
    #w is the width (height) of the patch
    # s is the stride
    #centroids: liste des centroids
    #affect : hard or triangle
    
    K=len(centroids) #K is the number of centroids
    
    window_shape = (w, w, 3)  
    
    #extraire tous les patchs de dimensions (w,w,3) avec un stepsize de s
    patches=view_as_windows(image, window_shape, step=s)
    
    #TODO preprocessing for each patch
    
    image_representation=np.zeros([patches.shape[0], patches.shape[1] ,K])
    
    #Compute activation function (TRIANGLE or HARD kmeans)
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            raw_patch=patches[i][j][0].reshape(-1, w*w*3)
            if(affect=='hard'):
                image_representation[i][j]=kmeanHard(raw_patch,centroids)
            if(affect=='triangle'):
                image_representation[i][j]=kmeanTriangle(raw_patch,centroids)
            
    halfr=round(patches.shape[0]/2)
    halfc=round(patches.shape[1]/2)     
    
    q1=np.zeros(K)
    q2=np.zeros(K)
    q3=np.zeros(K)
    q4=np.zeros(K)
    for i in range(K):
        
        q1[i]=sum(sum(image_representation[0:halfr, 0:halfc, i]))
        q2[i]=sum(sum(image_representation[halfr:, 0:halfc, i]))
        q3[i]=sum(sum(image_representation[0:halfr, halfc:, i]))
        q4[i]=sum(sum(image_representation[halfr: , halfc: , i]))
    

    return np.concatenate([q1, q2, q3, q4 ])


#fonction qui transforme toutes les images d'un dataset en array de vecteur de features
def extract_features(images, centroids, w=6, s=1)
    # images input is images=dict['data']
    #centroids: liste des centroids
    #w taille des patchs
    # s espace entre les patchs
    images = images.reshape(10000, 3, 32, 32).transpose(0,2,3,1)
    list_features=[]

    for img in images:
        features=transform_representation(img, w, s, len(centroids),centroids)   
        list_features.append(features)

    array_features=np.array(list_features)
    
    return array_features
    
    