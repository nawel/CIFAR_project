import math
import matplotlib
import matplotlib.pyplot as plt 


dict=unpickle('../dataset/cifar-10-batches-py/data_batch_1')
images=dict['data']
labels=dict['labels']
labels = np.asarray(labels)

NLABELS=10

########### Compute means and variances #############
means = {}
variances = {}
for lbl in range(NLABELS):
    subtrain = images[labels==lbl] # get the subpart of train for the class
    mean = subtrain.mean(axis=0)   # compute the mean       
    means[lbl]=mean                # store the mean vector in a dict 
    var = sum((subtrain[n] - mean)**2 
              for n in range(subtrain.shape[0]))/subtrain.shape[0] # compute the variance 
    variances[lbl]=var

########## compute priors ############################
def computePriors(labels):
    priors = {}
    priors = np.zeros([NLABELS,1])
    for lbl in range(NLABELS):
        priors[lbl]=labels[labels==lbl].shape[0]
    priors = priors/priors.sum()
    return priors

########## compute posteriors: P(Y=lbl/X=image) ############################
def computePosteriors(image):
    posteriors = np.zeros([NLABELS,1])
    for lbl in range(NLABELS):
            mean = means[lbl] 
            sigma2 = variances[lbl]
            non_null = sigma2!=0
            scale = 0.5*np.log(2*sigma2[non_null]*np.pi) # 1/2 * log(2*sigma^2*pi)
            expterm = -0.5*np.divide(np.square(image[non_null]-mean[non_null]) # -1/2 ((x-µ)/sigma)^2
                                     ,sigma2[non_null])
            llh = (expterm-scale).sum() #log de vraisemblance 
            post = llh + np.log(priors[lbl]) 
            posteriors[lbl]=post
    return posteriors

########## Inference  ############################
def inference(image, label):  
    correct=0.0    
    posteriors=computePosteriors(image)
    classe=np.argmax(posts)
    if classe==label:
        correct=1.0
    return classe, correct

