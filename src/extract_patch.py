def unpickle(file):
    """ Unpickle a file in a dictionnary
    """
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def extract_patches(batchfile, patch_w=6, patch_h=6,patch_c=3, max_patches=100):
    """ Extract random patches from a dataset
    patch_w: width
    patch_h: height
    patch_c: number of channels
    max_patches: maximum number of randomly extracted patches per image
    """
        
    import numpy as np
    from PIL import Image
    from sklearn.feature_extraction.image import extract_patches_2d
    
    #load the data
    dict=unpickle(batchfile)
    patches=[]
    rng = np.random.RandomState(0) #initialize random state
    
    images=dict['data'] #contains images in row-major shape
    images = images.reshape(10000, 3, 32, 32).transpose(0,2,3,1) #reshape the images to matrices of (32,32,3) 
    
    n_samples=len(images)
    patch_size =(patch_h,patch_w)
    #extracting patches for all images samples
    patches = [extract_patches_2d(images[i], patch_size,
                              max_patches, random_state=rng)
              for i in range(n_samples)]
    #reshape patches from matrices to vectors
    patches = np.array(patches).reshape(-1, patch_w*patch_h*patch_c)
    
    return patches
