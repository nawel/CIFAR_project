def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def extract_patches(batchfile, patch_w=6, patch_h=6,patch_c=3, max_patches=100):
    import numpy as np
    from PIL import Image
    from sklearn.feature_extraction.image import extract_patches_2d
    
    dict=unpickle(batchfile)
    patches=[]
    rng = np.random.RandomState(0)
    
    images=dict['data']
    images = images.reshape(10000, 3, 32, 32).transpose(0,2,3,1)
    
    n_samples=len(images)
    patch_size =(patch_h,patch_w)
    patches = [extract_patches_2d(images[i], patch_size,
                              max_patches, random_state=rng)
              for i in range(n_samples)]
    patches = np.array(patches).reshape(-1, patch_w*patch_h*patch_c)
    
    return patches