# coding: utf-8

import numpy as np
import random
from numpy import linalg as LA




#pour chaque centroid dans C, on affect un ou plusieur vecteur x appartenant à X.
def affectation(X, C):
    
    tabDist = {} #on crée un dictionnaire de ayant la structure suivante : tabDist={indexCentroid1: [xi,...xn], indexCentroid2:.... }
    i=0
    oldTmpDist = 0.0
    tmpDist=0.0
    bestIndex=0
    #on parcour tout les vecteur X
    for x in X:
        #on parcour tout les centroid dans mu
        for c in C:
            #on calcule la distance entre le vecteur x et  un centroid 
            print "x = "+str(x)+" "
            tmpDist = LA.norm(x-c)
            if tmpDist<oldTmpDist:  #si cette nouvelle distance est inférieur à l'ancien alors on le garde
                oldTmpDIst=tmpDist
                indexCentroid = i
            i=i+1

        #on ajoute le vecteur x dans le centroids qui lui corespond 
        try:
            tabDist[bestIndex].append(x)#on ajoute le vecteur dans le centroid ayant l'index = bestIndex
        except KeyError:
            #on crée un nouveau centroid dans le dictionaire si celui ci n'était pas encore créer
            tabDist[bestIndex] = [x]
        
        #on remet à jours les variables de compteur
        oldTmpDist=0.0
        bestIndex=0
        i=0
        
    return tabDist

def affectation2(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
            for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters




def misAjourCentroid(centroid):
    #on crée un nouveau tableau de centroid
    newCentroid = []
    #on trie la liste des centroid tout en récupérant les clé 
    cle= sorted(centroid.keys())
    for i in cle:
        #on fait la moyenne des valeur des vecteurs contant dans un centroid i
        newCentroid.append(np.mean(centroid[i], axis = 0)) 
        #newCentroid[centroid[i]] = np.mean(centroid[i], axis = 0)
    return newCentroid



def testConvergenceOptimal(newCen,oldCen):
    print "test convergence"
    return (set([tuple(a) for a in newCen]) == set([tuple(a) for a in oldCen])) 



#param : X=ensemble de vecteur, k le nb de centroids, nbIteration = nombre de itération pour chercher les cendroids
#retourn la liste des centroids
def findCentroid(X,k,nbIteration):
   
    #initialisation des centroids aléatoirement
    oldCen = random.sample(X, k)
    newCen = random.sample(X, k)
    #print "init new centroid :"+str(newCen)
    #print "init old centroid :"+str(oldCen)

    i=0
    #calcul la distance de entre un vecteur Xi avec tous ces centroids et on selectionne le plus petit 
    #while not testConverge(newCen,oldCen):
    while i<nbIteration:

        oldCen = newCen
       
        #on va affecter un centroid a chaque vecteur, la variable centroidListVect contiendra tout les centroids avec leur vecteurs de X
        centroidListVect = affectation2(X, newCen)
        #print "list vecteur "+str(centroidListVect)
        #mis a jours  tout les centroids
        newCen = misAjourCentroid(centroidListVect)
        
        #print "new centroid :"+str(newCen)
        #print "old centroid :"+str(oldCen)

        i=i+1
    return(newCen, centroidListVect)



#param : X = vecteur, C : list des centroids
#retourne 1 vecteur ,c'est à dire le resultat de FK
def kmeanHard(X,C):
    #on initialise le vectueur de fk à 0 de la même taille que le nombre de cendroid 
    resultX=np.array([0]*len(C))
    m=0.0
    indiceMin=0
    i=0
    for cle in C:
        tmp=LA.norm(cle-X)#on calcule la norme 2 entre le vecteur X et le centroid 
        if(m>tmp):#si la nouvelle distance est plus petite que l'ancien alors on met à jours le vecteur de sortie
            m=tmp
            #on stock l'indice du centroid
            indiceMin=i
        i=i+1
    #on affecte la valeur 1 au cendroid qui contient le vecteur X
    resultX[indiceMin]=1
    return resultX





#param : X = vecteur, C : list des centroids
#retourne 1 vecteur ,c'est à dire le resultat de FK
def kmeanTriangle(X,C):
    #on initialise le vectueur de fk à 0 de la même taille que le nombre de cendroid 
    resultX=np.array([0]*len(C))
    i=0
    for cle in C:
        #on affecte la valeur de la norme 2 entre le vecteur X et le centroid 
        resultX[i]=LA.norm(cle-X)#on calcule 
        i=i+1
   
    return resultX

def init_board_gauss(N, k):
    n = float(N)/k
    X = []
    for i in range(k):
        c = (random.uniform(-1, 1), random.uniform(-1, 1))
        s = random.uniform(0.05,0.5)
        x = []
        while len(x) < n:
            a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
            # Continue drawing points from the distribution in the range [-1,1]
            if abs(a) < 1 and abs(b) < 1:
                x.append([a,b])
        X.extend(x)
    #print X
    X = np.array(X)[:N]
    return X

def unpickle(file):
    """ Unpickle a file in a dictionnary
    """
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def extract_patches(batchfile="/Users/ozad/git/CIFAR_project/dataset/cifar-10-batches-py/test_batch", patch_w=6, patch_h=6,patch_c=3, max_patches=100):
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
    #print images
    n_samples=len(images)
    print n_samples
    patch_size =(patch_h,patch_w)
    print patch_size
    #extracting patches for all images samples
    patches = [extract_patches_2d(images[i], patch_size,
                              max_patches, random_state=rng)
              for i in range(n_samples)]
     #reshape patches from matrices to vectors
    patches = np.array(patches).reshape(-1, patch_w*patch_h*patch_c)
    
    return patches


#jeu de teste à petite échelle

#ici la dimension d'un vecteur est de 3 
a =np.array([[1,2, 3],[11,22, 97],[19,82, 3],[1,2, 3],[12,225, 44],[141,222, 97],[139,82, 33],[14,2, 3],[14,5, 31],[45,2, 41],[21,75, 90],[1,51, 42],[41,55, 4] ,[3,2,8],[13,22,4]])
#a =np.array([[1,2],[11,22],[19, 3],[1, 3],[1, 4],[45, 41],[75, 90],[151, 42],[55, 4] ,[32,8],[22,4]])

#on créer un vecteur vect
vect = np.array([1,2,3])
mu,clu=findCentroid(a,3,5)

print "Liste centroid "+str(mu)
print "Regroupement des vecteurs selon leur centroids respective : "+str(clu)

resultHard=kmeanHard(vect,mu)
print "Resultat kmean hard : "+str(resultHard)

resultTri=kmeanTriangle(vect,mu)
print "Resultat kmean hard : "+str(resultTri)



#jeu de test en grandeur nature
#X2=extract_patches()
#mu,clu=findCentroid(X2,3,5)