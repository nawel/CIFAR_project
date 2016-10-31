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

def testConverge(newCen,oldCen):
    return (set([tuple(a) for a in newCen]) == set([tuple(a) for a in oldCen])) 




#param : k le nb de centroids
#retourn la liste des centroids
def findCentroid(X,k):
   
    #initialisation des centroids aléatoirement
    oldCen = random.sample(X, k)
    newCen = random.sample(X, k)

    #calcul la distance de entre un vecteur Xi avec tous ces centroids et on selectionne le plus petit 
    while not testConverge(newCen,oldCen):
        oldCen = newCen
       
        #on va affecter un centroid a chaque vecteur, la variable centroidListVect contiendra tout les centroids avec leur vecteurs de X
        centroidListVect = affectation(X, newCen)
        
        #mis a jours  tout les centroids
        listeCentroid = misAjourCentroid(centroidListVect)
    
    return(listeCentroid, centroidListVect)


#param : X = vecteur, C : list des centroids, k nb of centroid
#retourne 1 vecteur c'est à dire le FK
def kmeanHard(X,C,k):
    resultX=K*[0]
    i=0
    m=0
    for c in range(C):
        tmp=LA.norm(c-X)#on calcule la norme 2 entre le vecteur X et le centroid 
        if(m>tmp):#si la nouvelle distance est plus petite que l'ancien alors on met à jours le vecteur de sortie
            m=tmp
            resultX=K*[0]
            resultX[i]=1
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
X = np.array(X)[:N]
return X



X = init_board_gauss(20,3)
mu,clu=findCentroid(X,4)