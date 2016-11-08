# coding: utf-8
import numpy as np
from matplotlib.pylab import *
import cPickle, gzip
import math
import matplotlib
import random


class Perceptron(object):
    """docstring for Perceptron"""
    def __init__(self):
        super(Perceptron, self).__init__()
        self.w=np.array([])#array list de poid
        self.epsilon = 0.2

    def intPoid(self,taille):
        self.w = [random.random() * 2 - 1 for _ in xrange(taille)] 
        #print self.w
        
    """calcule la sortie du perceptron : out"""
    def calculSortie(self, X):
        #On calcule la valeur de sortie : sum(wi*xi)
        out = sum([wi * xi for wi, xi in zip(self.w, X)]) 
        if out >= 0:
            return 1
        else:
            return -1
    
    """ return le sum WiXi pour de tout les perceptron"""
    def sumAllPerceptronWiXi(listeP, X):
        out=0.0
        for p in listeP:
            out = out+ sum([wi * xi for wi, xi in zip(p.w, X)]) 
        return out
    
    
    """ rerurn le label de X, en se basant sur le meileur(le plus proche de 1) output de chaque perceptron"""
    #softmax exp(Wi*xi)/sum(exp(Wi*xi(de tout les perceptron))) 
    def calculSortieVect(self,X,listeP):
        i=0
        tmp=0.0
        bonPerceptron=0
        for p in listeP:
            
            #On calcule la valeur de sortie : sum(wi*xi)
            out = sum([wi * xi for wi, xi in zip(p.w, X)]) 
            out = np.exp(out)/sumAllPerceptronWiXi(listeP,X)
            if out>tmp:
                tmp=out
                bonPerceptron=i
            i=i+1
        #on return l'indice du meilleur perceptron qui possède le score le plus proche de 1
        return bonPerceptron

    """Mettre à jours les poids dans w"""
    def updateWeights(self, X, erreurSortie):
        """
        out = sortie du réseau pour un vecteur X
        c=sortie attendu
        erreurSortie=c-out
        wi = wi + learningRate * (c - out) * xi
        iterError = (d - r)
        """
        self.w = [wi + self.epsilon *erreurSortie * xi for wi, xi in zip(self.w, X)]

        
    
    
    def train(self, data,nbIteration):
        fin = False
        iteration = 0
        while not fin:
            globalError = 0.0
            for x in data: # on parcourt tout les data d'apprentissage
                out = self.calculSortie(x)
                c=x[2] #on charge la valeur de sortie attendu
                #on compare la valeur de sortie attendu avec la valeur de sortie actuel
                if c != out: 
                    erreurSortie = c - out # on calcule l'erreur entre  la valeur de sortie attendu et la valeur de sortie actuel
                    self.updateWeights(x, erreurSortie)
                    globalError = abs(erreurSortie) +globalError
                    print "globalError = "+str(globalError)

            iteration += 1
            # on vérifie nos conditions d'arrêts
            if globalError == 0.0 or iteration >= nbIteration: 
                fin = True # on sort de la boucle
                print "globalError = "+str(globalError) + ' avec  iterations = %s' % iteration
     
    def trainVect(self, data,nbIteration,label,listeP):
        fin = False
        iteration = 0
        compt=0
        while not fin:
            compt=compt+1
            #if math.fmod(compt,100)==0:
            #   print "on est à : "+str(compt)+" data entréés"
            print "on est à : "+str(compt)+" data entréés"

            i=0
            globalError = 0.0
            for x in data: # on parcourt tout les data d'apprentissage
                out = self.calculSortieVect(x,listeP)
                #print "fin calcul sotie pour le vecteur : "+str(x)
                
                #on compare la valeur de sortie attendu avec la valeur de sortie actuel
                if label != out: 
                    erreurSortie = label - out # on calcule l'erreur entre  la valeur de sortie attendu et la valeur de sortie actuel
                    self.updateWeights(x, erreurSortie)
                    globalError = abs(erreurSortie) +globalError
                    #print "globalError = "+str(globalError)
                i=i+1
            iteration += 1
            # on vérifie nos conditions d'arrêts
            if globalError == 0.0 or iteration >= nbIteration: 
                fin = True # on sort de la boucle
                print "globalError = "+str(globalError) + ' avec  iterations = %s' % iteration

def generateData(n):
    """
    on crée des vecteurs 2 D avec un label dans la troisème colonne
    """
    xb = (np.random.rand(n) * 2 -1) / 2 - 0.5
    yb = (np.random.rand(n) * 2 -1) / 2 + 0.5
    xr = (np.random.rand(n) * 2 -1) / 2 + 0.5
    yr = (np.random.rand(n) * 2 -1) / 2 - 0.5
    inputs = []
    inputs.extend([[xb[i], yb[i], 1] for i in xrange(n)])
    inputs.extend([[xr[i], yr[i], -1] for i in xrange(n)])
    return np.array(inputs)


def main ():
    trainset = generateData(80) # train set generation
    testset = generateData(20) # test set generation
    
    #print "train set : "+str(trainset[0])
    #print type(trainset)
    p = Perceptron() # use a short
    #print p.w
    #on initialise le tableau poid
    p.intPoid(2)
    p.train(trainset,50)
    #print "train set après : "+str(trainset)
    #Perceptron test
    i=0
    toErreur=0.0
    
    for x in testset:
        i+=1
        c=x[2]
        r = p.calculSortie(x)
        #on compare la valeur de sortie attendu avec la valeur de sortie actuel
        if r != x[2]: 
            #on incrémente le compteur des erreurs
            toErreur +=1 
        if r == 1:
            #on dessine les points
            plot(x[0], x[1], 'ob')
        else:
            plot(x[0], x[1], 'or')
            
            
    #calculer le to d'erreur       
    toErreur = toErreur/i
    print "taux d'erreur : "+str(toErreur)
    
    
    
    # on dessine la ligne de séparation  
    
    d = p.w / norm(p.w) 
    print "d="+str(d)
    
    #on calcule la première coordonnée de la vecteur
    d1 = [d[1], -d[0]]
    
    #on calcule la deuxième coordonnée de la vecteur
    d2 = [-d[1], d[0]]
    plot([d1[0], d2[0]], [d1[1], d2[1]], '--k')
    show()


def mainVect(nbIte):
   
    import matplotlib.pyplot as plt 
    from ipywidgets import FloatProgress
    from IPython.display import display

    NLABELS=10
    # Load the dataset
    f = gzip.open('/Users/ozad/Desktop/ecole/masterAic/TC1/tp5/mnist.pkl.gz', 'rb')

    train_set, valid_set, test_set = cPickle.load(f)

    print str(len(train_set[0]))+" training examples"
    f.close()
    # exemple: 
    

    testset = test_set[0]
    labelsTest = test_set[1]
  
    images = train_set[0]
    print (len(images[0]))
    labelsTrain = train_set[1]
    
    listePercep=[]
    i=0
    for i in range(NLABELS) :
        print"début init perceptron "+str(i)
        p = Perceptron() # use a short
        print"début initialisation des poids"
        p.intPoid(len(images[0]))# on initialise les poid , avec la taille de 784
        print"fin initialisation des poids"
        subtrain = images[labelsTrain==i]
        print"début entrainement"
        p.trainVect(subtrain,nbIte,i,listePercep)
        print"fin entrainement"
        listePercep.append(p)#on ajoute le nouveau perceptron dans la liste 
        print"fin init perceptron "+str(i)
  
   
    toErreur=0.0  
    k=0
    i=0
    for i in range(NLABELS) :
        subtrain2=(testset[labelsTest==i])
        for x in testset:
            r = listePercep[i].calculSortieVect(x,listePercep)
            #on compare la valeur de sortie attendu avec la valeur de sortie actuel
            if r != i: 
                print "attendu = "+str(i)+" , renvoyé par le réseau : "+str(r)
                #on incrémente le compteur des erreurs
                toErreur +=1 
            k=k+1

    #calculer le to d'erreur       
    toErreur = toErreur/(k+1)
    print "taux d'erreur : "+str(toErreur)
  
    
    

mainVect(9)

# cProfile test.py pour surveiller les teste


 