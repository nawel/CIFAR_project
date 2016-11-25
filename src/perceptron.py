# coding: utf-8
import numpy as np
from matplotlib.pylab import *
import cPickle, gzip
import math
import matplotlib
import random
import matplotlib.pyplot as plt 
from ipywidgets import FloatProgress
from IPython.display import display
import threading
import time 


 
class Perceptron(object):
    """docstring for Perceptron"""
    def __init__(self):
        super(Perceptron, self).__init__()
        self.w=np.array([])#array list de poid
        self.epsilon = 0.5 # pas apprentissage
        self.biais=0.5
        self.name=""

    def intPoid(self,taille):
        self.w = [random.random() * 2 - 1 for _ in xrange(taille)] 
        #print self.w
        
    """calcule la sortie du perceptron : out"""
    def trainningPredic(self, X):
        #On calcule la valeur de sortie : sum(wi*xi)
        out = sum(self.w*X)+self.biais 
        #print "out = "+str(out)
        """ > 19 => t° erreur = 51%  sans le biais"""
        if out >= 0:
            return 1
        else:
            return -1
    
    """ return le sum WiXi pour de tout les perceptron"""
    def sumAllPerceptronWiXi(self,listeP, X):
        out=0.0
        for p in listeP:
            out = out+ np.exp(sum(p.w*X)) 
        return out
    
    
    """ return le label de X, en se basant sur le meileur(le plus proche de 1) output de chaque perceptron"""
    #softmax exp(Wi*xi)/sum(exp(Wi*xi(de tout les perceptron))) 
    def testPredic(self,X,listeP):
        i=0
        tmp=0.0
        bonPerceptron=0
        sumAllpercep = self.sumAllPerceptronWiXi(listeP,X)
        for p in listeP:
            
            #On calcule la valeur de sortie : sum(wi*xi)
            out = sum(p.w*X) 
            out = np.exp(out)/sumAllpercep
            if out>tmp:
                tmp=out
                bonPerceptron=i
            i=i+1
        #on return l'indice du meilleur perceptron qui possède le score le plus proche de 1
        return bonPerceptron

    """Mettre à jours les poids dans w"""
    def updateWeights(self, X, erreurSortie):

        self.w = [wi + self.epsilon *erreurSortie * xi for wi, xi in zip(self.w, X)]
        #print"erreur de sortie :"+str(erreurSortie)
        self.biais=self.biais+self.epsilon*erreurSortie

    """ entraine le modèle point """
    def train(self, data,nbIteration):
        fin = False
        iteration = 0
        while not fin:
            globalError = 0.0
            for x in data: # on parcourt tout les data d'apprentissage
                out = self.trainningPredic(x)
                c=x[2] #on charge la valeur de sortie attendu
                #on compare la valeur de sortie attendu avec la valeur de sortie actuel
                if c != out: 
                    erreurSortie = -1 # on calcule l'erreur entre  la valeur de sortie attendu et la valeur de sortie actuel
                    self.updateWeights(x, erreurSortie)
                    globalError = abs(erreurSortie) +globalError
                    print "globalError = "+str(globalError)

            iteration += 1
            # on vérifie nos conditions d'arrêts
            if globalError == 0.0 or iteration >= nbIteration: 
                fin = True # on sort de la boucle
                print "globalError = "+str(globalError) + ' avec  iterations = %s' % iteration
     
    def train3(self, data,nbIteration,label,listLabel):
        print "debut fct train"
        fin = False
        iteration = 0
        compt=0;
        while not fin:
            globalError = 0.0
            for x in data: # on parcourt tout les data d'apprentissage
                out = self.trainningPredic(x)
                #on compare la valeur de sortie attendu avec la valeur de sortie actuel
 
                y=-1
                if label==listLabel[compt] :
                    y=1#on met 1 dans y , la valeur qu'on souhaite que la prédicion retourne
                erreurSortie = y-out
                #if out == -1  and label==listLabel[compt]: 
 
                self.updateWeights(x, erreurSortie)
                globalError = abs(erreurSortie) +globalError
                compt=compt+1
            compt=0
            iteration += 1
            print "nb ité : "+str(iteration)
            # on vérifie nos conditions d'arrêts
            if globalError == 0.0 or iteration >= nbIteration: 
                fin = True # on sort de la boucle
 
    def train2(self, data,nbIteration):
        fin = False
        iteration = 0
        while not fin:
            globalError = 0.0
            for x in data: # on parcourt tout les data d'apprentissage
                out = self.trainningPredic(x)
                 #on compare la valeur de sortie attendu avec la valeur de sortie actuel
                if out != 1: 
                    erreurSortie = -1 # on prend le signe
                    
                    self.updateWeights(x, erreurSortie)
                    globalError = abs(erreurSortie) +globalError
                    #print "globalError = "+str(globalError)

            iteration += 1
            print "nb ité : "+str(iteration)
            # on vérifie nos conditions d'arrêts
            if globalError == 0.0 or iteration >= nbIteration: 
                fin = True # on sort de la boucle
                #print "globalError = "+str(globalError) + ' avec  iterations = %s' % iteration













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
    inputs.extend([[xr[i], yr[i], 1] for i in xrange(n)])
    return np.array(inputs)


def mainPt ():
    trainset = generateData(20) # train set generation
    testset = generateData(20) # test set generation
    
    print "train set 0 : "+str(trainset[0])
    print "train set : "+str(trainset)

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
        r = p.trainningPredic(x)
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










def mainVect2(nbIte,biaisInit,pas):



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

    print "phase initialisation"
    #phase création et initialisation des perceptron
    for i in range(NLABELS) :
        print"début init perceptron "+str(i)
        p = Perceptron() # use a short
        p.name=i
        #on init le biais
        p.biais=biaisInit
        #on init le pas apprentissage
        p.epsilon=pas
        #print"début initialisation des poids"
        p.intPoid(len(images[0]))# on initialise les poid , avec la taille de 784
        #print"fin initialisation des poids"
        listePercep.append(p)#on ajoute le nouveau perceptron dans la liste 
        #print"fin init perceptron "+str(i)
    print "fin initialisation"


    print "phase entrainement"
    #phase entrainement
    for i in range(NLABELS) :   
        """on entraine les 10 perceptron selon les label"""
        print "entrainement perceptron : "+str(listePercep[i].name) +" , i = "+str(i)
        subtrain = images[labelsTrain==i]
        #print"début entrainement"
        listePercep[i].train2(subtrain,nbIte)
        #print"fin entrainement"
    print "fin entrainement"

        


    print "phase test..."
    toErreur=0.0  
    k=0
    for i in range(NLABELS) :
        subtrain2=(testset[labelsTest==i])
        for x in subtrain2:
            #print "perceptron : "+str(listePercep[i].name) + " label : "+str(i)

            r = listePercep[i].testPredic(x,listePercep)
            #on compare la valeur de sortie attendu avec la valeur de sortie actuel
            if r != i: 
                #print "attendu = "+str(i)+" , renvoyé par le réseau : "+str(r)
                #on incrémente le compteur des erreurs
                toErreur +=1 
            k=k+1
            #TODO faire le t° d'erreur par classe

    #calculer le to d'erreur       
    toErreur = toErreur/(k+1)
    print "fin test."
    print "taux d'erreur : "+str(toErreur)









def mainVect3(nbIte,biaisInit,pas):



    NLABELS=10
    # Load the dataset
    f = gzip.open('/Users/ozad/Desktop/ecole/masterAic/TC1/tp5/mnist.pkl.gz', 'rb')

    train_set, valid_set, test_set = cPickle.load(f)

    print str(len(train_set[0]))+" training examples"
    f.close()
    # exemple: 


    testset = np.array(test_set[0])
    labelsTest =  np.array(test_set[1])

    images =  np.array(train_set[0])
    #print (len(images[0]))
    labelsTrain = np.array( train_set[1])

    listePercep=[]

    print "phase initialisation"
    #phase création et initialisation des perceptron
    for i in range(NLABELS) :
        print"début init perceptron "+str(i)
        p = Perceptron() # use a short
        p.name=i
        #on init le biais
        p.biais=biaisInit
        #on init le pas apprentissage
        p.epsilon=pas
        #print"début initialisation des poids"
        p.intPoid(len(images[0]))# on initialise les poid , avec la taille de 784
        #print"fin initialisation des poids"
        listePercep.append(p)#on ajoute le nouveau perceptron dans la liste 
        #print"fin init perceptron "+str(i)
    print "fin initialisation"


    print "phase entrainement"
    #phase entrainement
    """for i in range(NLABELS) :   
        #on entraine les 10 perceptron selon les label
        print "entrainement perceptron : "+str(listePercep[i].name) +" , i = "+str(i)
        listePercep[i].train3(images,nbIte,i,labelsTrain)
        #thread.start_new_thread( print_time, ("Thread-1", 2, ) )
    """
      
    threads = []
    t0=threading.Thread( target=listePercep[0].train3, args=(images,nbIte,0,labelsTrain) )
    threads.append(t0)

    t0.start()
    t1=threading.Thread( target=listePercep[1].train3, args=(images,nbIte,1,labelsTrain) )
    t1.start()
    threads.append(t1)
    t2=threading.Thread( target=listePercep[2].train3, args=(images,nbIte,2,labelsTrain) )
    t2.start()
    threads.append(t2)

    t3=threading.Thread( target=listePercep[3].train3, args=(images,nbIte,3,labelsTrain) )
    t3.start()
    threads.append(t3)

    t4=threading.Thread( target=listePercep[4].train3, args=(images,nbIte,4,labelsTrain) )
    t4.start()
    threads.append(t4)

    t5=threading.Thread( target=listePercep[5].train3, args=(images,nbIte,5,labelsTrain) )
    t5.start()
    threads.append(t5)

    t6=threading.Thread( target=listePercep[6].train3, args=(images,nbIte,6,labelsTrain) )
    t6.start()
    threads.append(t6)

    t7=threading.Thread( target=listePercep[7].train3, args=(images,nbIte,7,labelsTrain) )
    t7.start()
    threads.append(t7)

    t8=threading.Thread( target=listePercep[8].train3, args=(images,nbIte,8,labelsTrain) )
    t8.start()
    threads.append(t8)

    t9=threading.Thread( target=listePercep[9].train3, args=(images,nbIte,9,labelsTrain) )
    t9.start()
    threads.append(t9)


    exitFlag = 1

    # Wait for all threads to complete
    for t in threads:
        t.join()
    print "Exiting Main Thread"

    
    while(finAllThread!=10):
              time.sleep(1)
    
    
    print "fin entrainement"
   
        


    print "phase test..."
    toErreur=0.0  
    k=0
    for i in range(NLABELS) :
        subtrain2=(testset[labelsTest==i])
        print "tes label :"+str(i)
        for x in subtrain2:
            #print "perceptron : "+str(listePercep[i].name) + " label : "+str(i)

            r = listePercep[i].testPredic(x,listePercep)
            #on compare la valeur de sortie attendu avec la valeur de sortie actuel
            if r != i: 
                #print "attendu = "+str(i)+" , renvoyé par le réseau : "+str(r)
                #on incrémente le compteur des erreurs
                toErreur +=1 
            k=k+1
            #TODO faire le t° d'erreur par classe

    #calculer le to d'erreur       
    toErreur = toErreur/(k+1)
    print "fin test."
    print "taux d'erreur : "+str(toErreur)

    
    
    
#mainPt()
#meilleur param :  n200, 1, 0.45 : nbIte,biaisInit,pas
mainVect3(10,1,0.45)

# cProfile test.py pour surveiller les teste


 