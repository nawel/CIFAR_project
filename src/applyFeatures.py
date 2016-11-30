# coding: utf-8
# Author: Daro HENG <daro.heng@u-psud.fr>
# Licence: BSD 3 clause
from feature_extraction import extract_features
from perceptron import mainPerceptron
from kmeans import extract_centroid, unpickle
from sklearn.cluster import KMeans
import numpy as np
import pickle 
import cPickle


def applyFeatures(fileTrain,fileTest,DataTestAfterFeature,fileCentroid,DataTrainAfterFeature):
	
	##################### Création des centroid, puis chargement ###################
	#on extrait des centroid des data train
	extract_centroid(fileTrain,fileCentroid,500)
	
	#on charge les centroid
	Fcentroid = open(fileCentroid, 'rb')
	listCentroid=cPickle.load(Fcentroid)

	print "fin extraction des centroids"



	#################### Application des centroids sur les données test et train ############
	
		###### application sur les données train ###############
	#on charge les data train 
	data = unpickle(fileTrain)

	#on applique les features sur les data train
	dataTrain=extract_features(data['data'],listCentroid)
	
	#sauvegarde des data train après l'application des features 	
 	pickle.dump(dataTrain, open(DataTrainAfterFeature, "wb"))

	print "fin application des centroids sur les données train "


		###### application sur les données test ###############

	#on charge les data test 
	dataT = unpickle(fileTest)

	#on applique les features sur les data train
	dataTest=extract_features(dataT['data'],listCentroid)
	
	#sauvegarde des data train après l'application des features
 	pickle.dump(dataTest, open(DataTestAfterFeature, "wb"))

	print "fin application des centroids sur les données test "




 	################ chargement des données test et train ######################
	"""

	#on charge les data train
	Ftrain = open(DataTrainAfterFeature, 'rb')
	dataTrainAfterFeature= cPickle.load(Ftrain)

	#on charge les labels des data train
	labelsTrain=data['labels']
	labelsTrain = np.asarray(labelsTrain)

	print "fin chargement des données train "



	#on charge les data test
	Ftest = open(DataTestAfterFeature, 'rb')
	dataTestAfterFeature= cPickle.load(Ftest)

	#on charge les labels des data test
	labelsTest=dataT['labels']
	labelsTest = np.asarray(labelsTest)

	print "fin chargement des données test "




	print "Application sur les perceptron "

  	"""
 	#Appel au réseaux de perceptron
	#mainPerceptron(dataTrainAfterFeature,dataTestAfterFeature,labelsTrain,labelsTest,10,1,0.5,10)
	


	
applyFeatures('/Users/ozad/git/CIFAR_project/dataset/cifar-10-batches-py/data_batch_2','/Users/ozad/git/CIFAR_project/dataset/cifar-10-batches-py/test_batch','/Users/ozad/git/CIFAR_project/data50C/dataTestAfterFeature2.txt','/Users/ozad/git/CIFAR_project/data50C/centroidsData500.txt','/Users/ozad/git/CIFAR_project/data50C/dataTrainAfterFeature2.txt')







