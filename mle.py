from random import shuffle
import numpy
from scipy.stats import multivariate_normal
from scipy.spatial import KDTree
import scipy.stats
import statistics
import math
import numpy as np
from sklearn import svm

def MLE(_trainingdata, _testingdata):
	trainingdata0 = []
	trainingdata1 = []
	trainingdata2 = []
	trainingdata3 = []
	trainingdata4 = []
	trainingdata5 = []
	trainingdata6 = []
	trainingdata7 = []
	for entry in _trainingdata:
		if(entry[0]<-10):
			trainingdata0.append(entry[1:10])
		elif(entry[0]>-10 and entry[0]<-5):
			trainingdata1.append(entry[1:10])
		elif(entry[0]>-5 and entry[0]<-1):
			trainingdata2.append(entry[1:10])
		elif(entry[0]>-1 and entry[0]<=0):
			trainingdata3.append(entry[1:10])
		elif(entry[0]>0 and entry[0]<1):
			trainingdata4.append(entry[1:10])
		elif(entry[0]>1 and entry[0]<5):
			trainingdata5.append(entry[1:10])
		elif(entry[0]>5 and entry[0]<10):
			trainingdata6.append(entry[1:10])
		elif(entry[0]>10):
			trainingdata7.append(entry[1:10])

	means = numpy.mean(trainingdata0, axis=0)
	covs = numpy.cov(trainingdata0, y=None, rowvar=False)
	a = multivariate_normal(mean=means)

	means = numpy.mean(trainingdata1, axis=0)
	covs = numpy.cov(trainingdata1, y=None, rowvar=False)
	b = multivariate_normal(mean=means, cov=covs)

	means = numpy.mean(trainingdata2, axis=0)
	covs = numpy.cov(trainingdata2, y=None, rowvar=False)
	c = multivariate_normal(mean=means, cov=covs)

	means = numpy.mean(trainingdata3, axis=0)
	covs = numpy.cov(trainingdata3, y=None, rowvar=False)
	d = multivariate_normal(mean=means, cov=covs)

	means = numpy.mean(trainingdata4, axis=0)
	covs = numpy.cov(trainingdata4, y=None, rowvar=False)
	e = multivariate_normal(mean=means, cov=covs)

	means = numpy.mean(trainingdata5, axis=0)
	covs = numpy.cov(trainingdata5, y=None, rowvar=False)
	f = multivariate_normal(mean=means, cov=covs)

	means = numpy.mean(trainingdata6, axis=0)
	covs = numpy.cov(trainingdata6, y=None, rowvar=False)
	g = multivariate_normal(mean=means, cov=covs)

	means = numpy.mean(trainingdata7, axis=0)
	covs = numpy.cov(trainingdata7, y=None, rowvar=False)
	h = multivariate_normal(mean=means)

	pa = float(len(trainingdata0)/float(len(_trainingdata)))
	pb = float(len(trainingdata1)/float(len(_trainingdata)))
	pc = float(len(trainingdata2)/float(len(_trainingdata)))
	pd = float(len(trainingdata3)/float(len(_trainingdata)))
	pe = float(len(trainingdata4)/float(len(_trainingdata)))
	pf = float(len(trainingdata5)/float(len(_trainingdata)))
	pg = float(len(trainingdata6)/float(len(_trainingdata)))
	ph = float(len(trainingdata7)/float(len(_trainingdata)))

	successfulpreds = 0

	for item in _testingdata:
		observation = item[1:10]

		bottom = \
		((a.pdf(observation)*pa) + \
		(b.pdf(observation)*pb) + \
		(c.pdf(observation)*pc) + \
		(d.pdf(observation)*pd) + \
		(e.pdf(observation)*pe) + \
		(f.pdf(observation)*pf) + \
		(g.pdf(observation)*pg) + \
		(h.pdf(observation)*ph))

		proba = (pa * a.pdf(observation))/bottom
		probb = (pb * b.pdf(observation))/bottom
		probc = (pc * c.pdf(observation))/bottom
		probd = (pd * d.pdf(observation))/bottom
		probe = (pe * e.pdf(observation))/bottom
		probf = (pf * f.pdf(observation))/bottom
		probg = (pg * g.pdf(observation))/bottom
		probh = (ph * h.pdf(observation))/bottom
		probs = [proba, probb, probc, probd, probe, probf, probg, probh]
		maxp = 0.0
		i = 0
		maxpi = 0
		for prob in probs:
			if maxp < prob:
				maxp = prob
				maxpi = i
			i = i + 1
		if(maxpi == 0):
			if(item[0]<-10):
				successfulpreds = successfulpreds + 1
		elif(maxpi == 1):
			if(item[0]>-10 and item[0]<-5):
				successfulpreds = successfulpreds + 1
		elif(maxpi == 2):
			if(item[0]>-5 and item[0]<-1):
				successfulpreds = successfulpreds + 1
		elif(maxpi == 3):	
			if(item[0]>-1 and item[0]<0):
				successfulpreds = successfulpreds + 1
		elif(maxpi == 4):	
			if(item[0]>0 and item[0]<1):
				successfulpreds = successfulpreds + 1
		elif(maxpi == 5):	
			if(item[0]>1 and item[0]<5):
				successfulpreds = successfulpreds + 1
		elif(maxpi == 6):	
			if(item[0]>5 and item[0]<10):
				successfulpreds = successfulpreds + 1
		elif(maxpi == 7):
			if(item[0]>10):
				successfulpreds = successfulpreds + 1

	return float(successfulpreds)/float(len(_testingdata))

data = np.loadtxt('data', dtype=float, delimiter=',')

# n_classes=8
# spread = np.zeros(n_classes)
# for i in range(0,len(data)):

# 	spread[int(data[i][0])]+=1

# for i in range(0,n_classes):
# 	print ("Class "+str(i)+" has "+str(spread[i]))

# data = np.random.permutation(data);

print(len(data))

shuffle(data)

train_data = data[0:(int(len(data)/2))]
test_data = data[(int(len(data)/2)):]

print(MLE(train_data.tolist(), test_data.tolist()))
