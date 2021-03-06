# Avikshith Pilly & William Dunkerley

# START: OWN CODE
from random import shuffle
import numpy as np
from scipy.stats import multivariate_normal
import math


def actualClass(_eodpercent):
	if(_eodpercent<-10):
		return 0
	elif(_eodpercent>-10 and _eodpercent<=-5):
		return 1
	elif(_eodpercent>-5 and _eodpercent<=-1):
		return 2
	elif(_eodpercent>-1 and _eodpercent<=0):
		return 3
	elif(_eodpercent>0 and _eodpercent<=1):
		return 4
	elif(_eodpercent>1 and _eodpercent<=5):
		return 5
	elif(_eodpercent>5 and _eodpercent<=10):
		return 6
	elif(_eodpercent>10):
		return 7

def MLE(_trainingdata, _testingdata, _n):
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
			trainingdata0.append(entry[1:_n])
		elif(entry[0]>-10 and entry[0]<-5):
			trainingdata1.append(entry[1:_n])
		elif(entry[0]>-5 and entry[0]<-1):
			trainingdata2.append(entry[1:_n])
		elif(entry[0]>-1 and entry[0]<=0):
			trainingdata3.append(entry[1:_n])
		elif(entry[0]>0 and entry[0]<1):
			trainingdata4.append(entry[1:_n])
		elif(entry[0]>1 and entry[0]<5):
			trainingdata5.append(entry[1:_n])
		elif(entry[0]>5 and entry[0]<10):
			trainingdata6.append(entry[1:_n])
		elif(entry[0]>10):
			trainingdata7.append(entry[1:_n])

	means = np.mean(trainingdata0, axis=0)
	covs = np.cov(trainingdata0, y=None, rowvar=False)
	a = multivariate_normal(mean=means)

	means = np.mean(trainingdata1, axis=0)
	covs = np.cov(trainingdata1, y=None, rowvar=False)
	b = multivariate_normal(mean=means, cov=covs)

	means = np.mean(trainingdata2, axis=0)
	covs = np.cov(trainingdata2, y=None, rowvar=False)
	c = multivariate_normal(mean=means, cov=covs)

	means = np.mean(trainingdata3, axis=0)
	covs = np.cov(trainingdata3, y=None, rowvar=False)
	d = multivariate_normal(mean=means, cov=covs)

	means = np.mean(trainingdata4, axis=0)
	covs = np.cov(trainingdata4, y=None, rowvar=False)
	e = multivariate_normal(mean=means, cov=covs)

	means = np.mean(trainingdata5, axis=0)
	covs = np.cov(trainingdata5, y=None, rowvar=False)
	f = multivariate_normal(mean=means, cov=covs)

	means = np.mean(trainingdata6, axis=0)
	covs = np.cov(trainingdata6, y=None, rowvar=False)
	g = multivariate_normal(mean=means, cov=covs)

	means = np.mean(trainingdata7, axis=0)
	covs = np.cov(trainingdata7, y=None, rowvar=False)
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

	loss = []
	for i in range(8):
		loss.append([0]*8)

	for item in _testingdata:
		observation = item[1:_n]

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

		loss[actualClass(item[0])][maxpi] += 1

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

	# Confusion matrix printout
	# for row in loss:
	# 	print(row)
	# print("--------")
	return float(successfulpreds)/float(len(_testingdata))

def MLETrials(_trials, _companyfile):
	s = 0.0
	lst = []
	for i in range(_trials):
		data = np.loadtxt(_companyfile, dtype=float, delimiter=',')
		shuffle(data)
		train_data = data[0:(int(len(data)/2))]
		test_data = data[(int(len(data)/2)):]
		try:
			x = MLE(train_data.tolist(), test_data.tolist(), 10)
		except Exception as e:
			data = np.loadtxt(_companyfile, dtype=float, delimiter=',')
			shuffle(data)
			train_data = data[0:(int(len(data)/2))]
			test_data = data[(int(len(data)/2)):]
			x = MLE(train_data.tolist(), test_data.tolist(), 10)
		s = s + x
		lst.append(x)
	print(_companyfile)
	print("Mean: " + str(s/float(_trials)))
	print("Vari: " + str(np.var(lst)))

# Set n to 5 or 10
n = 5

if n == 5:
	MLETrials(10, '5msft')
	MLETrials(10, '5aapl')
	MLETrials(10, '5ibm')
	MLETrials(10, '5jpm')
elif n == 10:
	MLETrials(10, '10msft')
	MLETrials(10, '10aapl')
	MLETrials(10, '10ibm')
	MLETrials(10, '10jpm')

#END: OWN CODE
