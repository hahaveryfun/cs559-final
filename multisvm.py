import numpy as np
from sklearn import svm

data = np.loadtxt('aapl', dtype=float, delimiter=',')

n_classes=8

for i in range(0,len(data)):
        #classify data
        if (data[i][0]<-10):
                data[i][0]=0
        elif (data[i][0]<-5):
                data[i][0]=1
	elif (data[i][0]<-1):
		data[i][0]=2
	elif (data[i][0]< 0):
		data[i][0]=3
	elif (data[i][0]>10):
		data[i][0]=7
	elif (data[i][0]>5):
		data[i][0]=6
	elif (data[i][0]>1):
		data[i][0]=5
	elif (data[i][0]>0):
		data[i][0]=4
print data.shape
spread = np.zeros(n_classes)
for i in range(0,len(data)):
	spread[int(data[i][0])]+=1

for i in range(0,n_classes):
	print ("Class "+str(i)+" has "+str(spread[i]))

data = np.random.permutation(data);

train_data = data[0:len(data)/2,]
test_data = data[len(data)/2:,]


lin = svm.LinearSVC()

lin.fit(train_data[:,1:],train_data[:,0:1])

pred = lin.predict(test_data[:,1:])

classM = np.zeros((n_classes,n_classes))

for i in range(len(test_data)):
        p = int(pred[i])
        a = int(test_data[i,0])
        classM[a][p]+=1

print classM

print(lin.score(test_data[:,1:],test_data[:,0:1]))
