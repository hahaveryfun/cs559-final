import numpy as np
from sklearn import svm

data = np.loadtxt('msft', dtype=float, delimiter=',')

n_classes=2
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


print(lin.score(test_data[:,1:],test_data[:,0:1]))
