import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from cvxopt import matrix,solvers

data = np.loadtxt('data', dtype=float, delimiter=',')

#number of features
n=10
n_classes=2
spread = np.zeros(n_classes)
for i in range(0,len(data)):
	spread[int(data[i][0])]+=1

for i in range(0,n_classes):
	print ("Class "+str(i)+" has "+str(spread[i]))

data = np.random.permutation(data);

#use kernal trick with kernal equal to (xy)^2
#since we are not actualy inside svm we need to calculate the individual parts of the kernal
def kernal(x,y):
        return np.dot(x,y)
train_data = data[0:len(data)/2,]
test_data = data[len(data)/2:,]

#SVM code
z=[]
for d in train_data:
        #
        if (d[0]==0):
                z.append(-1)
        else:
                z.append(1)
z = np.ravel(np.array(z))
X=train_data[:,1:]
#not our Code
n_samples, n_features = X.shape

# Gram matrix
K = np.zeros((n_samples, n_samples))
for i in range(n_samples):
        for j in range(n_samples):
                K[i,j] = kernal(X[i], X[j])
        
P = matrix(np.outer(z,z) * K)
z=z[np.newaxis]
q = matrix(-1*np.ones(n_samples))
G = matrix(-1*np.diag(np.ones(n_samples)))
h = matrix(np.zeros(n_samples))
A = matrix(z,tc='d')
b = matrix(0.0)

sol = solvers.qp(P,q,G,h,A,b)

a = np.ravel(sol['x'])
#Choose this threshold randomly dont know what I should change it too
threshold = 800000
sv_t= a > threshold
ind = np.arange(len(a))[sv_t]
a=a[sv_t]
sv=X[sv_t]
sv_y=z.T[sv_t]

nsv = len(a)
#y intercept
b=0.0
for n in range(nsv):
        b+=sv_y[n]
        temp =K[ind[n],sv_t]
        b-=np.sum(a*sv_y*temp)
b/= len(a)

#Weight vector
w = np.zeros(n_features)
for n in range(nsv):
        w += a[n] * sv_y[n] * sv[n]
#Not our code

prediction=np.sign(np.dot(test_data[:,1:],w)+b)
correct = 0
wrong = 0
for i in range(len(test_data)):
        
        if (prediction[i]==-1 and test_data[i][0]==0):
                correct+=1
        elif (prediction[i]==1 and test_data[i][0]==1):
                correct+=1
        else:
                wrong+=1

print "wrong is " + str(wrong)
print "accuary is " + (correct)/float(correct+wrong)
        

lin = svm.LinearSVC()


lin.fit(train_data[:,1:],train_data[:,0:1])

print(lin.score(test_data[:,1:],test_data[:,0:1]))

