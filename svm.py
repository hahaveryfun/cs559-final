# Start our code
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from cvxopt import matrix,solvers

#number of features
n=10
n_classes=8
def getData(file):
        data = np.loadtxt(file, dtype=float, delimiter=',')
        for i in range(0,len(data)):
                #classify data
                if (data[i][0]<-10):
                        data[i][0]=0
                elif (data[i][0]>-10 and data[i][0]<-5):
                        data[i][0]=1
	        elif (data[i][0]>-5 and data[i][0]<-1):
		        data[i][0]=2
	        elif (data[i][0]>-1 and data[i][0]< 0):
		        data[i][0]=3
	        elif (data[i][0]>0 and data[i][0]<1):
		        data[i][0]=4
	        elif (data[i][0]>1 and data[i][0]<5):
		        data[i][0]=5
                elif(data[i][0]>5 and data[i][0]<10):
                        data[i][0]=6
	        elif (data[i][0]>10):
		        data[i][0]=7        
        spread = np.zeros(n_classes)
        for i in range(0,len(data)):
                spread[int(data[i][0])]+=1
                
        for i in range(0,n_classes):
                print ("Class "+str(i)+" has "+str(spread[i]))

        return np.random.permutation(data)

def svmOVA(train_data,test_data):
        predM = np.zeros((len(test_data),n_classes))
        # make an svm for each class
        for c in range(n_classes):
                print "Creating svm for class "+ str(c)
                #SVM code
                # z tell if sample should belong above or bellow the line
                z=[]
                for d in train_data:
                        if (d[0]==c):
                                z.append(1)
                        else:
                                z.append(-1)
                z = np.ravel(np.array(z))
                X=train_data[:,1:]
                n_samples, n_features = X.shape
                
                print "creating gram matrix with linear kernal"
                # Gram matrix
                #using linear kernal
                K = np.matmul(X,X.T)
                # End our code
                # Start their code
                P = matrix(np.outer(z,z) * K)
                z=z[np.newaxis]
                q = matrix(-1*np.ones(n_samples))
                G = matrix(-1*np.diag(np.ones(n_samples)))
                h = matrix(np.zeros(n_samples))
                A = matrix(z,tc='d')
                b = matrix(0.0)
                
                print "Begining convex optimization"
                sol = solvers.qp(P,q,G,h,A,b)
                print "Found support vectors for svm"
                a = np.ravel(sol['x'])
                # End their code
                # Start Our code
                #Choose this threshold randomly dont know what I should change it too
                #Allow 20 support vectors
                #threshold = a[np.argsort(a)][n_samples-20-1]
                threshold = 1e-5
                #normalizing before applying threshold since a is a very large number
                sv_t =a > threshold
                #sv_t= ((a-mean)/var) > 1e-5
                # End Our code
                ind = np.arange(len(a))[sv_t]
                a=a[sv_t]
                print "The amount of support vectors is "+str(len(a))
                sv=X[sv_t]
                sv_y=z.T[sv_t]
                
                nsv = len(a)
                #print "starting to calculate y intercept"
                #y intercept is zero because that seems to be working best for our code
                b=0.0
                #temp2=a*sv_y
                #for n in range(nsv):
                #        b+=sv_y[n]
                #        temp =K[ind[n],sv_t]
                #        b-=np.sum(temp2*temp)
                #b/= len(a)
                #print "found y intercept"
                #Weight vector
                #linear kernal
                print "using svm to predict samples"
                # Start their code
                w = np.zeros(n_features)
                for n in range(nsv):
                        w += a[n] * sv_y[n] * sv[n]
                y_predict = np.dot(X,w)
                # End their code
                # Start our code
                #b=-1*(np.amax(y_predict)+np.amin(y_predict))/2.
                for j in range(len(y_predict)):
                        predM[j][c]=np.sign(y_predict[j]+b)
        correct = 0
        wrong = 0

        svmA = np.zeros(n_classes)
        for i in range(len(test_data)):
                # todo might be better to get weight vector and y intercept instead of prediction
                # assign to closest svm
                prediction=[]
                for j in range(len(predM[i])):
                        if (predM[i][j]==1 and test_data[i][0]==j):
                                prediction.append(j)
                                svmA[j]+=1
                        elif (predM[i][j]==-1 and test_data[i][0]!=j):
                                svmA[j]+=1
                if (len(prediction)!=1):
                        wrong+=1
                        continue
                prediction=prediction[0]
                #prediction = np.argmax(predM[i])
                if (test_data[i][0]==prediction):
                        correct+=1
                else:
                        wrong+=1
                        
        #print "wrong is " + str(wrong)
        total = correct+wrong
        accuracy=(correct)/float(total)
        print "accuary is " + str(accuracy)
        for i in range(len(svmA)):
                a = svmA[i]/float(total)
                print "accuracy of svm for class "+str(i)+ " is " +str(a)
        return accuracy
n=10

data = getData('aapl')
s1=[]
for i in range(n):
        train_data=data[0:len(data)/2,]
        test_data=data[0:len(data)/2,]
        s1.append(svmOVA(train_data,test_data))
        data=np.random.permutation(data)

data = getData('msft')
s2=[]
for i in range(n):
        train_data=data[0:len(data)/2,]
        test_data=data[0:len(data)/2,]
        s2.append(svmOVA(train_data,test_data))
        data=np.random.permutation(data)

data = getData('ibm')
s3=[]
for i in range(n):
        train_data=data[0:len(data)/2,]
        test_data=data[0:len(data)/2,]
        s3.append(svmOVA(train_data,test_data))
        data=np.random.permutation(data)

data = getData('jpm')
s4=[]
for i in range(n):
        train_data=data[0:len(data)/2,]
        test_data=data[0:len(data)/2,]
        s4.append(svmOVA(train_data,test_data))
        data=np.random.permutation(data)

m1=np.mean(s1)
v1=np.var(s1)
print "aapl mean= "+str(m1)+" var="+str(v1)
m2=np.mean(s2)
v2=np.var(s2)
print "msft mean= "+str(m2)+" var="+str(v2)
m1=np.mean(s3)
v1=np.var(s3)
print "ibm mean= "+str(m3)+" var="+str(v3)
m1=np.mean(s4)
v1=np.var(s4)
print "jpm mean= "+str(m4)+" var="+str(v4)
# End our Code
