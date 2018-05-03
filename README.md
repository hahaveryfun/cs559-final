# cs559-final
Final Project repository for CS 559 

## Avikshith Pilly & William Dunkerley

## data.py
#### Description:
This is the file that we download the data from using alphago.

#### Usage:
python data.py <N>
This is the variable that we use to determine the amount of features(n-1) you want each sample to have.
N is the total amount of days per sample including the label.

Additionally, within the request URL for Alpha Vantage, you can place a different ticker for a different company to request other company data. 'msft' has been provided in the example

#### Output:
Creates a file called data. It is up to you to rename to correct file like aapl or ibm.


## mle.py

#### Usage:
python mle.py <N>
This N variable can be either 5 or 10.
If n=5, it will run MLE over the examples where there are 4 features.
If n=10, it will run MLE over the examples where there are 9 features.

#### Description:
The file that does the maximum liklihood estimation.

#### Requires:
Needs these files: 5aapl,5msft,5ibm,5jpm,10aapl,10msft,10ibm,10jpm (All have been provided)

#### Output:
The accuracy of MLE classification for each of the companies: Apple, Microsoft, IBM and JP Morgan Chase at both n=5 and n=10

## svm.py
#### Description:
The file that does the support vector machine classification.

#### Usage:
python svm.py <N>

<N> can be either 5 or 10
If n=5, it will run SVM over the examples where there are 4 features.
If n=10, it will run SMV over the examples where there are 9 features.

#### Requires:
Needs these files:
5aapl,5msft,5ibm,5jpm,10aapl,10msft,10ibm,10jpm (All have been provided)

#### Output:
The accuracy of SVM classification for each of the companies: Apple, Microsoft, IBM and JP Morgan Chase
