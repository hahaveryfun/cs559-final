import requests
import json
from datetime import datetime
import collections

url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=AAPL&apikey=FLFAVC3T1WTRQOFD'
# FLFAVC3T1WTRQOFD
url = url+"&outputsize=full"

r = requests.get(url, data={})
data = r.json()

final = {}

#create dictionary
# keys day
# value stock values at the end of each day
for item in data['Time Series (Daily)']:
	datekey = datetime.strptime(item, '%Y-%m-%d')
	final[datekey] = data['Time Series (Daily)'][item]['4. close']

#sort dictionary based on keys
od = collections.OrderedDict(sorted(final.items()))

#list to store data
lst = []

#create a list with first column the date , second  column of 
for k, v in od.items(): 
	lst.append([k, v, 0.0])

#convert stock values to percentage increases or decreases
for i in range(1, len(lst)):
	lst[i][-1] = 100.0*((float(lst[i][1]) - float(lst[i-1][1]))/float(lst[i-1][1]))

# n defines how big we want to make our features
n = 10
vectors = []
for i in range(n, len(lst)):
	newvec = []
	for j in range(1, n):
		newvec.append(lst[i-j])
	# index 0 (so, lst[i]) of newvec is the ACTUAL rate of growth this particular day
	newvec = [lst[i]] + newvec
	vectors.append(newvec)

finalvectors = []

for vector in vectors:
	temp = []
	for attribute in vector:
		#grab stock percentage from each attribute
		temp.append(attribute[-1])
		print(str(attribute[0]) + ": " + str(attribute[-1]))
	finalvectors.append(temp)
	print("----------")

#convert first columns as idenitfier of which class data belongs too
#classes 
#0 is loss
#1 is gain
for v in finalvectors:
	if (v[0]<0):
		v[0]=0
	else:
		v[0]=1

f = open("data","w")
for v in finalvectors:
	s=str(v[0])	
	for i in range(1,n):
		s+=","		
		s+=str(v[i])
	s+="\n"
	f.write(s)
		
f.close()
for item in finalvectors:
	print(item)
