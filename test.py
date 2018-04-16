import requests
import json
from datetime import datetime
import collections

url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=MSFT&apikey=FLFAVC3T1WTRQOFD'
# FLFAVC3T1WTRQOFD
url = url+"&outputsize=full"

r = requests.get(url, data={})
data = r.json()

final = {}

for item in data['Time Series (Daily)']:
	datekey = datetime.strptime(item, '%Y-%m-%d')
	final[datekey] = data['Time Series (Daily)'][item]['4. close']

od = collections.OrderedDict(sorted(final.items()))

lst = []

for k, v in od.items(): 
	lst.append([k, v, 0.0])

for i in range(1, len(lst)):
	lst[i][-1] = 100.0*((float(lst[i][1]) - float(lst[i-1][1]))/float(lst[i-1][1]))

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
		temp.append(attribute[-1])
		print(str(attribute[0]) + ": " + str(attribute[-1]))
	finalvectors.append(temp)
	print("----------")

for item in finalvectors:
	print(item)