#given a set of documents, this code groups documents that have similar content
#movie review data set has been used for the purpose
#-------------------------------------------

import numpy as np
import nltk
import re
import copy
import math
import collections
import random
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
stemmer=SnowballStemmer("english")
Max=9999
cnum=3
data=[]
allwords=[]
term_freq_matrix=[]
tf_idf=[]
cluster_data=[]
centroids=[]
initial_head=[]
stop_words=set(stopwords.words("english"))


def tokenize_and_stem(text):
	tokens=[word for sent in nltk.sent_tokenize(text.decode('utf-8')) for word in nltk.word_tokenize(sent) if word not in stop_words]
	filtered_tokens=[]
	for token in tokens:
		if re.search('[a-zA-Z]',token):
			filtered_tokens.append(token.lower())
	stems=[stemmer.stem(t) for t in filtered_tokens]
	return stems

def create_term_freq_matrix():
	for ele in data:
		lis=[]
		temp=collections.Counter(ele)
		for x in allwords:
			if x in temp:
				lis.append(temp[x])
			else:
				lis.append(0)
		term_freq_matrix.append(lis)

def create_tfidf():
	num_doc_withthis_term={}
	for ele in allwords:
		counter=0
		for x in data:
			if ele in x:
				counter+=1
		num_doc_withthis_term[ele]=counter
	for ele in term_freq_matrix:
		lis=[]
		for i in range(len(allwords)):
			lis.append(ele[i]*math.log(len(data)/float(num_doc_withthis_term[allwords[i]])))                             # formulae: tf* log(no of doc/no of doc that contains the term)
		tf_idf.append(lis)

def cosine_sim(lis1,lis2):
	nu=0
#	print lis1
#	print lis2
	for i in range(len(lis1)):
		nu+=lis1[i]*lis2[i]
	a=math.sqrt(sum([i**2 for i in lis1]))
	b=math.sqrt(sum([i**2 for i in lis2]))
	return nu/(a*b)


def update_clusters():
	for i in range(len(cluster_data)):
		m=Max
		newcluster=0
		for j in range(cnum):
			sim=cosine_sim(cluster_data[i][0],centroids[j][0])
			if(sim<m):
				m=sim
				newcluster=j
		cluster_data[i][1]=newcluster

def recalculate_centroid():
	for j in range(cnum):
		x=[[0 for i in range(len(cluster_data[0][0]))],j]
		totalincluster=0
		for k in range(len(cluster_data)):
			if(cluster_data[k][1]==j):
				for i in range(len(cluster_data[0][0])):
					x[0][i]+=cluster_data[k][0][i]
					totalincluster+=1
		if(totalincluster>0):
			centroids[j][0]=[ele/totalincluster for ele in x[0]]

def Kmeans(num):
	cnum=num
	global initial_head
	count=0
	initial_head=[random.randrange(len(tf_idf)) for i in range(cnum)]
	for i in range(len(tf_idf)):
		if i in initial_head:
			centroids.append([tf_idf[i],count])
			cluster_data.append([tf_idf[i],count])
			count+=1
		else:
			cluster_data.append([tf_idf[i],None])

	c1=copy.copy(centroids[0])
	c2=copy.copy(centroids[1])
	c3=copy.copy(centroids[2])
	update_clusters()
	recalculate_centroid()
	
	while((c1!=centroids[0] or c2!=centroids[1]) or c3!=(centroids[2])):
		c1=centroids[0]
		c2=centroids[1]
		c3=centroids[2]
		recalculate_centroid()
		update_clusters()

	
		

f=open("input_data.txt")
text=f.read()
for para in text.split("BREAKS HERE"):
	temp=tokenize_and_stem(para)
	if temp:
		data.append(temp)
data=data[:50]
for lis in data:
	for ele in lis:
		if ele not in allwords:
			allwords.append(ele)


create_term_freq_matrix()
create_tfidf()
Kmeans(3)

print "cluster 0"	
for i in range(len(cluster_data)):
	if cluster_data[i][1]==0:
		print 'document: ',i

print "cluster 1"	
for i in range(len(cluster_data)):
	if cluster_data[i][1]==1:
		print 'document: ',i


print "cluster 2 includes"	
for i in range(len(cluster_data)):
	if cluster_data[i][1]==2:
		print 'document: ',i
