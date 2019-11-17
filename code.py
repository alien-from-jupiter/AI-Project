import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score #works

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import numpy as np
import matplotlib.pyplot as plt

import random
#from random import seed
#from random import randint
# seed random number generator


data = pd.read_csv("Training.csv")
#data.head()
#data.columns
#len(data.columns)
#len(data['prognosis'].unique())
df = pd.DataFrame(data)
#df.head()
#len(df)
cols = df.columns
cols = cols[:-1]
#cols
#len(cols)
x = df[cols]
y = df['prognosis']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

mnb = MultinomialNB()
mnb = mnb.fit(x_train, y_train)
mnb.score(x_test, y_test)

'''print ("cross result========")
scores = cross_validation.cross_val_score(mnb, x_test, y_test, cv=3)
print (scores)
print (scores.mean())'''
test_data = pd.read_csv("Testing.csv")
#test_data.head()
testx = test_data[cols]
testy = test_data['prognosis']
mnb.score(testx, testy)
#dt.__getstate__()
dt = DecisionTreeClassifier(criterion = "entropy", random_state = 42)
dt=dt.fit(x_train,y_train)

importances = dt.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols
for f in range(10):
   print("%d. feature %d - %s (%f)" % (f + 1, indices[f], features[indices[f]] ,importances[indices[f]]))

feature_dict = {}
for i,f in enumerate(features):
    feature_dict[f] = i
#feature_dict['redness_of_eyes']

r = random.randrange(0,len(testx)+1,1)
print(r+2)
sample_x = testx.iloc[r,:].values

#print(testy.iloc[r,1])

#sample_x = [i/52 if i ==52 else 1 for i in range(len(features))]
#print(len(sample_x))
sample_x = np.array(sample_x).reshape(1,len(sample_x))
#print(sample_x)

#print(dt.predict(sample_x))
ypred = dt.predict(sample_x)
print(ypred)
#print(dt.predict_proba(sample_x))
#print(accuracy_score(testy.iloc[r,:],ypred)*100)
d = pd.read_csv("doc.csv")

a = d.iloc[0:41,0].values
#print(a)
b = d.iloc[0:41,1].values
c = d.iloc[0:41,2].values
for i in range(0,41):
	#print(ypred)
	#print(a[0]==ypred)
	if a[i] == ypred:
		print("Consult the doctor : ",b[i])
		print("Link : ",c[i])