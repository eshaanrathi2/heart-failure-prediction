# -*- coding: utf-8 -*-
"""
**Application of Machine Learning Classification algorithms on heart failure clinical records data** <br><br>

link to colab:

https://colab.research.google.com/drive/1fkbDp6XX-vJtpG8BXNVuEdlLKtxog9i8?usp=sharing

#Structure of work:

* Load dataset and libraries in environment

* First look at the dataset

* Dataset exploration and visualisation

* Feature selection

* Machine learning modelling and classification

* Conclusions and learning outcomes

* References

#A brief background of dataset and the project:
The dataset consists of several features on which Heart failure depends, and depending on those features, it tells that whether the person suffered fatality or not.
The following work aims to understand the dataset, relation between various features and Death_Event, and use Machine learning algorithm(s) to predict whether a person will die or not, given a set of entries for those attributes.<br>

Dataset source : https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records

# Load dataset and libraries in environment
"""

#We stored the dataset in our Google drive. Will be mounting our G-drive to use it.
from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd #for dataframe manipulation
import matplotlib.pyplot as plt #for figures
# %matplotlib inline
#for figures

import numpy as np #n-d array manipulation

import plotly.express as px #for plots
import plotly.graph_objects as go # for plots
import plotly.io as pio
pio.renderers
pio.renderers.default = "jpg"

import seaborn as sns #for plots

from sklearn import preprocessing #for preprocessing
import statsmodels.api as sm #for feature selection
from sklearn.feature_selection import SelectKBest, f_classif #for feature selection

#machine learning dependencies:
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

#dependencies for ORCA and colab
!pip install plotly>=4.0.0
!wget https://github.com/plotly/orca/releases/download/v1.2.1/orca-1.2.1-x86_64.AppImage -O /usr/local/bin/orca
!chmod +x /usr/local/bin/orca
!apt-get install xvfb libgtk2.0-0 libgconf-2-4

"""<br> We now load dataset and store it in a dataframe called "data".<br><br>"""

data = pd.read_csv('/content/drive/MyDrive/ids-project/heart.csv')

"""#First look at the dataset

**Concise summary of the Dataset:** <br><br>
"""

data.info()

"""<br> We observe that there are 299 tuples, each of 13 attributes in the dataset. 10 attributes have data of int64 type and 3 have that of float64 type. Moreover, none of the entries are null.

<br>Meanings, measurement units, and intervals of each feature of the dataset (given by original donors of the dataset) :
"""

from IPython.display import Image
Image(filename="/content/drive/MyDrive/ids-project/table_meanings.png")

"""<br><br>Overview of the first 5 records in the dataset:<br><br>"""

data.head()

"""<br><br>Overview of the last 5 records in the dataset:"""

data.tail()

"""<br><br>Descriptive statistics of data: (i.e. count, central tendency, quartiles, max and min)"""

data.describe()

"""<br><br>**Q. Are there any duplicate records (rows)? If found, should they be removed?**

They will take space and consume time while running the algorithms later. Hence if found, we will remove them.
"""

duplicate_rows = data[data.duplicated()]
print("number of duplicate rows: ", duplicate_rows.shape[0])

"""No duplicate rows were found in the dataset.

# Dataset exploration and visualisation

Q. What is the ratio of sujects who died to the total subjects evaluated?<br><br>
"""

fig = px.pie(data, names='DEATH_EVENT', title='Distributon of death event in subjects', width = 500, height=500)
fig.show(renderer="svg")

"""There's an imbalance in the distribution of DEATH_EVENT. Fatal and Non-fatal cases are roughly in 1:2 ratio.

#Now we look at various features, their inter-relation and impact on target(i.e. DEATH_EVENT)

Q. What is the distribution of subjects with their age? How many died at a particular age?<br><br>
"""

fig = px.histogram(data, title='distribution of age with DEATH_EVENT', x="age", color="DEATH_EVENT", nbins = 50,width=800,height=500)
fig.show(renderer="svg")

"""<br>Q. is there any anomaly present in the distribution of age of subjects? If yes, should it be removed?"""

fig = px.box(
    data, 
    x="DEATH_EVENT", 
    y="age", 
    points='all',
    title='box plot of age and DEATH_EVENT',
    width=400,
    height=600    
)

fig.show(renderer="svg")

"""We observe that the quartiles of age for DEATH_EVENT = 1 are higher than DEATH_EVENT = 0. Moreover, it lies away from the cluster of the corresponding observations. We remove this outlier from the dataset."""

print("shape of data frame with anomaly: ", data.shape)
data = data.drop(data[(data['age'] > 85) & (data['DEATH_EVENT'] == 0)].index)
print("shape of data frame after removing anomaly: ", data.shape)

"""<br>Q. What is the ratio of the subjects who had anaemia?<br><br>"""

fig = px.pie(data, names='anaemia', title='Distributon of anaemia in subjects', width = 400, height=400, hole = .4)
fig.show(renderer="svg")

"""Q. How many died out of those who had anaemia? How many didn't? What can be inferred?<br><br>"""

anaemia_p = data[data["anaemia"]==1]
anaemia_n = data[data["anaemia"]==0]

anaemia_p_non_f = anaemia_p[data["DEATH_EVENT"]==0]
anaemia_p_f = anaemia_p[data["DEATH_EVENT"]==1]
anaemia_n_non_f = anaemia_n[data["DEATH_EVENT"]==0]
anaemia_n_f = anaemia_n[data["DEATH_EVENT"]==1]

VALUES = [len(anaemia_p_non_f), len(anaemia_p_f), len(anaemia_n_non_f), len(anaemia_n_f)]
LABELS = ["anaemia positive & non-fatal", "anaemia positive & fatal", "anaemia negative & non-fatal", "anaemia negative & fatal"]

fig = px.pie(data, values=VALUES, names=LABELS, title='Distribution of anaemia with DEATH_EVENT in subjects', width = 500, height=400, hole = .4)
fig.show(renderer="svg")

"""We observe that fatality vs non-fatality ratio was found to be slightly higher in anaemia postive subjects than anaemia negative subjects. Therefore, anaemia can be a good feature for classifying DEATH_EVENT. We will verify this later.<br><br>

<br><br>Q. What is the distribution of subjects with creatinine_phosphokinase content? How many died at a particular level of creatinine_phosphokinase content?<br><br>
"""

fig = px.histogram(data, title='distribution of creatinine_phosphokinase with DEATH_EVENT', x="creatinine_phosphokinase", color="DEATH_EVENT", nbins=50, width=1000,height=500)
fig.show(renderer="svg")

"""Q. is there any anomaly present in the distribution of creatinine_phosphokinase in subjects? If yes, should it be removed?<br><br>"""

fig = px.box(
    data, 
    x="DEATH_EVENT", 
    y="creatinine_phosphokinase", 
    points='all',
    title='box plot of creatinine_phosphokinase and DEATH_EVENT',
    width=600,
    height=600    
)

fig.show(renderer="svg")

"""We observe outliers in DEATH_EVENT = 1 and creatinine_phosphokinase > 7000, but we don't have robust reason to remove them. It's possible that they might tell us something critical about data. We leave them untouched.<br><br>

Q. What is the ratio of the subjects who had diabetes?<br><br>
"""

fig = px.pie(data, names='diabetes', title='Distributon of diabetes in subjects', width = 400, height=400, hole = .4)
fig.show(renderer="svg")

"""Q. How many died out of those who had diabetes? How many didn't? What can be inferred?<br><br>"""

p = data[data["diabetes"]==1]
n = data[data["diabetes"]==0]

p_non_f = p[data["DEATH_EVENT"]==0]
p_f = p[data["DEATH_EVENT"]==1]
n_non_f = n[data["DEATH_EVENT"]==0]
n_f = n[data["DEATH_EVENT"]==1]

VALUES = [len(p_non_f), len(p_f), len(n_non_f), len(n_f)]
LABELS = ["diabetes positive & non-fatal", "diabetes positive & fatal", "diabetes negative & non-fatal", "diabetes negative & fatal"]

fig = px.pie(data, values=VALUES, names=LABELS, title='Distribution of diabetes with DEATH_EVENT in subjects', width = 500, height=400, hole = .4)
fig.show(renderer="svg")

"""We observe that fatality vs non-fatality ratio was found to be similar in both the types of subjects, i.e. diabetes postive and dibetes negative.(roughly diving fatal and non-fatal subjects in 1:2 ratio in both). Therefore, diabetes doesn't seem to be a good feature for classifying DEATH_EVENT.<br><br>

<br>Q. What is the distribution of subjects with content of ejection_fraction? How many died at a particular level of ejection_fraction content?<br><br>
"""

fig = px.histogram(data, title='distribution of ejection_fraction with DEATH_EVENT', x="ejection_fraction", color="DEATH_EVENT", nbins=80, width=1000,height=500)
fig.show(renderer="svg")

"""Q. is there any anomaly present in the distribution of ejection_fraction in subjects? If yes, should it be removed?<br><br>"""

fig = px.box(
    data, 
    x="DEATH_EVENT", 
    y="ejection_fraction", 
    points='all',
    title='box plot of ejection_fraction and DEATH_EVENT',
    width=600,
    height=600    
)

fig.show(renderer="svg")

"""We observe that the ejection_fraction's Median, Q1 and Q3 are higher when Death_EVENT is 0. The record with ejection_fraction = 80 and DEATH_EVENT = 0 is is in accordance to the above observation inspite of being away from it's corresponding cluster. We don't have robust reasoning to remove this record. We keep it untouched.

Q. What is the ratio of the subjects who had high blood pressure?
"""

#high_blood_pressure
fig = px.pie(data, names='high_blood_pressure', title='Distributon of high_blood_pressure in subjects', width = 400, height=400, hole = .4)
fig.show(renderer="svg")

"""Q. How many died out of those who had high blood pressure? How many didn't? What can be inferred?"""

p = data[data["high_blood_pressure"]==1]
n = data[data["high_blood_pressure"]==0]

p_non_f = p[data["DEATH_EVENT"]==0]
p_f = p[data["DEATH_EVENT"]==1]
n_non_f = n[data["DEATH_EVENT"]==0]
n_f = n[data["DEATH_EVENT"]==1]

VALUES = [len(p_non_f), len(p_f), len(n_non_f), len(n_f)]
LABELS = ["high_blood_pressure positive & non-fatal", "high_blood_pressure positive & fatal", "high_blood_pressure negative & non-fatal", "high_blood_pressure negative & fatal"]

fig = px.pie(data, values=VALUES, names=LABELS, title='Distribution of high_blood_pressure with DEATH_EVENT in subjects', width = 600, height=400, hole = .4)
fig.show(renderer="svg")

"""We observe that fatality vs non-fatality ratio was found to be slightly higher in high_blood_pressure postive subjects than high_blood_pressure negative subjects. Therefore, anaemia can be a good feature for classifying DEATH_EVENT. We will verify this later.

Q. What is the distribution of subjects with their platelet content? How many died at a particular level of platelet content?
"""

#platelets
fig = px.histogram(data, title='distribution of platelets with DEATH_EVENT', x="platelets", color="DEATH_EVENT", nbins=80, width=1000,height=400)
fig.show(renderer="svg")

"""Q. Is there any anomaly present in the distribution of platelet content in subjects? If yes, should it be removed?"""

fig = px.box(
    data, 
    x="DEATH_EVENT", 
    y="platelets", 
    points='all',
    title='box plot of platelets and DEATH_EVENT',
    width=500,
    height=600    
)

fig.show(renderer="svg")

"""The quartiles of platelets for both DEATH_EVENT categories are almost same. The only difference in the distribution is that, DEATH_EVENT = 0 has a compact distribution (near it's box) than DEATH_EVENT = 1.

Clearly, the records with plateletsn > 700000 and DEATH_EVENT = 0 are anomalies. We can safely remove them. Similarly for the record with platelets > 600000 and DEATH_EVENT = 1.
"""

print("shape of data frame with anomaly: ", data.shape)
data = data.drop(data[(data['platelets'] > 700000) & (data['DEATH_EVENT'] == 0)].index)
data = data.drop(data[(data['platelets'] > 600000) & (data['DEATH_EVENT'] == 1)].index)
print("shape of data frame after removing anomaly: ", data.shape)

"""Q. What is the distribution of subjects with their serum_creatinine content? How many died at a particular level of serum_creatinine content?"""

#serum_creatinine
fig = px.histogram(data, title='distribution of serum_creatinine with DEATH_EVENT', x="serum_creatinine", color="DEATH_EVENT", nbins=50, width=1000,height=400)
fig.show(renderer="svg")

"""Q. Is there any anomaly present in the distribution of serum_creatinine content in subjects? If yes, should it be removed?"""

fig = px.box(
    data, 
    x="DEATH_EVENT", 
    y="serum_creatinine", 
    points='all',
    title='box plot of serum_creatinine and DEATH_EVENT',
    width=500,
    height=500    
)

fig.show(renderer="svg")

"""The box for DEATH_EVENT = 1 is more wider than that of DEATH_EVENT = 0 and has higher median and upper fences.

The record with serum_creatinine = 9.4 and DEATH_EVENT = 1 is in accordance to the above rule inspite of being abnormally away from it's cluster. We don't have enough reasoning to remove this record.
Whereas the record with serum_creatinine = 6.1 and DEATH_EVENT = 0 is anomalous. We remove it. 
"""

print("shape of data frame with anomaly: ", data.shape)
data = data.drop(data[(data['serum_creatinine'] > 6) & (data['DEATH_EVENT'] == 0)].index)
print("shape of data frame after removing anomaly: ", data.shape)

"""Q. What is the distribution of subjects with their serum_sodium content? How many died at a particular level of serum_sodium content?"""

#serum_sodium
fig = px.histogram(data, title='distribution of serum_sodium with DEATH_EVENT', x="serum_sodium", color="DEATH_EVENT", nbins=50, width=1000,height=400)
fig.show(renderer="svg")

"""Q. Is there any anomaly present in the distribution of serum_sodium content in subjects? If yes, should it be removed?"""

fig = px.box(
    data, 
    x="DEATH_EVENT", 
    y="serum_sodium",
    points='all',
    title='box plot of serum_sodium and DEATH_EVENT',
    width=500,
    height=500    
)

fig.show(renderer="svg")

"""We observe that, cluster for serum_sodium has higher values for DEATH_EVENT = 0 than DEATH_EVENT = 1. But The record with serum_sodium = 113 and DEATH_EVENT = 0 is clearly violating the above trend and it is also very far away from its cluster. We remove this anomaly."""

print("shape of data frame with anomaly: ", data.shape)
data = data.drop(data[(data['serum_sodium'] < 115) & (data['DEATH_EVENT'] == 0)].index)
print("shape of data frame after removing anomaly: ", data.shape)

"""Q. What is the ratio of the subjects who are male (or female)?"""

#sex
fig = px.pie(data, names='sex', title='Distributon of sex in subjects', width = 400, height=400, hole = .4)
fig.show(renderer="svg")

"""Q. How many died out of those who were male (or female)? How many didn't? What can be inferred?"""

p = data[data["sex"]==1]
n = data[data["sex"]==0]

p_non_f = p[data["DEATH_EVENT"]==0]
p_f = p[data["DEATH_EVENT"]==1]
n_non_f = n[data["DEATH_EVENT"]==0]
n_f = n[data["DEATH_EVENT"]==1]

VALUES = [len(p_non_f), len(p_f), len(n_non_f), len(n_f)]
LABELS = ["male & non-fatal", "male & deceased", "female & non-fatal", "female & deceased"]

fig = px.pie(data, values=VALUES, names=LABELS, title='Distribution of sex with DEATH_EVENT in subjects', width = 500, height=400, hole = .4)
fig.show(renderer="svg")

"""For male subjects, fatality and non fatality is in 1:2 ratio. This is almost same to that in female subjects. Therefore, sex doesn't seem to be a good feature to classify the DEATH_EVENT.

Q. What is the ratio of the subjects who smoked?
"""

#smoking
fig = px.pie(data, names='smoking', title='Distributon of smoking in subjects', width = 500, height=400, hole = .4)
fig.show(renderer="svg")

"""Q. How many died out of those who smoked? How many didn't? What can be inferred?"""

p = data[data["smoking"]==1]
n = data[data["smoking"]==0]

p_non_f = p[data["DEATH_EVENT"]==0]
p_f = p[data["DEATH_EVENT"]==1]
n_non_f = n[data["DEATH_EVENT"]==0]
n_f = n[data["DEATH_EVENT"]==1]

VALUES = [len(p_non_f), len(p_f), len(n_non_f), len(n_f)]
LABELS = ["smoker & non-fatal", "smoker & deceased", "non-smoker & non-fatal", "non-smoker & deceased"]

fig = px.pie(data, values=VALUES, names=LABELS, title='Distribution of smoking with DEATH_EVENT in subjects', width = 500, height=400, hole = .4)
fig.show(renderer="svg")

"""For subjects who smoke, fatality and non fatality is in 1:2 ratio. This is almost same to that in non-smoker subjects. Therefore, smoking doesn't seem to be a good feature to classify the DEATH_EVENT.

We drop 'time' attribute, since it signifies time (in days) of death or discharging. But if one has to use model for prediction, he won't be able to provide value of time, since he doesn't know when will subject get discharged in future.
"""

data = data.drop(['time'], axis=1)

"""<br><br>Brief overview of data till now"""

data.info()

"""#Feature selection

**Q. Do we need all features?**

No, we must discard (possibly none) those features who have partial (or poor) dependencies on the target (DEATH_EVENT).

**Q. Why should we remove them?**

Removing useless features would save space, will be computationally cheaper and give higher accuracy while applying classification algorithms.<br>

<br>In the above section we observed histograms, box plots, pie charts to get insights, anomalies and common trends in the data.
But we didn't strongly say that whether the feature is good enough to classify DEATH_EVENT correctly or not, because we need more reasoning for it.

We will use stronger technique(s) for that purpose.

<br>**Q. Should the features be brought on same scale(range). How?**

First we will scale each feature in [0,1] so that we can assess them on same scale. Feature scaling also speeds up the computation while running ML algorithms.
We didn't do this before, because we had to visualise how the original data looks.<br><br>
"""

cols_to_norm = data.columns[0:11].tolist()
data[cols_to_norm] = data[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

"""Overview of Data frame after scaling :"""

data.head()

"""**Q. Which features should you remove? How do you choose them?**

We employ two methods for selecting features.
<br><br>

1. **Wrapper method** : It uses a greedy approach. It makes a subset of combinations of features from all, runs dummy ML algorithm on them and iteratively (or recursively) selects optimal features us. Performance metric used is pvalue.

2. **Select K best (sklearn)** : It internally uses functions like anova, chi2, mutual_info_regression etc and select features according to the k highest scores. Since we are dealing with numerical data which has continuous values for a field and we have multiple features in the data set, we can use **ANOVA**.

Wrapper method :
"""

X = data.iloc[:,0:11]  
y = data.iloc[:,-1]
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_max)
    else:
        break
print(cols)

"""Select K best :"""

best = SelectKBest(score_func=f_classif, k=10) #f_classif computes the ANOVA F-value for the provided sample
fit = best.fit(X,y)

scores = pd.DataFrame(fit.scores_)
col = pd.DataFrame(X.columns)

feature = pd.concat([col,scores],axis=1)
feature.columns = ['Feature','Score']  #naming the dataframe columns
print(feature.nlargest(10,'Score'))

"""From both of the above methods we see that serum_creatinine, ejection_fraction and age are best possible features to classify DEATH_EVENT.

#Machine learning modelling and classification

Train test split :
"""

features = ["serum_creatinine", "ejection_fraction", "age"]
x = data[features]
y = data["DEATH_EVENT"]
acc = []
acc_t = []
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25)

"""<br>Helper function for plotting confusion matrix :<br><br>"""

def plot_cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')

"""<br><br>
Decision tree classifier:

* Q. What?

 It is a tree based algorithm which splits the source set, constituting the root node of the tree, into subsets—which constitute the successor children. The splitting is based on a set of splitting rules based on classification features.

* Q. Why?

 Because it can solve classification problem. Its easy to employ and one of the foremost ML algorithms which is computationally cheap as well.

* Q.How?

 We use sklearn.tree module to build the classifier.
"""

model = DecisionTreeClassifier(max_depth = 3, criterion='entropy', max_leaf_nodes=3)
model.fit(train_x, train_y)
pred = model.predict(test_x)
acc_ = accuracy_score(test_y, pred)*100
acc.append(acc_)
acc_t_ = model.score(train_x, train_y)*100
acc_t.append(acc_t_)
print("train accuracy with Decision Tree Classifier = ", acc_t_)
print("test accuracy with Decision Tree Classifier = ", acc_)

"""Confusion matrix for DecisionTreeClassifier :"""

plot_cm(test_y, pred)

"""<br><br>
Random forest :

* Q. What?

 It employs multiple decision trees and combines them into one ensemble model.

* Q. Why?

 Because it can solve classification problem. It does not rely on the feature importance given by a single decision tree, it rather looks into multiple decison trees. Therefore gives better results compared to a decison tree.

* Q.How?

 We use sklearn.ensemble class to build the classifier.
"""

model = RandomForestClassifier()
model.fit(train_x, train_y)
pred = model.predict(test_x)
acc_ = accuracy_score(test_y, pred)*100
acc.append(acc_)
acc_t_ = model.score(train_x, train_y)*100
acc_t.append(acc_t_)
print("train accuracy with Random Forest Classifier = ", acc_t_)
print("test accuracy with Random Forest Classifier = ", acc_)

"""Confusion matrix for Random Forest Classifier"""

plot_cm(test_y, pred)

"""<br><br>
K nearest neighbours classifier:

* Q. What?

 The algorithm assumes that similar things exist in close proximity. In other words, similar things are near to each other.

* Q. Why?

 Because it can solve classification problem and works good enough for fewer features.

* Q.How?

 We use sklearn.neighbors module to build the classifier. We will look for ideal value of k say from [1,20].
"""

training_accuracy = []
test_accuracy = []
neighbors_settings = range(1, 20)
for n_neighbors in neighbors_settings:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(train_x,train_y)
    training_accuracy.append(knn.score(train_x,train_y))
    test_accuracy.append(knn.score(test_x,test_y))
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.savefig('knn_compare_model')

"""Accuracies diverge from k = 5, we can choose this value for k , for this split of data."""

model = KNeighborsClassifier(n_neighbors=5)
model.fit(train_x, train_y)
pred = model.predict(test_x)
acc_ = accuracy_score(test_y, pred)*100
acc.append(acc_)
acc_t_ = model.score(train_x, train_y)*100
acc_t.append(acc_t_)
print("train accuracy with K nearest neighbours Classifier = ", acc_t_)
print("test accuracy with K Nearest Neighbors Classifier = ", acc_)

"""Confusion matrix for Random Forest Classifier"""

plot_cm(test_y, pred)

"""<br><br>

Naive Bayes classifier :

* Q. What?
 It is a probabilistic classifier based on Bayes' theorem with strong (naïve) independence assumptions between the features.

* Q. Why?

 Because it can solve classification problem and works good for small dataset like this and small number of features.

* Q.How?

 We use sklearn.naive_bayes module to build the classifier.
"""

model = GaussianNB()
model.fit(train_x, train_y)
pred = model.predict(test_x)
acc_ = accuracy_score(test_y, pred)*100
acc.append(acc_)
acc_t_ = model.score(train_x, train_y)*100
acc_t.append(acc_t_)
print("train accuracy with Naive Bayes = ", acc_t_)
print("test accuracy with Naive Bayes classifier = ", acc_)

"""Confusion matrix for naive bayes"""

plot_cm(test_y, pred)

"""<br><br>

Support vector machine classification:

* Q. What?
 This model (linear in nature) is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible.

* Q. Why?

 Because it can solve classification problem and works good for numerical data type, like this dataframe.

* Q.How?

 We use sklearn.svm module to build the classifier.
"""

model = SVC()
model.fit(train_x, train_y)
pred = model.predict(test_x)
acc_ = accuracy_score(test_y, pred)*100
acc.append(acc_)
acc_t_ = model.score(train_x, train_y)*100
acc_t.append(acc_t_)
print("train accuracy with Support vector machine Classifier = ", acc_t_)
print("test accuracy with Support vector machine classifier = ", acc_)

"""Confusion matrix for Support vector machine classifier"""

plot_cm(test_y, pred)

"""**Comparisions of Machine learning algorithms used:**"""

classifiers=["Decision Tree", "Random Forest", "KNN", "Naive Bayes", "SVM"]
acc_graph = pd.DataFrame({
  "classifiers": classifiers,
  "acc_t": acc_t,
})
plt.figure(figsize=(10, 7))
splot=sns.barplot(x='classifiers',y='acc_t',data=acc_graph,ci=None)
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.1f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
plt.xlabel("Machine Learning classification algorithm", size=14)
plt.ylabel("accuracy on train set", size=14)
plt.show()

classifiers=["Decision Tree", "Random Forest", "KNN", "Naive Bayes", "SVM"]
acc_graph = pd.DataFrame({
  "classifiers": classifiers,
  "acc": acc,
})
plt.figure(figsize=(10, 7))
splot=sns.barplot(x='classifiers',y='acc',data=acc_graph,ci=None)
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.1f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
plt.xlabel("Machine Learning classification algorithm", size=14)
plt.ylabel("accuracy on test set", size=14)
plt.show()

"""<br><br>

We observe that for train set, Random forest gave best results, whereas for test set all algorithms gave almost same accuracies.
We know that Random forest combines multiple decision trees and are henceforth give promising results.

#Conclusions and learning outcomes

* We were able to understand the features in Heart failure clinical records Data Set, it's shape and datatypes.
* We learnt about distribution of values in various features, their central tendencies, deviations and anomalies. 
* We were able to remove few anomalies and cleaned the data.
* We carried out feature selection via two methods.
* We were able to use several Machine Learning algorithms and understood which worked best for this data.

#References

https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

https://scikit-learn.org/stable/modules/naive_bayes.html

https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html

https://scikit-learn.org/stable/modules/feature_selection.html

https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=random%20fores#sklearn.ensemble.RandomForestClassifier

https://github.com/brpy/colab-pdf

https://plotly.com/python/pie-charts/

https://plotly.com/python/bar-charts/

https://plotly.com/python/histograms/

https://seaborn.pydata.org/generated/seaborn.barplot.html

#For downloading this colab notebook :
"""

!apt-get install inkscape

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# #!wget -nc https://raw.githubusercontent.com/brpy/colab-pdf/master/colab_pdf.py
# #from colab_pdf import colab_pdf
# colab_pdf('IDS-Report.ipynb')
