# C:\workspaces\Datascientest\Streamlit\Cours\Titanic

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn import linear_model
from sklearn import ensemble
from sklearn.metrics import confusion_matrix


df=pd.read_csv("train.csv")

st.title("Projet de classification binaire Titanic")
st.sidebar.title("Sommaire")
pages=["Exploration", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0] : 
    st.write("### Introduction")

    st.dataframe(df.head(10))
    st.write(df.shape)
    st.dataframe(df.describe())

    if st.checkbox("Afficher les NA") :
        st.dataframe(df.isna().sum())

if page == pages[1] :
  st.write("### DataVizualization")

  fig = plt.figure()
  sns.countplot(x = 'Survived', data = df)
  st.pyplot(fig)

  fig = plt.figure()
  sns.countplot(x = 'Sex', data = df)
  plt.title("Répartition du genre des passagers")
  st.pyplot(fig)

  fig = plt.figure()
  sns.countplot(x = 'Pclass', data = df)
  plt.title("Répartition des classes des passagers")
  st.pyplot(fig)

  fig = sns.displot(x = 'Age', data = df)
  plt.title("Distribution de l'âge des passagers")
  st.pyplot(fig)

  fig = plt.figure()
  plt.title("Survie en fonction du sexe")
  sns.countplot(x = 'Survived', hue='Sex', data = df)
  st.pyplot(fig)

  fig = sns.catplot(x='Pclass', y='Survived', data=df, kind='point')
  plt.title("Survie en fonction de la classe")
  st.pyplot(fig)

  fig = sns.lmplot(x='Age', y='Survived', hue="Pclass", data=df)
  plt.title("Survie en fonction de l'âge et de la classe")
  st.pyplot(fig)

  ma_palette = sns.light_palette("seagreen", 4)
  fig, ax = plt.subplots()
  df2 = df[["Survived","Pclass","Sex","Age","SibSp","Parch","Fare"]]
  df2['Sex'] = df2['Sex'].replace({'male': 1, 'female': 0})
  sns.heatmap(df2.corr(), annot=True,cmap=ma_palette, ax=ax)
  plt.title("Matrice de corrélation")
  st.write(fig)

if page == pages[2] : 
  st.write("### Modélisation")

  df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
  y = df['Survived']
  X_cat = df[['Pclass', 'Sex',  'Embarked']]
  X_num = df[['Age', 'Fare', 'SibSp', 'Parch']]

  for col in X_cat.columns:
    X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
  for col in X_num.columns:
    X_num[col] = X_num[col].fillna(X_num[col].median())
    X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)
    X = pd.concat([X_cat_scaled, X_num], axis = 1)

  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
  X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])

  def prediction(classifier):
    if classifier == 'Random Forest':
        clf = ensemble.RandomForestClassifier()
    elif classifier == 'SVC':
        clf = svm.SVC()
    elif classifier == 'Logistic Regression':
        clf = linear_model.LogisticRegression()
    clf.fit(X_train, y_train)
    return clf
  
  def scores(clf, choice):
    if choice == 'Accuracy':
        return clf.score(X_test, y_test)
    elif choice == 'Confusion matrix':
        return confusion_matrix(y_test, clf.predict(X_test))
        # pd.crosstab(y_test,clf.predict(X_test),rownames=['Realité'],colnames=['Prédiction'])
    
  choix = ['Random Forest', 'SVC', 'Logistic Regression']
  option = st.selectbox('Choix du modèle', choix)
  st.write('Le modèle choisi est :', option)

  clf = prediction(option)
  display = st.radio('Que souhaitez-vous montrer ?', ('Accuracy', 'Confusion matrix'))
  if display == 'Accuracy':
    st.write(scores(clf, display))
  elif display == 'Confusion matrix':
    st.dataframe(scores(clf, display))