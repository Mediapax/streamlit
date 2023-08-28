# ré-échantillonage

import streamlit as st
from PIL import Image
import re

import pandas as pd
import numpy as np
from joblib import load

import matplotlib.pyplot as plt


def reechantillonage():
    st.title('Ré-échantillonage') 

    # chemin d'accès aux modèles enregistrés
    filename_path = './models/resampling/'

    # introduction
    st.markdown("Nous avons observé que notre variable cible présente une distribution déséquilibrée (~80/20). Il est donc naturel d'envisager un léger ré-équilibrage par sous-échantillonnage ou par sur-échantillonnage.")

    st.markdown("Dans le cadre de cette étude, deux méthodes de ré-échantillonnage seront considérées:")
    st.markdown("1. `RandomUnderSampler`")
    st.markdown('2. `RandomOverSampler`')
    
    st.markdown("Plusieurs modèles de classifications sont également testés:")
    st.markdown("1. `LogisticRegression`")
    st.markdown('2. `KNeighborsClassifier`')
    st.markdown('3. `DecisionTreeClassifier`')
    st.markdown('4. `RandomForestClassifier`')

    # résumé des résultats
    st.subheader("Résumé des résultats obtenus:")
    image = Image.open(filename_path+'results_summary.png')
    st.image(image)

    # Observation de l'influence du taux de ré-échantillonage:
    # - sur le nombre de lignes
    # - sur les métriques
    
    st.divider()
    st.subheader("Impact du taux de ré-échantillonage sur le modèle")
    st.markdown("Nous étudions ici l'impact du taux de ré-échantillonnage sur les différentes métriques (pour la classe 1).")
    st.markdown("Pour des raisons de temps de cacul, l'étude est ici faite avec une simple `LogisticRegression`. Nous avons vu précédemment que les autres modèles n'améliorent pas significativement les scores malgré des temps de calculs bien plus conséquents.")

    st.markdown("**Affichage des résultats**")

    undersampler_study, oversampler_study = load(filename_path+"resampling_study.dataframe")

    fig, axs = plt.subplots(2,2,figsize=(10,9), sharex=False, sharey=False)
    
    ax = axs[0,0]
    ax.plot(undersampler_study.ratio, undersampler_study.nb_obs)
    ax.set_title("Undersampling")
    ax.set_xlabel("Ratio")
    ax.set_ylabel("Nombre de lignes")
    
    ax = axs[0,1]
    ax.plot(oversampler_study.ratio, oversampler_study.nb_obs)
    ax.set_title("Oversampling")
    ax.set_xlabel("Ratio")
    ax.set_ylabel("Nombre de lignes")
    
    ax = axs[1,0]
    ax.plot(undersampler_study.ratio, undersampler_study.precision, label='Precision')
    ax.plot(undersampler_study.ratio, undersampler_study.f1_score, label='F1')
    ax.plot(undersampler_study.ratio, undersampler_study.recall, label='Recall')
    ax.set_title('Scores en fonction du taux de ré-échantillonage')
    ax.set_xlabel('Undersampling ratio')
    ax.set_ylabel('Score')
    ax.set_ylim(0.5,1)
    ax.legend()
    
    
    ax = axs[1,1]
    ax.plot(oversampler_study.ratio, oversampler_study.precision, label='Precision')
    ax.plot(oversampler_study.ratio, oversampler_study.f1_score, label='F1')
    ax.plot(oversampler_study.ratio, oversampler_study.recall, label='Recall')
    ax.set_title('Scores en fonction du taux de ré-échantillonage')
    ax.set_xlabel('Oversampling ratio')
    ax.set_ylabel('Score')
    ax.set_ylim(0.5,1)
    ax.legend()
    
    fig.tight_layout()

    st.pyplot(fig)
    
    st.markdown("**Commentaire**")
    st.markdown("Nous obtenons des résultats similaires avec les méthodes `RandomUnderSampler` et `RandomOverSampler`. Pour des raisons de temps de calculs, nous préfèrerons donc nous concentrer sur l'under-sampling si besoin est.")
    
