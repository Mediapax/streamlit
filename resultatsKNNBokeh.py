# resultatsKNNBokeh

def resultatsKNNBokeh(faccuracy,fprecision, frecall, ff1score, fspecificity, metricsUtilisees,nbVoisinsMax):
    import streamlit as st

    from resizeImage import resizeImage, loadImage
    from PIL import Image
    from displayBackground import displayBackground

    import pandas as pd
    import numpy as np
    from numpy.lib.utils import source
    import matplotlib.pyplot as plt
    import warnings

    from bokeh.util.warnings import BokehUserWarning
    from bokeh.models import TabPanel, Tabs
    from bokeh.models import ColumnDataSource
    from bokeh.models.tools import HoverTool,LassoSelectTool
    from bokeh.models.ranges import DataRange1d
    from bokeh.colors import RGB
    from bokeh.models import LinearColorMapper
    from bokeh.models import PanTool,ResetTool,HoverTool,WheelZoomTool,SaveTool,BoxZoomTool
    from bokeh.models import Legend
    from bokeh.models import ColumnDataSource, GeoJSONDataSource, HoverTool
    from bokeh.plotting import figure
    from bokeh.layouts import row,column


    # chargement des données
    chemin_local = "./techniquesML/knn/"

    #accuracy
    path = chemin_local+faccuracy
    accuracy_v1 = pd.read_csv(path,sep=',',index_col=0)

    #precision_v1.0.csv
    path = chemin_local+fprecision
    precision_v1 = pd.read_csv(path,sep=',',index_col=0)

    #recall_v1.0.csv
    path = chemin_local+frecall
    recall_v1 = pd.read_csv(path,sep=',',index_col=0)

    #f1score_v1.0.csv
    path = chemin_local+ff1score
    f1score_v1 = pd.read_csv(path,sep=',',index_col=0)

    #specificity
    path = chemin_local+fspecificity
    specificity_v1 = pd.read_csv(path,sep=',',index_col=0)

    accuracy_v1["evaluateur"] = "accuracy_v1"
    precision_v1["evaluateur"] = "precision_v1"
    recall_v1["evaluateur"] = "recall_v1"
    f1score_v1["evaluateur"] = "f1score_v1"
    specificity_v1["evaluateur"] = "specificity_v1"

    st.write(accuracy_v1.head(15))
    st.write("")

    st.write(precision_v1.head(15))
    st.write("")

    st.write(recall_v1.head(15))
    st.write("")

    st.write(f1score_v1.head(15))
    st.write("")

    st.write(specificity_v1.head(15))
    st.write("")

    st.write(metricsUtilisees)
    st.write("")

    st.write(str(nbVoisinsMax))
    st.write("")


    # affichage des données avec Bokeh
    params = {}
    params['nb_voisins'] = np.arange(2,nbVoisinsMax,1)
    params['metric'] = metricsUtilisees # ['l1','l2','manhattan','nan_euclidean','minkowski','chebyshev','cityblock','cosine','euclidean']
    # 'haversine' ==> valide seulement en 2D
    params['weights'] = ['uniform','distance']
        
    mes_couleurs = ['magenta','maroon','mediumaquamarine','mediumblue','mediumorchid','mediumpurple','mediumseagreen',
                    'mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','gold',
                    'goldenrod','navy','grey','darksalmon','red']
    # 9 metrics & 2 weights = 18 curves = 18 colors
    # https://docs.bokeh.org/en/2.4.2/docs/reference/colors.html#bokeh-colors-named

    # Attention : au moins UNE couleur de plus que le nb courbes !!!
    # on utilise la couleur supplémentaire pour colorer le pt quand la souris passe dessus


    source1 = ColumnDataSource(accuracy_v1)
    source2 = ColumnDataSource(precision_v1)
    source3 = ColumnDataSource(recall_v1)
    source4 = ColumnDataSource(f1score_v1)
    source5 = ColumnDataSource(specificity_v1)

    tt1 = [("k", "@k"),("metric", "@metric"),("weights","@weights"),("valeur", "@valeur")]

    #tools='hover,xpan,xwheel_zoom,box_zoom,save,reset'
    tools=[HoverTool(),BoxZoomTool(), PanTool(),ResetTool()]

    ##################### Accuracy
    p1 = figure(width=800, height=600, x_axis_label='k', y_axis_label='valeur',title="Accuracy",
            toolbar_location = "below",
            tooltips = tt1,tools = tools)

    chgt_couleur1 = []
    hover1 = []
    legend_item1 = []
    g1 = [] # liste des graphiques à dessiner sur la figure

    for i,dist in enumerate(accuracy_v1['metric'].unique()):
        for j,w in enumerate(accuracy_v1['weights'].unique()):
            ma_source = ColumnDataSource(accuracy_v1[(accuracy_v1['metric']==dist) & (accuracy_v1['weights']==w)])
            g1.append(p1.line(x='k',y='valeur',source = ma_source,color=mes_couleurs[i+j],
                            line_width=2)) #,legend_label=dist+'--'+w))
            g1.append(p1.circle(x='k',y='valeur',source = ma_source,color=mes_couleurs[i+j],
                                size=6)) #,legend_label=dist+'--'+w))

            # chgt de couleur qd la souris passe sur un pt
            chgt_couleur1.append(p1.circle(x='k',y='valeur',source = ma_source,
                            size=15,
                            alpha=0, # alpha=0 ==> invisible tant que la souris ne va pas sur le point
                            hover_fill_color=mes_couleurs[-1], hover_alpha=0.7)) #,legend_label=dist+'--'+w))

            hover1.append(HoverTool(tooltips=tt1,mode='mouse',renderers=[chgt_couleur1[-1]]))
            legend_item1.append((dist+'--'+w,[g1[-1],g1[-2],chgt_couleur1[-1]]))

    p1.legend.visible = False
    legend1 = Legend(items=legend_item1,location="center",click_policy="hide")

    p1.add_layout(legend1, 'right')

    tab1 = TabPanel(child=p1, title="Accuracy")

    ##################### Precision
    p2 = figure(width=800, height=600,x_axis_label='k', y_axis_label='valeur',title="Precision",
                x_range = p1.x_range,
                toolbar_location = "below",
                tooltips = tt1,tools = tools)

    chgt_couleur2 = []
    hover2 = []
    legend_item2 = []
    g2 = [] # liste des graphiques à dessiner sur la figure

    for i,dist in enumerate(precision_v1['metric'].unique()):
        for j,w in enumerate(precision_v1['weights'].unique()):
            ma_source = ColumnDataSource(precision_v1[(precision_v1['metric']==dist) & (precision_v1['weights']==w)])
            g2.append(p2.line(x='k',y='valeur',source = ma_source,color=mes_couleurs[i+j],line_width=2))
            g2.append(p2.circle(x='k',y='valeur',source = ma_source,color=mes_couleurs[i+j],size=6))

            # chgt de couleur qd la souris passe sur un pt
            chgt_couleur2.append(p2.circle(x='k',y='valeur',source = ma_source,
                            size=15,
                            alpha=0, # alpha=0 ==> invisible tant que la souris ne va pas sur le point
                            hover_fill_color=mes_couleurs[-1], hover_alpha=0.7))

            hover2.append(HoverTool(tooltips=tt1,mode='mouse',renderers=[chgt_couleur2[-1]]))
            legend_item2.append((dist+'--'+w,[g2[-1],g2[-2],chgt_couleur2[-1]]))

    p2.legend.visible = False
    legend2 = Legend(items=legend_item2,location="center",click_policy="hide")

    p2.add_layout(legend2, 'right')

    tab2 = TabPanel(child=p2, title="Precision")

    # ##################### Recall
    p3 = figure(width=800, height=600,x_axis_label='k', y_axis_label='valeur',title="Recall",
                x_range = p1.x_range,
                toolbar_location = "below",
                tooltips = tt1,tools = tools)

    chgt_couleur3 = []
    hover3 = []
    legend_item3 = []
    g3 = [] # liste des graphiques à dessiner sur la figure

    for i,dist in enumerate(recall_v1['metric'].unique()):
        for j,w in enumerate(recall_v1['weights'].unique()):
            ma_source = ColumnDataSource(recall_v1[(recall_v1['metric']==dist) & (recall_v1['weights']==w)])
            g3.append(p3.line(x='k',y='valeur',source = ma_source,color=mes_couleurs[i+j],line_width=2))
            g3.append(p3.circle(x='k',y='valeur',source = ma_source,color=mes_couleurs[i+j],size=6))

            # chgt de couleur qd la souris passe sur un pt
            chgt_couleur3.append(p3.circle(x='k',y='valeur',source = ma_source,
                            size=15,
                            alpha=0, # alpha=0 ==> invisible tant que la souris ne va pas sur le point
                            hover_fill_color=mes_couleurs[-1], hover_alpha=0.7))

            hover3.append(HoverTool(tooltips=tt1,mode='mouse',renderers=[chgt_couleur3[-1]]))
            legend_item3.append((dist+'--'+w,[g3[-1],g3[-2],chgt_couleur3[-1]]))

    p3.legend.visible = False
    legend3 = Legend(items=legend_item3,location="center",click_policy="hide")

    p3.add_layout(legend3, 'right')

    tab3 = TabPanel(child=p3, title="Recall")

    # ##################### F1-Score
    p4 = figure(width=800, height=600,x_axis_label='k', y_axis_label='valeur',title="F1-Score",
                x_range = p1.x_range,
                toolbar_location = "below",
            tooltips = tt1,tools = tools)

    chgt_couleur4 = []
    hover4 = []
    legend_item4 = []
    g4 = [] # liste des graphiques à dessiner sur la figure

    for i,dist in enumerate(f1score_v1['metric'].unique()):
        for j,w in enumerate(f1score_v1['weights'].unique()):
            ma_source = ColumnDataSource(f1score_v1[(f1score_v1['metric']==dist) & (f1score_v1['weights']==w)])
            g4.append(p4.line(x='k',y='valeur',source = ma_source,color=mes_couleurs[i+j],line_width=2))
            g4.append(p4.circle(x='k',y='valeur',source = ma_source,color=mes_couleurs[i+j],size=6))

            # chgt de couleur qd la souris passe sur un pt
            chgt_couleur4.append(p4.circle(x='k',y='valeur',source = ma_source,
                            size=15,
                            alpha=0, # alpha=0 ==> invisible tant que la souris ne va pas sur le point
                            hover_fill_color=mes_couleurs[-1], hover_alpha=0.7))

            hover4.append(HoverTool(tooltips=tt1,mode='mouse',renderers=[chgt_couleur4[-1]]))
            legend_item4.append((dist+'--'+w,[g4[-1],g4[-2],chgt_couleur4[-1]]))

    p4.legend.visible = False
    legend4 = Legend(items=legend_item4,location="center",click_policy="hide")

    p4.add_layout(legend4, 'right')

    tab4 = TabPanel(child=p4, title="F1-Score")

    # ##################### Specificity
    p5 = figure(width=800, height=600,x_axis_label='k', y_axis_label='valeur',title="Specificity",
                x_range = p1.x_range,
                toolbar_location = "below",
            tooltips = tt1,tools = tools)

    chgt_couleur5 = []
    hover5 = []
    legend_item5 = []
    g5 = [] # liste des graphiques à dessiner sur la figure

    for i,dist in enumerate(specificity_v1['metric'].unique()):
        for j,w in enumerate(specificity_v1['weights'].unique()):
            ma_source = ColumnDataSource(specificity_v1[(specificity_v1['metric']==dist) & (specificity_v1['weights']==w)])
            g5.append(p5.line(x='k',y='valeur',source = ma_source,color=mes_couleurs[i+j],line_width=2))
            g5.append(p5.circle(x='k',y='valeur',source = ma_source,color=mes_couleurs[i+j],size=6))

            # chgt de couleur qd la souris passe sur un pt
            chgt_couleur5.append(p5.circle(x='k',y='valeur',source = ma_source,
                            size=15,
                            alpha=0, # alpha=0 ==> invisible tant que la souris ne va pas sur le point
                            hover_fill_color=mes_couleurs[-1], hover_alpha=0.7))

            hover5.append(HoverTool(tooltips=tt1,mode='mouse',renderers=[chgt_couleur5[-1]]))
            legend_item5.append((dist+'--'+w,[g5[-1],g5[-2],chgt_couleur5[-1]]))
            
    p5.legend.visible = False
    legend5 = Legend(items=legend_item5,location="center",click_policy="hide")

    p5.add_layout(legend5, 'right')

    tab5 = TabPanel(child=p5, title="Specificity")

    # ##################### Accuracy en fct du F1-Score

    p6 = figure(width=800, height=600,x_axis_label='F1-Score', y_axis_label='Accuracy',
    #             title="Accuracy en fct du F1-Score",
                toolbar_location = "below",
            x_range = DataRange1d(bounds='auto'),
            y_range = DataRange1d(bounds='auto'))

    acc_f1 = accuracy_v1.copy(deep=True)
    acc_f1.rename(columns={'valeur':'accuracy'},inplace=True)

    acc_f1['f1score'] = f1score_v1['valeur']
    acc_f1['precision'] = precision_v1['valeur']
    acc_f1['recall'] = recall_v1['valeur']
    acc_f1['specificity'] = specificity_v1['valeur']

    acc_f1['id_couleur'] = round(17 * (acc_f1['accuracy'] * acc_f1['f1score']**2),0).astype(int)
    # couleurs arbitraires pour différencier :
    # les meilleurs compromis F1Score/Accuracy
    # les compromis moyens
    # les mauvais compromis

    source6 = ColumnDataSource(acc_f1)

    ma_cmap = LinearColorMapper(palette=mes_couleurs, 
                                low = min(acc_f1['id_couleur']), 
                                high = max(acc_f1['id_couleur']))
                
    p6.circle(x='f1score',y='accuracy',
            fill_color= {"field":"id_couleur", "transform":ma_cmap},
            line_color=None,
            source = source6,
            size=6)

    # chgt de couleur qd la souris passe sur un pt
    chgt_couleur6 = p6.circle(x='f1score',y='accuracy',source = source6,
                            size=15,
                            alpha=0, # alpha=0 ==> invisible tant que la souris ne va pas sur le point
                            hover_fill_color=mes_couleurs[-1], hover_alpha=0.7)

    tt2 = [("k", "@k"),("metric", "@metric"),("weights","@weights"),("f1score", "@f1score"),("accuracy","@accuracy"),
        ("recall","@recall"),("precision","@precision"),("specificity","@specificity")]

    hover6 = HoverTool(
            tooltips=tt2,
            mode='mouse',
            renderers=[chgt_couleur6])

    p6.add_tools(hover6)

    tab6 = TabPanel(child=p6, title="Accuracy en fct du F1-Score")

    # ##################### Affichages

    # h = column(row(p1,p2),row(p3,p4),row(p5,p6))
    # show(h)

    tabs = Tabs(tabs=[tab6,tab1,tab2,tab3,tab4,tab5])

    st.bokeh_chart(tabs)

    
