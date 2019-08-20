import dash
import dash_html_components as html
import dash_core_components as dcc

import plotly.plotly as py
import plotly.graph_objs as go
import plotly
import json
from textwrap import dedent as d
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd  
import numpy as np  
import pandas as pd
from sklearn.preprocessing import scale
import scipy.cluster.hierarchy as shc
#warnings.filterwarnings("ignore")
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
import os

values = pd.read_csv('Intensity.values.csv')
values.index = values['Metabolite']
del values['Metabolite']

values_t = values.T
values_t.index.name = 'Sample'


#plt.figure(figsize=(10, 7))  
#plt.title("Customer Dendograms")  

dend_metabolite = shc.dendrogram(shc.linkage(values, method='ward'),labels=values.index)  
dend_metabolite_order = dend_metabolite['ivl']


dend_sample = shc.dendrogram(shc.linkage(values_t, method='ward'),labels=values_t.index)  
dend_sample_order = dend_sample['ivl']

df = values[dend_sample_order]
df = df.reindex(dend_metabolite_order)



values_t = values.T
values_t.index.name = 'Sample'

#plt.figure(figsize=(10, 7))  
#plt.title("Customer Dendograms")  

dend_metabolite = shc.dendrogram(shc.linkage(values, method='ward'),labels=values.index)  
dend_metabolite_order = dend_metabolite['ivl']


dend_sample = shc.dendrogram(shc.linkage(values_t, method='ward'),labels=values_t.index)  
dend_sample_order = dend_sample['ivl']

df = values[dend_sample_order]
df = df.reindex(dend_metabolite_order)
#df.index.name = 'Metabolite'



df_zscore = df.apply(
            lambda V: scale(V,axis=0,with_mean=True, with_std=True,copy=False),axis=1)


trace = go.Heatmap(z=np.array(df_zscore),x=dend_sample_order,y=dend_metabolite_order)
data=[trace]



###volcano plot
comparison = pd.read_csv('KO_vs_WT.corrected.csv')
comparison_sig = (comparison.loc[comparison['ttest_pval'] < 0.05])
comparison_insig = (comparison.loc[comparison['ttest_pval'] >= 0.05])

trace0 = go.Scatter(
    #x = -np.log10(comparison_sig['ttest_pval']),
    x = comparison_sig['Log2FoldChange'],
    #y = comparison_sig['Log2FoldChange'],
    y = -np.log10(comparison_sig['ttest_pval']),
    name = 'Above',
    mode = 'markers',
    text= comparison_sig['Metabolite'],
    marker = dict(
        size = 10,
        color = 'rgba(152, 0, 0, .8)',
        line = dict(
            width = 2,
            color = 'rgb(0, 0, 0)'
        )
    )
)

trace1 = go.Scatter(
    #x = -np.log10(comparison_insig['ttest_pval']),
    x = comparison_insig['Log2FoldChange'],
    y = -np.log10(comparison_insig['ttest_pval']),
    #y = comparison_insig['Log2FoldChange'],
    name = 'Below',
    mode = 'markers',
    text= comparison_insig['Metabolite'],
    marker = dict(
        size = 10,
        color = 'rgba(255, 182, 193, .9)',
        line = dict(
            width = 2,
        )
    )
)

volcano_data = [trace0, trace1]









#comparison
directory = os.getcwd()
result = directory+"/KO_vs_WT.corrected.csv"
comparison = pd.read_csv(result)




app = dash.Dash()
app.layout = html.Div([
    html.Div([
        html.Div([
            html.H3('Heatmap'),
            dcc.Graph(id='g1', figure={
            'data': data,
            'layout': {
                'clickmode': 'event+select','width': '500', 'display': 'inline-block', 'height': '500', 'padding': '0 20',

            },
        })
        ], className="five columns"),

        tortorfasdf
        ], className="six columns"),
    ], className="row")
])

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

if __name__ == '__main__':
    app.run_server(debug=True)

