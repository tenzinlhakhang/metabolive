import plotly.plotly as py
import plotly.graph_objs as go
import plotly
import json
from textwrap import dedent as d

import dash
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



### Perform Hierarchical Clustering on Metabolites & Samples
### Z-score Normalize Heatmap
### Generate Plotly Heatmap


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



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = html.Div([
    dcc.Graph(
        id='basic-interactions',
        figure={'data': data,'layout': {'title': 'Dash Data Visualization'},
        'layout': {
                'clickmode': 'event+select'}
        })]
    ),
html.Div(className='row', children=[
        html.Div([
            dcc.Markdown(d("""
                **Zoom Data**

                Zoom over values in the graph.
            """)),
            html.Pre(id='relayoutData', style=styles['pre'])
        ], className='three columns')
    ])



@app.callback(
    Output('relayout-data', 'children'),
    [Input('basic-interactions', 'relayoutData')])

def display_selected_data(relayoutData):
    return json.dumps(relayoutData, indent=2)

if __name__ == '__main__':
    app.run_server(debug=True)

#plotly.offline.plot(data, filename='Heatmap.html')
