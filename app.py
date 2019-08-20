import dash_bootstrap_components as dbc
import dash_html_components as html
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import pandas as pd
import metabolyze as met
from dash.dependencies import Input, Output
from functools import reduce
import dash_html_components as html
import dash_core_components as dcc
import pandas as pd
import metabolyze as met
from functools import reduce
import warnings
import pandas as pd
import itertools
import scipy
import scipy.stats
import numpy as np
from functools import reduce
import re
import numpy 
import subprocess as sp
import os
import sys
import time
import dash
import dash_table
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input, State
import plotly.graph_objs as go
from sklearn.preprocessing import scale
import scipy.cluster.hierarchy as shc


import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  
import itertools
import string
#import RNAseq
import numpy as np
import pandas as pd
from scipy import stats
import numpy as np
import sys, json
from time import sleep
from scipy import *
from scipy import stats
from sklearn.decomposition import PCA
import plotly.plotly as py
#%matplotlib inline
import plotly
#plotly.offline.init_notebook_mode() # To embed plots in the output cell of the notebook
import plotly.graph_objs as go
import os
import operator
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from math import sqrt
from itertools import combinations
from matplotlib import rcParams





COLORS8 = [
'#1F77B4',
'#26A9E0',
'#75DBA7',
'#2CA02C',
'#9467BD',
'#FF0000',
'#FF7F0E',
'#E377C2',
]

COLORS = ['#1F78B4','#E31A1C','#FF7F00','#6A3D9A','#B15928','#666666', 'k']
COLORS10 = [
'#1f77b4',
'#ff7f0e',
'#2ca02c',
'#d62728',
'#9467bd',
'#8c564b',
'#e377c2',
'#7f7f7f',
'#bcbd22',
'#17becf',
]

COLORS20 = [
'#1f77b4',
'#aec7e8',
'#ff7f0e',
'#ffbb78',
'#2ca02c',
'#98df8a',
'#d62728',
'#ff9896',
'#9467bd',
'#c5b0d5',
'#8c564b',
'#c49c94',
'#e377c2',
'#f7b6d2',
'#7f7f7f',
'#c7c7c7',
'#bcbd22',
'#dbdb8d',
'#17becf',
'#9edae5',
]

COLORS20b = [
'#393b79',
'#5254a3',
'#6b6ecf',
'#9c9ede',
'#637939',
'#8ca252',
'#b5cf6b',
'#cedb9c',
'#8c6d31',
'#bd9e39',
'#e7ba52',
'#e7cb94',
'#843c39',
'#ad494a',
'#d6616b',
'#e7969c',
'#7b4173',
'#a55194',
'#ce6dbd',
'#de9ed6',
]





result = met.Analysis('skeleton_output.tsv','Groups.csv')
dropdown_comparison = [' vs. '.join(list(x)) for x in result.dme_comparisons()]
#print("met_comparison",result.dme_comparisons())
dropdown_list = []
for dropdown,met_comparison in zip(dropdown_comparison,result.dme_comparisons()):
    drop_dict = {}
    drop_dict['label'] = dropdown
    met_list = list(met_comparison)
    met_list = ','.join("'{0}'".format(comparison) for comparison in met_list)
    drop_dict['value'] = met_list
    dropdown_list.append(drop_dict)


  

#table = dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True)

#print(values)
#print(values)

# navbar = dbc.NavbarSimple(
#     children=[
#         dbc.NavItem(dbc.NavLink("Link", href="#")),
#         dbc.DropdownMenu(
#             nav=True,
#             in_navbar=True,
#             label="Menu",
#             children=[
#                 dbc.DropdownMenuItem("Entry 1"),
#                 dbc.DropdownMenuItem("Entry 2"),
#                 dbc.DropdownMenuItem(divider=True),
#                 dbc.DropdownMenuItem("Entry 3"),
#             ],
#         ),
#     ],
#     brand="Demo",
#     brand_href="#",
#     sticky="top",
# )

logo = 'data:image/png;base64,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'



search_bar = dbc.Row(
    [
        dbc.Col(dbc.Input(type="search", placeholder="Search")),
        dbc.Col(
            dbc.Button("Search", color="primary", className="ml-2"),
            width="auto",
        ),
    ],
    no_gutters=True,
    className="ml-auto flex-nowrap mt-3 mt-md-0",
    align="center",
)

navbar = dbc.Navbar(
    [
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=logo, height="50px")),
                    dbc.Col(dbc.NavbarBrand("Metabolive", className="ml-2")),
                ],
                align="center",
                no_gutters=True,
            ),
            href="www.github.com",
        ),
        dbc.NavbarToggler(id="navbar-toggler"),
        dbc.Collapse(search_bar, id="navbar-collapse", navbar=True),
    ],
    color="dark",
    dark=True,
)



jumbotron = dbc.Jumbotron(
    [
        html.H1("Metabolive Beta", className="display-3"),
        html.P(
            "Beta testing for the interactive analysis of Metabolyze results."
        ),
    ]
)

#"'1e6 per mL','RPMI_PBS'"
#[{'label': 'OLFR2 vs. GPI1', 'value': ['OLFR2', 'GPI1']}]

button = html.Div(
    [
        dbc.Button("Submit", id="button"),
        html.Span(id="example-output"),
    ]
)


dropdown = dcc.Dropdown(
        id='my-dropdown',
        #options=[{'label':'OLFR2 vs. GPI1', 'value': "'OLFR2','GPI1'"}, {'label': 'GPI1 vs. OLFR2', 'value': "'GPI1','OLFR2'"}],
        options=dropdown_list,
        #value=["test"]
    )

#graphs = html.Div(id='graph',className="row")
table = html.Div(id='output-data-upload')
hidden_div = html.Div(id='intermediate-value', style={'display': 'none'})
volcano_graph = html.Div(id='volcano_graph')
heatmap_graph = html.Div(id='heatmap_graph')
pca_graph = html.Div(id='pca_graph')
#graph = html.Div(id='graph')
# graph_volcano = html.Div(id='graph',className="row"),

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([navbar,jumbotron,dropdown,button,hidden_div,volcano_graph,heatmap_graph,pca_graph,table])
app.config['suppress_callback_exceptions']=True



@app.callback(
    dash.dependencies.Output('intermediate-value', 'children'),
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('my-dropdown', 'value')])


def update_output(n_clicks,value):
    standard = pd.read_table(result.data)
    detection_column_index = standard.columns.get_loc("detections")

    #print("valuedawg",value)

    standard = standard.iloc[:,0:detection_column_index]
    standard.index = standard['Metabolite']
    del standard['Metabolite']


    matrices = []    
    sample_groups = result.get_groups()
    #print (comparison[0])
    #value = value.replace("'", "")

    value_list = value.split(',')
    #value_list_strip = [value.replace(" ", "") for value in value_list]
    value_list_final = [val[1:-1] for val in value_list]
    #value_list = value_list.replace("'", "")
    #test_condition = 'GPI1'
    #ids = (sample_groups[test_condition]) 
    #print('wtf')
    comparison_ids = []

    matrices = []
    for condition in value_list_final:
        #print("condition",condition)
        if condition in sample_groups:
            test = condition
            #real_condition = condition.replace("'", "")
            ids = (sample_groups[test]) 
            #print (ids)
            matrices.append((result.get_imputed_full_matrix(result.get_matrix(ids=ids),param='detected')))
            comparison_ids.append(ids)
    #print(matrices)
    #print("yowtf",matrices[0].shape[1])
    group_sample_number =  int((matrices[0].shape)[1])
    group_sample_number_2 = int(group_sample_number+ ((matrices[1].shape)[1]))
                



    df_m = reduce(lambda left,right: pd.merge(left,right,left_index=True, right_index=True), matrices)
    intensity_matrix = reduce(lambda left,right: pd.merge(left,right,left_index=True, right_index=True), matrices)

    

    # plot = pca_plot(df_m,result.samplesheet,value_list_final)
    # pca_fig = plot
    ### dendrogram heatmap calculation




    

    blank_matrix = pd.DataFrame(result.get_matrix(result.get_ids('Blank')))
    #blank_matrix.to_csv(results_folder+'Tables/'+'blank_intensity.csv')
    blank_threshold = pd.DataFrame(blank_matrix.mean(axis=1)*3)+10000
    blank_threshold['Metabolite'] = blank_threshold.index
    blank_threshold.columns = ['blank_threshold','Metabolite']

    #print(df_m.head())
    df_m['ttest_pval'] = ((scipy.stats.ttest_ind(df_m.iloc[:, :group_sample_number], df_m.iloc[:, group_sample_number:group_sample_number_2], axis=1))[1])
    df_m['1/pvalue'] = float(1)/df_m['ttest_pval']      
    group_1_df = (pd.DataFrame(df_m.iloc[:, :group_sample_number]))
    group_2_df = (pd.DataFrame(df_m.iloc[:, group_sample_number:group_sample_number_2]))
    
    
    
    df_m[value_list_final[0]+'_Mean'] = (group_1_df.mean(axis=1))
    df_m[value_list_final[1]+'_Mean'] = (group_2_df.mean(axis=1))
    
    df_m['Log2FoldChange'] =  np.log2(((group_1_df.mean(axis=1)))/((group_2_df.mean(axis=1))))
    df_m['LogFoldChange'] =  (((group_1_df.mean(axis=1)))/((group_2_df.mean(axis=1))))
    # df_m['Metabolite'] = df_m.index


    final_df_m = pd.merge(standard, df_m, left_index=True, right_index=True)
    final_df_m = pd.merge(final_df_m,blank_threshold,left_index=True, right_index=True)
    # # Add detection column

    for col in blank_matrix.columns:

        final_df_m[col] = blank_matrix[col].values


    final_df_m['combined_mean'] = (final_df_m[value_list_final[0]+'_Mean']+final_df_m[value_list_final[1]+'_Mean'])/2
    final_df_m['impact_score'] = (((2**abs(final_df_m['Log2FoldChange']))*final_df_m['combined_mean'])/final_df_m['ttest_pval'])/1000000
    final_df_m.impact_score = final_df_m.impact_score.round()
    final_df_m['impact_score'] = final_df_m['impact_score'].fillna(0)



    sig_genes = final_df_m.loc[final_df_m['ttest_pval'] < 0.05].index
    

    # data = cluster_plot(intensity_matrix,sig_genes)


    detection_dict = {}
            
    comparison_matrix = group_1_df.join(group_2_df, how='outer')
    
    
    for index, row in comparison_matrix.iterrows():
        test_list = []
        #print (row)
        #print(index)
        row_intensity = (pd.DataFrame(row))
        blankthresh = blank_threshold.loc[index, ['blank_threshold']][0]
        detected = (row_intensity[row_intensity > float(blankthresh)].count())
        detected = (detected[0])
        detection_dict[index] = detected

    detection_df = pd.DataFrame(list(detection_dict.items()))
    detection_df.columns = ['Metabolite','Detection']
    detection_df.index = detection_df['Metabolite']

    final_df_m = pd.merge(final_df_m,detection_df,left_index=True, right_index=True)
    # volcano df
    
    #final_df_m = final_df_m.fillna('NA')
    final_df_m = final_df_m[np.isfinite(final_df_m['ttest_pval'])]

    return (final_df_m.to_json(),intensity_matrix.to_json())

    # volcano_df = final_df_m[np.isfinite(final_df_m['ttest_pval'])]
    # sig_volcano = volcano_df.loc[volcano_df['ttest_pval'] < 0.05]
    # insig_volcano = volcano_df.loc[volcano_df['ttest_pval'] > 0.05]


    # volcano_fig_1 = go.Scatter(
    # #x = -np.log10(comparison_insig['ttest_pval']),


    # x = sig_volcano['Log2FoldChange'],
    # y = -np.log10(sig_volcano['ttest_pval']),
    # #y = comparison_insig['Log2FoldChange'],
    # text = sig_volcano.index,
    # name = 'Significant',
    # mode = 'markers',
    # marker = dict(
    #     size = 10,
    #     color = 'rgba(152, 0, 0, .8)',
    #     line = dict(
    #         width = 2,)))

    # volcano_fig_2 = go.Scatter(
    # #x = -np.log10(comparison_insig['ttest_pval']),


    # x = insig_volcano['Log2FoldChange'],
    # y = -np.log10(insig_volcano['ttest_pval']),
    # #y = comparison_insig['Log2FoldChange'],
    # text = insig_volcano.index,
    # name = 'Insignificant',
    # mode = 'markers',
    # marker = dict(
    #     size = 10,
    #     color = 'rgba(255, 182, 193, .9)',
    #     line = dict(
    #         width = 2,)))


    # volcano_data = [volcano_fig_1,volcano_fig_2]
    
    # figure = html.Div([
    #             #html.H3('Volcano Plot',style={'textAlign': 'center'}),
    #             dcc.Graph(id='g2', figure={
    #             'data': volcano_data,
    #             'layout': {
    #                 'clickmode': 'event+select','width': '450', 'display': 'inline-block', 'height': '450', 'padding': '0 20',

    #             },
    #         })], className="four columns")
        
    # figure_two = html.Div([
    #         #html.H3('Heatmap',style={'textAlign': 'center'}),
    #         dcc.Graph(id='g3', figure={
    #         'data': data,
    #         'layout': {
    #             'clickmode': 'event+select','width': '450', 'display': 'inline-block', 'height': '450', 'padding': '0 20',

    #         },
    #     })], className="four columns")

    # figure_three = html.Div([
    #         #html.H3('PCA',style={'textAlign': 'center'}),
    #         dcc.Graph(id='g4', figure={
    #         'data': pca_fig,
    #         'layout': {
    #             'clickmode': 'event+select','width': '500', 'display': 'inline-block', 'height': '450', 'padding': '0 20',

    #         },
    #     })], className="four columns")

    # packed_figures = [figure, figure_two,figure_three]




    ## heatmap dendrogram calculation
    #table = dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True)

# #
#     html_object =  html.Div([
#             dbc.Table.from_dataframe(final_df_m.iloc[:, : 10], striped=True, bordered=True, hover=True),
#             ])

#     return(html_object)
###
#

@app.callback(
    dash.dependencies.Output('output-data-upload', 'children'),
    [dash.dependencies.Input('intermediate-value', 'children')])

def update_table(jsonified_cleaned_data):
    dff = pd.read_json(jsonified_cleaned_data[0]) # or, more generally json.loads(jsonified_cleaned_data)
    html_object =  html.Div([
            dbc.Table.from_dataframe(dff.iloc[:, 10: 20], striped=True, bordered=True, hover=True),
            ])
    return (html_object)


@app.callback(
    dash.dependencies.Output('volcano_graph', 'children'),
    [dash.dependencies.Input('intermediate-value', 'children')])

def update_volcano(jsonified_cleaned_data):

    final_df_m = pd.read_json(jsonified_cleaned_data[0])
    volcano_df = final_df_m[np.isfinite(final_df_m['ttest_pval'])]
    sig_volcano = volcano_df.loc[volcano_df['ttest_pval'] < 0.05]
    insig_volcano = volcano_df.loc[volcano_df['ttest_pval'] > 0.05]


    volcano_fig_1 = go.Scatter(
    #x = -np.log10(comparison_insig['ttest_pval']),


    x = sig_volcano['Log2FoldChange'],
    y = -np.log10(sig_volcano['ttest_pval']),
    #y = comparison_insig['Log2FoldChange'],
    text = sig_volcano.index,
    name = 'Significant',
    mode = 'markers',
    marker = dict(
        size = 10,
        color = 'rgba(152, 0, 0, .8)',
        line = dict(
            width = 2,)))

    volcano_fig_2 = go.Scatter(
    #x = -np.log10(comparison_insig['ttest_pval']),


    x = insig_volcano['Log2FoldChange'],
    y = -np.log10(insig_volcano['ttest_pval']),
    #y = comparison_insig['Log2FoldChange'],
    text = insig_volcano.index,
    name = 'Insignificant',
    mode = 'markers',
    marker = dict(
        size = 10,
        color = 'rgba(255, 182, 193, .9)',
        line = dict(
            width = 2,)))


    volcano_data = [volcano_fig_1,volcano_fig_2]
    
    volcano = html.Div([
                #html.H3('Volcano Plot',style={'textAlign': 'center'}),
                dcc.Graph(id='g2', figure={
                'data': volcano_data,
                'layout': {
                    'clickmode': 'event+select','width': '450', 'display': 'inline-block', 'height': '450', 'padding': '0 20',

                },
            })], className="four columns")

    return(volcano)


@app.callback(
    dash.dependencies.Output('heatmap_graph', 'children'),
    [dash.dependencies.Input('intermediate-value', 'children')])

def update_heatmap(jsonified_cleaned_data):
    final_df_m = pd.read_json(jsonified_cleaned_data[0])
    matrix = pd.read_json(jsonified_cleaned_data[1])
    sig_genes = final_df_m.loc[final_df_m['ttest_pval'] < 0.05].index
    def cluster_plot(matrix,genes):

        values = matrix

        values = values.loc[values.index.isin(genes)]
        values_t = values.T
        values_t.index.name = 'Sample'

        #values_t.to_csv('values_t.csv')
        #plt.figure(figsize=(10, 7))  
        #plt.title("Customer Dendograms")  

        dend_metabolite = shc.dendrogram(shc.linkage(values, method='ward'),labels=values.index)  
        dend_metabolite_order = dend_metabolite['ivl']


        dend_sample = shc.dendrogram(shc.linkage(values_t, method='ward'),labels=values_t.index)  
        dend_sample_order = dend_sample['ivl']

        df = values[dend_sample_order]
        df = df.reindex(dend_metabolite_order)



        df_zscore = df.apply(
                    lambda V: scale(V,axis=0,with_mean=True, with_std=True,copy=False),axis=1)


        trace = go.Heatmap(z=np.array(df_zscore),x=dend_sample_order,y=dend_metabolite_order)
        data=[trace]
        return(data)
    data = cluster_plot(matrix,sig_genes)

    heatmap_figure = html.Div([
        #html.H3('Heatmap',style={'textAlign': 'center'}),
        dcc.Graph(id='g3', figure={
        'data': data,
        'layout': {
            'clickmode': 'event+select','width': '450', 'display': 'inline-block', 'height': '450', 'padding': '0 20',

        },
    })], className="four columns")

    return(heatmap_figure)

@app.callback(
    dash.dependencies.Output('pca_graph', 'children'),
    [dash.dependencies.Input('intermediate-value', 'children')],
    [dash.dependencies.State('my-dropdown', 'value')])

def update_pca(jsonified_cleaned_data,value):
    
    matrix = pd.read_json(jsonified_cleaned_data[1])
   
    value_list = value.split(',')
    #value_list_strip = [value.replace(" ", "") for value in value_list]
    value_list_final = [val[1:-1] for val in value_list]
    def perform_PCA(fpkmMatrix, standardize=3, log=True):
    ## preprocessing of the fpkmMatrix
        if log:
            fpkmMatrix = np.log10(fpkmMatrix + 1.)
        if standardize == 2: # standardize along rows/genes
            fpkmMatrix = stats.zscore(fpkmMatrix, axis=1)
        elif standardize == 1: # standardize along cols/samples
            fpkmMatrix = stats.zscore(fpkmMatrix, axis=0)

        ## remove genes with NaNs
        fpkmMatrix = fpkmMatrix[~np.isnan(np.sum(fpkmMatrix, axis=1))]

        pca = PCA(n_components=None)
        ## get variance captured
        pca.fit(fpkmMatrix.T)
        variance_explained = pca.explained_variance_ratio_[0:3]
        variance_explained *= 100
        ## compute PCA and plot
        pca_transformed = pca.transform(fpkmMatrix.T)
        return (variance_explained, pca_transformed)


##
    def pca_plot(matrix,samplesheet,conditions):


        # directory = matrix_path.split('/')[0]
        # print(directory)
        expr_df = matrix.astype(float)

        variance_explained, pca_transformed = perform_PCA(expr_df.values, standardize=2, log=True)
        

        meta_df = pd.read_csv(samplesheet)
        meta_df = meta_df.loc[meta_df['Group'] != 'Blank']
        meta_df = meta_df.loc[meta_df['Group'].isin(conditions)]

        
        meta_df['File'].str.replace('.mzXML','')

        meta_df.index = meta_df['id']
        #print(meta_df)
        meta_df['x'] = pca_transformed[:,0]
        meta_df['y'] = pca_transformed[:,1]
        meta_df['z'] = pca_transformed[:,2]

        conditions = meta_df['Group'].unique().tolist()
        #platforms = meta_df['platform'].unique().tolist()
        SYMBOLS = ['circle']
        COLORS = COLORS20

        data = [] 
        #print(meta_df)

        for (condition), meta_df_sub in meta_df.groupby(['Group']):
            #print(condition)
            # Iteratate through samples grouped by condition and platform
            display_name = '%s' % (condition)
            # Initiate a Scatter3d instance for each group of samples specifying their coordinates
            # and displaying attributes including color, shape, size and etc.
            trace = go.Scatter3d(
                x=meta_df_sub['x'],
                y=meta_df_sub['y'],
                z=meta_df_sub['z'],
                text=meta_df_sub.index,
                mode='markers',
                marker=dict(
                    size=10,
                    color=COLORS[conditions.index(condition)], # Color by infection status
                    opacity=.8,
                ),
                name=display_name,
            )
            
            data.append(trace)

        # Configs for layout and axes
        # layout=dict(height=1000, width=1200, 
        #             title='3D PCA plot',
        #             scene=dict(
        #                 xaxis=dict(title='PC1 (%.2f%% variance)' % variance_explained[0]),
        #                 yaxis=dict(title='PC2 (%.2f%% variance)' % variance_explained[1]),
        #                 zaxis=dict(title='PC3 (%.2f%% variance)' % variance_explained[2])
        #                )
        # )
        
        return(data)
    pca_fig = pca_plot(matrix,result.samplesheet,value_list_final)
    pca_div = html.Div([
            #html.H3('PCA',style={'textAlign': 'center'}),
            dcc.Graph(id='g4', figure={
            'data': pca_fig,
            'layout': {
                'clickmode': 'event+select','width': '500', 'display': 'inline-block', 'height': '450', 'padding': '0 20',

            },
        })], className="four columns")

    return(pca_div)


if __name__ == '__main__':
    app.run_server(debug=True)

