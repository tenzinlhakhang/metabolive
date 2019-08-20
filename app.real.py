import dash
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



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config['suppress_callback_exceptions']=True

result = met.Analysis('skeleton_output.tsv','Groups.csv')
dropdown_comparison = [' vs. '.join(list(x)) for x in result.dme_comparisons()]

dropdown_list = []
for dropdown,met_comparison in zip(dropdown_comparison,result.dme_comparisons()):
    drop_dict = {}
    drop_dict['label'] = dropdown
    drop_dict['value'] = met_comparison
    dropdown_list.append(drop_dict)





app.layout = html.Div([
    html.H1('Metabolyze Web App Test'),
    dcc.Dropdown(
        id='my-dropdown',
        options=[{'label': 'OLFR2 vs. GPI1', 'value': "'OLFR2','GPI1'"}, {'label': 'GPI1 vs. OLFR2', 'value': "'GPI1','OLFR2'"}],
        value=["test"]
    ),
    html.Div(id='output-container'),
    html.Button('Submit', id='button'),
    html.Div(id='output-data-upload'),
    dcc.Graph(id='graph'),
])



# @app.callback([
#     Output('output-container', 'children'),
#     Output('graph', 'figure'),
# ], [Input('data-dropdown', 'n_clicks')
# ], [State('my-dropdown', 'value')])

@app.callback(
    dash.dependencies.Output('output-container', 'children'),
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('my-dropdown', 'value')])

def update_output(n_clicks,value):

    standard = pd.read_table(result.data)
    detection_column_index = standard.columns.get_loc("detections")


    standard = standard.iloc[:,0:detection_column_index]
    standard.index = standard['Metabolite']
    del standard['Metabolite']


    matrices = []    
    sample_groups = result.get_groups()
    #print (comparison[0])
    #value = value.replace("'", "")
    value_list = value.split(',')
    value_list_strip = [value.replace(" ", "") for value in value_list]
    value_list_final = [val[1:-1] for val in value_list_strip]
    #value_list = value_list.replace("'", "")
    #test_condition = 'GPI1'
    #ids = (sample_groups[test_condition]) 
    
    comparison_ids = []

    matrices = []
    for condition in value_list_final:
        if condition in sample_groups:
            test = condition
            #real_condition = condition.replace("'", "")
            ids = (sample_groups[test]) 
            #print (ids)
            matrices.append((result.get_imputed_full_matrix(result.get_matrix(ids=ids),param='detected')))
            comparison_ids.append(ids)
            
    #print("yowtf",matrices[0].shape[1])
    group_sample_number =  int((matrices[0].shape)[1])
    group_sample_number_2 = int(group_sample_number+ ((matrices[1].shape)[1]))
                
         

    df_m = reduce(lambda left,right: pd.merge(left,right,left_index=True, right_index=True), matrices)
    blank_matrix = pd.DataFrame(result.get_matrix(result.get_ids('Blank')))
    #blank_matrix.to_csv(results_folder+'Tables/'+'blank_intensity.csv')
    blank_threshold = pd.DataFrame(blank_matrix.mean(axis=1)*3)+10000
    blank_threshold['Metabolite'] = blank_threshold.index
    blank_threshold.columns = ['blank_threshold','Metabolite']

    
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
    final_df_m = final_df_m.fillna('NA')
    
    html_object =  html.Div([
            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in final_df_m.columns],
                data=final_df_m.to_dict("rows"),
                style_table={'overflowX': 'scroll','maxHeight': '300px'},
                style_cell={'width': '150px',
                'height': '60px',
                'textAlign': 'left'}),


            ])
    return(html_object)




if __name__ == '__main__':
    app.run_server(debug=True)