import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


df = pd.read_csv('Intensity.values.csv')

test = pd.DataFrame()
colnames = [col for col in list(df) if col != 'Metabolite']
test  = pd.DataFrame()
for col in colnames:
    
    new_df = pd.DataFrame(df[['Metabolite',col]])
    #print("###",contin_var)
    #new_df['Continuous_variable'] = str(contin_var)
    new_df['Expression'] = col
    #new_df['Continuous_variable'] = str(contin_var)
    #new_df.loc[147] = ['Continuous_Variable', contin_var,col ]
    new_df.columns = ['Metabolite','Expression','Sample']  
    #print(new_df.tail())
    test = pd.concat([test,new_df])
    
    

    

#available_indicators = df['Indicator Name'].unique()
available_indicators = test['Metabolite'].unique()

app.layout = html.Div([
    html.Div([

        html.Div([
            dcc.Dropdown(
                id='xaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='Beta-Alanine'
            )
            
        ],
        style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='yaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='Continuous_Variable'
            )
            
        ],style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),

    dcc.Graph(id='indicator-graphic')

])

@app.callback(
    dash.dependencies.Output('indicator-graphic', 'figure'),
    [dash.dependencies.Input('xaxis-column', 'value'),
     dash.dependencies.Input('yaxis-column', 'value')]
     )
     
def update_graph(xaxis_column_name, yaxis_column_name):
    dff = test
    return {
        'data': [go.Scatter(
            x=dff[dff['Metabolite'] == xaxis_column_name]['Expression'],
            y=dff[dff['Metabolite'] == yaxis_column_name]['Expression'],
            text=dff[dff['Metabolite'] == yaxis_column_name]['Sample'],
            mode='markers',
            marker={
                'size': 18,
                'opacity': 0.5,
                'line': {'width': 1.5, 'color': 'black'}
            }
        )],
        'layout': go.Layout(
            xaxis={
                'title': xaxis_column_name
            },
            yaxis={
                'title': yaxis_column_name
            },
            margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
            hovermode='closest'
        )
    }


if __name__ == '__main__':
    app.run_server(debug=True)


