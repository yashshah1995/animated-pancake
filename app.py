# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 19:00:56 2020

@author: Sachin Nandakumar

This file serves to provide the structure, design and functionality of RUSKLAINER web application

"""


#####################################################################################################################################
#   Import Libraries
#####################################################################################################################################

import dash_bootstrap_components as dbc
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import base64
from plot_creation import Plot_Creator
import logging
import flask
import copy
import global_vars
import numpy as np

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

#####################################################################################################################################
#   Configuration of dash app
#####################################################################################################################################

external_stylesheets =  ['https://codepen.io/chriddyp/pen/bWLwgP.css']
server = flask.Flask(__name__)
app = dash.Dash(__name__, 
                # prevent_initial_callbacks=True,
                server=server, 
                external_stylesheets= [dbc.themes.BOOTSTRAP],
                meta_tags=[
                    {"name": "viewport", "content": "width=device-width, initial-scale=1"}
                ],
            )
            
# external_stylesheets)
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/brPBPO.css"})
# app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/dZVMbK.css'})
app.config['suppress_callback_exceptions'] = True

#####################################################################################################################################
#   Initialization of global variables
#####################################################################################################################################
legend_img = global_vars.LEGEND_FILENAME
tool_screenshot = global_vars.TOOL_SCREENSHOT_FILENAME
encoded_legend_image = base64.b64encode(open(legend_img, 'rb').read())
encoded_tool_screenshot = base64.b64encode(open(tool_screenshot, 'rb').read())

# fig = go.Figure()

kw_global = global_vars.KW_GLOBAL
num_sample_global = global_vars.NUM_SAMPLE_GLOBAL
model_global = global_vars.MODEL_GLOBAL
instance_global = global_vars.INSTANCE_GLOBAL
text_global = global_vars.TEXT_GLOBAL
explainer_global = global_vars.EXPLAINER_GLOBAL
data_dict = global_vars.DATA_DICT

current_model = ''
current_instance = -2
explanation_data = None

plot_data = Plot_Creator()

# explanation_data = plot_data.data_for_explanationchart(model_global, instance_global)
# score_data = plot_data.data_for_chart1(model_global, kw_global, num_sample_global, instance_global)
# value_data = plot_data.data_for_chart2(model_global, kw_global, num_sample_global, instance_global)
# dataframe_for_feature_value_table = plot_data.data_for_table(model_global, kw_global, num_sample_global, instance_global)
# explanation_data = plot_data.data_for_explanationchart(model_global, instance_global) 

#####################################################################################################################################
#   App Layout
#####################################################################################################################################

app.title = 'Rusklainer'
app.layout = html.Div([
        html.Div(
            ############################################
            #   Web Page Banner: Title
            ############################################
            className='banner',
            children=[
                html.Div(
                    className='container scalable',
                    children = [
                        html.Div([
                         html.H6(
                            id='banner-title',
                            children=html.Span([html.Span('RUSKLAINER', style={'fontSize': 30, 'fontWeight': 'bold'}), html.Span(' | ', style={'fontSize': 35,}), 'Identification of contributing features towards ' , html.Span('RU', style={'fontWeight': 'bold'}), 'pture ri', html.Span('SK', style={'fontWeight': 'bold'}), ' prediction using ',  
                                                 html.Span('L', style={'fontWeight': 'bold'}), 'ime expl', html.Span('AINER', style={'fontWeight': 'bold'})]),
                            style={
                                "color": "#F7F9FA"
                            }),
                        ]),
                    ]
                ),
        ]),
                                                
        ############################################
        #   Initialize Web Page Tabs
        ############################################
        dcc.Tabs(
                id="tabs-with-classes",
                value='tab-1',
                parent_className='custom-tabs',
                className='custom-tabs-container',
                children=[
                        dcc.Tab(
                                label='Home',
                                value='tab-1',
                                className='custom-tab',
                                selected_className='custom-tab--selected'
                                ),
                        dcc.Tab(
                                label='Help',
                                value='tab-2',
                                className='custom-tab',
                                selected_className='custom-tab--selected',
                                ),
                         ], colors={
                        }),
        html.Div(id='tabs-content-classes'),
        ])

#####################################################################################################################################
#   Tab-wise callback function
#####################################################################################################################################

@app.callback(Output('tabs-content-classes', 'children'),
              [Input('tabs-with-classes', 'value')])
def render_content(tab):
    ############################################
    #   Tab 1: 'Home'
    ############################################
    if tab == 'tab-1':
        return html.Div([
                html.Div([
                dbc.Card(
                dbc.CardBody([
                    # dcc.Loading(
                        dbc.Row([
                            html.Br(),
                            dbc.Col([
                                    ############################################
                                    #   Dropdown for selecting ML models
                                    ############################################
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div(id='model', style={'display':'none'}),
                                            html.H6(id='select-ml-model', children='Select ML model'),
                                            dcc.Dropdown(
                                                id='model_dropdown',
                                                options=[
                                                        {'label': 'XGBoost', 'value': 'XGB',},
                                                        {'label': 'Support Vector Machine', 'value': 'SVM'},
                                                        {'label': 'Random Forest', 'value': 'RForest'}
                                                        ],
                                                value= model_global,
                                                searchable=False,
                                            ),
                                        ])
                                    ], style={"width": "100%"},),
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div(id='instances', style={'display':'none'}),
                                            html.H6(id='select-instances', children='Select instance'),
                                            dbc.Input(
                                                id='inp',
                                                type="number",
                                                max=350,
                                                placeholder="number b/w 0 & 350",
                                                value= instance_global,
                                                debounce=True,
                                            ),
                                        ])
                                    ], style={"width": "100%"},),
                                            
                                    # dbc.Card([
                                    #     dbc.CardBody([
                                    #         html.Div(id='kw_update', style={'display':'none'}),
                                    #         html.H6(id='select-kernelwidth', children='Select kernel width'),
                                    #         dcc.Dropdown(
                                    #             id='kw_dropdown',
                                    #             options=[
                                    #                 {'label': '0.40', 'value': 0.40},
                                    #                 {'label': '0.55', 'value': 0.55},
                                    #                 {'label': '0.60', 'value': 0.60},
                                    #                 {'label': '0.65', 'value': 0.65},
                                    #                 {'label': '2.0', 'value': 2.0}
                                    #                 ],
                                    #             value= kw_global,
                                    #             searchable=False,
                                    #         )
                                    #     ])
                                    # ], style={"width": "100%"},),
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div(id='sample size slider', style={'display':'none'}),
                                            html.H6(id='select-sample-size', children='Select sample size'),
                                            dcc.Slider(
                                                    id='sample_slider',
                                                    min=0,
                                                    max=2,
                                                    step=1,
                                                    marks={
                                                            0: '5000',
                                                            1: '10000',
                                                            2: '15000'
                                                    },
                                                    value=num_sample_global,
                                            )
                                        ])
                                    ], style={"width": "100%"},),
                                            
                                    ############################################
                                    #   Information display for model performance
                                    ############################################
                                    
                                    dbc.Card(
                                        dbc.CardBody([
                                            html.H6(children='Performance of Model', style={'textAlign': 'center',}),
                                            html.Br(),
                                            html.Div([
                                                html.P(id='my_text',children=text_global)
                                            ],style={'fontFamily':'verdana', 'font-size':12}),
                                            html.Div(id='performance_update', style={'display':'none'}),
                                        ]),
                                    ),
                                    html.Div(id='target'),
                                ], width=2),

                                ############################################
                                #   Feature Importance Graph
                                ############################################
                                
                                dbc.Col([
                                    dbc.Card(
                                    dbc.CardBody(
                                        dbc.Row([
                                                html.H4('LIME Explanation'),
                                                ], justify="center", align="center"),)),
                                    dbc.Card(
                                    dbc.CardBody(
                                        dcc.Loading(
                                        children=[
                                            dbc.Row([
                                                html.Div(
                                                [
                                                    html.Span('RUPTURED', style={'color': '#990000', 'fontWeight': 'bold'}),
                                                    html.Span(style={"margin-left": "20px"}),
                                                    html.Span('NON-RUPTURED', style={'color': '#004D00', 'fontWeight': 'bold'})
                                                ]),
                                            ], justify="center", align="center"),
                                            dbc.Row([
                                                html.Div([], id='LIME_div'),
                                                
                                                html.Label('Feature Effect: Coefficients of LIME explanations', style={'textAlign':'center', 'fontFamily':'verdana', 'font-size':14} ),
                                                html.Br(),
                                                html.Div([
                                                    dcc.Markdown('''
                                                        ** The above legend indicates the contribution of each feature to the respective classes. Subsequently, it also indicates how much each feature supports/contradicts the black box prediction.
                                                    '''),
                                                ], style={'fontSize':10, 'textAlign': 'center'}),
                                            ], justify="center", align="center"),
                                        
                                            dbc.Row([
                                                html.Div([
                                                    html.P(id='explainer text',children=[explainer_global])
                                                ], style={'fontSize':15, 'textAlign': 'center'}),
                                                html.Div(id='target2'),
                                            ]),
                                        
                                            dbc.Row([
                                                html.Div([
                                                    dbc.Button('More Explanation',
                                                        id = 'collapse-button1',
                                                        className = 'mb-1',
                                                        color = 'dark',
                                                    ),
                                                ]),
                                            ], justify="center", align="center"),
                                        ],
                                    ))),
                                    dbc.Card(
                                    dbc.CardBody(
                                        dbc.Collapse(
                                            dcc.Loading(
                                            dbc.Row([
                                                html.Div(
                                                    dcc.Graph(
                                                        id='expl_plot',
                                                        # figure = fig,
                                                ), id='explanation_plot', style={'width': '100%'}),
                                            html.Div(
                                                children=[
                                                html.Div(id='kw_slider', style={'display':'none'}),
                                                dcc.Slider(
                                                    id='slider_updatemode',
                                                    min=0.01,
                                                    marks={i: {'label': '{}'.format(i)} for i in list(np.arange(0.01,1.51,0.15).round(2))},
                                                    max=1.5,
                                                    # value=global_vars.OPTIMAL_KW,
                                                    step=0.01,
                                                    # updatemode='drag'
                                                )], style={'width': '87%'}),
                                                ], justify="center", align="center")), id = 'collapse1',
                                        ), 
                                    ),)
                                ], width=7),
                                        
                                ############################################
                                #   Black box prediction prob. graph
                                ############################################
                                dbc.Col([
                                    dbc.Card(
                                    dbc.CardBody(
                                    # html.Div([
                                        html.Div([
                                            html.H6(children='Black-box prediction probabilities'),
                                            dcc.Loading(
                                                html.Div(dcc.Graph(
                                                        id='prediction probability',
                                                ), id='predprob_fig_div'),),
                                            html.Label('Prediction Probability of Rupture status', style={'textAlign':'center', 'fontFamily':'verdana', 'font-size':12} ),
                                        ], id='prediction_prob_div',)
                                    )),
                                ############################################
                                #   Feature value display of instance
                                ############################################
                                dbc.Card(
                                dbc.CardBody(
                                    html.Div([
                                        html.H6(id='feature-label', children='Feature values of Instance {}'.format(instance_global)),
                                        dcc.Loading(
                                            html.Div([
                                                dash_table.DataTable(
                                                    id='feature table',
                                                )
                                            ], id='feature_table_div'),
                                        ),
                                    ])
                                )
                                ),
                            ], width=3),
                        ], no_gutters = True)
                    #)
                    ]
                ), color="light")
                ]),
        ])
    ############################################
    #   Tab 2: 'Help'
    ############################################
    elif tab == 'tab-2':
        return html.Div([
            dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    html.H6("LIME aims to create local models – one for each observation of the dataset to be explained. These local models operate in the neighbourhood of the instance (of the dataset) to be explained.", className="card-title"),
                    html.P("Given below are a step-by-step video tutorial and detailed description of this application. It should give you a better understanding of what each component represents."),
                ], justify="center", align="center")
            ]),
            ]),
            dbc.Card(
            dbc.CardBody([            
                dbc.Row([
                    dbc.Col([
                        html.Div(
                            html.Center(
                                children=[
                                        html.Video(
                                            controls=True,
                                            id = 'tutorial-video',
                                            src="/assets/vid.mp4",
                                            autoPlay=False,
                                            width="60%",
                                        ),
                                ]
                            )
                        )
                    ], width=10),
                    dbc.Col([
                        dbc.Button(
                            "For detailed description",
                            id="collapse-button",
                            className="mb-3",
                            color = 'dark'
                        ),
                    ], width=2, align="center",),
                ], align="center",)])),
                dbc.Card(
                dbc.CardBody(
                    dbc.Collapse(
                        dbc.Row([
                            dbc.Col(
                                html.Img(src='data:image/png;base64,{}'.format(encoded_tool_screenshot.decode()), style={'width': '100%',}),
                            width=7),
                            dbc.Col([
                                dbc.Card(
                                dbc.CardBody([
                                    html.Div([
                                        html.H5('Description'),
                                        dcc.Markdown('''
                                            **1)** Select model - This dropdown is provided with 3 black-box models (XGBoost, Support Vector Machine & Random Forest) having the best performance evaluated against other 5 models using nested cross-validation.\n
                                            **2)** Select instance - Each instance corresponds to the aneurysm data of a patient in the dataset. The LIME model explains the black box prediction of the selected instance. Since the training dataset consists of 351 patients, this input box accepts values from 0 to 350.\n
                                            *Select variables for LIME prediction:* \n
                                            **3)** The kernel width determines how large the neighbourhood is: A small kernel width means that an instance must be very close to influence the local model, a larger kernel width means that instances that are farther away also influence the model. However, there is no best way to determine the width. The chosen kernel width of **0.65** was where the surrogate model was found to converge locally before global convergence when inspected over a range of values to find stable locality. \n
                                            **4)** The sample size allows you to choose the number of samples to be perturbed around the instance of interest **(2)** within the kernel width **(3)**.\n
                                            *Hence LIME model draws **s** number of samples from a normal distribution around the instance **i** within the kernel width **k**. * \n
                                            **5)** Displays the performance of the selected model **(1)** and the corresponding hyperparameter settings chosen for the model using cross-validation.\n
                                            **6)** This graph shows the contribution of features towards the black-box model's **(1)** prediction of rupture status of aneurysm of the selected instance **(2)**. The importance/effect of a feature is evaluated by its coefficients in the LIME model (ridge regression). The features are sorted based on the importance of the predicted class.\n
                                            **7)** Displays the information on whether the instance is predicted correctly/incorrectly by the black-box model on comparison with the actual rupture status from the train set.\n
                                            **8)** This graph shows the black-box model's prediction probabilities of rupture status of aneurysm of the instance\n
                                            **9)** Displays the original value of each feature of the instance selected **(2)**. The values of categorical features (Multiple, Localisation & Side) are encoded to numbers before running the machine learning models. Displayed are the encoded as well as the real values of those features.
                                        '''),
                                    ], style={'fontSize': 12},),
                                ])),
                            ]),
                        ]),
                        id="collapse",
                    )
                )),
        ]),
                    

#####################################################################################################################################
#
#   CALLBACK FUNCTIONS OF EACH COMPONENT
#
#####################################################################################################################################
                                 
# black-box model dropdown callback
@app.callback(Output('model', 'children'),
              [Input('model_dropdown', 'value')])
def handle_dropdown_value(model_dropdown):
    return model_dropdown

# instance number input box callback
@app.callback(Output('instances', 'children'),
              [Input('inp', 'value')])
def update_instance_value(value):
    if value is not None:
        return value
    else:
        return 0
    

# sample size slider callback
@app.callback(Output('sample size slider', 'children'),
              [Input('sample_slider', 'value')])
def update_sample_value(value):
    return value
    

# model performance info text callback
@app.callback(Output('my_text', 'children'),
              [Input('model_dropdown', 'value')])
def update_parameter_text(model_dropdown):
    global text_global
    text_global = plot_data.update_parameter_text(model_dropdown) 
    return text_global



def update_explainer_text(model_dropdown, inp, kw_dropdown, sample_slider):
#    global explainer_global
    if inp is None or inp == 0:
        inp = 0
    text = plot_data.update_explainer_text(model_dropdown, kw_dropdown, sample_slider, inp) 
    if ' correctly' in text:
        text1, text2 = text.split('correctly')
        return html.Span([text1 , html.Span(' correctly ', style={'color': 'green', 'fontWeight': 'bold'}), text2])
    else:
        text1, text2 = text.split('incorrectly')
        return html.Span([text1 , html.Span(' incorrectly ', style={'color': 'red', 'fontWeight': 'bold'}), text2])



def update_LIME_plot(model_dropdown, inp, kw_slider, sample_slider):
    if inp is None or inp == 0:
        inp = 0
        
    LIME_fig = {'data': plot_data.data_for_chart1(model_dropdown, kw_slider, sample_slider, inp),
                'layout': {
                    # 'height': '50%',
                    'margin': {'l': 180, 'b': 20, 't': 30, 'r': 30},
                    'xaxis' : {'range': [-.25,.25]},
                    'yaxis' : dict(autorange="reversed"),
                    'paper_bgcolor' : 'rgba(0,0,0,0)',
                    'plot_bgcolor' : 'rgba(0,0,0,0)'
                    }
                }    
    return dcc.Graph(
        id='LIME',
        figure = LIME_fig
    )
        
        
def global_store(model_dropdown, kw_dropdown, sample_slider, inp):
    value_data = plot_data.data_for_chart2(model_dropdown, kw_dropdown, sample_slider, inp)
    return value_data

def generate_figure(model_dropdown, kw_dropdown, sample_slider, inp, figure):
    fig = copy.deepcopy(figure)
    value_data = global_store(model_dropdown, kw_dropdown, sample_slider, inp)
    fig['data'][0]['x'] = [round(value_data[2],3), round(value_data[1],3)]
    fig['layout'] = {'height': 120, 'margin': {'l': 45, 'b': 25, 't': 30, 'r': 10}, 'xaxis': {'range': [0, 1]}, 'paper_bgcolor' : 'rgba(0,0,0,0)',  'plot_bgcolor' : 'rgba(0,0,0,0)', 'font' : {'size':10}}
    return fig

# detailed explanation collapsable
@app.callback(
    Output("collapse1", "is_open"),
    [Input("collapse-button1", "n_clicks")],
    [State("collapse1", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

def update_blackbox_prediction_prob(model_dropdown, inp, kw_dropdown, sample_slider):
    global value_data, instance_global
    if inp is None or inp == 0:
        inp = 0
        
    pred_fig = generate_figure(
        model_dropdown, 
        kw_dropdown, 
        sample_slider, 
        inp, 
        {'data': [
            go.Bar(
                y = ['Non-rup', 'Rup'],
                marker= dict(
                    color=['#004c00', '#990000']),
                orientation='h'
                )],
        })    
    
    return dcc.Graph(
        id='prediction probability',
        figure=pred_fig,
        )
        
# feature value display table label callback
@app.callback(Output('feature-label', 'children'),
              [Input('inp', 'value')])
def update_label(value):
    if value is not None:
        return 'Feature values of Instance {}'.format(value)
    else:
        return 'Feature values of Instance {}'.format(0)        
        
def update_table(model_dropdown, inp, kw_dropdown, sample_slider):
    global dataframe_for_feature_value_table, instance_global
    if inp is None or inp == 0:
        inp = 0
    dataframe_for_feature_value_table = plot_data.data_for_table(model_dropdown, kw_dropdown, sample_slider, inp)
    
    return dash_table.DataTable(
        id='feature table',
        columns=[{"name": i, "id": i} for i in dataframe_for_feature_value_table.iloc[:, : 2].columns],
        data=dataframe_for_feature_value_table.to_dict("rows"),
        style_table={
            'maxHeight': '100%',
            'width': '100%',
            'minWidth': '100%',
        },
        style_cell={
            'fontFamily': 'verdana',
            'textAlign': 'left',
            'font_size': '13px',
            'whiteSpace': 'inherit',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },  
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ]
    )
 
    
# feature importance graph callback
@app.callback(
        [Output('explanation_plot', 'children'), Output('expl_plot', 'selectedData'), 
         Output('slider_updatemode', 'value'), Output('LIME_div', 'children'), Output('explainer text', 'children'),
         Output('feature_table_div', 'children'), Output('predprob_fig_div', 'children')],
        [Input('expl_plot', 'figure'), Input('expl_plot', 'selectedData'),
         Input('slider_updatemode', 'value'), Input('model_dropdown', 'value'), 
         Input('inp', 'value'), Input('sample_slider', 'value')])

def explanation_callback(fig, clicked_point, kw, model_dropdown, inp, sample_slider):
    if inp is None or inp == 0:
        inp = 0
        
    print('in explanation_callback: ', kw) 
    print('clicked_point in explanation_callback: ', clicked_point)
    
    global current_model, current_instance, explanation_data
    
    if current_model != model_dropdown or current_instance != inp:   
        explanation_data_ = plot_data.data_for_explanationchart(model_dropdown, inp) 
        current_model = model_dropdown 
        current_instance = inp
        explanation_data = explanation_data_
        kw = explanation_data[1]
        fig = go.Figure() 
    
        for trace in explanation_data[0]:
            fig.add_trace(trace)
            
        print("fig data: ", len(fig.data))
    else:
        explanation_data = explanation_data
        # if clicked_point:
        #     kw = clicked_point['points'][0]['x']
        fig = go.Figure(fig)
        
    if clicked_point:
        kw = clicked_point['points'][0]['x']
        clicked_point = None
    elif kw == None:
        kw = explanation_data[1]
    
    print('in explanation_callback: ', kw) 
    print('clicked_point in explanation_callback: ', clicked_point)    
    print('kw: ', kw)
    
    fig.update_layout(
        xaxis=dict(
            range=[0.01, 1.5], 
            fixedrange=True, 
            tickmode = 'array',
            tickvals = list(np.arange(0.01,1.51,0.15).round(2)),
            title = dict(text = "Kernel Width"),
            autorange=False),
        yaxis = dict(
            range=[explanation_data[3][1] - 0.15, explanation_data[3][0] + 0.15], 
            autorange=False,
            title = dict(text = "R squared")
            ),
        margin = {'t': 30, 'b': 10},
        # hovertemplate="<br>".join([
        #     # "ColX: %{x}",
        #     # "ColY: %{y}",
        #     "global: %{customdata[0]}",
        #     "local: %{customdata[1]}",
        #     "Col3: %{customdata[2]}",
        # ]),
        hovermode = 'x unified',
        shapes = [{
            'type': 'line',
            'x0': kw,
            'y0': explanation_data[3][1] - 0.15,
            'x1': kw,
            'y1': explanation_data[3][0] + 0.15,
            'line': {
                'color': '#00cc96',
                'width': 2,
                'dash' : 'dashdot'
            }},],
        paper_bgcolor='rgba(0,0,0,0)',
        clickmode='event+select',
        legend=dict(y=1.1, orientation='h')
        # legend_orientation='h',
    )        
    
    # print(fig['layout'])
    
    explainer_text = update_explainer_text(model_dropdown, inp, kw, sample_slider)
    dt = update_table(model_dropdown, inp, kw, sample_slider)
    pred_graph = update_blackbox_prediction_prob(model_dropdown, inp, kw, sample_slider)
    LIME_plot = update_LIME_plot(model_dropdown, inp, kw, sample_slider)

    
    print(explainer_text)
    
    # expl_config = { 'modeBarButtonsToRemove': ['sendDataToCloud', '!autoScale2d', 'resetScale2d', 'hoverClosestCartesian', 'hoverCompareCartesian', 'lasso2d', 'select2d'], 'displaylogo': False, 'showTips': True }
    
    return dcc.Graph(id='expl_plot', figure=fig), clicked_point, kw, LIME_plot, explainer_text, dt, pred_graph

if __name__ == '__main__':
    app.run_server(debug=True)
    # , dev_tools_ui=False,dev_tools_props_check=False