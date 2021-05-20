# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 22:29:09 2020

@author: Sachin Nandakumar

This file serves to provide data for graphs & texts (dynamic features of UI) in the required format for app.py file


"""

#####################################################################################################################################
#   Import Libraries
#####################################################################################################################################

from lime_processor import LIME_Explainer
import global_vars
import pandas as pd
import numpy as np
import dash_html_components as html
import plotly.graph_objs as go


class Plot_Creator:
    
    def __init__(self):
        self.lime = LIME_Explainer()
        self.kernel_dict = {0: '0.40', 1: '0.55', 2: '0.6', 3: '0.65', 4: '2.0'}
        self.kernel_width_vector = list(np.arange(0.01,1.91,0.01).round(2))
        # self.kernel_width_vector = list(np.arange(0.1,1.5,0.1).round(2))
        self.sample_dict = {0: '5000', 1: '10000', 2: '15000'}
        self.score = {}
        self.values = {}
        self.score = {}
        self.get_score()
        self.de2en_features = {'Multipel': 'multiple', 'Lokalisation': 'localisation', 'Max. Höhe (Hmax)':'max. height (Hmax)', 'Trägergefäß (T2)': 'parent vessel (T2)', 'Seite':'side', ' Trägergefäß (T1)':'parent vessel (T1)', 'Size ratio (Hmax/T)': 'size ratio (Hmax/H)'}
        self.lower_case_features = {'Width (W)': 'width (W)', 'Neck (D)':'neck (D)'}
    
    def get_score(self, model_dropdown='xgb', kw_dropdown='0.65', sample_slider=0, inp=121):
        '''
            Common function for calling main() of LIME explainer (lime_processor.py) and update variables.
            Also, provided with initial values
        '''
        self.score, self.values, self.probability, self.label = self.lime.run_LIME(model_dropdown, kw_dropdown, int(self.sample_dict[sample_slider]), int(inp))
        
        
    def data_for_chart1(self, model_dropdown, kw_dropdown, sample_slider, inp):
        '''
            Function to return data for feature importance graph by initiating
            get_score() method with updated variables & creating traces
        '''
        self.get_score(model_dropdown, kw_dropdown, sample_slider, inp)
        
        ruptured_class_data = {}
        unruptured_class_data = {}
        
        for key, value in self.score.items():
            if key in self.de2en_features.keys():
                key = self.de2en_features[key]
            elif key in self.lower_case_features.keys():
                key = self.lower_case_features[key]
            else:
                key = key.lower()
            if value<0:
                value = round(value, 5)
                new_key = str(key)+'='+str(abs(value))
                ruptured_class_data[new_key] = value
            else:
                value = round(value, 5)
                new_key = str(key)+'='+str(abs(value))
                unruptured_class_data[new_key] = value
        
        if ruptured_class_data:
            ruptured_class_data = {k: v for k, v in sorted(ruptured_class_data.items(), reverse=True, key=lambda item: item[1])}
            
        if unruptured_class_data:
            unruptured_class_data = {k: v for k, v in sorted(unruptured_class_data.items(), reverse=True, key=lambda item: item[1])}
        
        trace2 = {
        "uid": "1b22a06c-4601-4add-a435-9cc8ada81d21", 
        "text": '', 
        "type": "bar", 
        "x": list(ruptured_class_data.values()), 
        "y": list(ruptured_class_data), 
        'mode': 'lines',
        "marker": { "color": '#990000' },
        "orientation": "h",
        'showlegend': False,
        'hoverinfo': 'none'
        }
        trace1 = {
        "uid": "1f978ddf-8dba-49ec-a3c4-4fd1abc7a1e0", 
        "text": '', 
        "type": "bar", 
        "x": list(unruptured_class_data.values()), 
        "y": list(unruptured_class_data), 
        'mode': 'lines',
        "marker": { "color": '#004c00' },
        "orientation": "h",
        'showlegend': False,
        'hoverinfo': 'none'
        }
        return ([trace1, trace2])
    
    
    def data_for_explanationchart(self, model_dropdown, inp):
        '''
            Function to return data for feature importance graph by initiating
            get_score() method with updated variables & creating traces
        '''
        
        optimal_kw, krwidval, glob_smooth, loc_smooth = self.lime.get_glo_loc_plot(self.kernel_width_vector, model_dropdown, inp)
        
        # max_limit = max([max(local_smoothened_values[:,1]), max(global_smoothened_values[:,1])])
        # min_limit = min([min(local_smoothened_values[:,1]), min(global_smoothened_values[:,1])])
        
        max_limit = max([max(loc_smooth), max(glob_smooth)])
        min_limit = min([min(loc_smooth), min(glob_smooth)])
        
        
        # print([min(local_smoothened_values[:,1]), min(global_smoothened_values[:,1])])
        
        
        print(len(glob_smooth))
        
        # trace1_col = ['#004c00'] * len(self.kernel_width_vector)
        # trace1_col[krwidval] = '#000000'
        trace1 = go.Scatter(
            # x = self.kernel_width_vector, 
            x = self.kernel_width_vector[:len(glob_smooth)],
            # y = global_smoothened_values[:,1], 
            y = glob_smooth, 
            mode = 'lines+markers',
            marker = { "color": '#EF553C' },
            name = "global",
            # "orientation": "h",
            # 'showlegend': False,
            hoverinfo = 'y',
            # hovermode = 'closest'
        )
        
        # trace2_col = ['#990000'] * len(self.kernel_width_vector)
        # trace2_col[krwidval] = '#000000'
        trace2 = go.Scatter(
            # x = self.kernel_width_vector,
            x = self.kernel_width_vector[:len(loc_smooth)],
            # y = local_smoothened_values[:,1], 
            y = loc_smooth, 
            mode = 'lines+markers',
            marker = { "color": '#636EFA' },
            name = "local",
            # "orientation": "h",
            # 'showlegend': False,
            hoverinfo = 'y',
            # hovermode = 'closest'
        )
        
        print('optimal kw: ', optimal_kw)
        print(self.kernel_width_vector[krwidval])
        
        trace3 = go.Scatter(
            x=[optimal_kw], 
            y=[min_limit - 0.10], 
            # y=0, 
            mode='markers',
            hoverinfo = 'x',
            marker = dict(symbol='triangle-down',
                             size=16, #[0, 16, 0, 16,0, 16],
                             color='#000000'),
            # dict(line=dict(color='black', width = 1),
            #                       symbol = 'x-dot',
            #                       size = 14,
            #                       color = '#d8ffd8'),
            # line=go.scatter.Line(color="gray"),
            name = 'Optimal Kernel Width',
            # showlegend=True
        )
        
        return ([trace1, trace2, trace3], optimal_kw, self.kernel_width_vector, [max_limit, min_limit], optimal_kw/0.01)
    
    
    def data_for_chart2(self, model_dropdown, kw_dropdown, sample_slider, inp):
        '''
            Function to return model prediction probability data by 
            initiating get_score() method with updated variables
        '''
        self.get_score(model_dropdown, kw_dropdown, sample_slider, inp)
        return self.probability
    
    
    
    
    def data_for_table(self, model_dropdown, kw_dropdown, sample_slider, inp):
        '''
            Function to create dataframe with data required to display table of
            feature values
        '''
        self.get_score(model_dropdown, kw_dropdown, sample_slider, inp)
        rows_list = []
        for key, value in self.values.items():
            if key in self.de2en_features.keys():
                key = self.de2en_features[key]
            elif key in self.lower_case_features.keys():
                key = self.lower_case_features[key]
            else:
                key = key.lower()
            value = round(value, 3)
            
            if key.strip() in self.score:
                if self.score[key.strip()]<0:
                    status = 1
                else:
                    status = 2  
            else:
                status = 0
                
            if key.strip() in global_vars.UNIT_DICT.keys():
                value = str(value) + ' ' +global_vars.UNIT_DICT[key.strip()]
            elif key.strip() in global_vars.TYPE_DICT.keys():
                value = str(value + 1) + ' (' +global_vars.TYPE_DICT[key.strip()][value+1] + ')'
            elif not key.strip() == 'Size ratio (Hmax/H)':
                value = str(value) + global_vars.DEGREE_SIGN
                
            row_wise_dict = {'Features': key, 'Values': value, 'Status': status}
            rows_list.append(row_wise_dict)
        return pd.DataFrame(rows_list)
    
    
    
    
    def update_parameter_text(self, model_dropdown):
        '''
            Function to update the model performance text area
        '''
        if model_dropdown.lower() == 'xgb':
            return html.P([html.Span('Model', style={'fontWeight': 'bold'}), ':', html.Span('XGBoost', style={'fontWeight': 'bold'}), html.Br(), html.Span('Accuracy', style={'fontWeight': 'bold'}), ':', html.Span('70.3', style={'fontWeight': 'bold'}), html.Br(), html.Br(), html.Span('Best Hyperparameter Settings', style={'fontWeight': 'bold'}), ': ', html.Br(), '  n_estimators: 100', html.Br(), '  eta: 0.3', html.Br(), '  colsample_bytree: 1 ', html.Br(), '  max_depth: 2', html.Br(), '  gamma: 4', html.Br(), '  subsample: 1', html.Br(), '  min_child_weight: 4', html.Br(), '  objective: binary:logistic', html.Br(), '  sketch_eps: 0.5', html.Br(), '  tree_method: approx'])
        elif model_dropdown.lower() == 'svm':
            return html.P([html.Span('Model', style={'fontWeight': 'bold'}), ':', html.Span('Support Vector Machine', style={'fontWeight': 'bold'}), html.Br(), html.Span('Accuracy', style={'fontWeight': 'bold'}), ':', html.Span('67', style={'fontWeight': 'bold'}), html.Br(), html.Br(), html.Span('Best Hyperparameter Settings', style={'fontWeight': 'bold'}), ': ', html.Br(), '  C: 0.2', html.Br(), '  gamma: auto', html.Br(), '  kernel: linear'])
        else:
            return html.P([html.Span('Model', style={'fontWeight': 'bold'}), ':', html.Span('Random Forest', style={'fontWeight': 'bold'}), html.Br(), html.Span('Accuracy', style={'fontWeight': 'bold'}), ':', html.Span('66.4', style={'fontWeight': 'bold'}), html.Br(), html.Br(), html.Span('Best Hyperparameter Settings', style={'fontWeight': 'bold'}), ': ', html.Br(), '  criterion: entropy', html.Br(), '  max_depth: 4', html.Br(), '  min_samples_leaf: 2 ', html.Br(), '  min_samples_split: 3', html.Br(), '  n_estimators: 800', html.Br(), '  bootstrap: True', html.Br(), '  oob_score: True'])
    
    # 'Accuracy: 70.3'
    # html.Span('Accuracy', style={'fontWeight': 'bold'}), ':', html.Span('70.3', style={'fontWeight': 'bold'})
    
    
    def update_explainer_text(self, model_dropdown, kw_dropdown, sample_slider, inp):
        '''
            Function to update the explainer text
        '''
        self.get_score(model_dropdown, kw_dropdown, sample_slider, inp)
        
        if model_dropdown.lower() == 'xgb':
            model = 'XGBoost'
        elif model_dropdown.lower() == 'svm':
            model = 'Support Vector Machine'
        else:
            model = 'Random Forest'
        if self.label == 0:
            actual_status = 'Ruptured'
        else:
            actual_status = 'Non-ruptured'
        if self.probability[1] > self.probability[2]:
            predicted_status = 'Ruptured'
        else:
            predicted_status = 'Non-ruptured'
        if actual_status == predicted_status:
            prediction = 'correctly'
        else:
            prediction = 'incorrectly'
        probability = str(round(max(self.probability.values()),4))
        return 'The model ({}) predicts the instance {} of the training set as {} with probability {}. The actual class is {} which means that the model has {} classified the instance!'.format(model, inp, predicted_status, probability , actual_status, prediction)




#####################################################################################################################################
#   Code for Unit-testing
#####################################################################################################################################

# uncomment below code for testing this file

# plt = Plot_Creator()
#print(plt.data_for_table('xgb', 0, 0, 121))
#plt.data_for_table('xgb', 0, 0, 101)
#plt.update_explainer_text('xgb', 121)

# plt.data_for_explanationchart('xgb', 121)