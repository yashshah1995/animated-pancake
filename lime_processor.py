"""
@author: Yash Shah

This file is the core file for LIME model. 

"""



#####################################################################################################################################
#   Import Libraries
#####################################################################################################################################

import numpy as np
import pandas as pd
from sklearn import preprocessing 
import pickle
import lime
import lime.lime_tabular
import global_vars
from functools import partial
import matplotlib.pyplot as plt 
from sklearn.metrics import pairwise_distances, r2_score
from tqdm.auto import tqdm

# The lime package is at C:\Users\Sachin Nandakumar\AppData\Roaming\Python\Python37\site-packages\sklearn\linear_model\ridge.py

class LIME_Explainer:
    def __init__(self):
        self.encoder = pickle.load(open(global_vars.ENCODER, 'rb'))
        self.R_squared_dict = {}
        self.categorical_features = [0,1,2]
        self.categorical_names = {}
        
    def preprocess_train(self, model):
        # , scaled_encoded, anuer, anuer1, X3
        '''
            This function does basic preprocessing of the files fetched from get_data 
        '''

        filename = 'pickled_models/'+model.lower()+'.pkl'
        model = pickle.load(open(filename, 'rb'))
        
        scaled_encoded = pd.read_csv(global_vars.SCALED_ENCODED_CSV,header=None)
        
        anuer = pd.read_csv(global_vars.DATA)
        
        # anuer = anuer[:, :3]
        
        label_col = anuer['Rupturiert']
        anuer = anuer.drop(columns = ['Rupturiert'])
        
        anuer1 = pd.read_csv(global_vars.DATA1)
        anuer1 = anuer1.drop(columns = ['Rupturiert'])
        # scaling_anuer = anuer1.copy()
        

        for feature in self.categorical_features:
            le = preprocessing.LabelEncoder()
            le.fit(anuer.iloc[:, feature])
            anuer.iloc[:, feature] = le.transform(anuer.iloc[:, feature])
            anuer1.iloc[:, feature] = le.transform(anuer1.iloc[:, feature])
            self.categorical_names[feature] = list(range(le.classes_.max()))
            
        le= preprocessing.LabelEncoder()
        le.fit(label_col)
        label_col = le.transform(label_col)
        class_names = le.classes_
        
        return model, scaled_encoded, anuer, anuer1, label_col, le, class_names
    
    
    
    def explainer(self, model, x1, x2, class_names, 
                  instance_no, krw, numsamp, RETURN="all"):
        '''
            This function runs the tabular explainer for LIME and outputs the scores, attributes and predictions
        '''
        predict_fn = lambda x: model.predict_proba(self.encoder.transform(x)).astype(float)
        
        explainer1 = lime.lime_tabular.LimeTabularExplainer(np.array(x2),feature_names = x2.columns.tolist(),
                                                     class_names=class_names,
                                                     mode="classification",
                                                    categorical_features=self.categorical_features, 
                                                    categorical_names=self.categorical_names,
                                                    feature_selection = 'forward_selection', 
                                                    sample_around_instance=True,
                                                    discretize_continuous=False, kernel_width=krw,random_state=42) 
        
        
        exp1 = explainer1.explain_instance(x2.iloc[instance_no,:], predict_fn)
        
        
        if exp1.score>1 or exp1.score<0:
            print(f'score: {exp1.score}, instance: {instance_no}, krw: {krw}')
        
        if RETURN == "R2":
            if instance_no == 0:
                self.R_squared_dict[krw] = [exp1.score]
            else:
                self.R_squared_dict[krw].append(exp1.score)
            return self.R_squared_dict
        
        check_list = exp1.as_list()
        colus = x2.columns.tolist()
        temp_frame = []
        for j in range(len(check_list)):
            contained = [x for x in colus if x in check_list[j][0]]
            temp_frame.append(contained[0])
        
        empty = {k:[] for k in temp_frame}
        
        for j in range(len(check_list)):
            contained = [x for x in colus if x in check_list[j][0]]
            empty[contained[0]] = check_list[j][1]
        
        score = {k: v for k, v in sorted(empty.items(), key=lambda item: abs(item[1]),reverse=True)}
        
        store_attribute = dict(x2.iloc[instance_no,:])
        
        val = np.array(x1.iloc[instance_no,:])[np.newaxis]
        preddict = {k:[] for k in [1,2]}
        preddict[1] = model.predict_proba(val)[0][0]
        preddict[2] = model.predict_proba(val)[0][1]
        
        return exp1.score, score, store_attribute, preddict, exp1.weights, exp1.distances
    
    def run_LIME(self, model, kernel_width, sample_size, instance_number):
        '''
            main()
        '''
        # model, processed, anuer, anuer1, label_col = self.get_data(model)
        model, processed, anuer, anuer1, label_col, _,  class_names = self.preprocess_train(model) #, processed, anuer, anuer1, label_col
        R_2, score_dict, attri_dict, probab_dict, weights, distances = self.explainer(model, processed, anuer, class_names,
                                                    instance_number, kernel_width, sample_size, RETURN="all")
        
        
        
        return score_dict, attri_dict, probab_dict, label_col[instance_number]
    
    
    def get_scaler(self, anuer):
        scaler = preprocessing.StandardScaler(with_mean=False)
        scaler.fit(anuer.values)
        for index, col in enumerate(['Multipel', 'Lokalisation', 'Seite']):
            scaler.mean_[index] = 0
            scaler.scale_[index] = 1
            
        return scaler
    
    
    def get_black_box_predictors(self, anuer, model, label_col, le):
        
        encoded_set = pd.DataFrame(self.encoder.fit_transform(anuer))
        
        encoded_set.columns = ['f'+str(i) for i in range(22)]
        predict_fn = lambda x: model.predict_proba(self.encoder.transform(x)).astype(float)
        
        
        # Get the prediction probabilities of original instances
        blackbox_preds = model.predict(encoded_set)
        # blackbox_preds_1 = pd.DataFrame(blackbox_preds)
        blackbox_pred_prob = model.predict_proba(encoded_set)
        blackbox_pred_prob_1 = pd.DataFrame(blackbox_pred_prob)
        
        # le= preprocessing.LabelEncoder()
        # le.fit(blackbox_preds)
        # label_col = le.transform(label_col)
        blackbox_preds = le.transform(blackbox_preds)
        
        class_names = le.classes_
        
        return blackbox_preds, blackbox_pred_prob_1, predict_fn, class_names
        
    
    
    def kernel(self, d, kernel_width):
        return np.sqrt(np.exp(-(d * 2) / kernel_width * 2))
    
    def get_original_instance(self, instance, model): 
        # model, processed, anuer, anuer1, label_col = self.get_data(model)
        model, _, anuer, anuer1, label_col, le, _ = self.preprocess_train(model)
        anuer_copy = anuer1.copy()
        data_row = anuer_copy.iloc[instance, :]
        
        anuer_copy["new"] = range(1,len(anuer_copy)+1)
        anuer_copy.loc[instance,'new'] = 0
        anuer_copy = anuer_copy.sort_values("new").reset_index(drop='True').drop('new', axis=1)
        
        for index,col in enumerate(['Multipel', 'Lokalisation', 'Seite']):
           anuer_copy.loc[anuer_copy[col] != data_row[col], col] = 0
           anuer_copy.loc[anuer_copy[col] == data_row[col], col] = 1
    
        scaler = self.get_scaler(anuer1)
    
        scaled_rest_rows = (anuer_copy - scaler.mean_) / scaler.scale_ 
         
        data_row1 = scaled_rest_rows.iloc[0, :]
        
        distances = pairwise_distances(scaled_rest_rows, data_row1.values.reshape(1,-1), 
                                               metric='euclidean').ravel()
        
        blackbox_preds, blackbox_pred_prob_1, predict_fn, class_names = self.get_black_box_predictors(anuer, model, label_col, le)
        
        selected_probs = pd.DataFrame(blackbox_pred_prob_1[blackbox_preds[instance]].to_numpy())
        
        selected_probs["new"] = range(1,len(selected_probs)+1)
        selected_probs.loc[instance,'new'] = 0
        selected_probs = selected_probs.sort_values("new").reset_index(drop='True').drop('new', axis=1)
    
        selected_probs = selected_probs[0].to_numpy()
        
        return scaled_rest_rows, selected_probs, distances, anuer, anuer1, class_names, predict_fn, blackbox_preds

    
    def smoothTriangle(self, data, degree):
        triangle=np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1])) # up then down
        smoothed=[]
    
        for i in range(degree, len(data) - degree * 2):
            point=data[i:i + len(triangle)] * triangle
            smoothed.append(np.sum(point)/np.sum(triangle))
        # Handle boundaries
        smoothed=[smoothed[0]]*int(degree + degree/2) + smoothed
        while len(smoothed) < len(data):
            smoothed.append(smoothed[-1])
        return smoothed
    
    def runningMeanFast(self, x, N):
        return np.convolve(x, np.ones((N,))/N, mode = 'valid')[(N-1):]

    def get_glo_loc_plot(self, kernel_width_vector, model, instance):
    
        scaled_rest_rows, selected_probs, distances, anuer, anuer1, class_names, predict_fn, blackbox_preds = self.get_original_instance(instance, model)
        localvals = []
        globalvals = []
        
        for kw in tqdm(kernel_width_vector):
            # print("---------------")
            # print(kw)
            explainer1 = lime.lime_tabular.LimeTabularExplainer(training_data = np.array(anuer1),
                                           feature_names = anuer.columns.tolist(),
                                           class_names=class_names,
                                           categorical_features = self.categorical_features, 
                                           categorical_names = self.categorical_names,
                                           feature_selection = 'none',
                                           sample_around_instance=True,
                                           discretize_continuous=False, 
                                           kernel_width = kw,
                                           random_state=42)
    
            exp1 = explainer1.explain_instance(anuer1.iloc[instance,:].values, 
                                               predict_fn, num_samples = 5000, 
                                               num_features = 12,
                                               labels = (blackbox_preds[instance],))  
            
            pertubed_data = exp1.scaled_data
            pertub_label = exp1.label_col
            perturb_weights = exp1.weights
            
            kernel_fn = partial(self.kernel, kernel_width=kw)
            weights = kernel_fn(distances)
            
            globalvals.append(r2_score(selected_probs, exp1.easy_model.predict(scaled_rest_rows), weights))
            localvals.append(r2_score(pertub_label, exp1.easy_model.predict(pertubed_data), perturb_weights))
            
            
        
        localvals = np.asarray(localvals)
        globalvals = np.asarray(globalvals)
        
        localvals[localvals <= -1] = -1.0
        localvals[localvals >= 1] = 1
    
        globalvals[globalvals <= -1] = -1.0
        globalvals[globalvals >= 1] = 1
    
        glob_smooth = self.runningMeanFast(globalvals, 5) 
        loc_smooth = self.runningMeanFast(localvals, 5) 
        
        print(f'Length of globalvals: {len(glob_smooth)}, and localvals: {len(loc_smooth)}')
        
        neg_counter = -1
    
        for vals in range(len(kernel_width_vector[:len(loc_smooth)])):
        
            if (glob_smooth[neg_counter] < loc_smooth[neg_counter]):
                if (glob_smooth[-1] < loc_smooth[-1]):
                    neg_counter-=1
                    continue
                else:
                    if (glob_smooth[-1] < loc_smooth[-1]):
                        statement = 'optimal KrW might not be stable'
                        krwidval = neg_counter+1
                        break
                    else:
                        statement = ''
                        krwidval = neg_counter+1
                        break
            elif (glob_smooth[neg_counter] == loc_smooth[neg_counter]):
                
                if (glob_smooth[-1] < loc_smooth[-1]):
                    statement = 'optimal KrW might not be stable'
                    krwidval = neg_counter+1
                    break
                else:
                    statement = ''
                    krwidval = neg_counter+1
                    break
            else:
                neg_counter-=1
        
        if(neg_counter*-1 == len(kernel_width_vector[:len(loc_smooth)])+1):
            statement = 'only global convergence stable'
            krwidval = -1
        
        
        # #plt.plot(kernel_width_vector, localvals,'.b-',label = 'local')
        # plt.legend()
        # plt.suptitle("local vs. global for instance {} \n optimal_value {} \n {} \n actual {} predicted {}".format(instance, kernel_width_vector[:len(loc_smooth)][krwidval], statement, label_col[get_instance], blackbox_preds[get_instance]))    


        
    
        # neg_counter = -1
        
        # lowessglob = sm.nonparametric.lowess(globalvals, kernel_width_vector, frac=0.26)
        # lowessloc = sm.nonparametric.lowess(localvals, kernel_width_vector, frac=0.26)
        
        # moving_avg_global = self.smoothTriangle(globalvals, 5)
        # moving_avg_local = self.smoothTriangle(localvals, 5)
        
        # globalvals = savgol_filter(globalvals, 17, 3)
        # localvals = savgol_filter(localvals, 17, 3)
    
        # for index in list(reversed(range(len(kernel_width_vector)))):
        #     if globalvals[index] <= localvals[index]:
        #         krwidval = index
        #         break
        #     else:
        #         neg_counter-=1
        
        # for index in list(reversed(range(len(kernel_width_vector)))):
        #     if (lowessglob[:,1][index] <= lowessloc[:,1][index]):
        #         # if (lowessglob[:,1][-1] < lowessloc[:,1][-1]):
        #         krwidval = index
        #         break
            
        # print(moving_avg_global)
            
        # for index in list(reversed(range(len(kernel_width_vector)))):
        #     if (moving_avg_global[index] <= moving_avg_local[index]):
        #         # if (lowessglob[:,1][-1] < lowessloc[:,1][-1]):
        #         krwidval = index
        #         break
                    # neg_counter-=1
                    # continue
                # else:
                    # if (lowessglob[:,1][-1] < lowessloc[:,1][-1]):
                        # statement = 'optimal KrW might not be stable'
                        # krwidval = index
                        # break
                    # else:
                        # statement = ''
                        # krwidval = index
                        # break
            # elif (lowessglob[:,1][index] == lowessloc[:,1][index]):
            #     if (lowessglob[:,1][-1] < lowessloc[:,1][-1]):
            #         # statement = 'optimal KrW might not be stable'
            #         krwidval = index
            #         break
            #     else:
            #         # statement = ''
            #         krwidval = index
            #         break
            # else:
            #     neg_counter-=1
            
    
        # print(krwidval, lowessglob[:,1][krwidval], lowessloc[:,1][krwidval], )
        # print(krwidval, moving_avg_global[krwidval], moving_avg_local[krwidval], )
        
        # fig, axes = plt.subplots(1,2)
        # # #plt.plot(kernel_width_vector, globalvals,'.r-',label = 'global')
        # # #plt.plot(kernel_width_vector, smooth(globalvals,15),'.g-',label = 'global_smooth')
        # axes[0].plot(kernel_width_vector, globalvals,'.r-',label = 'global')
        # axes[0].plot(kernel_width_vector, localvals,'.b-',label = 'local')
        # axes[1].plot(kernel_width_vector[:len(glob_smooth)], glob_smooth,'.r-',label = 'globalsmoothed')
        # axes[1].plot(kernel_width_vector[:len(loc_smooth)], loc_smooth,'.b-',label = 'localsmoothed')
        
        # # #lowess = sm.nonparametric.lowess(globalvals, kernel_width_vector, frac=0.2)
        # # #plt.plot(lowess[:, 0], lowess[:, 1],'.g-',label = 'global-smoothed')
        # # #plt.plot(kernel_width_vector, moving_average(globalvals, 4),'.g-',label = 'global-smoothed')
        
        
        # # #plt.plot(kernel_width_vector, localvals,'.b-',label = 'local')
        # plt.legend()
        # plt.suptitle("local vs. global for instance {} \n optimal_value {}".format(instance, kernel_width_vector[krwidval]))   
        
        # # because range takes [0, 1.5) whereas the kernelwidth takes [0.01, 0.5]. One value less for kernelwidth
        optimal_kw = kernel_width_vector[:len(loc_smooth)][krwidval]
        # #  - 0.01
    
        # # print(localvals, lowessloc[:,1])
        # # return optimal_kw, krwidval, lowessglob, lowessloc
        
        
        print(f'optimal kw: {optimal_kw}')
        
        return optimal_kw, krwidval, glob_smooth, loc_smooth
    
    # def run_LIME_(self, model, total_instances):
        # '''
            # main()
        # '''
        
        # # def warn(*args, **kwargs):
        # #     pass
        # # import warnings
        # # warnings.warn = warn
        
        # kernel_width_vector = list(np.arange(0.01,1.2,0.01).round(2))
        
        # R_squared = {}
        # model, processed, anuer, label_col = self.get_data(model)
        # processed, anuer, label_col, class_names, categorical_names, categorical_features1 = self.preprocess_train(model, processed, anuer, label_col)
        
        # for i in tqdm(range(total_instances)):
            # for kernel_width in kernel_width_vector:
                # R_squared = self.explainer(model, processed, anuer, categorical_features1, 
                                            # categorical_names, class_names, i, kernel_width, 5000, RETURN = "R2")
        # return R_squared
     
        
    
    
#####################################################################################################################################
#   Code for Testing single explanation
#####################################################################################################################################

# lime_exp = LIME_Explainer()
# kernel_width_values = list(np.arange(0.01,1.51,0.01).round(2))
# score_dict, attri_dict, probab_dict, x = lime_exp.run_LIME(model='xgb', kernel_width=0.06, sample_size=5, instance_number=10)
# x = lime_exp.get_glo_loc_plot(kernel_width_values, 'xgb', 105)
# d = x[4]
# w = x[5]
# l_s = x[6]
# g_s = x[7]
# w_ = [(i, w[i]) for i, element in enumerate(w) if element!=0]
# print(R_2, score_dict, attri_dict, probab_dict)
    
# html = exp1.as_html()
# with open("121_.html", "w") as file:
#     file.write(html)
