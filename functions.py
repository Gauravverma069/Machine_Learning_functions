
# Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import root_mean_squared_error,r2_score,mean_squared_error,root_mean_squared_log_error,mean_absolute_error,mean_squared_log_error
from sklearn.linear_model import LinearRegression

from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from scipy.stats import yeojohnson
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler,MinMaxScaler

    
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Lasso



def best_tts(X,y,stratify=None,shuffle=True,model = LinearRegression(),method = root_mean_squared_error):
    global train_r2_
    train_r2_,train_r2_ts,train_r2_rs = 0,0,0
    test_r2_,test_r2_ts,test_r2_rs = 0,0,0
    adj_r2_train_,adj_r2_train_ts,adj_r2_train_rs = 0,0,0
    adj_r2_test_,adj_r2_test_ts,adj_r2_test_rs = 0,0,0
    train_evaluation_,train_evaluation_ts,train_evaluation_rs =100,100,100
    test_evaluation_,test_evaluation_ts,test_evaluation_rs = 100,100,100
    for k in range(65,90):
        i = k/100
        for j in range(1,100):
            X_train,X_test,y_train,y_test = tts(X,y[X.index],test_size = i, random_state = j, stratify= stratify,shuffle=shuffle)
            
            model = model
            model.fit(X_train,y_train) # model fitting
            y_pred_train = model.predict(X_train) # model prediction for train
            y_pred_test = model.predict(X_test) # model prediction for test

            train_r2 = r2_score(y_train, y_pred_train) # evaluating r2 score for train
            if train_r2_ < train_r2:
                train_r2_ = train_r2
                train_r2_ts = i
                train_r2_rs = j
            
            test_r2 = r2_score(y_test, y_pred_test)  # evaluating r2 score for test
            if test_r2_ < test_r2:
                test_r2_ = test_r2
                test_r2_ts = i
                test_r2_rs = j

            n_r_train, n_c_train = X_train.shape # getting no of rows and columns of train data
            n_r_test,  n_c_test = X_test.shape # getting no of rows and columns of test data

            adj_r2_train = 1 - ((1 - train_r2)*(n_r_train - 1)/ (n_r_train - n_c_train - 1))  # evaluating adjusted r2 score for train
            if adj_r2_train_ < adj_r2_train:
                adj_r2_train_ = adj_r2_train
                adj_r2_train_ts = i
                adj_r2_train_rs = j
                
            adj_r2_test = 1 - ((1 - test_r2)*(n_r_test - 1)/ (n_r_test - n_c_test - 1)) # evaluating adjusted r2 score for test
            if adj_r2_test_ < adj_r2_test:
                adj_r2_test_ = adj_r2_test
                adj_r2_test_ts = i
                adj_r2_test_rs = j
                
            train_evaluation = method(y_train, y_pred_train) # evaluating train error
            if train_evaluation_ > train_evaluation:
                train_evaluation_ = train_evaluation
                train_evaluation_ts = i
                train_evaluation_rs = j
                
            test_evaluation = method(y_test, y_pred_test) # evaluating test error
            if test_evaluation_ > test_evaluation:
                test_evaluation_ = test_evaluation
                test_evaluation_ts = i
                test_evaluation_rs = j
    return print("based on train_r2",train_r2_,"test_size",train_r2_ts,"random_state",train_r2_rs, "\n",
    "based on test_r2_",test_r2_,"test_size",test_r2_ts,"random_state",test_r2_rs, "\n",
    "based on adj_r2_train_",adj_r2_train_,"test_size",adj_r2_train_ts,"random_state",adj_r2_train_rs, "\n",
    "based on adj_r2_test_",adj_r2_test_,"test_size",adj_r2_test_ts,"random_state",adj_r2_test_rs,"\n",
    "based on train_evaluation_",train_evaluation_,"test_size",train_evaluation_ts,"random_state",train_evaluation_rs, "\n",
    "based on test_evaluation_",test_evaluation_,"test_size",test_evaluation_ts,"random_state",test_evaluation_rs)
            


# creating an empty dataFrame to store evaluation data
evaluation_df = pd.DataFrame({"evaluation_df_method" :[],
                                  "model": [],# model displays regression model
                                  "method": [],# method display evaluation metrics used
                                  "train_r2": [],# train r2 shows train R2 score
                                  "test_r2": [],# test r2 shows test R2 Score
                                  "adjusted_r2_train": [],# adjusted_r2_train shows adjusted r2 score for train
                                  "adjusted_r2_test": [],# adjusted_r2_test shows adjusted r2 score for test
                                  "train_evaluation": [],# train_evaluation shows train evaluation score by used method
                                  "test_evaluation" : []# test_evaluation shows test evaluation score by used method
                                })

# creating a function for evaluation of regression data
def reg_evaluation(evaluation_df_method,X_train,X_test,y_train,y_test,model = LinearRegression(),method = root_mean_squared_error):# input parameters from train_test_split , model and method for evaluation.
    model = model
    model.fit(X_train,y_train) # model fitting
    y_pred_train = model.predict(X_train) # model prediction for train
    y_pred_test = model.predict(X_test) # model prediction for test
    
    train_r2 = r2_score(y_train, y_pred_train) # evaluating r2 score for train
    test_r2 = r2_score(y_test, y_pred_test)  # evaluating r2 score for test
    
    n_r_train, n_c_train = X_train.shape # getting no of rows and columns of train data
    n_r_test,  n_c_test = X_test.shape # getting no of rows and columns of test data
    
    adj_r2_train = 1 - ((1 - train_r2)*(n_r_train - 1)/ (n_r_train - n_c_train - 1))  # evaluating adjusted r2 score for train
    adj_r2_test = 1 - ((1 - test_r2)*(n_r_test - 1)/ (n_r_test - n_c_test - 1)) # evaluating adjusted r2 score for test

    train_evaluation = method(y_train, y_pred_train) # evaluating train error
    test_evaluation = method(y_test, y_pred_test) # evaluating test error
    
    if method == root_mean_squared_error:
        a = "root_mean_squared_error"
    elif method ==root_mean_squared_log_error:
        a = "root_mean_squared_log_error"
    elif method == mean_absolute_error:
        a = "mean_absolute_error"
    elif method == mean_squared_error:
        a = "mean_squared_error"
    elif method == mean_squared_log_error:
        a = "mean_squared_log_error"    
    
    # declaring global dataframes
    global evaluation_df,temp_df
    
    # creating temporary dataframe for concating in later into main evaluation dataframe
    temp_df = pd.DataFrame({"evaluation_df_method" :[evaluation_df_method],
                                "model": [model],
                                  "method": [a],
                                  "train_r2": [train_r2],
                                  "test_r2": [test_r2],
                                  "adjusted_r2_train": [adj_r2_train],
                                  "adjusted_r2_test": [adj_r2_test],
                                  "train_evaluation": [train_evaluation],
                                  "test_evaluation" : [test_evaluation]
                                })
    evaluation_df = pd.concat([evaluation_df,temp_df]).reset_index(drop = True)
    
    return evaluation_df # returning evaluation_df



# creating a function for null_handling with different methods for null value imputing, categorical columns encoding and evaluation 
def null_handling(df,y ,ord_cols = None,categories='auto',model = LinearRegression(),method = root_mean_squared_error,test_size = 0.25, random_state = 42):
    global knn_imputed_df,si_mean_imputed_df,si_median_imputed_df,si_most_frequent_imputed_df,iter_imputed_df,knn_imputed_df_with_dropped_cat_missing_val
    global si_mean_imputed_df_with_dropped_cat_missing_val,si_median_imputed_df_with_dropped_cat_missing_val
    global si_most_frequent_imputed_df_with_dropped_cat_missing_val,iter_imputed_df_with_dropped_cat_missing_val, miss_val_dropped_df
    global num_cols,cat_cols
    num_cols = df.select_dtypes(exclude = "O").columns 
    cat_cols = df.select_dtypes(include = "O").columns
    if df.isnull().sum().sum() ==0:
        print("No null Values in DataFrame")
        if ord_cols != None:
            # ordinal encoding 
            ord = OrdinalEncoder(categories=categories,handle_unknown = "use_encoded_value",unknown_value = -1,dtype = "float")
            df[ord_cols] = ord.fit_transform(df[ord_cols])
            
            ohe_cols = list(set(cat_cols) - set(ord_cols))
            
            # one hot encoding
            ohe = OneHotEncoder(sparse_output = False,handle_unknown = "ignore")   
            pd.options.mode.chained_assignment = None
            df.loc[:, ohe.get_feature_names_out()] = ohe.fit_transform(df[ohe_cols])
            df.drop(columns = ohe_cols,inplace = True)
            
            pd.options.mode.chained_assignment = 'warn'
            X_train, X_test, y_train, y_test = tts(df,y[df.index],test_size = 0.25, random_state = 42)
            reg_evaluation("baseline dataframe",X_train,X_test,y_train,y_test,model = model,method = method)
        else:
            ohe = OneHotEncoder(sparse_output = False,handle_unknown = "ignore")
            pd.options.mode.chained_assignment = None
            df.loc[:, ohe.get_feature_names_out()] = ohe.fit_transform(df[cat_cols])
            df.drop(columns = cat_cols,inplace = True)
            
            pd.options.mode.chained_assignment = 'warn'
            
            X_train, X_test, y_train, y_test = tts(df,y[df.index],test_size = 0.25, random_state = 42)
            reg_evaluation("baseline dataframe",X_train,X_test,y_train,y_test,model = model,method = method)
    else:
        # applying various imputing methods on numerical columns
        knn_imputer = KNNImputer(n_neighbors = 5) 
        knn_imputed_num_df = pd.DataFrame(data = knn_imputer.fit_transform(df[num_cols]),columns = knn_imputer.get_feature_names_out())
        si_imputer = SimpleImputer(strategy = "mean")
        si_mean_imputed_num_df = pd.DataFrame(data = si_imputer.fit_transform(df[num_cols]),columns = si_imputer.get_feature_names_out())
        si_imputer = SimpleImputer(strategy = "median")
        si_median_imputed_num_df = pd.DataFrame(data = si_imputer.fit_transform(df[num_cols]),columns = si_imputer.get_feature_names_out())
        si_imputer = SimpleImputer(strategy = "most_frequent")
        si_most_frequent_imputed_num_df = pd.DataFrame(data = si_imputer.fit_transform(df[num_cols]),columns = si_imputer.get_feature_names_out())
        iter_imputer = IterativeImputer(max_iter = 200,random_state= 42)
        iter_imputed_num_df = pd.DataFrame(data = iter_imputer.fit_transform(df[num_cols]),columns = iter_imputer.get_feature_names_out())
        
        # treating missing values in categorical columns
        si_imputer = SimpleImputer(strategy = "most_frequent")
        si_most_frequent_imputed_cat_df = pd.DataFrame(data = si_imputer.fit_transform(df[cat_cols]),columns = si_imputer.get_feature_names_out())
        cat_miss_val_dropped_df = df[cat_cols].dropna()
        
        # creating a dataframe with dropping missing values
        miss_val_dropped_df = df.dropna()
        
        # creating dataframe by concating numerical df with "most frequent" imputing of categorical column
        knn_imputed_df              =(pd.concat([knn_imputed_num_df,si_most_frequent_imputed_cat_df],axis = 1))
        si_mean_imputed_df          =(pd.concat([si_mean_imputed_num_df,si_most_frequent_imputed_cat_df],axis = 1))
        si_median_imputed_df        =(pd.concat([si_median_imputed_num_df,si_most_frequent_imputed_cat_df],axis = 1))
        si_most_frequent_imputed_df =(pd.concat([si_most_frequent_imputed_num_df,si_most_frequent_imputed_cat_df],axis = 1))
        iter_imputed_df             =(pd.concat([iter_imputed_num_df,si_most_frequent_imputed_cat_df],axis = 1))
        # creating dataframe by concating numerical df with dropping of missing values from categorical column
        knn_imputed_df_with_dropped_cat_missing_val              =(pd.concat([knn_imputed_num_df.loc[cat_miss_val_dropped_df.index],cat_miss_val_dropped_df],axis = 1))
        si_mean_imputed_df_with_dropped_cat_missing_val          =(pd.concat([si_mean_imputed_num_df.loc[cat_miss_val_dropped_df.index],cat_miss_val_dropped_df],axis = 1))
        si_median_imputed_df_with_dropped_cat_missing_val        =(pd.concat([si_median_imputed_num_df.loc[cat_miss_val_dropped_df.index],cat_miss_val_dropped_df],axis = 1))
        si_most_frequent_imputed_df_with_dropped_cat_missing_val =(pd.concat([si_most_frequent_imputed_num_df.loc[cat_miss_val_dropped_df.index],cat_miss_val_dropped_df],axis = 1))
        iter_imputed_df_with_dropped_cat_missing_val             =(pd.concat([iter_imputed_num_df.loc[cat_miss_val_dropped_df.index],cat_miss_val_dropped_df],axis = 1))
        
        # list of dataframes
        
        list_df_after_missing_values= [knn_imputed_df,
                                si_mean_imputed_df,
                                si_median_imputed_df,
                                si_most_frequent_imputed_df,
                                iter_imputed_df,
                                knn_imputed_df_with_dropped_cat_missing_val,
                                si_mean_imputed_df_with_dropped_cat_missing_val,
                                si_median_imputed_df_with_dropped_cat_missing_val,
                                si_most_frequent_imputed_df_with_dropped_cat_missing_val,
                                iter_imputed_df_with_dropped_cat_missing_val,
                                miss_val_dropped_df]
        list_df_after_missing_values_names= ["knn_imputed_df",
                                "si_mean_imputed_df",
                                "si_median_imputed_df",
                                "si_most_frequent_imputed_df",
                                "iter_imputed_df",
                                "knn_imputed_df_with_dropped_cat_missing_val",
                                "si_mean_imputed_df_with_dropped_cat_missing_val",
                                "si_median_imputed_df_with_dropped_cat_missing_val",
                                "si_most_frequent_imputed_df_with_dropped_cat_missing_val",
                                "iter_imputed_df_with_dropped_cat_missing_val",
                                "miss_val_dropped_df"]
        if ord_cols != None:
            # ordinal encoding 
            ord = OrdinalEncoder(categories=categories,handle_unknown = "use_encoded_value",unknown_value = -1,dtype = "float")
            knn_imputed_df[ord_cols] = ord.fit_transform(knn_imputed_df[ord_cols])
            si_mean_imputed_df[ord_cols] = ord.fit_transform(si_mean_imputed_df[ord_cols])
            si_median_imputed_df[ord_cols] = ord.fit_transform(si_median_imputed_df[ord_cols])
            si_most_frequent_imputed_df[ord_cols] = ord.fit_transform(si_most_frequent_imputed_df[ord_cols])
            iter_imputed_df[ord_cols] = ord.fit_transform(iter_imputed_df[ord_cols])
            knn_imputed_df_with_dropped_cat_missing_val[ord_cols] = ord.fit_transform(knn_imputed_df_with_dropped_cat_missing_val[ord_cols])
            si_mean_imputed_df_with_dropped_cat_missing_val[ord_cols] = ord.fit_transform(si_mean_imputed_df_with_dropped_cat_missing_val[ord_cols])
            si_median_imputed_df_with_dropped_cat_missing_val[ord_cols] = ord.fit_transform(si_median_imputed_df_with_dropped_cat_missing_val[ord_cols])
            si_most_frequent_imputed_df_with_dropped_cat_missing_val[ord_cols] = ord.fit_transform(si_most_frequent_imputed_df_with_dropped_cat_missing_val[ord_cols])
            iter_imputed_df_with_dropped_cat_missing_val[ord_cols] = ord.fit_transform(iter_imputed_df_with_dropped_cat_missing_val[ord_cols])
            miss_val_dropped_df[ord_cols] = ord.fit_transform(miss_val_dropped_df[ord_cols])
            
            ohe_cols = list(set(cat_cols) - set(ord_cols))
            
            # one hot encoding
            ohe = OneHotEncoder(sparse_output = False,handle_unknown = "ignore")
            
            pd.options.mode.chained_assignment = None
            for i in list_df_after_missing_values:
                i.loc[:, ohe.get_feature_names_out()] = ohe.fit_transform(i[ohe_cols])
                i.drop(columns = ohe_cols,inplace = True)
            
            pd.options.mode.chained_assignment = 'warn'
            
            for j,i in enumerate(list_df_after_missing_values):
                
                # train test and splitting
                X_train, X_test, y_train, y_test = tts(i,y[i.index],test_size = test_size, random_state = random_state)
                reg_evaluation(list_df_after_missing_values_names[j],X_train,X_test,y_train,y_test,model = model,method = method)
        else:
            ohe = OneHotEncoder(sparse_output = False,handle_unknown = "ignore")
            pd.options.mode.chained_assignment = None
            for i in list_df_after_missing_values:
                i.loc[:, ohe.get_feature_names_out()] = ohe.fit_transform(i[cat_cols])
                i.drop(columns = cat_cols,inplace = True)
            
            pd.options.mode.chained_assignment = 'warn'
            
            for j,i in enumerate(list_df_after_missing_values):
    
                X_train, X_test, y_train, y_test = tts(i,y[i.index],test_size = test_size, random_state = random_state)
                reg_evaluation(list_df_after_missing_values_names[j],X_train,X_test,y_train,y_test,model = model,method = method)
            
    return evaluation_df # returning evaluating dataframe


def visual(df,num_cols = [],plot = "boxplot", x=None, y=None,hue = None,orient=None, color=None, palette=None,bins='auto',
           dodge=False,markers=True,estimator='mean',stat='count',logistic=False,order=1, logx=False,kind='scatter'):
    if type(num_cols) == list:
        if len(num_cols) == 0:
            num_col = df.select_dtypes(exclude = "object").columns.tolist()
        
        else:
            num_col = num_cols.tolist()
    else:
        if num_cols.tolist() == None:
            num_col = df.select_dtypes(exclude = "object").columns.tolist()
        
        else:
            num_col = num_cols.tolist()
    for col in num_col:
        if plot == "boxplot":
            sns.boxplot(df,x = col,y = y,orient = orient,color = color, palette = palette,hue = hue)
            plt.show()
        elif plot == "histplot":
            sns.histplot(df,x = col,kde = True,hue = hue,y = y,bins='auto',color = color, palette = palette)
            plt.show()
        elif plot == "violinplot":
            sns.violinplot(df,x = col,y = y,orient = orient,color = color, palette = palette,hue = hue)
            plt.show()
        elif plot == "stripplot":
            sns.stripplot(df,x = col,y = y,orient = orient,color = color, palette = palette,hue = hue,dodge=dodge)
            plt.show()
        elif plot == "swarmplot":
            sns.swarmplot(df,x = col,y = y,orient = orient,color = color, palette = palette,hue = hue,dodge=dodge)
            plt.show()
        elif plot == "scatterplot":
            sns.swarmplot(df,x = col,y = y,orient = orient,color = color, palette = palette,hue = hue)
            plt.show()
        elif plot == "lineplot":
            sns.lineplot(df,x = col,y = y,color = color, palette = palette,hue = hue,markers= markers)
            plt.show()
        elif plot == "barplot":
            sns.barplot(df,x = col,y = y,color = color, palette = palette,hue = hue,estimator=estimator)
            plt.show()
        elif plot == "countplot":
            sns.countplot(df,x = col,y = y,color = color, palette = palette,hue = hue,stat=stat)
            plt.show()
        elif plot == "regplot":
            sns.regplot(df,x = col,y = y,color = color,logistic=logistic,order=order, logx=logx)
            plt.show()
        elif plot == "heatmap":
            sns.heatmap(df[num_cols].corr(),annot =True)
            plt.show()
            break
        elif plot == "pairplot":
            sns.pairplot(df[num_cols],hue =hue,kind=kind)
            plt.show()
            break
            
            
def detect_outliers(df,num_cols = []):
    global outlier_df,zscore_cols,outlier_indexes,iqr_cols
    outlier_df = pd.DataFrame({"method" :[],"columns name":[],"upper limit":[],
                           "lower limit":[],"no of Rows":[],"percentage outlier":[]})
    if type(num_cols) == list:
        if len(num_cols)!=0:
            num_cols = num_cols
        else:
            num_cols = df.select_dtypes(exclude = "object").columns.tolist()
    else:
        if num_cols.tolist() != None:
            num_cols = num_cols
        else:
            num_cols = df.select_dtypes(exclude = "object").columns.tolist()
    zscore_cols = []
    iqr_cols = []
    outlier_indexes =[]
    for col in num_cols:
        skewness = df[col].skew()
        if -0.5 <= skewness <= 0.5:
            method = "zscore"
            zscore_cols.append(col)

        else:
            method = "iqr"
            iqr_cols.append(col)
    if len(zscore_cols) >0:
        for col in zscore_cols:
            mean = df[col].mean()
            std = df[col].std()
            ul = mean + (3*std)
            ll = mean - (3*std)
            mask = (df[col] < ll) | (df[col] > ul)
            temp = df[mask]

            Zscore_index = temp.index.tolist()
            outlier_indexes.extend(Zscore_index)

            if len(temp)>0:

                temp_df = pd.DataFrame({["method"] : "ZScore",
                "columns name" : [col],
                "upper limit" : [round(ul,2)],
                "lower limit" :[ round(ll,2)],
                "no of Rows" : [len(temp)],
                "percentage outlier" : [round(len(temp)*100/len(df),2)]})
                
                outlier_df = pd.concat([outlier_df,temp_df]).reset_index(drop = True)

    else:
        print("No columns for Zscore method")
       
    
    if len(iqr_cols) >0:
        for col in iqr_cols:
            q3 = df[col].quantile(.75)
            q1 = df[col].quantile(.25)
            IQR = q3 -q1
            ul = q3 + 1.5*IQR
            ll = q1 - 1.5*IQR
            mask = (df[col] < ll) | (df[col] > ul)
            temp = df[mask]

            IQR_index = temp.index.tolist()
            outlier_indexes.extend(IQR_index)

            if len(temp)>0:
                list(outlier_indexes).append(list(IQR_index))

                temp_df1 = pd.DataFrame({"method" : ["IQR"],
                "columns name" : [col],
                "upper limit" : [round(ul,2)],
                "lower limit" : [round(ll,2)],
                "no of Rows": [len(temp)],
                "percentage outlier" : [round((len(temp)*100/len(df)),2)]
                                    })
          
                outlier_df = pd.concat([outlier_df,temp_df1]).reset_index(drop = True)
            
    else:
        print("No columns for IQR method")

       
    outlier_indexes = list(set(outlier_indexes))
    
    return outlier_df



def outlier_handling(df,y,outlier_indexes = [],outlier_cols = None,model = LinearRegression(),method = root_mean_squared_error,test_size = 0.25, random_state = 42):
    num_col = df.select_dtypes(exclude = "O").columns
    
    global outliers_dropped_df,log_transformed_df,sqrt_transformed_df,yeo_johnson_transformed_df,rank_transformed_df
    global std_scaler_df,winsorize_transformed_df,inverse_log_transformed_winsorize_df,inverse_sqrt_transformed_winsorize_df,minmaxscaler_df
    
    if len(outlier_indexes) ==0:
        print("no outlier indexes passed")
        outliers_dropped_df = df.copy()
    else:
        outliers_dropped_df = df.drop(index =outlier_indexes)
    
    if outlier_cols != None:
                
        if df[outlier_cols][df[outlier_cols] <0].sum().sum() == 0:
            log_transformed_df = df.copy()
            log_transformed_df[outlier_cols] = np.log(log_transformed_df[outlier_cols] + 1e-5)
            sqrt_transformed_df = df.copy()
            sqrt_transformed_df[outlier_cols] = np.sqrt(sqrt_transformed_df[outlier_cols] + 1e-5)
            inverse_log_transformed_winsorize_df = log_transformed_df.copy()
            inverse_sqrt_transformed_winsorize_df = sqrt_transformed_df.copy()
            for column in outlier_cols:
                inverse_log_transformed_winsorize_df[column] =  np.exp(winsorize(inverse_log_transformed_winsorize_df[column], limits=[0.05, 0.05]))
                inverse_sqrt_transformed_winsorize_df[column] =  (winsorize(inverse_sqrt_transformed_winsorize_df[column], limits=[0.05, 0.05]))**2
        else:
            print("df have values less than zero")
        std_scaler_df = df.copy()
        std_scaler_df[outlier_cols] = StandardScaler().fit_transform(std_scaler_df[outlier_cols])
        
        minmaxscaler_df = df.copy()
        minmaxscaler_df[outlier_cols] = MinMaxScaler().fit_transform(minmaxscaler_df[outlier_cols])
 
        yeo_johnson_transformed_df = df.copy()
        for column in outlier_cols:
            try:
                yeo_johnson_transformed_df[column], lambda_ = yeojohnson(yeo_johnson_transformed_df[column])

            except :
                yeo_johnson_transformed_df[column] = yeo_johnson_transformed_df[column]

                print(f"Yeo-Johnson transformation failed for column '{column}'. Original data used.")
            # yeo_johnson_transformed_df[column], lambda_ = yeojohnson(yeo_johnson_transformed_df[column])
        rank_transformed_df = df.copy()
        rank_transformed_df[outlier_cols] = rank_transformed_df[outlier_cols].rank()
        winsorize_transformed_df = df.copy()
        for column in outlier_cols:
            winsorize_transformed_df[column] = winsorize(winsorize_transformed_df[column], limits=[0.05, 0.05])
            
            
        
    else:
        
        
        if df[num_col][df[num_col] <0].sum().sum() == 0:
            log_transformed_df = df.copy()
            log_transformed_df[num_col] = np.log(log_transformed_df[num_col] + 1e-5)
            sqrt_transformed_df = df.copy()
            sqrt_transformed_df[num_col] = np.sqrt(sqrt_transformed_df[num_col] + 1e-5)
            inverse_log_transformed_winsorize_df = log_transformed_df.copy()
            inverse_sqrt_transformed_winsorize_df = sqrt_transformed_df.copy()
            for column in num_col:
                inverse_log_transformed_winsorize_df[column] =  np.exp(winsorize(inverse_log_transformed_winsorize_df[column], limits=[0.05, 0.05]))
                inverse_sqrt_transformed_winsorize_df[column] =  (winsorize(inverse_sqrt_transformed_winsorize_df[column], limits=[0.05, 0.05]))**2
        else:
            
            print("df have values less than zero")
            
        std_scaler_df = df.copy()
        std_scaler_df[outlier_cols] = StandardScaler().fit_transform(std_scaler_df[outlier_cols])
        
        minmaxscaler_df = df.copy()
        minmaxscaler_df[outlier_cols] = MinMaxScaler().fit_transform(minmaxscaler_df[outlier_cols])
        
        yeo_johnson_transformed_df = df.copy()
        for column in num_col:
            try:
                yeo_johnson_transformed_df[column], lambda_ = yeojohnson(yeo_johnson_transformed_df[column])

            except :
                yeo_johnson_transformed_df[column] = yeo_johnson_transformed_df[column]

                print(f"Yeo-Johnson transformation failed for column '{column}'. Original data used.")
            # yeo_johnson_transformed_df[column], lambda_ = yeojohnson(yeo_johnson_transformed_df[column])
        rank_transformed_df = df.copy()
        rank_transformed_df[num_col] = rank_transformed_df[num_col].rank()
        winsorize_transformed_df = df.copy()
        for column in num_col:
            winsorize_transformed_df[column] = winsorize(winsorize_transformed_df[column], limits=[0.05, 0.05])
             
    if (df[num_col][df[num_col] <0].sum().sum() == 0):        
        outlier_handled_df = [std_scaler_df,minmaxscaler_df,outliers_dropped_df,log_transformed_df,sqrt_transformed_df,yeo_johnson_transformed_df,
                              rank_transformed_df,winsorize_transformed_df,inverse_log_transformed_winsorize_df,inverse_sqrt_transformed_winsorize_df]
    
        outlier_handled_df_name = ["std_scaler_df","minmaxscaler_df","outliers_dropped_df", "log_transformed_df","sqrt_transformed_df", "yeo_johnson_transformed_df","rank_transformed_df","winsorize_transformed_df",
                                   "inverse_log_transformed_winsorize_df", "inverse_sqrt_transformed_winsorize_df"]
    elif df[outlier_cols][df[outlier_cols] <0].sum().sum() == 0:
        outlier_handled_df = [std_scaler_df,minmaxscaler_df,outliers_dropped_df,log_transformed_df,sqrt_transformed_df,yeo_johnson_transformed_df,
                              rank_transformed_df,winsorize_transformed_df,inverse_log_transformed_winsorize_df,inverse_sqrt_transformed_winsorize_df]
    
        outlier_handled_df_name = ["std_scaler_df","minmaxscaler_df","outliers_dropped_df","log_transformed_df", "sqrt_transformed_df","yeo_johnson_transformed_df","rank_transformed_df",
                                   "winsorize_transformed_df","inverse_log_transformed_winsorize_df","inverse_sqrt_transformed_winsorize_df"]
    
    else:
        outlier_handled_df = [std_scaler_df,minmaxscaler_df,outliers_dropped_df,yeo_johnson_transformed_df,rank_transformed_df,winsorize_transformed_df]
    
        outlier_handled_df_name = ["std_scaler_df","minmaxscaler_df","outliers_dropped_df","yeo_johnson_transformed_df","rank_transformed_df","winsorize_transformed_df"]
 
    for j,i in enumerate(outlier_handled_df):
        X_train, X_test, y_train, y_test = tts(i,y[i.index],test_size = test_size, random_state = random_state)
        reg_evaluation(outlier_handled_df_name[j],X_train,X_test,y_train,y_test,model = model,method = method)
            
    return evaluation_df # returning evaluating dataframe
                               


def feature_selection(X_train, y_train,alpha = 0.05,model = LinearRegression(),method = root_mean_squared_error):
    global pval_cols,coef_cols,pval_and_coef_cols,mi_cols,corr_u_cols,corr_l_cols,vif_cols,lasso_cols

    model = sm.OLS(y_train, sm.add_constant(X_train))
    model_fit = model.fit()
    pval_cols = model_fit.pvalues[model_fit.pvalues > 0.05].index.tolist()
    coef_cols = model_fit.params[abs(model_fit.params) < 0.001].index.tolist()
    pval_and_coef_cols = list(set(coef_cols) | set(pval_cols))

    mi_scores = mutual_info_regression(X_train, y_train)
    mi = pd.DataFrame()

    mi["col_name"] = X_train.columns
    mi["mi_score"] = mi_scores

    mi_cols = mi[mi.mi_score ==0].col_name.values.tolist()

    corr = X_new.corr()
    
    corru= pd.DataFrame(np.triu(corr),columns = corr.columns , index = corr.index)
    corr_u_cols = corru[corru[(corru > 0.5 )& (corru <1)].any()].index.tolist()
    
    corrl= pd.DataFrame(np.tril(corr),columns = corr.columns , index = corr.index)
    corr_l_cols = corrl[corrl[(corrl > 0.5 )& (corrl <1)].any()].index.tolist()
    
    X_new_vif = sm.add_constant(X_train)
    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X_new_vif.columns
    vif["VIF"] = [variance_inflation_factor(X_new_vif.values, i) for i in range(X_new_vif.shape[1])]

    vif_cols = vif[vif.VIF >10].variables.tolist()

    # lasso
    if alpha == "best":
        
        lasso_len = []
        alpha_i = []
        for i in range(1,1000,5):
            j = i/10000

            model_lasso = Lasso(alpha=j)
            model_lasso.fit(X_train, y_train)
            col_df = pd.DataFrame({
                "col_name": X_train.columns,
                "lasso_coef": model_lasso.coef_
            })
            a = len(col_df[col_df.lasso_coef ==0])
            lasso_len.append(a)
            alpha_i.append(j)
        for i in zip(lasso_len,alpha_i):
            print(i)
        input_alpha = float(input("enter alpha"))
        model_lasso = Lasso(alpha=input_alpha)
        model_lasso.fit(X_train, y_train)
        col_df = pd.DataFrame({
            "col_name": X_train.columns,
            "lasso_coef": model_lasso.coef_
        })

        lasso_cols =col_df[col_df.lasso_coef ==0].col_name.tolist()
    else:
        model_lasso = Lasso(alpha=alpha)
        model_lasso.fit(X_train, y_train)
        col_df = pd.DataFrame({
            "col_name": X_train.columns,
            "lasso_coef": model_lasso.coef_
        })

        lasso_cols =col_df[col_df.lasso_coef ==0].col_name.tolist()
        
    feature_cols = [pval_cols,coef_cols,pval_and_coef_cols,mi_cols,corr_u_cols,corr_l_cols,vif_cols,lasso_cols]
    feature_cols_name = ["pval_cols","coef_cols","pval_and_coef_cols","mi_cols","corr_u_cols","corr_l_cols","vif_cols","lasso_cols"]
    
    for i,j in enumerate(feature_cols):
        reg_evaluation(f"{feature_cols_name[i]} dropped" ,X_train.drop(columns = j),X_test.drop(columns = j),y_train,y_test,model = model,method = method)
    return evaluation_df    
