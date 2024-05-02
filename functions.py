

# Importing Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# creating a function for outlier detection in a DataFrame.
def detect_outliers(df): 
# outlier Detection for easy access for distance based Machine Learning models.
    
    global outlier_df,num_cols,zscore_cols,outlier_indexes,iqr_cols  # creating a global variables
    outlier_df = pd.DataFrame({"method" :[],"columns name":[],
                               "upper limit":[],"lower limit":[],
                               "no of Rows":[],"percentage outlier":[]})  # empty dataframe to store results
    
    
    num_cols = df.select_dtypes(exclude = "object").columns.tolist() # excluding Object Datatypes
    zscore_cols = []
    iqr_cols = []
    
    outlier_indexes =[]
    for col in num_cols:
        skewness = df[col].skew() 
        if -0.5 <= skewness <= 0.5: # checking skewness for method selection
            method = "zscore"
            zscore_cols.append(col) # appending columns in Z_score cols where method is Z_score

        else:
            method = "iqr"
            iqr_cols.append(col) # appending columns in IQR cols where method is IQR
            
    # using Zscore method for finding outliers 
        # taking upper limit and lower limit 3 standard deviation
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
       
    # using IQR method for finding outliers 
        # taking upper whisker and lower whisker using IQR method
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

       
    outlier_indexes = set(outlier_indexes) # after function call, outlier index can be called seperately
                                           # as global variable is defined.
    
    return outlier_df  # returning Outlier DataFrame

# Visual function using Seaborn for graph plotting.
def visual(df,plot = "boxplot", x=None, y=None,hue = None,orient=None, color=None, palette=None,bins='auto',
           dodge=False,markers=True,estimator='mean',stat='count',logistic=False,order=1, logx=False,kind='scatter'):
               # parameters for Graph plotting               
    num_cols = df.select_dtypes(exclude = "object").columns.tolist() # selecting numerical columns from DataFrane
    for col in num_cols:
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
            sns.swarmplot(df,x = col,y = y,orient = orient,color = color, palette = palette,hue = hue,markers=markers)
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

# creating an empty dataFrame to store evaluation data
evaluation_df = pd.DataFrame({"model": [],# model displays regression model
                                  "method": [],# method display evaluation metrics used
                                  "train_r2": [],# train r2 shows train R2 score
                                  "test_r2": [],# test r2 shows test R2 Score
                                  "adjusted_r2_train": [],# adjusted_r2_train shows adjusted r2 score for train
                                  "adjusted_r2_test": [],# adjusted_r2_test shows adjusted r2 score for test
                                  "train_evaluation": [],# train_evaluation shows train evaluation score by used method
                                  "test_evaluation" : []# test_evaluation shows test evaluation score by used method
                                })

# creating a function for evaluation of regression data
def reg_evaluation(X_train,X_test,y_train,y_test,model = LinearRegression(),method = root_mean_squared_error):# input parameters from train_test_split , model and method for evaluation.
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
    temp_df = pd.DataFrame({"model": [model],
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


# importing important libraries
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder

# creating a function for null_handling with different methods for null value imputing, categorical columns encoding and evaluation 
def null_handling(df,ord_cols = None,categories='auto',model = LinearRegression(),method = root_mean_squared_error):
    num_cols = df.select_dtypes(exclude = "O").columns 
    cat_cols = df.select_dtypes(include = "O").columns
    if df.isnull().sum().sum() ==0:
        print("No null Values in DataFrame")
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
                X_train, X_test, y_train, y_test = tts(i,y[i.index],test_size = 0.25, random_state = 42)
                reg_evaluation(list_df_after_missing_values_names[j],X_train,X_test,y_train,y_test)
        else:
            ohe = OneHotEncoder(sparse_output = False,handle_unknown = "ignore")
            pd.options.mode.chained_assignment = None
            for i in list_df_after_missing_values:
                i.loc[:, ohe.get_feature_names_out()] = ohe.fit_transform(i[cat_cols])
                i.drop(columns = cat_cols,inplace = True)
            
            pd.options.mode.chained_assignment = 'warn'
            
            for j,i in enumerate(list_df_after_missing_values):
    
                X_train, X_test, y_train, y_test = tts(i,y[i.index],test_size = 0.25, random_state = 42)
                reg_evaluation(list_df_after_missing_values_names[j],X_train,X_test,y_train,y_test,model = model,method = method)
            
    return evaluation_df # returning evaluating dataframe
