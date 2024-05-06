# Machine_Learning_functions

A machine learning function repository for a centralized collection of various machine learning algorithms, models, and utilities designed to perform specific tasks or solve particular problems. 

## Functions List:

1.  [_**Best Train Test Split**_](#best_tts)
2.  [_**Regression Evaluation**_](#reg_eval)
3.  [_**Null value handling with encoding and Evaluation**_](#null_handling)
4.  [_**Visual**_](#visual)
5.  [_**Detect Outliers**_](#Detect_Outliers)
6.  [_**Oultier Handling and Evaluation**_](#outlier_handling)
7.  [_**Feature Selection and Evaluation**_](#feature_selection)

## Functions in Detail:

<a name="reg_eval"></a>
### 1. **Best Train Test Split**

This function is a utility designed to assist in determining the optimal combination of the test_size and random_state parameters for splitting a dataset into training and testing sets using the train_test_split function from scikit-learn.

   _Initiation by_ - best_tts().
   
   _Input Parameters_ - (X,y,stratify=None,shuffle=True,model = LinearRegression(),method = root_mean_squared_error).
   
   _Function Return_ - best test_size and random_state combination based on r2,adjustedr2 and evaluation.
   
<a name="reg_eval"></a>
### 2. **Regression Metrics Evaluation** 

This Python function provides a comprehensive evaluation of regression models by calculating various metrics to assess model performance and reliability.

   _Initiation by_ - reg_evaluation().
   
   _Input Parameters_ - (evaluation_df_method,X_train,X_test,y_train,y_test,model = LinearRegression(),method = root_mean_squared_error).
   
   _Function Return_ - evaluation_df (pandas Dataframe).
   1. evaluation_df_method : evaluation method user input.
   2. model - regression model used for model training.
   3. method - method for evaluating errors (default - root_mean_squared_error) other methods available are(root_mean_squared_log_error,mean_absolute_error,mean_squared_error,mean_squared_log_error).
   4. train_r2/test_r2 - r2 score for train/test data respectively.
   5. adjusted_r2_train/adjusted_r2_test - adjusted r2 score for train and test respectively.![adjusted r2 score](https://github.com/Gauravverma069/Machine_Learning_functions/assets/121911821/f0ca18e3-ed20-4f99-ba5a-2425ea68f5b5)
   6. train_evaluation/test_evaluation - evaluation score for train and test

<a name="null_handling"></a>
### 3. **Null value handling with encoding and Evaluation** 

This repository contains a Python function for imputing missing values in datasets. Missing data is a common problem in data analysis and can significantly affect the results of statistical analyses or machine learning models. This function provides a simple yet effective methods for handling missing values.while also encoding categorical columns, and evaluating with reg_evaluation function.

   _Initiation by_ - null_handling().
   
   _Input Parameters_ - (df,y ,ord_cols = None,categories='auto',model = LinearRegression(),method = root_mean_squared_error,test_size = 0.25, random_state = 42).

   _Function Return_ - evaluation_df (pandas Dataframe).

   1.create 11 dataframes with different combinations for imputing and missing value handling.
   
      1. knn_imputed_df, (KNN imputed dataframe with categorical columns imputed with most frequest)
      2. si_mean_imputed_df, (simple imputer method -mean with categorical columns imputed with most frequest)
      3. si_median_imputed_df, (simple imputer method -median with categorical columns imputed with most frequest)
      4. si_most_frequent_imputed_df, (simple imputer method -most frequent with categorical columns imputed with most frequest)
      5. iter_imputed_df, (iterative imputation with categorical columns imputed with most frequest)
      6. knn_imputed_df_with_dropped_cat_missing_val, (KNN imputed dataframe with categorical columns missing values dropped)
      7. si_mean_imputed_df_with_dropped_cat_missing_val, (simple imputer method -mean with categorical columns missing values dropped)
      8. si_median_imputed_df_with_dropped_cat_missing_val, (simple imputer method -median with categorical columns missing values dropped)
      9. si_most_frequent_imputed_df_with_dropped_cat_missing_val, (simple imputer method -most frequent with categorical columns missing values dropped)
      10. iter_imputed_df_with_dropped_cat_missing_val, (iterative imputation  with categorical columns missing values dropped)
      11. miss_val_dropped_df, (all missing values dropped)
   
   2. ordinal encoding and onehotencoding of categorical columns
   3. train_test_split and evaluation dataframe returns
    ![evaluation_df](https://github.com/Gauravverma069/Machine_Learning_functions/assets/121911821/0f972a95-8a2f-4704-b250-0f9219c050a9)
      
<a name="visual"></a>
### 4. **Visuals** 

A versatile function that takes input data and generates clear, customizable plots, allowing users to visualize relationships, trends, and patterns effortlessly.

   _Initiation by_ - visual()
   
   _Input Parameters_ - (df,num_cols = [],plot = "boxplot", x=None, y=None,hue = None,orient=None, color=None, palette=None,bins='auto',
           dodge=False,markers=True,estimator='mean',stat='count',logistic=False,order=1, logx=False,kind='scatter')
           
   _Function Return_ - visual graph by selecting different plot types (default = "boxplot")
   
   _visual plots_ -
   1. boxplot - variable parameters(y, orient, color, palette, hue)
   2. histplot - variable parameters(hue, y, bins, color, palette)
   3. violinplot - variable parameters(y, orient, color, palette, hue)
   4. stripplot - variable parameters(y, orient, color, palette, hue, dodge)
   5. swarmplot - variable parameters(y, orient, color, palette, hue, dodge)
   6. scatterplot - variable parameters(y, orient, color, palette, hue, markers)
   7. lineplot - variable parameters(y, color, palette, hue, markers)
   8. barplot - variable parameters(y, color, palette, hue, estimator)
   9. countplot - variable parameters(y, color, palette, hue, stat)
   10. regplot - variable parameters(y, color, logistic, order, logx)
   11. heatmap - variable parameters(None)
   12. pairplot -  variable parameters(hue =hue,kind=kind)
for detailed study of parameters and plots please refer offical site link- [Seaborn](https://seaborn.pydata.org/api.html))

<a name="Detect_Outliers"></a>
### 5. **Detect Outliers** 

The detect_outliers function is designed to identify outliers within a dataset. Outliers are data points that significantly deviate from the rest of the data distribution and may represent errors, anomalies, or rare events. This function employs statistical techniques or machine learning algorithms to detect such outliers.

   _Initiation by_ - detect_outliers()
   
   _Input Parameters_ (df,num_cols = []) 
   
   _Function Return_ - Outlier Dataframe(outlier_df)
     1. **method** - _Z_score/IQR_
     2. **columns name** - _name of column of outlier_
     3. **Upper limit** - _for Z_score_- (mean + 3* standard Deviation)
                      _for IQR_ - (Q3 + 1.5*IQR)
     4. **Lower limit** - _for Z_score_- (mean - 3* standard Deviation)
                      _for IQR_ - (Q1 - 1.5*IQR)
     5. **No of rows** - _no of rows of Outliers_
     6. **percentage outlier** - _percentage of outlier_

Additional Function returns
  1. **outlier_indexes** - return list of outlier Indexes
  2. **num_cols** - return list of nemerical columns
  3. **zscore_cols** - list of columns using method Z_score
  4. **iqr_cols** - list of columns using method IQR

<a name="outlier_handling"></a>
### 6. **Oultier Handling and Evaluation** 

This function is a utility designed to preprocess the dataset by handling outliers using specified techniques and evaluate a machine learning model's performance.

   _Initiation by_ - outlier_handling()
   
   _Input Parameters_ (df,y,outlier_indexes = [],outlier_cols = None,model = LinearRegression(),method = root_mean_squared_error,test_size = 0.25, random_state = 42) 
   
   _Function Return_ - evaluation_df (pandas Dataframe).
   
     1. std_scaler_df, (using Standard scaler)
     2. minmaxscaler_df, (using min max scaler)
     3. outliers_dropped_df, (dropping outliers)
     4. log_transformed_df, (log transformation of data)
     5. sqrt_transformed_df, (square root transformation of data)
     6. yeo_johnson_transformed_df, (yeo johnson transformation of data)
     7. rank_transformed_df, (rank transformation of data)
     8. winsorize_transformed_df, (winsorization of outliers to 0.05 quantile)
     9.inverse_log_transformed_winsorize_df, (log transformation, winsorization and inverse transformation)
     10. inverse_sqrt_transformed_winsorize_df, (square root transformation, winsorization and inverse transformation)

<a name="feature_selection"></a>
### 6. **Feature Selection and Evaluation** 

This function is a utility designed to perform feature selection and evaluate a machine learning model's performance.

   _Initiation by_ - feature_selection()
   
   _Input Parameters_ (X_train, y_train,alpha = 0.05,model = LinearRegression(),method = root_mean_squared_error) 
   
   _Function Return_ - evaluation_df (pandas Dataframe).
   
     1. pval_cols, (columns where p_values <0.05)
     2. coef_cols, (columns where coef <0.001)
     3. pval_and_coef_cols, (columns of both p_values and coef.)
     4. mi_cols, (columns by Mutual Information ,where mi_score = 0 )
     5. corr_u_cols, (columns by correlation using upper triangle columns and corr() >0.5)
     6. corr_l_cols, (columns by correlation using lower triangle columns and corr() >0.5)
     7. vif_cols, (columns by variance_inflation_factor where VIF > 10)
     8. lasso_cols, (columns by finding lasso coef. where lasso_coef =0)

     


   

