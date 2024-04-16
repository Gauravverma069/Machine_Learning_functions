# outlier Detection for easy access for distance based Machine Learning models.

# Importing Required Libraries

import pandas as pd
import numpy as np

# creating a function for outlier detection in a DataFrame.
def detect_outliers(df): 
    
    global outlier_df  # creating a global variable for outlier dataframe
    outlier_df = pd.DataFrame({"method" :[],"columns name":[],
                               "upper limit":[],"lower limit":[],
                               "no of Rows":[],"percentage outlier":[]})  # empty dataframe to store results
    
    
    num_cols = df.select_dtypes(exclude = "object").columns.tolist() # excluding Object Datatypes
    zscore_cols = []
    iqr_cols = []
    global outlier_indexes # creating a global variable for outlier indexes
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

