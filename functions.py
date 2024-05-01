

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
