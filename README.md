# Machine_Learning_functions

A machine learning function repository for a centralized collection of various machine learning algorithms, models, and utilities designed to perform specific tasks or solve particular problems. 

## Functions List:
1.  [_**Detect Outliers**_](#Detect_Outliers)
2.  [_**Visual**_](#visual)
## Functions in Detail:

<a name="Detect_Outliers"></a>
1. **Detect Outliers** - The detect_outliers function is designed to identify outliers within a dataset. Outliers are data points that significantly deviate from the rest of the data distribution and may represent errors, anomalies, or rare events. This function employs statistical techniques or machine learning algorithms to detect such outliers.
   _Initiation by_ - detect_outliers(df)
   _Input Parameters_ (df) - Pandas DataFrame
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

<a name="visual"></a>
2. **Visuals** - A versatile function that takes input data and generates clear, customizable plots, allowing users to visualize relationships, trends, and patterns effortlessly.
   _Initiation by_ - visual(df)
   _Input Parameters_ - (df,plot = "boxplot", x=None, y=None,hue = None,orient=None, color=None, palette=None,bins='auto',
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


