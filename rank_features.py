# Use this code to predict the percentage tip expected after a trip in NYC green taxi
# The code is a predictive model that was built and trained on top of the Gradient Boosting Classifer and 
# the Random Forest Gradient both provided in scikit-learn

# The input: 
#    pandas.dataframe with columns: This should be in the same format as downloaded from the website

# The data frame go through the following pipeliine:
# 1. Cleaning
# 2. Creation of derived variables
# 3. Making predictions

# The output:
#    pandas.Series, two files are saved on disk,  submission.csv and cleaned_data.csv respectively.

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import os, json, requests, pickle
from scipy.stats import skew
from shapely.geometry import Point,Polygon,MultiPoint,MultiPolygon
from scipy.stats import ttest_ind, f_oneway, lognorm, levy, skew, chisquare
from sklearn.preprocessing import normalize, scale
from sklearn import metrics
from tabulate import tabulate #pretty print of tables. source: http://txt.arboreus.com/2013/03/13/pretty-print-tables-in-python.html
from shapely.geometry import Point,Polygon,MultiPoint

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor

import pdb

# import scikit learn libraries
from sklearn import cross_validation, metrics   #model optimization and valuation tools
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import warnings
warnings.filterwarnings('ignore')

# create a function to check if a location is located inside Upper Manhattan
def is_within_bbox(loc,poi):
    """
    This function returns 1 if a location loc(lat,lon) is located inside a polygon of interest poi
    loc: tuple, (latitude, longitude)
    poi: shapely.geometry.Polygon, polygon of interest
    """
    return 1*(Point(loc).within(poi))


def clean_data(adata):
    """
    This function cleans the input dataframe adata:
    . drop Ehail_fee [99% transactions are NaNs]
    . impute missing values in Trip_type
    . replace invalid data by most frequent value for RateCodeID and extra
    . encode categorical to numeric
    . rename pickup and dropff time variables (for later use)
    
    input:
        adata: pandas.dataframe
    output: 
        pandas.dataframe

    """
    ## make a copy of the input
    data = adata.copy()
    ## drop Ehail_fee: 99% of its values are NaNs
    if 'Ehail_fee' in data.columns:
        data.drop('Ehail_fee',axis=1,inplace=True)

    ##  replace missing values in Trip_type with the most frequent value 1
    #data['Trip_type '] = data['Trip_type '].replace(np.NaN,1)
    
    ## replace all values that are not allowed as per the variable dictionary with the most frequent allowable value
    # remove negative values from Total amound and fare_amount
    print "Negative values found and replaced by their abs"
    print "total_amount", 100*data[data.total_amount<0].shape[0]/float(data.shape[0]),"%"
    print "fare_amount", 100*data[data.fare_amount<0].shape[0]/float(data.shape[0]),"%"
    print "Improvement_surcharge", 100*data[data.improvement_surcharge<0].shape[0]/float(data.shape[0]),"%"
    print "tip_amount", 100*data[data.tip_amount<0].shape[0]/float(data.shape[0]),"%"
    print "tolls_amount", 100*data[data.tolls_amount<0].shape[0]/float(data.shape[0]),"%"
    print "mta_tax", 100*data[data.mta_tax<0].shape[0]/float(data.shape[0]),"%"
    data.total_amount = data.total_amount.abs()
    data.fare_amount = data.fare_amount.abs()
    data.improvement_surcharge = data.improvement_surcharge.abs()
    data.tip_amount = data.tip_amount.abs()
    data.tolls_amount = data.tolls_amount.abs()
    data.mta_tax = data.mta_tax.abs()
    
    # RateCodeID
    indices_oi = data[~((data.RateCodeID>=1) & (data.RateCodeID<=6))].index
    data.loc[indices_oi, 'RateCodeID'] = 2 # 2 = Cash payment was identified as the common method
    print round(100*len(indices_oi)/float(data.shape[0]),2),"% of values in RateCodeID were invalid.--> Replaced by the most frequent 2"
    
    # extra
    indices_oi = data[~((data.extra==0) | (data.extra==0.5) | (data.extra==1))].index
    data.loc[indices_oi, 'extra'] = 0 # 0 was identified as the most frequent value
    print round(100*len(indices_oi)/float(data.shape[0]),2),"% of values in extra were invalid.--> Replaced by the most frequent 0"
    
    # total_amount: the minimum charge is 2.5, so I will replace every thing less than 2.5 by the median 11.76 (pre-obtained in analysis)
    indices_oi = data[(data.total_amount<2.5)].index
    data.loc[indices_oi,'total_amount'] = 11.76
    print round(100*len(indices_oi)/float(data.shape[0]),2),"% of values in total amount worth <$2.5.--> Replaced by the median 1.76"
    
    # encode categorical to numeric (I avoid to use dummy to keep dataset small)
    if data.store_and_fwd_flag.dtype.name != 'int64':
        data['store_and_fwd_flag'] = (data.store_and_fwd_flag=='Y')*1
    
    # rename time stamp variables and convert them to the right format
    print "renaming variables..."
    data.rename(columns={'tpep_pickup_datetime':'pickup_dt','tpep_dropoff_datetime':'Dropoff_dt'},inplace=True)
    print "converting timestamps variables to right format ..."
    data['pickup_dt'] = data.pickup_dt.apply(lambda x:dt.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
    data['Dropoff_dt'] = data.Dropoff_dt.apply(lambda x:dt.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
    
    print "Done cleaning"
    return data



# Function to run the feature engineering
def engineer_features(adata):
    """
    This function create new variables based on present variables in the dataset adata. It creates:
    . Week: int {1,2,3,4,5}, Week a transaction was done
    . Week_day: int [0-6], day of the week a transaction was done
    . Month_day: int [0-30], day of the month a transaction was done
    . Hour: int [0-23], hour the day a transaction was done
    . Shift type: int {1=(7am to 3pm), 2=(3pm to 11pm) and 3=(11pm to 7am)}, shift of the day  
    . Speed_mph: float, speed of the trip
    . Tip_percentage: float, target variable
    . With_tip: int {0,1}, 1 = transaction with tip, 0 transction without tip
    
    input:
        adata: pandas.dataframe
    output: 
        pandas.dataframe
    """
    
    # make copy of the original dataset
    data = adata.copy()
    
    # derive time variables
    print "deriving time variables..."
    ref_week = dt.datetime(2015,9,1).isocalendar()[1] # first week of september in 2015
    data['Week'] = data.pickup_dt.apply(lambda x:x.isocalendar()[1])-ref_week+1
    data['Week_day']  = data.pickup_dt.apply(lambda x:x.isocalendar()[2])
    data['Month_day'] = data.pickup_dt.apply(lambda x:x.day)
    data['Hour'] = data.pickup_dt.apply(lambda x:x.hour)
    #data.rename(columns={'pickup_hour':'Hour'},inplace=True)

    # create shift variable:  1=(7am to 3pm), 2=(3pm to 11pm) and 3=(11pm to 7am)
    data['Shift_type'] = np.NAN
    data.loc[data[(data.Hour>=7) & (data.Hour<15)].index,'Shift_type'] = 1
    data.loc[data[(data.Hour>=15) & (data.Hour<23)].index,'Shift_type'] = 2
    data.loc[data[data.Shift_type.isnull()].index,'Shift_type'] = 3
    
    # Trip duration 
    print "deriving Trip_duration..."
    data['trip_duration'] = (data['Dropoff_dt'].sub(data['pickup_dt'])).apply(lambda x:x.total_seconds()/60.)
    
    print "deriving direction variables..."
    # create direction variable Direction_NS. 
    # This is 2 if taxi moving from north to south, 1 in the opposite direction and 0 otherwise
    data['Direction_NS'] = (data.pickup_latitude>data.dropoff_latitude)*1+1
    indices = data[(data.pickup_latitude == data.dropoff_latitude) & (data.pickup_latitude!=0)].index
    data.loc[indices,'Direction_NS'] = 0

    # create direction variable Direction_EW. 
    # This is 2 if taxi moving from east to west, 1 in the opposite direction and 0 otherwise
    data['Direction_EW'] = (data.pickup_longitude>data.dropoff_longitude)*1+1
    indices = data[(data.pickup_longitude == data.dropoff_longitude) & (data.pickup_longitude!=0)].index
    data.loc[indices,'Direction_EW'] = 0
    
    # create variable for Speed
    print "deriving Speed. Make sure to check for possible NaNs and Inf vals..."
    data['Speed_mph'] = data.trip_distance/(data.trip_duration/60)
    # replace all NaNs values and values >240mph by a values sampled from a random distribution of 
    # mean 12.9 and  standard deviation 6.8mph. These values were extracted from the distribution
    indices_oi = data[(data.Speed_mph.isnull()) | (data.Speed_mph>240)].index
    data.loc[indices_oi,'Speed_mph'] = np.abs(np.random.normal(loc=12.9,scale=6.8,size=len(indices_oi)))
    print "Feature engineering done! :-)"
    
    # Create a new variable to check if a trip originated in Upper Manhattan
    #print "checking where the trip originated..."
    #data['U_manhattan'] = data[['pickup_latitude','pickup_longitude']].apply(lambda r:is_within_bbox((r[0],r[1])),axis=1)
    
    # create tip percentage variable
    data['Tip_percentage'] = 100*data.tip_amount/data.total_amount
    
    # create with_tip variable
    data['With_tip'] = (data.Tip_percentage>0)*1

    return data


# Functions for exploratory data analysis
def visualize_continuous(df,label,method={'type':'histogram','bins':20},outlier='on'):
    """
    function to quickly visualize continous variables
    df: pandas.dataFrame 
    label: str, name of the variable to be plotted. It should be present in df.columns
    method: dict, contains info of the type of plot to generate. It can be histogram or boxplot [-Not yet developped]
    outlier: {'on','off'}, Set it to off if you need to cut off outliers. Outliers are all those points
    located at 3 standard deviations further from the mean
    """
    # create vector of the variable of interest
    v = df[label]
    # define mean and standard deviation
    m = v.mean()
    s = v.std()
    # prep the figure
    fig,ax = plt.subplots(1,2,figsize=(14,4))
    ax[0].set_title('Distribution of '+label)
    ax[1].set_title('Tip % by '+label)
    if outlier=='off': # remove outliers accordingly and update titles
        v = v[(v-m)<=3*s]
        ax[0].set_title('Distribution of '+label+'(no outliers)')
        ax[1].set_title('Tip % by '+label+'(no outliers)')
    if method['type'] == 'histogram': # plot the histogram
        v.hist(bins = method['bins'],ax=ax[0])
    if method['type'] == 'boxplot': # plot the box plot
        df.loc[v.index].boxplot(label,ax=ax[0])
    ax[1].plot(v,df.loc[v.index].Tip_percentage,'.',alpha=0.4)
    ax[0].set_xlabel(label)
    ax[1].set_xlabel(label)
    ax[0].set_ylabel('Count')
    ax[1].set_ylabel('Tip (%)')

def visualize_categories(df,catName,chart_type='histogram',ylimit=[None,None]):
    """
    This functions helps to quickly visualize categorical variables. 
    This functions calls other functions generate_boxplot and generate_histogram
    df: pandas.Dataframe
    catName: str, variable name, it must be present in df
    chart_type: {histogram,boxplot}, choose which type of chart to plot
    ylim: tuple, list. Valid if chart_type is histogram
    """
    print catName
    cats = sorted(pd.unique(df[catName]))
    if chart_type == 'boxplot': #generate boxplot
        generate_boxplot(df,catName,ylimit)
    elif chart_type == 'histogram': # generate histogram
        generate_histogram(df,catName)
    else:
        pass
    
    #=> calculate test statistics
    groups = df[[catName,'Tip_percentage']].groupby(catName).groups #create groups
    tips = df.Tip_percentage
    if len(cats)<=2: # if there are only two groups use t-test
        print ttest_ind(tips[groups[cats[0]]],tips[groups[cats[1]]])
    else: # otherwise, use one_way anova test
        # prepare the command to be evaluated
        cmd = "f_oneway("
        for cat in cats:
            cmd+="tips[groups["+str(cat)+"]],"
        cmd=cmd[:-1]+")"
        print "one way anova test:", eval(cmd) #evaluate the command and print
    print "Frequency of categories (%):\n",df[catName].value_counts(normalize=True)*100
    
def test_classification(df,label,yl=[0,50]):
    """
    This function test if the means of the two groups with_tip and without_tip are different at 95% of confidence level.
    It will also generate a box plot of the variable by tipping groups
    label: str, label to test
    yl: tuple or list (default = [0,50]), y limits on the ylabel of the boxplot
    df: pandas.DataFrame (default = data)
    
    Example: run <visualize_continuous(data,'fare_amount',outlier='on')>
    """
    
    if len(pd.unique(df[label]))==2: #check if the variable is categorical with only two  categores and run chisquare test
        vals=pd.unique(df[label])
        gp1 = df[df.With_tip==0][label].value_counts().sort_index()
        gp2 = df[df.With_tip==1][label].value_counts().sort_index()
        print "t-test if", label, "can be used to distinguish transaction with tip and without tip"
        print chisquare(gp1,gp2)
    elif len(pd.unique(df[label]))>=10: #other wise  run the t-test
        df.boxplot(label,by='With_tip')
        plt.ylim(yl)
        plt.show()
        print "t-test if", label, "can be used to distinguish transaction with tip and without tip"
        print "results:",ttest_ind((df[df.With_tip==0][label].values.astype(int)),(df[df.With_tip==1][label].values.astype(int)),axis=0, equal_var=False)
    else:
        pass

def generate_boxplot(df,catName,ylimit):
    """
    generate boxplot of tip percentage by variable "catName" with ylim set to ylimit
    df: pandas.Dataframe
    catName: str
    ylimit: tuple, list
    """
    df.boxplot('Tip_percentage',by=catName)
    #plt.title('Tip % by '+catName)
    plt.title('')
    plt.ylabel('Tip (%)')
    if ylimit != [None,None]:
        plt.ylim(ylimit)
    plt.show()

def generate_histogram(df,catName):
    """
    generate histogram of tip percentage by variable "catName" with ylim set to ylimit
    df: pandas.Dataframe
    catName: str
    ylimit: tuple, list
    """
    cats = sorted(pd.unique(df[catName]))
    colors = plt.cm.jet(np.linspace(0,1,len(cats)))
    hx = np.array(map(lambda x:round(x,1),np.histogram(df.Tip_percentage,bins=20)[1]))
    fig,ax = plt.subplots(1,1,figsize = (15,4))
    for i,cat in enumerate(cats):
        vals = df[df[catName] == cat].Tip_percentage
        h = np.histogram(vals,bins=hx)
        w = 0.9*(hx[1]-hx[0])/float(len(cats))
        plt.bar(hx[:-1]+w*i,h[0],color=colors[i],width=w)
    plt.legend(cats)
    plt.yscale('log')
    plt.title('Distribution of Tip by '+catName)
    plt.xlabel('Tip (%)')


# creatWelche function to assign each coordinates point to its borough
def find_borough(lat,lon):
    """
    return the borough of a location given its latitude and longitude
    lat: float, latitude
    lon: float, longitude
    """
    boro = 0 # initialize borough as 0
    for k,v in boros.iteritems(): # update boro to the right key corresponding to the parent polygon
        if v['polygon'].contains(Point(lon,lat)):
            boro = k
            break # break the loop once the borough is found
    return [boro]


# define a function that help to train models and perform cv
def modelfit(alg,dtrain,predictors,target,scoring_method,performCV=True,printFeatureImportance=True,cv_folds=5):
    """
    This functions train the model given as 'alg' by performing cross-validation. It works on both regression and classification
    alg: sklearn model
    dtrain: pandas.DataFrame, training set
    predictors: list, labels to be used in the model training process. They should be in the column names of dtrain
    target: str, target variable
    scoring_method: str, method to be used by the cross-validation to valuate the model
    performCV: bool, perform Cv or not
    printFeatureImportance: bool, plot histogram of features importance or not
    cv_folds: int, degree of cross-validation
    """
    # train the algorithm on data
    alg.fit(dtrain[predictors],dtrain[target])
    #predict on train set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    if scoring_method == 'roc_auc':
        dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    #perform cross-validation
    if performCV:
        cv_score = cross_validation.cross_val_score(alg,dtrain[predictors],dtrain[target],cv=cv_folds,scoring=scoring_method)
        #print model report
        print "\nModel report:"
        if scoring_method == 'roc_auc':
            print "Accuracy:",metrics.accuracy_score(dtrain[target].values,dtrain_predictions)
            print "AUC Score (Train):",metrics.roc_auc_score(dtrain[target], dtrain_predprob)
        if (scoring_method == 'mean_squared_error'):
            print "Accuracy:",metrics.mean_squared_error(dtrain[target].values,dtrain_predictions)
    if performCV:
        print "CV Score - Mean : %.7g | Std : %.7g | Min : %.7g | Max : %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score))
    #print feature importance
    if printFeatureImportance:
        if dir(alg)[0] == '_Booster': #runs only if alg is xgboost
            feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
        else:
            feat_imp = pd.Series(alg.feature_importances_,predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar',title='Feature Importances')
        plt.ylabel('Feature Importe Score')
        plt.show()

# optimize n_estimator through grid search
def optimize_num_trees(alg,param_test,scoring_method,train,predictors,target):
    """
    This functions is used to tune paremeters of a predictive algorithm
    alg: sklearn model,
    param_test: dict, parameters to be tuned
    scoring_method: str, method to be used by the cross-validation to valuate the model
    train: pandas.DataFrame, training data
    predictors: list, labels to be used in the model training process. They should be in the column names of dtrain
    target: str, target variable
    """
    gsearch = GridSearchCV(estimator=alg, param_grid = param_test, scoring=scoring_method,n_jobs=2,iid=False,cv=5)
    gsearch.fit(train[predictors],train[target])
    return gsearch

# plot optimization results
def plot_opt_results(alg):
    cv_results = []
    for i in range(len(param_test['n_estimators'])):
        cv_results.append((alg.grid_scores_[i][1],alg.grid_scores_[i][0]['n_estimators']))
    cv_results = pd.DataFrame(cv_results)
    plt.plot(cv_results[1],cv_results[0])
    plt.xlabel('# trees')
    plt.ylabel('score')
    plt.title('optimization report')



def predict_tip(transaction):
    """
    This function predicts the percentage tip expected on 1 transaction
    transaction: pandas.dataframe, this should have been cleaned first and feature engineered
    """
    # define predictors labels as per optimization results
    cls_predictors = ['payment_type','total_amount','trip_duration','Speed_mph','mta_tax',
                      'extra','Hour','Direction_NS', 'Direction_EW']

    reg_predictors = ['payment_type','total_amount','trip_duration','Speed_mph','mta_tax',
                      'extra','Hour','Direction_NS', 'Direction_EW']    

    # classify transactions
    clas = gs_cls.best_estimator_.predict(transaction[cls_predictors])
    
    # predict tips for those transactions classified as 1
    return clas*gs_rfr.best_estimator_.predict(transaction[reg_predictors])



# Download the September 2015 dataset
if os.path.exists('yellow_tripdata_2015-06.csv'): # Check if the dataset is present on local disk and load it
    data = pd.read_csv('yellow_tripdata_2015-06.csv')
else: # Download dataset if not available on disk
    url = "https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2015-06.csv"
    data = pd.read_csv(url)
    data.to_csv(url.split('/')[-1])

# Print the size of the dataset
print "Number of rows:", data.shape[0]
print "Number of columns: ", data.shape[1]

# create backup dataset
backup_data = data.copy()



# clean a loaded dataset
data = clean_data(data)


data_extended = engineer_features(data)




## code to compare the two Tip_percentage identified groups
# split data in the two groups
data1 = data_extended[data_extended.Tip_percentage>0]
data2 = data_extended[data_extended.Tip_percentage==0]

# generate histograms to compare
fig,ax=plt.subplots(1,2,figsize=(14,4))
data1.Tip_percentage.hist(bins = 20,normed=True,ax=ax[0])
ax[0].set_xlabel('Tip (%)')
ax[0].set_title('Distribution of Tip (%) - All transactions')

data1.Tip_percentage.hist(bins = 20,normed=True,ax=ax[1])
ax[1].set_xlabel('Tip (%)')
ax[1].set_title('Distribution of Tip (%) - Transaction with tips')
ax[1].set_ylabel('Group normed count')
plt.savefig('Question4_target_varc.jpeg',format='jpeg')
plt.show()








continuous_variables=['total_amount','fare_amount','trip_distance','trip_duration','tolls_amount','Speed_mph','Tip_percentage']
cor_mat = data1[continuous_variables].corr()
plt.imshow(cor_mat, cmap='gray')
plt.xticks(range(len(continuous_variables)),continuous_variables,rotation='vertical')
plt.yticks(range(len(continuous_variables)),continuous_variables)
plt.colorbar()
plt.title('Correlation between continuous variables')
plt.show()
#print cor_mat



# visualization of the payment_type
visualize_categories(data1,'payment_type','histogram',[13,20])




## OPTIMIZATION & TRAINING OF THE CLASSIFIER


print "Optimizing the classifier..."

train = data1.copy()
train = train.loc[np.random.choice(train.index,size=100000,replace=False)]
indices = data1.index[~data1.index.isin(train.index)]
test = data1.loc[np.random.choice(indices,size=100000,replace=False)]

train['ID'] = train.index
IDCol = 'ID'
target = 'Tip_percentage'

predictors = ['payment_type','total_amount','trip_duration','Speed_mph','mta_tax',
              'extra','Hour','Direction_NS', 'Direction_EW']



# Random Forest
tic = dt.datetime.now()

# optimize n_estimator through grid search
param_test = {'n_estimators':range(50,200,25)} # define range over which number of trees is to be optimized

# initiate classification model
rfr = RandomForestRegressor()

# get results of the search grid
gs_rfr = optimize_num_trees(rfr,param_test,'mean_squared_error',train,predictors,target)

# print optimization results
print gs_rfr.grid_scores_, gs_rfr.best_params_, gs_rfr.best_score_


# cross validate the best model with optimized number of estimators
modelfit(gs_rfr.best_estimator_,train,predictors,target,'mean_squared_error')

# save the best estimator on disk as pickle for a later use
with open('my_rfr_reg2.pkl','wb') as fid:
    pickle.dump(gs_rfr.best_estimator_,fid)
    fid.close()

ypred = gs_rfr.best_estimator_.predict(test[predictors])

print 'RFR test mse:',metrics.mean_squared_error(ypred,test.Tip_percentage)
print 'RFR r2:', metrics.r2_score(ypred,test.Tip_percentage)
print dt.datetime.now()-tic
plot_opt_results(gs_rfr)

feat_imp = pd.Series(gs_rfr.best_estimator_.feature_importances_,predictors).sort_values(ascending=False)
feat_imp.plot(kind='bar',title='Feature Importances')
plt.ylabel('Feature Importance Score')
plt.ylim([0,1])
plt.show()



