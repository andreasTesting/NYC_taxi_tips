# Print the size of the dataset

import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import os, json, requests, pickle
from scipy.stats import skew
from shapely.geometry import MultiLineString, Polygon, Point, MultiPoint, MultiPolygon, LinearRing
from scipy.stats import ttest_ind, f_oneway, lognorm, levy, skew, chisquare
#import scipy.stats as st
from sklearn.preprocessing import normalize, scale
from tabulate import tabulate #pretty print of tables. source: http://txt.arboreus.com/2013/03/13/pretty-print-tables-in-python.htmls
import pandas as pd

from scipy.spatial import Delaunay
import numpy as np
import math, warnings
from shapely.ops import cascaded_union, polygonize

import json
from pprint import pprint
import collections
import urllib

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


def plot_area_tips_histogram(data,poi,count,area_name='None'):
    """
    This function returns 1 if a location loc(lat,lon) is located inside a polygon of interest poi
    poi: shapely.geometry.Polygon, polygon of interest
    """

    tic = dt.datetime.now()
    # Create a new variable to check if a trip originated in Upper Manhattan
    data['U_area'] = data[['pickup_latitude','pickup_longitude']].apply(lambda r:is_within_bbox((r[0],r[1]),poi),axis=1)
    print "Processing time ", dt.datetime.now()-tic

    # create a vector to contain Tip percentage for
    v1 = data[(data.U_area==0) & (data.tip_percentage>0)].tip_percentage
    v2 = data[(data.U_area==1) & (data.tip_percentage>0)].tip_percentage

    # generate bins and histogram values
    bins = np.histogram(v1,bins=3)[1]
    h1 = np.histogram(v1,bins=bins)
    h2 = np.histogram(v2,bins=bins)

    # generate the plot
    # First suplot: visualize all data with outliers
    fig,ax = plt.subplots(1,1,figsize=(10,5))
    w = .4*(bins[1]-bins[0])
    ax.bar(bins[:-1],h1[0],width=w,color='b')
    ax.bar(bins[:-1]+w,h2[0],width=w,color='g')
    ax.set_yscale('log')
    ax.set_xlabel('Tip (%)')
    ax.set_ylabel('Count')
    ax.set_title('Tip')
    ax.legend(['Non-Area','Area'],title='origin')
    filename = 'hist_area_tips_' + area_name + '_' + str(count)
    plt.savefig(filename,format='jpeg')
    ttest = ttest_ind(v1,v2,equal_var=False)
    print 't-test results:', 

    return ttest


# Get external geolocation data for geo accosiations
if os.path.exists('d085e2f8d0b54d4590b1e7d1f35594c1pediacitiesnycneighborhoods.geojson.json'): # Check if the dataset is present on local disk and load it
    with open('d085e2f8d0b54d4590b1e7d1f35594c1pediacitiesnycneighborhoods.geojson.json') as data_file:    
        NYC_geo_data = json.load(data_file)
else: # Download geolocation dataset if not available on disk
    url = "http://data.beta.nyc//dataset/0ff93d2d-90ba-457c-9f7e-39e47bf2ac5f/resource/35dd04fb-81b3-479b-a074-a27a37888ce7/download/d085e2f8d0b54d4590b1e7d1f35594c1pediacitiesnycneighborhoods.geojson"
    urllib.urlretrieve (url, "d085e2f8d0b54d4590b1e7d1f35594c1pediacitiesnycneighborhoods.geojson.json")
    with open('d085e2f8d0b54d4590b1e7d1f35594c1pediacitiesnycneighborhoods.geojson.json') as data_file:    
        NYC_geo_data = json.load(data_file)    


dict_level1_len = len(NYC_geo_data['features'])

borough_list = []
for i in range(dict_level1_len):
    borough_list.append(NYC_geo_data['features'][i]['properties']['borough'])
borough_list_unique = list(set(borough_list))

neighborhood_list = []
for i in range(dict_level1_len):
    neighborhood_list.append(NYC_geo_data['features'][i]['properties']['neighborhood'])
neighborhood_list_unique = list(set(neighborhood_list))


# Download the June 2015 dataset
if os.path.exists('yellow_tripdata_2015-06.csv'): # Check if the dataset is present on local disk and load it
    data = pd.read_csv('yellow_tripdata_2015-06.csv')
else: # Download dataset if not available on disk
    url = "https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2015-06.csv"
    data = pd.read_csv(url)
    data.to_csv(url.split('/')[-1])


print "Number of rows:", data.shape[0]
print "Number of columns: ", data.shape[1]


data = data[(data.total_amount>=2.5)] #cleaning
data['tip_percentage'] = 100*data.tip_amount/data.total_amount
print "Summary: Tip percentage\n",data.tip_percentage.describe()

# prepare dictionary for the boroughs in NYC
df_boroughs = pd.DataFrame(borough_list_unique)
df_boroughs = df_boroughs.rename(columns={0: "NYC_borough"}) 
df_boroughs['poly_list'] = np.empty((len(df_boroughs), 0)).tolist()
dict_boroughs = df_boroughs.set_index('NYC_borough').to_dict()


# for every borough collect the points of interest of the respective sub-areas 
for  i in range(len(neighborhood_list)):   
      area_name = NYC_geo_data['features'][i]['properties']['neighborhood']
      borough_name = NYC_geo_data['features'][i]['properties']['borough']
      poly_indices = NYC_geo_data['features'][i]['geometry']['coordinates']
      array_dim = np.asarray(poly_indices).shape[1] * np.asarray(poly_indices).shape[2] 
      poly_indices_array = np.asarray(poly_indices).reshape(array_dim,)
      poly_indices_list = list(poly_indices_array)
      it = iter(poly_indices_list)
      poly_list_of_tuples = zip(it, it)
      poi = Polygon(poly_list_of_tuples)
      dict_boroughs['poly_list'][borough_name].append(poi)
      
# create histogram of tip level for each borough (as origin), compared to all other boroughs
for  i in range(len(borough_list_unique)): 
      borough_name = borough_list_unique[i]  
      borough_poly = cascaded_union(dict_boroughs['poly_list'][borough_name])
      borough_poly_conv_hull = borough_poly.convex_hull
      borough_poly_conv_hull_ind = np.asarray(borough_poly_conv_hull.exterior)
      borough_poly_conv_hull_ind[:,[0, 1]] = borough_poly_conv_hull_ind[:,[1, 0]]
      borough_poly_conv_hull = Polygon(borough_poly_conv_hull_ind)
      plot_area_tips_histogram(data,borough_poly_conv_hull,i,borough_name)




