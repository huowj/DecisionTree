#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 12:51:50 2018

@author: hwj
"""

import math
import numpy as np
import pandas as pd
from scipy.stats import mode


titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
titanic.to_csv("/hwj/DataMining/DecisionTree/datasets/titanic.csv")
#print(titanic.head())

######so we should transform the dataset#############

age_class=[]
for i in range(titanic.shape[0]):
    if titanic.age[i]<19:
        age_class.append('children')
    elif titanic.age[i]<40:
        age_class.append('adult')
    else:
        age_class.append('old')

#boat_class=[]
#for i in range(titanic.shape[0]):
#    if math.isnan(titanic.boat[i]):
#        boat_class.append(1)
#    else:
#        boat_class.append(0)
        
titanic_transformation=pd.concat([titanic,pd.DataFrame(age_class,columns=['age_class'])],axis=1)


def ExpectedInformation(y):
    values_y={}
    y_list=y.tolist()
    for i in y_list:
        if y_list.count(i)>1:
            values_y[i]=y_list.count(i)
    total_y=np.sum(list(values_y.values()))
    E_I=0
    for key in values_y.keys():
        proportion_y=values_y[key]/total_y
        E_I-=proportion_y*math.log(proportion_y,2)
    return(E_I)



def gain(x,y):
    data=pd.DataFrame(list(zip(x,y)),columns=[x.name,y.name])
    E_I_y=ExpectedInformation(y)
    x_list=x.tolist()
    values_x={}
    for i in x_list:
        if x_list.count(i)>1:
            values_x[i]=x_list.count(i)
    total_x=np.sum(list(values_x.values()))
    E_I_x=0
    for key in values_x.keys():
        data1_y=data[data[x.name]==key][y.name]
        proportion_x=values_x[key]/total_x
        E_I_x+=proportion_x*ExpectedInformation(data1_y)
    gain_x=E_I_y-E_I_x
    print(x.name,gain_x)
    return(gain_x)
        

  
def ID3(x_dataset,y_dataset):
    y_name=y_dataset.name
    if len(list(set(y_dataset)))==1:
        return(y_dataset[0])
    x_columns=x_dataset.columns
    if len(x_columns)==0:
        return(mode(y_dataset)[0][0])
    gain_index=0
    attribute=''
    for item in x_columns:
        if gain(x_dataset[item],y_dataset)>gain_index:
            gain_index=gain(x_dataset[item],y_dataset)
            attribute=item
    attribute_values=list(set(x_dataset[attribute]))
    DecisionTree={attribute:{}}
    data_now=pd.concat([x_dataset,pd.DataFrame(y_dataset)],axis=1)
    for value in attribute_values:
        data_temp=data_now[data_now[attribute]==value].drop([attribute],axis=1)
        x_dataset_temp=data_temp.drop([y_name],axis=1)
        y_dataset_temp=data_temp[y_name]
        DecisionTree[attribute][value]=ID3(x_dataset_temp,y_dataset_temp)
    return(DecisionTree)
        
    
        
#    x_columns=x_dataset.columns
#    gain_index=0
#    attribute=''
#    for item in x_columns:
#        if gain(x_dataset[item],y_dataset)>gain_index:
#            gain_index=gain(x_dataset[item],y_dataset)
#            attribute=item
#    print(gain_index,attribute)
        
    
   
if __name__ == "__main__":
    x_train=titanic_transformation.iloc[:,[1,10,11]]
    y_train=titanic_transformation.survived
#    ExpectedInformation(titanic.survived)
#    gain(titanic.boat,titanic.survived)
    a=ID3(x_train,y_train)
    print(a)
    