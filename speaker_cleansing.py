# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 11:56:31 2020
@author: dfuent
"""
import pandas as pd
import os
import numpy as np

def speaker_map(row):

    if type(row['Transcript']) is str:
        if row['Transcript'].find(':') > 0:
            f = row['Transcript'].find(':')
        else: 
            f = row['Transcript'].find(';')
        if f > 0:            
            return row['Transcript'][0:f]
        else:
            return np.nan

def find_colon(row):
    i = 0
    try:
        for r in row['Transcript'].split(' '): 
            i += 1
            if r.find(':') > 0:
                if i < 4:
                    return i
                else:
                    return 0
        return 0
    except:
        return 0

df = pd.read_csv('Transcripts_df.csv')

print(df.columns)

df['Speaker_Clean'] = df.apply(speaker_map, axis = 1)


l = ['MR. ', 'MRS.', 'MS.']

for i in l:
    df['Speaker_Clean'] = df['Speaker_Clean'].str.replace(i, '')
print(df.head())

df['Speaker_Clean'] = df['Speaker_Clean'].fillna(method='ffill')
#df['Speaker_Clean'] = np.where(df['Speaker_Clean'].isnull(), df['Speaker'], df['Speaker_Clean'])
df['Speaker_fin'] = np.where(df['Speaker_Clean'].str.split(' ').str.len() > 3, df['Speaker'], df['Speaker_Clean'])
df['Speaker_fin'] = np.where(df['Speaker_fin'].isnull(), df['Speaker_Clean'], df['Speaker_fin'])
df['word_count'] = df['Transcript'].str.split(' ').str.len()
df['flag'] = df.apply(find_colon, axis = 1)

df['word_count'] = df['word_count'] - df['flag']
df = df.drop(['flag', 'Speaker'], axis=1)
df.fillna('xx')

print(df.head())
print(df.dtypes)

pd.Series(df['Speaker_fin'].unique()).to_csv('speakers.csv')

speaker_map = pd.read_csv('speaker_map.csv')

print(speaker_map.head())
print(speaker_map.dtypes)

#df.join(a.set_index('LabelName'), on='LabelName')

df = df.join(speaker_map.set_index('Speaker_fin'), rsuffix='_map', on = 'Speaker_fin')

#df.drop(columns=['0'])

#df = df.merge(speaker_map,  on='Speaker_fin', suffixes=('','_map'))

df.to_csv('Transcripts_final.csv')
