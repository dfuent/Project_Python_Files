# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 11:56:31 2020
@author: dfuentes
"""
import pandas as pd
import os
import numpy as np
import math

def speaker_map(row): # function to find if a transcript row contains a speaker

    if type(row['Transcript']) is str: # if a transcript line is a string
        
        # there are two main instances of whether a transcript line contains
        # a speaker: if there's a colon in the line, or if there's a semi-colon
        # There are some other instances, but we clean those up later via 
        # a mapping file 
        
        if row['Transcript'].find(':') > 0: # find returns -1 if ':' isn't found
            f = row['Transcript'].find(':') # return the index of the colon
        else: 
            f = row['Transcript'].find(';') # same as above except w/ semi-colon
        if f > 0:            
            return row['Transcript'][0:f] # if either ';' or ':' were found, return the name
        else:
            return np.nan # if neither found, return nan


# there are some lines in the transcripts that have colons when a speaker is 
# making a point (i.e. not a name). We want to flag where the speaker_map returns
# a name, not one of these instances. We use this later to pull out the 
# actual transcript words and not just the name

def find_colon(row): 
    i = 0
    try:
        for r in row['Transcript'].split(' '): 
            i += 1
            if r.find(':') > 0:
                
                # return the position of the colon if it's found in one of the
                # first 4 words
                if i < 4:
                    return i 
                else: # return 0 if the colon shows up later
                    return 0
        # If we get here, there isn't a colon later in the word        
        return 0
    except:
        return 0
    
# functions are now defined. Will use those later in the apply    

df = pd.read_csv('Transcripts_df.csv')

print(df.columns)

# apply the function to isolate the speaker
df['Speaker_Clean'] = df.apply(speaker_map, axis = 1)


# Need to remove formality from the names
l = ['MR. ', 'MRS.', 'MS.']


# Replace the formal greetings
for i in l:
    df['Speaker_Clean'] = df['Speaker_Clean'].str.replace(i, '')

# We want to forward fill the speaker column if blank
df['Speaker_Clean'] = df['Speaker_Clean'].fillna(method='ffill')
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
clean_map = pd.read_csv('map_file_final.csv')

#print(speaker_map.head())
#print(speaker_map.dtypes)


df = df.join(speaker_map.set_index('Speaker_fin'), rsuffix='_map', on = 'Speaker_fin')

df2 = pd.read_csv('2020debates.csv')
df2['word_count'] = df2['Transcript'].str.split(' ').str.len()

df = df.append(df2)

print(df.columns)

df['Debate_date'] = df['Debate'].str.replace('Debate', '').str.strip().str.replace('Transcripts', '').str.replace('Transcript', '').str.strip().str.replace('First Half', '').str.strip().str.replace('Second Half', '').str.strip()

df['Key'] = df['Debate_date'] + ' | ' + df['Speaker_standardized'].str.strip()

df = df.join(clean_map.set_index('Key'), rsuffix='_map', on = 'Key')

df.drop(['Speaker_fin', '0'], axis = 1, inplace = True)

df['Transcript'] = df['Transcript'].str.upper()

df.to_csv('Transcripts_full.csv')

# create yearly files and individual candidate files

year_list = df['Year'].unique()
speaker_list = df['Actual Speaker'].unique()

abs_dir = os.getcwd()

rel_dir = os.path.join(abs_dir, '\Yearly_Files\\')

for i in year_list:
    if math.isnan(i) == False:
        df_year = df[df['Year'] == i]
        df_year.to_csv(abs_dir + '\Yearly_Files\Transcripts_' + str(int(i)) + '.csv')
        for j in speaker_list:
            df_y_spk = df_year[df_year['Actual Speaker'] == j]
            if len(df_y_spk.index) > 0:
                df_y_spk.to_csv(abs_dir + '\Yearly_Files\Transcripts_' + j + '_' + str(int(i)) + '.csv')
            else:
                pass
    else:
        pass
 

