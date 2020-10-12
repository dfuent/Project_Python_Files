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
   

df = pd.read_csv('Transcripts_df.csv')

print(df.columns)

df['Speaker_Clean'] = df.apply(speaker_map, axis = 1)


l = ['MR. ', 'MRS.', 'MS.']

for i in l:
    df['Speaker_Clean'] = df['Speaker_Clean'].str.replace(i, '')
print(df.head())

df['Speaker_Clean'] = np.where(df['Speaker_Clean'].isnull(), df['Speaker'], df['Speaker_Clean'])
df['Speaker_fin'] = np.where(df['Speaker_Clean'].str.split(' ').str.len() > 3, df['Speaker'], df['Speaker_Clean'])

df.to_csv('Test Speaker.csv')

# test case to find China in each row in csv. next i'm going to investigate how to search against a list of words
tdf['China Count'] = tdf.Transcript.str.count('China')
# tdf.to_csv('new_data.csv')
# print(tdf.head())
wordPivot = tdf.pivot_table(index = ['Speaker_fin'], values = ['China Count'], aggfunc = 'sum') # pivot table aggregates total number of word mentions across a speaker
# print(wordPivot)
wordPivot.to_csv('testwords.csv') 

#creation of dataframes by election cycle and by candidate/party
#Begin 1960
# create Nixon/GOP dataframe for 1960
Nixondf = df[df["Speaker_fin"]=='NIXON']
Nixondf

# create Kennedy/Dem dataframe for 1960
Kennedydf = df[df["Speaker_fin"]=='KENNEDY']
Kennedydf # the copyright information gets attributed to Kennedy

# create 1960 both candidates dataframe
debates_1960= Nixondf.append(Kennedydf)
debates_1960

#Begin 1976
#September 23, 1976 debate
Sept23_1976_debate = df[df["Debate"]=='September 23, 1976 Debate Transcript']
Sept23_1976_debate

#October 6, 1976 debate
Oct6_1976_debate = df[df["Debate"]=='October 6, 1976 Debate Transcript']
Oct6_1976_debate

#October 22, 1976 debate
Oct22_1976_debate = df[df["Debate"]=='October 22, 1976 Debate Transcript']
Oct22_1976_debate

#full debate transcript for 1976, includes moderator and any other non-candidate speech
debates_1976 = Sept23_1976_debate.append([Oct6_1976_debate, Oct22_1976_debate])
debates_1976

#for Ford/GOP 1976; Ford76 is also what we have for the GOP in 1976 because we don't have the VP debate transcript

Ford76 = debates_1976[(debates_1976["Speaker_fin"]=='FORD')]
Ford76 #row 810 attributes speech from a Mr. Kraft to Ford

#Carter/Dem 1976 debates; Carter76 is also what we have for Dems in 1976 because we don't have the VP debate transcript
Carter76 = debates_1976[(debates_1976["Speaker_fin"]=='CARTER')]
Carter76 # includes (barely audible) in Carter word count

#both candidates/parties 1976 (without moderator/others)
allcandidates76 = Ford76.append([Carter76])
allcandidates76
