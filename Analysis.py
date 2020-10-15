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
Nixondf = df[df["Speaker_standardized"]=='Richard Nixon']
Nixondf

# create Kennedy/Dem dataframe for 1960
Kennedydf = df[df["Speaker_standardized"]=='John F. Kennedy']
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

Ford76 = debates_1976[(debates_1976["Speaker_standardized"]=='Gerald Ford')]
Ford76 #row 810 attributes speech from a Mr. Kraft to Ford

#Carter/Dem 1976 debates; Carter76 is also what we have for Dems in 1976 because we don't have the VP debate transcript
Carter76 = debates_1976[(debates_1976["Speaker_standardized"]=='Jimmy Carter')]
Carter76 # includes (barely audible) in Carter word count

#both candidates/parties 1976 (without moderator/others)
allcandidates76 = Ford76.append([Carter76])
allcandidates76

#Begin 1980 debates
#October 28, 1980
Oct28_1980_debate = df[df["Debate"]=='October 28, 1980 Debate Transcript']
Oct28_1980_debate

#September 21, 1980 debate
Sept21_1980_debate = df[df["Debate"]=='September 21, 1980 Debate Transcript']
Sept21_1980_debate

#full debate transcript 1980, includes moderators and any other speakes in addition to candidates
debates_1980 = Oct28_1980_debate.append([Sept21_1980_debate])
debates_1980

#for Carter/Dems 1980; no VP debate in 1980
Carter80 = debates_1980[(debates_1980["Speaker_standardized"]=='Jimmy Carter')]
Carter80

#for Reagan/GOP in 1980, no VP debate
Reagan80 = debates_1980[(debates_1980["Speaker_standardized"]=='Ronald Reagan')]
Reagan80

#for Anderson/Ind. 1980, no VP debate in 1980
Anderson80 = debates_1980[(debates_1980["Speaker_standardized"]=='Ronald Reagan')]
Anderson80

#all candidates 1980
allcandidates80 = Anderson80.append([Reagan80, Carter80]) 
allcandidates80

#Begin 1984 debates
#October 7, 1984
Oct7_1984_debate = df[df["Debate"]=='October 7, 1984 Debate Transcript']
Oct7_1984_debate

#October 21, 1984
Oct21_1984_debate = df[df["Debate"]=='October 21, 1984 Debate Transcript']
Oct21_1984_debate 

#October 11, 1984 (VP debate)
Oct11_1984_debate = df[df["Debate"]=='October 11, 1984 Debate Transcript']
Oct11_1984_debate 

#all debate transcripts 1984, includes moderator and any other speakers
debates_1984 = Oct11_1984_debate.append([Oct21_1984_debate, Oct7_1984_debate])
debates_1984

#Democratic party 1984 (Mondale)
Mondale84 = debates_1984[(debates_1984["Speaker_standardized"]=='Walter Mondale')]
Mondale84

#Dem party 1984 (Ferraro)
Ferraro84 = debates_1984[(debates_1984["Speaker_standardized"]=='Geraldine Ferraro')]
Ferraro84

#all Dem transcript 1984
Dem1984 = Mondale84.append([Ferraro84])
Dem1984

#Republican party 1984 (Reagan)
Reagan84 = debates_1984[(debates_1984["Speaker_standardized"]=='Ronald Reagan')]
Reagan84

#Republican party 1984 (Bush)
Bush84 = debates_1984[(debates_1984["Speaker_standardized"]=='George W. Bush')]
Bush84

#full Republican party 1984
GOP1984 = Reagan84.append([Bush84])
GOP1984

allcandidates84 = GOP1984.append([Dem1984]) 
allcandidates84

#Begin 1988 debates
#October 5, 1988
Oct5_1988_debate = df[df["Debate"]=='October 5, 1988 Debate Transcripts']
Oct5_1988_debate

#September 25, 1988 debate
Sept25_1988_debate = df[df["Debate"]=='September 25, 1988 Debate Transcript']
Sept25_1988_debate

#October 13, 1988
Oct13_1988_debate = df[df["Debate"]=='October 13, 1988 Debate Transcript']
Oct13_1988_debate

#full debate transcript 1988, includes moderators and any other speakes in addition to candidates
debates_1988 = Oct13_1988_debate.append([Sept25_1988_debate, Oct5_1988_debate])
debates_1988

#Bush 1988
Bush88 = debates_1988[(debates_1988["Speaker_standardized"]=='George W. Bush')]
Bush88

#Quayle 1988
Quayle88 = debates_1988[(debates_1988["Speaker_standardized"]=='Dan Quayle')]
Quayle88

#Dukakis 1988
Dukakis88 = debates_1988[(debates_1988["Speaker_standardized"]=='Michael Dukakis')]
Dukakis88

#Bentsen 1988
Bentsen88 = debates_1988[(debates_1988["Speaker_standardized"]=='Lloyd Bentsen')]
Bentsen88

#GOP 1988
GOP1988 = Bush88.append([Quayle88]) 
GOP1988

#Dem 1988
Dem1988 = Dukakis88.append([Bentsen88]) 
Dem1988

#allcandidates 1988
allcandidates88 = GOP1988.append([Dem1988]) 
allcandidates88

#Begin 1992
#Oct 15, 1992 second half
Oct15_1992_2ndhalf_debate = df[df["Debate"]=='October 15, 1992 Second Half Debate Transcript']
Oct15_1992_2ndhalf_debate

#Oct 15 first half, 1992 debate
Oct15_1992_1sthalf_debate = df[df["Debate"]=='October 15, 1992 First Half Debate Transcript']
Oct15_1992_1sthalf_debate

#October 19, 1992
Oct19_1992_debate = df[df["Debate"]=='October 19, 1992 Debate Transcript']
Oct19_1992_debate

#October 11, 1992 first half
Oct11_1992_1sthalf_debate = df[df["Debate"]=='October 11, 1992 First Half Debate Transcript']
Oct11_1992_1sthalf_debate

#October 11, 1992 second half 
Oct11_1992_2ndhalf_debate = df[df["Debate"]=='October 11, 1992 Second Half Debate Transcript']
Oct11_1992_2ndhalf_debate

#October 13, 1992
Oct13_1992_debate = df[df["Debate"]=='October 13, 1992 Debate Transcript']
Oct13_1992_debate

#full debate transcript 1992, includes moderators and any other speakes in addition to candidates
debates_1992 = Oct13_1992_debate.append([Oct15_1992_1sthalf_debate, Oct15_1992_2ndhalf_debate, Oct11_1992_1sthalf_debate, Oct11_1992_2ndhalf_debate, Oct19_1992_debate])
debates_1992

#Bush1992
Bush92 = debates_1992[(debates_1992["Speaker_standardized"]=='George W. Bush')]
Bush92

#Quayle1992
Quayle92 = debates_1992[(debates_1992["Speaker_standardized"]=='Dan Quayle')]
Quayle92

#Clinton1992
Clinton92 = debates_1992[(debates_1992["Speaker_standardized"]=='Bill Clinton')]
Clinton92

#Gore1992
Gore92 = debates_1992[(debates_1992["Speaker_standardized"]=='Al Gore')]
Gore92

#Perot1992
Perot92 = debates_1992[(debates_1992["Speaker_standardized"]=='Ross Perot')]
Perot92

#Stockdale1992
Stockdale92 = debates_1992[(debates_1992["Speaker_standardized"]=='Adm. James Stockdale')]
Stockdale92

#GOP1992
GOP1992=Bush92.append([Quayle92])
GOP1992

#Dem1992

Dem1992=Clinton92.append([Gore92])
Dem1992

#Ind1992
Ind1992=Perot92.append([Stockdale92])
Ind1992

#all candidates 1992
allcandidates92 = GOP1992.append([Dem1992, Ind1992]) 
allcandidates92

#begin 1996 debates
#Oct 6, 1996
Oct6_1996_debate = df[df["Debate"]=='October 6, 1996 Debate Transcript']
Oct6_1996_debate

#Oct 16, 1996
Oct16_1996_debate = df[df["Debate"]=='October 16, 1996 Debate Transcript']
Oct16_1996_debate

#Oct 9, 1996
Oct9_1996_debate = df[df["Debate"]=='October 9, 1996 Debate Transcript']
Oct9_1996_debate

#full debate transcript 1996, includes moderators and any other speakes in addition to candidates
debates_1996 = Oct9_1996_debate.append([Oct6_1996_debate, Oct16_1996_debate])
debates_1996

#Clinton96
Clinton96 = debates_1996[(debates_1996["Speaker_standardized"]=='Bill Clinton')]
Clinton96

#Gore96
Gore96 = debates_1996[(debates_1996["Speaker_standardized"]=='Al Gore')]
Gore96

#Dole96
Dole96 = debates_1996[(debates_1996["Speaker_standardized"]=='Bob Dole')]
Dole96

#Kemp96
Kemp96 = debates_1996[(debates_1996["Speaker_standardized"]=='Jack Kemp')]
Kemp96

#GOP1996
GOP1996=Dole96.append([Kemp96])
GOP1996

#Dem1996
Dem1996=Clinton96.append([Gore96])
Dem1996

#all candidates 1996
allcandidates96 = GOP1996.append([Dem1996]) 
allcandidates96
