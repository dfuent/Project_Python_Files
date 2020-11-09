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

#top 200 words
allcandidates60_freq = pd.Series(' '.join(debates_1960.Transcript).split()).value_counts()[:200]
allcandidates60_freq

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

#top 200 words
allcandidates76_freq= pd.Series(' '.join(allcandidates76.Transcript).split()).value_counts()[:200]
allcandidates76_freq

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

#top 200 words
allcandidates84_freq= pd.Series(' '.join(allcandidates84.Transcript).split()).value_counts()[:200]
allcandidates84_freq

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

#top 200 words
allcandidates96_freq= pd.Series(' '.join(allcandidates96.Transcript).split()).value_counts()[:200]
allcandidates96_freq

#Begin 2000
#October 5, 2000
Oct5_2000_debate = df[df["Debate"]=='October 5, 2000 Debate Transcript']
Oct5_2000_debate

#October 17, 2000
Oct17_2000_debate = df[df["Debate"]=='October 17, 2000 Debate Transcript']
Oct17_2000_debate

#October 3, 2000
Oct3_2000_debate = df[df["Debate"]=='October 3, 2000 Transcript']
Oct3_2000_debate

#October 11, 2000
Oct11_2000_debate = df[df["Debate"]=='October 11, 2000 Debate Transcript']
Oct11_2000_debate

#full debate transcript 2000, includes moderators and any other speakes in addition to candidates
debates_2000 = Oct5_2000_debate.append([Oct17_2000_debate, Oct3_2000_debate, Oct11_2000_debate])
debates_2000

#Gore2000
Gore00 = debates_2000[(debates_2000["Speaker_standardized"]=='Al Gore')]
Gore00

#Lieberman2000
Lieberman00 = debates_2000[(debates_2000["Speaker_standardized"]=='Joseph Lieberman')]
Lieberman00

#Bush2000
Bush00 = debates_2000[(debates_2000["Speaker_standardized"]=='George W. Bush')]
Bush00

#Cheney2000
Cheney00 = debates_2000[(debates_2000["Speaker_standardized"]=='Dick Cheney')]
Cheney00

#GOP2000
GOP2000=Bush00.append([Cheney00])
GOP2000

#Dem2000
Dem2000=Gore00.append([Lieberman00])
Dem2000

#all candidates 2000
allcandidates00 = GOP2000.append([Dem2000]) 
allcandidates00

#top 200 words
allcandidates00_freq= pd.Series(' '.join(allcandidates00.Transcript).split()).value_counts()[:200]
allcandidates00_freq

#Begin 2004
#October 8, 2004
Oct8_2004_debate = df[df["Debate"]=='October 8, 2004 Debate Transcript']
Oct8_2004_debate

#September 30, 2004
Sept30_2004_debate = df[df["Debate"]=='September 30. 2004 Debate Transcript']
Sept30_2004_debate

#October 5, 2004
Oct5_2004_debate = df[df["Debate"]=='October 5, 2004 Transcript']
Oct5_2004_debate

#October 13, 2004
Oct13_2004_debate = df[df["Debate"]=='October 13, 2004 Debate Transcript']
Oct13_2004_debate

#full debate transcript 2004, includes moderators and any other speakes in addition to candidates
debates_2004 = Oct8_2004_debate.append([Sept30_2004_debate, Oct5_2004_debate, Oct13_2004_debate])
debates_2004

#Bush2004
Bush04 = debates_2004[(debates_2004["Speaker_standardized"]=='George W. Bush')]
Bush04

#Cheney2004
Cheney04 = debates_2004[(debates_2004["Speaker_standardized"]=='Dick Cheney')]
Cheney04

#Kerry2004
Kerry04 = debates_2004[(debates_2004["Speaker_standardized"]=='John Kerry')]
Kerry04

#Bush2004
Bush04 = debates_2004[(debates_2004["Speaker_standardized"]=='George W. Bush')]
Bush04

#Edwards2004
Edwards04 = debates_2004[(debates_2004["Speaker_standardized"]=='John Edwards')]
Edwards04

#GOP2004
GOP2004=Bush04.append([Cheney04])
GOP2004

#Dem2004
Dem2004=Kerry04.append([Edwards04])
Dem2004

#all candidates 2004
allcandidates04 = GOP2004.append([Dem2004]) 
allcandidates04

#top 200 words
allcandidates04_freq= pd.Series(' '.join(allcandidates04.Transcript).split()).value_counts()[:200]
allcandidates04_freq

#Begin 2008
#October 15, 2008
Oct15_2008_debate = df[df["Debate"]=='October 15, 2008 Debate Transcript']
Oct15_2008_debate

#October 2, 2008
Oct2_2008_debate = df[df["Debate"]=='October 2, 2008 Debate Transcript']
Oct2_2008_debate

#Septmber 26, 2008
Sept26_2008_debate = df[df["Debate"]=='September 26, 2008 Debate Transcript']
Sept26_2008_debate

#October 7, 2008
Oct7_2008_debate = df[df["Debate"]=='October 7, 2008 Debate Transcript']
Oct7_2008_debate

#full debate transcript 2008, includes moderators and any other speakes in addition to candidates
debates_2008 = Oct15_2008_debate.append([Oct2_2008_debate, Sept26_2008_debate, Oct7_2008_debate])
debates_2008

#McCain2008
McCain08 = debates_2008[(debates_2008["Speaker_standardized"]=='John McCain')]
McCain08

#Palin2008
Palin08 = debates_2008[(debates_2008["Speaker_standardized"]=='Sarah Palin')]
Palin08

#Obama2008
Obama08 = debates_2008[(debates_2008["Speaker_standardized"]=='Barack Obama')]
Obama08

#Biden2008
Biden08 = debates_2008[(debates_2008["Speaker_standardized"]=='Joe Biden')]
Biden08

#GOP2008
GOP2008=McCain08.append([Palin08])
GOP2008

#Dem2008

Dem2008=Obama08.append([Biden08])
Dem2008

#allcandidates2008
allcandidates08 = GOP2008.append([Dem2008]) 
allcandidates08

#top 200 words
allcandidates08_freq= pd.Series(' '.join(allcandidates08.Transcript).split()).value_counts()[:200]
allcandidates08_freq

#Begin 2012
#October 16, 2012
Oct16_2012_debate = df[df["Debate"]=='October 16, 2012 Debate Transcript']
Oct16_2012_debate

#October 3, 2012
Oct3_2012_debate = df[df["Debate"]=='October 3, 2012 Debate Transcript']
Oct3_2012_debate

#October 22, 2012
Oct22_2012_debate = df[df["Debate"]=='October 22, 2012 Debate Transcript']
Oct22_2012_debate

#October 11, 2012
Oct11_2012_debate = df[df["Debate"]=='October 11, 2012 Debate Transcript']
Oct11_2012_debate

#full debate transcript 2012, includes moderators and any other speakes in addition to candidates
debates_2012 = Oct16_2012_debate.append([Oct3_2012_debate, Oct22_2012_debate, Oct11_2012_debate])
debates_2012

#Obama2012
Obama12 = debates_2012[(debates_2012["Speaker_standardized"]=='Barack Obama')]
Obama12

#Biden2012
Biden12 = debates_2012[(debates_2012["Speaker_standardized"]=='Joe Biden')]
Biden12

#Romney2012
Romney12 = debates_2012[(debates_2012["Speaker_standardized"]=='Mitt Romney')]
Romney12

#Ryan2012
Ryan12 = debates_2012[(debates_2012["Speaker_standardized"]=='Paul Ryan')]
Ryan12

#GOP2012
GOP2012=Romney12.append([Ryan12])
GOP2012

#Dem2012
Dem2012=Obama12.append([Biden12])
Dem2012

#all candidates 2012
allcandidates12 = GOP2012.append([Dem2012]) 
allcandidates12

#top 200 words
allcandidates12_freq= pd.Series(' '.join(allcandidates12.Transcript).split()).value_counts()[:200]
allcandidates12_freq

#Begin2016
#October 4, 2016
Oct4_2016_debate = df[df["Debate"]=='October 4, 2016 Debate Transcript']
Oct4_2016_debate

#September 26, 2016
Sept26_2016_debate = df[df["Debate"]=='September 26, 2016 Debate Transcript']
Sept26_2016_debate

#October 19, 2016
Oct19_2016_debate = df[df["Debate"]=='October 19, 2016 Debate Transcript']
Oct19_2016_debate

#October 9, 2016
Oct9_2016_debate = df[df["Debate"]=='October 9, 2016 Debate Transcript']
Oct9_2016_debate

#full debate transcript 2016, includes moderators and any other speakes in addition to candidates
debates_2016 = Oct4_2016_debate.append([Sept26_2016_debate, Oct19_2016_debate, Oct9_2016_debate])
debates_2016

#Trump2016
Trump16 = debates_2016[(debates_2016["Speaker_standardized"]=='Donald Trump')]
Trump16

#Pence16
Pence16 = debates_2016[(debates_2016["Speaker_standardized"]=='Mike Pence')]
Pence16

#Hillary Clinton16 --labled as Bill in data
Clinton16 = debates_2016[(debates_2016["Speaker_standardized"]=='Bill Clinton')]
Clinton16

#Kaine16
Kaine16 = debates_2016[(debates_2016["Speaker_standardized"]=='Tim Kaine')]
Kaine16

#GOP 2016
GOP2016 = Trump16.append([Pence16]) 
GOP2016

#Dem 2016
Dem2016 = Clinton16.append([Kaine16]) 
Dem2016

#allcandidates 2016
allcandidates16 = GOP2016.append([Dem2016]) 
allcandidates16

#top 200 words
allcandidates16_freq= pd.Series(' '.join(allcandidates16.Transcript).split()).value_counts()[:200]
allcandidates16_freq

tdf['Jobs Count'] = tdf['Transcript'].str.count('jobs')
tdf['Economy Count'] = tdf.Transcript.str.count('economy')
tdf['Growth Count'] = tdf.Transcript.str.count('growth') # modify these to include capital as well
tdf['Terrorist Count'] = tdf.Transcript.str.count('terrorist')

tdf['AfterComma'] = tdf['Debate'].str.split(',').str[1]
tdf['AfterComma'] = tdf['AfterComma'].str.lstrip()
# print(AfterComma)
tdf['Year'] = tdf['AfterComma'].str[:4].fillna(0).astype(int)
# print(Year)

# tdf.to_csv('econanalysis.csv')

jobsPivot = tdf.pivot_table(index = ['Year'], values = ['Jobs Count'], aggfunc = 'sum')
macro_df = pd.read_csv('US_economic_data_annual.csv')
new_tdf = pd.merge(jobsPivot,macro_df,on='Year',how='right')
new_tdf.to_csv('econanalysis.csv')

terroristPivot = tdf.pivot_table(index = ['Year'], values = ['Terrorist Count'], aggfunc = 'sum')
# print(terroristPivot)
