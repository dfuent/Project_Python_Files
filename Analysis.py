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

###########################Caprill's analysis###################################3
#import packages for dataframes
import pandas as pd
import os
import numpy as np

#read in csv of transcript and related data
df = pd.read_csv('Transcripts_clean.csv')

#return all unique values in Debate column
debatesdates=df.Debate.unique()
debatesdates

#creation of dataframes by election cycle and by candidate/party
#Begin 1960
# create Nixon/GOP dataframe for 1960
Nixondf = df[df["Speaker_standardized"]=='Richard Nixon']
Nixondf

#testing length
len(Nixondf)

# create Kennedy/Dem dataframe for 1960
Kennedydf = df[df["Speaker_standardized"]=='John F. Kennedy']
Kennedydf # the copyright information gets attributed to Kennedy

#testing length
len(Kennedydf)

# create 1960 both candidates dataframe
debates_1960= Nixondf.append(Kennedydf)
debates_1960

#testing length
len(debates_1960)

#top 200 words
allcandidates60_freq = pd.Series(' '.join(debates_1960.Transcript).split()).value_counts()[:200]
allcandidates60_freq

#analysis of specific words from all candidates 
debates_60_nuclear = debates_1960.Transcript.str.count('NUCLEAR')
debates_60_nuclear.sum()

debates_60_atomic = debates_1960.Transcript.str.count('ATOMIC')
debates_60_atomic.sum()

debates_60_proliferation = debates_1960.Transcript.str.count('PROLIFERATION')
debates_60_proliferation.sum()

debates_60_nonproliferation = debates_1960.Transcript.str.count('NONPROLIFERATION')
debates_60_nonproliferation.sum()

debates_60_non_proliferation = debates_1960.Transcript.str.count('NON-PROLIFERATION')
debates_60_non_proliferation.sum()

debates_60_armsrace = debates_1960.Transcript.str.count('ARMS RACE')
debates_60_armsrace.sum()

debates_60_women = debates_1960.Transcript.str.count('WOMEN')
debates_60_women.sum()

debates_60_job = debates_1960.Transcript.str.count('JOB')
debates_60_job.sum()

debates_60_jobs = debates_1960.Transcript.str.count('JOBS')
debates_60_jobs.sum()

debates_60_unemployment = debates_1960.Transcript.str.count('UNEMPLOYMENT')
debates_60_unemployment.sum()

debates_60_world = debates_1960.Transcript.str.count('WORLD')
debates_60_world.sum()

debates_60_foreign = debates_1960.Transcript.str.count('FOREIGN')
debates_60_foreign.sum()

debates_60_Iraq = debates_1960.Transcript.str.count('IRAQ')
debates_60_Iraq.sum()

debates_60_Soviet= debates_1960.Transcript.str.count('SOVIET')
debates_60_Soviet.sum()

debates_60_Soviets= debates_1960.Transcript.str.count('SOVIETS')
debates_60_Soviets.sum()

debates_60_Russia= debates_1960.Transcript.str.count('RUSSIA')
debates_60_Russia.sum()

#analysis of specific word counts by candidate/party
Nixondf_nuclear = Nixondf.Transcript.str.count('NUCLEAR')
Nixondf_nuclear.sum()

GOP60_atomic = Nixondf.Transcript.str.count('ATOMIC')
GOP60_atomic.sum()

GOP60_proliferation = Nixondf.Transcript.str.count('PROLIFERATION')
GOP60_proliferation.sum()

GOP60_nonproliferation = Nixondf.Transcript.str.count('NONPROLIFERATION')
GOP60_nonproliferation.sum()

GOP60_non_proliferation = Nixondf.Transcript.str.count('NON-PROLIFERATION')
GOP60_non_proliferation.sum()

GOP60_armsrace = Nixondf.Transcript.str.count('ARMS RACE')
GOP60_armsrace.sum()

GOP60_women = Nixondf.Transcript.str.count('WOMEN')
GOP60_women.sum()

GOP60_job = Nixondf.Transcript.str.count('JOB')
GOP60_job.sum()

GOP60_jobs = Nixondf.Transcript.str.count('JOBS')
GOP60_jobs.sum()

GOP60_unemployment = Nixondf.Transcript.str.count('UNEMPLOYMENT')
GOP60_unemployment.sum()

GOP60_world = Nixondf.Transcript.str.count('WORLD')
GOP60_world.sum()

GOP60_foreign = Nixondf.Transcript.str.count('FOREIGN')
GOP60_foreign.sum()

DEM60_nuclear = Kennedydf.Transcript.str.count('NUCLEAR')
DEM60_nuclear.sum()

DEM60_atomic = Kennedydf.Transcript.str.count('ATOMIC')
DEM60_atomic.sum()

DEM60_proliferation = Kennedydf.Transcript.str.count('PROLIFERATION')
DEM60_proliferation.sum()

DEM60_nonproliferation = Kennedydf.Transcript.str.count('NONPROLIFERATION')
DEM60_nonproliferation.sum()

DEM60_non_proliferation = Kennedydf.Transcript.str.count('NON-PROLIFERATION')
DEM60_non_proliferation.sum()

DEM60_armsrace = Kennedydf.Transcript.str.count('ARMS RACE')
DEM60_armsrace.sum()

DEM60_women = Kennedydf.Transcript.str.count('WOMEN')
DEM60_women.sum()

DEM60_job = Kennedydf.Transcript.str.count('JOB')
DEM60_job.sum()

DEM60_jobs = Kennedydf.Transcript.str.count('JOBS')
DEM60_jobs.sum()

DEM60_unemployment = Kennedydf.Transcript.str.count('UNEMPLOYMENT')
DEM60_unemployment.sum()

DEM60_world = Kennedydf.Transcript.str.count('WORLD')
DEM60_world.sum()

DEM60_foreign = Kennedydf.Transcript.str.count('FOREIGN')
DEM60_foreign.sum()


#Begin 1976
#September 23, 1976 debate
Sept23_1976_debate = df[df["Debate"]=='September 23, 1976 Debate Transcript']
Sept23_1976_debate

#testing length
len(Sept23_1976_debate)

#October 6, 1976 debate
Oct6_1976_debate = df[df["Debate"]=='October 6, 1976 Debate Transcript']
Oct6_1976_debate

#testing length
len(Oct6_1976_debate)

#October 22, 1976 debate
Oct22_1976_debate = df[df["Debate"]=='October 22, 1976 Debate Transcript']
Oct22_1976_debate

#testing length
len(Oct22_1976_debate)

#full debate transcript for 1976, includes moderator and any other non-candidate speech
debates_1976 = Sept23_1976_debate.append([Oct6_1976_debate, Oct22_1976_debate])
debates_1976

#testing length
len(debates_1976)

#for Ford/GOP 1976; Ford76 is also what we have for the GOP in 1976 because we don't have the VP debate transcript

Ford76 = debates_1976[(debates_1976["Speaker_standardized"]=='Gerald Ford')]
Ford76 #row 810 attributes speech from a Mr. Kraft to Ford

#testing length
len(Ford76)

#Carter/Dem 1976 debates; Carter76 is also what we have for Dems in 1976 because we don't have the VP debate transcript
Carter76 = debates_1976[(debates_1976["Speaker_standardized"]=='Jimmy Carter')]
Carter76 # includes (barely audible) in Carter word count

#testing length
len(Carter76)

#both candidates/parties 1976 (without moderator/others)
allcandidates76 = Ford76.append([Carter76])
allcandidates76

#testing length
len(allcandidates76)

#top 200 words
allcandidates76_freq= pd.Series(' '.join(allcandidates76.Transcript).split()).value_counts()[:200]
allcandidates76_freq

#analysis of specific words in debate year by all candidates
debates_1976_nuclear = debates_1976.Transcript.str.count('NUCLEAR')
debates_1976_nuclear.sum()

debates_1976_atomic = debates_1976.Transcript.str.count('ATOMIC')
debates_1976_atomic.sum()
	
debates_1976_proliferation = debates_1976.Transcript.str.count('PROLIFERATION')
debates_1976_proliferation.sum()
	
debates_1976_nonproliferation = debates_1976.Transcript.str.count('NONPROLIFERATION')
debates_1976_nonproliferation.sum()	

debates_1976_non_proliferation = debates_1976.Transcript.str.count('NON-PROLIFERATION')
debates_1976_non_proliferation.sum()	

debates_1976_armsrace = debates_1976.Transcript.str.count('ARMS RACE')
debates_1976_armsrace.sum()
	
debates_1976_women = debates_1976.Transcript.str.count('WOMEN')
debates_1976_women.sum()
	
debates_1976_job = debates_1976.Transcript.str.count('JOB')
debates_1976_job.sum()	

debates_1976_jobs = debates_1976.Transcript.str.count('JOBS')
debates_1976_jobs.sum()
	
debates_1976_unemployment = debates_1976.Transcript.str.count('UNEMPLOYMENT')
debates_1976_unemployment.sum()	

debates_1976_world = debates_1976.Transcript.str.count('WORLD')
debates_1976_world.sum()
	
debates_1976_foreign = debates_1976.Transcript.str.count('FOREIGN')
debates_1976_foreign.sum()	

debates_1976_Iraq = debates_1976.Transcript.str.count('IRAQ')
debates_1976_Iraq.sum()
	
debates_1976_Soviet= debates_1976.Transcript.str.count('SOVIET')
debates_1976_Soviet.sum()	

debates_1976_Soviets= debates_1976.Transcript.str.count('SOVIETS')
debates_1976_Soviets.sum()
	
debates_1976_Russia= debates_1976.Transcript.str.count('RUSSIA')
debates_1976_Russia.sum()

#analysis of words by party

Ford76_nuclear = Ford76.Transcript.str.count('NUCLEAR')
Ford76_nuclear.sum()

Carter76_nuclear = Carter76.Transcript.str.count('NUCLEAR')
Carter76_nuclear.sum()

Ford76_atomic = Ford76.Transcript.str.count('ATOMIC')
Ford76_atomic.sum()

Carter76_atomic = Carter76.Transcript.str.count('ATOMIC')
Carter76_atomic.sum()

Ford76_proliferation = Ford76.Transcript.str.count('PROLIFERATION')
Ford76_proliferation.sum()

Carter76_proliferation = Carter76.Transcript.str.count('PROLIFERATION')
Carter76_proliferation.sum()

Ford76_nonproliferation = Ford76.Transcript.str.count('NONPROLIFERATION')
Ford76_nonproliferation.sum()

Carter76_nonproliferation = Carter76.Transcript.str.count('NONPROLIFERATION')
Carter76_nonproliferation.sum()

Ford76_non_proliferation = Ford76.Transcript.str.count('NON-PROLIFERATION')
Ford76_non_proliferation.sum()

Carter76_non_proliferation = Carter76.Transcript.str.count('NON-PROLIFERATION')
Carter76_non_proliferation.sum()

Ford76_armsrace = Ford76.Transcript.str.count('ARMS RACE')
Ford76_armsrace.sum()

Carter76_armsrace = Carter76.Transcript.str.count('ARMS RACE')
Carter76_armsrace.sum()

Ford76_women = Ford76.Transcript.str.count('WOMEN')
Ford76_women.sum()

Carter76_women = Carter76.Transcript.str.count('WOMEN')
Carter76_women.sum()

Ford76_job = Ford76.Transcript.str.count('JOB')
Ford76_job.sum()

Carter76_job = Carter76.Transcript.str.count('JOB')
Carter76_job.sum()

Ford76_jobs = Ford76.Transcript.str.count('JOBS')
Ford76_jobs.sum()

Carter76_jobs = Carter76.Transcript.str.count('JOBS')
Carter76_jobs.sum()

Ford76_unemployment = Ford76.Transcript.str.count('UNEMPLOYMENT')
Ford76_unemployment.sum()

Carter76_unemployment = Carter76.Transcript.str.count('UNEMPLOYMENT')
Carter76_unemployment.sum()

Ford76_world = Ford76.Transcript.str.count('WORLD')
Ford76_world.sum()

Carter76_world = Carter76.Transcript.str.count('WORLD')
Carter76_world.sum()

Ford76_foreign = Ford76.Transcript.str.count('FOREIGN')
Ford76_foreign.sum()

Carter76_foreign = Carter76.Transcript.str.count('FOREIGN')
Carter76_foreign.sum()

#Begin 1980 debates
#October 28, 1980
Oct28_1980_debate = df[df["Debate"]=='October 28, 1980 Debate Transcript']
Oct28_1980_debate

#testing length
len(Oct28_1980_debate)

#September 21, 1980 debate
Sept21_1980_debate = df[df["Debate"]=='September 21, 1980 Debate Transcript']
Sept21_1980_debate

#testing length
len(Sept21_1980_debate)

#full debate transcript 1980, includes moderators and any other speakes in addition to candidates
debates_1980 = Oct28_1980_debate.append([Sept21_1980_debate])
debates_1980

#testing length
len(debates_1980)

#for Carter/Dems 1980; no VP debate in 1980
Carter80 = debates_1980[(debates_1980["Speaker_standardized"]=='Jimmy Carter')]
Carter80

#testing length
len(Carter80)

#for Reagan/GOP in 1980, no VP debate
Reagan80 = debates_1980[(debates_1980["Speaker_standardized"]=='Ronald Reagan')]
Reagan80

#testing length
len(Reagan80)

#for Anderson/Ind. 1980, no VP debate in 1980
Anderson80 = debates_1980[(debates_1980["Speaker_standardized"]=='Ronald Reagan')]
Anderson80

#testing length
len(Anderson80)

#all candidates 1980
allcandidates80 = Anderson80.append([Reagan80, Carter80]) 
allcandidates80

#testing length
len(allcandidates80)

#Begin 1984 debates
#October 7, 1984
Oct7_1984_debate = df[df["Debate"]=='October 7, 1984 Debate Transcript']
Oct7_1984_debate

#testing length
len(Oct7_1984_debate)

#October 21, 1984
Oct21_1984_debate = df[df["Debate"]=='October 21, 1984 Debate Transcript']
Oct21_1984_debate 

#testing length
len(Oct21_1984_debate) 

#October 11, 1984 (VP debate)
Oct11_1984_debate = df[df["Debate"]=='October 11, 1984 Debate Transcript']
Oct11_1984_debate 

#testing length
len(Oct11_1984_debate) 

#all debate transcripts 1984, includes moderator and any other speakers
debates_1984 = Oct11_1984_debate.append([Oct21_1984_debate, Oct7_1984_debate])
debates_1984

#testing length
len(debates_1984)

#Democratic party 1984 (Mondale)
Mondale84 = debates_1984[(debates_1984["Speaker_standardized"]=='Walter Mondale')]
Mondale84

#testing length
len(Mondale84)

#Dem party 1984 (Ferraro)
Ferraro84 = debates_1984[(debates_1984["Speaker_standardized"]=='Geraldine Ferraro')]
Ferraro84

#testing length
len(Ferraro84)

#all Dem transcript 1984
Dem1984 = Mondale84.append([Ferraro84])
Dem1984

#testing length
len(Dem1984)

#Republican party 1984 (Reagan)
Reagan84 = debates_1984[(debates_1984["Speaker_standardized"]=='Ronald Reagan')]
Reagan84

#testing length
len(Reagan84)

#Republican party 1984 (Bush)
Bush84 = debates_1984[(debates_1984["Speaker_standardized"]=='George W. Bush')]
Bush84

#testing length
len(Bush84)

#full Republican party 1984
GOP1984 = Reagan84.append([Bush84])
GOP1984

#testing length
len(GOP1984)

allcandidates84 = GOP1984.append([Dem1984]) 
allcandidates84

#testing length
len(allcandidates84)

#top 200 words
allcandidates84_freq= pd.Series(' '.join(allcandidates84.Transcript).split()).value_counts()[:200]
allcandidates84_freq

#analysis of particular words used by candidates overall in 1984
debates_84_nuclear = debates_1984.Transcript.str.count('NUCLEAR')
debates_84_nuclear.sum()	

debates_84_atomic = debates_1984.Transcript.str.count('ATOMIC')
debates_84_atomic.sum()	

debates_84_proliferation = debates_1984.Transcript.str.count('PROLIFERATION')
debates_84_proliferation.sum()
	
debates_84_nonproliferation = debates_1984.Transcript.str.count('NONPROLIFERATION')
debates_84_nonproliferation.sum()
	
debates_84_non_proliferation = debates_1984.Transcript.str.count('NON-PROLIFERATION')
debates_84_non_proliferation.sum()
	
debates_84_armsrace = debates_1984.Transcript.str.count('ARMS RACE')
debates_84_armsrace.sum()
	
debates_84_women = debates_1984.Transcript.str.count('WOMEN')
debates_84_women.sum()	

debates_84_job = debates_1984.Transcript.str.count('JOB')
debates_84_job.sum()	

debates_84_jobs = debates_1984.Transcript.str.count('JOBS')
debates_84_jobs.sum()	

debates_84_unemployment = debates_1984.Transcript.str.count('UNEMPLOYMENT')
debates_84_unemployment.sum()	

debates_84_world = debates_1984.Transcript.str.count('WORLD')
debates_84_world.sum()	

debates_84_foreign = debates_1984.Transcript.str.count('FOREIGN')
debates_84_foreign.sum()	

debates_84_Iraq = debates_1984.Transcript.str.count('IRAQ')
debates_84_Iraq.sum()	

debates_84_Soviet= debates_1984.Transcript.str.count('SOVIET')
debates_84_Soviet.sum()	

debates_84_Soviets= debates_1984.Transcript.str.count('SOVIETS')
debates_84_Soviets.sum()	

debates_84_Russia= debates_1984.Transcript.str.count('RUSSIA')
debates_84_Russia.sum()

#analysis of words by party

GOP84_nuclear = GOP84.Transcript.str.count('NUCLEAR')
GOP84_nuclear.sum()

DEM1984_nuclear = DEM1984.Transcript.str.count('NUCLEAR')
DEM1984_nuclear.sum()

GOP84_atomic = GOP84.Transcript.str.count('ATOMIC')
GOP84_atomic.sum()

DEM1984_atomic = DEM1984.Transcript.str.count('ATOMIC')
DEM1984_atomic.sum()

GOP84_proliferation = GOP84.Transcript.str.count('PROLIFERATION')
GOP84_proliferation.sum()

DEM1984_proliferation = DEM1984.Transcript.str.count('PROLIFERATION')
DEM1984_proliferation.sum()

GOP1984_nonproliferation = GOP1984.Transcript.str.count('NONPROLIFERATION')
GOP1984_nonproliferation.sum()

DEM1984_nonproliferation = DEM1984.Transcript.str.count('NONPROLIFERATION')
DEM1984_nonproliferation.sum()

GOP84_non_proliferation = GOP84.Transcript.str.count('NON-PROLIFERATION')
GOP84_non_proliferation.sum()

DEM1984_non_proliferation = DEM1984.Transcript.str.count('NON-PROLIFERATION')
DEM1984_non_proliferation.sum()

GOP84_armsrace = GOP84.Transcript.str.count('ARMS RACE')
GOP84_armsrace.sum()

DEM1984_armsrace = DEM1984.Transcript.str.count('ARMS RACE')
DEM1984_armsrace.sum()

GOP84_women = GOP84.Transcript.str.count('WOMEN')
GOP84_women.sum()

DEM1984_women = DEM8194.Transcript.str.count('WOMEN')
DEM1984_women.sum()

GOP84_job = GOP84.Transcript.str.count('JOB')
GOP84_job.sum()

DEM1984_job = DEM1984.Transcript.str.count('JOB')
DEM1984_job.sum()

GOP84_jobs = GOP84.Transcript.str.count('JOBS')
GOP84_jobs.sum()

DEM1984_jobs = DEM1984.Transcript.str.count('JOBS')
DEM1984_jobs.sum()

GOP84_unemployment = GOP84.Transcript.str.count('UNEMPLOYMENT')
GOP84_unemployment.sum()

DEM1984_unemployment = DEM1984.Transcript.str.count('UNEMPLOYMENT')
DEM1984_unemployment.sum()

GOP84_world = GOP84.Transcript.str.count('WORLD')
GOP84_world.sum()

DEM1984_world = DEM1984.Transcript.str.count('WORLD')
DEM1984_world.sum()

GOP84_foreign = GOP84.Transcript.str.count('FOREIGN')
GOP84_foreign.sum()

DEM1984_foreign = DEM1984.Transcript.str.count('FOREIGN')
DEM1984_foreign.sum()



#Begin 1988 debates
#October 5, 1988
Oct5_1988_debate = df[df["Debate"]=='October 5, 1988 Debate Transcripts']
Oct5_1988_debate

#testing length
len(Oct5_1988_debate)

#September 25, 1988 debate
Sept25_1988_debate = df[df["Debate"]=='September 25, 1988 Debate Transcript']
Sept25_1988_debate

#testing length
len(Sept25_1988_debate)

#October 13, 1988
Oct13_1988_debate = df[df["Debate"]=='October 13, 1988 Debate Transcript']
Oct13_1988_debate

#testing length
len(Oct13_1988_debate)

#full debate transcript 1988, includes moderators and any other speakes in addition to candidates
debates_1988 = Oct13_1988_debate.append([Sept25_1988_debate, Oct5_1988_debate])
debates_1988

#testing length
len(debates_1988)

#Bush 1988
Bush88 = debates_1988[(debates_1988["Speaker_standardized"]=='George W. Bush')]
Bush88

#testing length
len(Bush88)

#Quayle 1988
Quayle88 = debates_1988[(debates_1988["Speaker_standardized"]=='Dan Quayle')]
Quayle88

#testing length
len(Quayle88)

#Dukakis 1988
Dukakis88 = debates_1988[(debates_1988["Speaker_standardized"]=='Michael Dukakis')]
Dukakis88

#testing length
len(Dukakis88)

#Bentsen 1988
Bentsen88 = debates_1988[(debates_1988["Speaker_standardized"]=='Lloyd Bentsen')]
Bentsen88

#testing length
len(Bentsen88)

#GOP 1988
GOP1988 = Bush88.append([Quayle88]) 
GOP1988

#testing length
len(GOP1988)

#Dem 1988
Dem1988 = Dukakis88.append([Bentsen88]) 
Dem1988

#testing length
len(Dem1988)

#allcandidates 1988
allcandidates88 = GOP1988.append([Dem1988]) 
allcandidates88

#testing length
len(allcandidates88)

#Begin 1992
#Oct 15, 1992 second half
Oct15_1992_2ndhalf_debate = df[df["Debate"]=='October 15, 1992 Second Half Debate Transcript']
Oct15_1992_2ndhalf_debate

#testing length
len(Oct15_1992_2ndhalf_debate)

#Oct 15 first half, 1992 debate
Oct15_1992_1sthalf_debate = df[df["Debate"]=='October 15, 1992 First Half Debate Transcript']
Oct15_1992_1sthalf_debate

#testing length
len(Oct15_1992_1sthalf_debate)

#October 19, 1992
Oct19_1992_debate = df[df["Debate"]=='October 19, 1992 Debate Transcript']
Oct19_1992_debate

#testing length
len(Oct19_1992_debate)

#October 11, 1992 first half
Oct11_1992_1sthalf_debate = df[df["Debate"]=='October 11, 1992 First Half Debate Transcript']
Oct11_1992_1sthalf_debate

#testing length
len(Oct11_1992_1sthalf_debate)

#October 11, 1992 second half 
Oct11_1992_2ndhalf_debate = df[df["Debate"]=='October 11, 1992 Second Half Debate Transcript']
Oct11_1992_2ndhalf_debate

#testing length
len(Oct11_1992_2ndhalf_debate)

#October 13, 1992
Oct13_1992_debate = df[df["Debate"]=='October 13, 1992 Debate Transcript']
Oct13_1992_debate

#testing length
len(Oct13_1992_debate)

#full debate transcript 1992, includes moderators and any other speakes in addition to candidates
debates_1992 = Oct13_1992_debate.append([Oct15_1992_1sthalf_debate, Oct15_1992_2ndhalf_debate, Oct11_1992_1sthalf_debate, Oct11_1992_2ndhalf_debate, Oct19_1992_debate])
debates_1992

#testing length
len(debates_1992)

#Bush1992
Bush92 = debates_1992[(debates_1992["Speaker_standardized"]=='George W. Bush')]
Bush92

#testing length
len(Bush92)

#Quayle1992
Quayle92 = debates_1992[(debates_1992["Speaker_standardized"]=='Dan Quayle')]
Quayle92

#testing length
len(Quayle92)

#Clinton1992
Clinton92 = debates_1992[(debates_1992["Speaker_standardized"]=='Bill Clinton')]
Clinton92

#testing length
len(Clinton92)

#Gore1992
Gore92 = debates_1992[(debates_1992["Speaker_standardized"]=='Al Gore')]
Gore92

#testing length
len(Gore92)

#Perot1992
Perot92 = debates_1992[(debates_1992["Speaker_standardized"]=='Ross Perot')]
Perot92

#testing length
len(Perot92)

#Stockdale1992
Stockdale92 = debates_1992[(debates_1992["Speaker_standardized"]=='Adm. James Stockdale')]
Stockdale92

#testing length
len(Stockdale92)

#GOP1992
GOP1992=Bush92.append([Quayle92])
GOP1992

#testing length
len(GOP1992)

#Dem1992

Dem1992=Clinton92.append([Gore92])
Dem1992

#testing length
len(Dem1992)

#Ind1992
Ind1992=Perot92.append([Stockdale92])
Ind1992

#testing length
len(Ind1992)

#all candidates 1992
allcandidates92 = GOP1992.append([Dem1992, Ind1992]) 
allcandidates92

#testing length
len(allcandidates92)

#begin 1996 debates
#Oct 6, 1996
Oct6_1996_debate = df[df["Debate"]=='October 6, 1996 Debate Transcript']
Oct6_1996_debate

#testing length
len(Oct6_1996_debate)

#Oct 16, 1996
Oct16_1996_debate = df[df["Debate"]=='October 16, 1996 Debate Transcript']
Oct16_1996_debate

#testing length
len(Oct16_1996_debate)

#Oct 9, 1996
Oct9_1996_debate = df[df["Debate"]=='October 9, 1996 Debate Transcript']
Oct9_1996_debate

#testing length
len(Oct9_1996_debate)

#full debate transcript 1996, includes moderators and any other speakes in addition to candidates
debates_1996 = Oct9_1996_debate.append([Oct6_1996_debate, Oct16_1996_debate])
debates_1996

#testing length
len(debates_1996)

#Clinton96
Clinton96 = debates_1996[(debates_1996["Speaker_standardized"]=='Bill Clinton')]
Clinton96

#testing length
len(Clinton96)

#Gore96
Gore96 = debates_1996[(debates_1996["Speaker_standardized"]=='Al Gore')]
Gore96

#testing length
len(Gore96)

#Dole96
Dole96 = debates_1996[(debates_1996["Speaker_standardized"]=='Bob Dole')]
Dole96

#testing length
len(Dole96)

#Kemp96
Kemp96 = debates_1996[(debates_1996["Speaker_standardized"]=='Jack Kemp')]
Kemp96

#testing length
len(Kemp96)

#GOP1996
GOP1996=Dole96.append([Kemp96])
GOP1996

#testing length
len(GOP1996)

#Dem1996
Dem1996=Clinton96.append([Gore96])
Dem1996

#testing length
len(Dem1996)

#all candidates 1996
allcandidates96 = GOP1996.append([Dem1996]) 
allcandidates96

#testing length
len(allcandidates06)

#top 200 words
allcandidates96_freq= pd.Series(' '.join(allcandidates96.Transcript).split()).value_counts()[:200]
allcandidates96_freq

#analysis of particular words for candidates overall in 1996	
debates_1996_nuclear = debates_1996.Transcript.str.count('NUCLEAR')
debates_1996_nuclear.sum()	

debates_1996_atomic = debates_1996.Transcript.str.count('ATOMIC')
debates_1996_atomic.sum()
	

debates_1996_proliferation = debates_1996.Transcript.str.count('PROLIFERATION')
debates_1996_proliferation.sum()	

debates_1996_nonproliferation = debates_1996.Transcript.str.count('NONPROLIFERATION')
debates_1996_nonproliferation.sum()	

debates_1996_non_proliferation = debates_1996.Transcript.str.count('NON-PROLIFERATION')
debates_1996_non_proliferation.sum()	

debates_1996_armsrace = debates_1996.Transcript.str.count('ARMS RACE')
debates_1996_armsrace.sum()	

debates_1996_women = debates_1996.Transcript.str.count('WOMEN')
debates_1996_women.sum()	

debates_1996_job = debates_1996.Transcript.str.count('JOB')
debates_1996_job.sum()	

debates_1996_jobs = debates_1996.Transcript.str.count('JOBS')
debates_1996_jobs.sum()
	
debates_1996_unemployment = debates_1996.Transcript.str.count('UNEMPLOYMENT')
debates_1996_unemployment.sum()	

debates_1996_world = debates_1996.Transcript.str.count('WORLD')
debates_1996_world.sum()

debates_1996_foreign = debates_1996.Transcript.str.count('FOREIGN')
debates_1996_foreign.sum()	

debates_1996_Iraq = debates_1996.Transcript.str.count('IRAQ')
debates_1996_Iraq.sum()	

debates_1996_Soviet= debates_1996.Transcript.str.count('SOVIET')
debates_1996_Soviet.sum()	

debates_1996_Soviets= debates_1996.Transcript.str.count('SOVIETS')
debates_1996_Soviets.sum()	

debates_1996_Russia= debates_1996.Transcript.str.count('RUSSIA')
debates_1996_Russia.sum()

#analysis of words by party

GOP1996_nuclear = GOP1996.Transcript.str.count('NUCLEAR')
GOP1996_nuclear.sum()

DEM1996_nuclear = DEM1996.Transcript.str.count('NUCLEAR')
DEM1996_nuclear.sum()

GOP1996_atomic = GOP1996.Transcript.str.count('ATOMIC')
GOP1996_atomic.sum()

DEM1996_atomic = DEM1996.Transcript.str.count('ATOMIC')
DEM1996_atomic.sum()

GOP1996_proliferation = GOP1996.Transcript.str.count('PROLIFERATION')
GOP1996_proliferation.sum()

DEM1996_proliferation = DEM1996.Transcript.str.count('PROLIFERATION')
DEM1996_proliferation.sum()

GOP1996_nonproliferation = GOP1996.Transcript.str.count('NONPROLIFERATION')
GOP1996_nonproliferation.sum()

DEM1996_nonproliferation = DEM1996.Transcript.str.count('NONPROLIFERATION')
DEM1996_nonproliferation.sum()

GOP1996_non_proliferation = GOP1996.Transcript.str.count('NON-PROLIFERATION')
GOP1996_non_proliferation.sum()

DEM1996_non_proliferation = DEM1996.Transcript.str.count('NON-PROLIFERATION')
DEM1996_non_proliferation.sum()

GOP1996_armsrace = GOP1996.Transcript.str.count('ARMS RACE')
GO19P96_armsrace.sum()

DEM1996_armsrace = DEM1996.Transcript.str.count('ARMS RACE')
DEM1996_armsrace.sum()

GOP1996_women = GOP1996.Transcript.str.count('WOMEN')
GOP1996_women.sum()

DEM1996_women = DEM1996.Transcript.str.count('WOMEN')
DEM1996_women.sum()

GOP1996_job = GOP1996.Transcript.str.count('JOB')
GOP1996_job.sum()

DEM1996_job = DEM1996.Transcript.str.count('JOB')
DEM1996_job.sum()

GOP1996_jobs = GOP1996.Transcript.str.count('JOBS')
GOP1996_jobs.sum()

DEM1996_jobs = DEM1996.Transcript.str.count('JOBS')
DEM1996_jobs.sum()

GOP1996_unemployment = GOP1996.Transcript.str.count('UNEMPLOYMENT')
GOP1996_unemployment.sum()

DEM1996_unemployment = DEM1996.Transcript.str.count('UNEMPLOYMENT')
DEM1996_unemployment.sum()

GOP1996_world = GOP1996.Transcript.str.count('WORLD')
GOP1996_world.sum()

DEM1996_world = DEM1996.Transcript.str.count('WORLD')
DEM1996_world.sum()

GOP1996_foreign = GOP1996.Transcript.str.count('FOREIGN')
GOP1996_foreign.sum()

DEM1996_foreign = DEM1996.Transcript.str.count('FOREIGN')
DEM1996_foreign.sum()


#Begin 2000
#October 5, 2000
Oct5_2000_debate = df[df["Debate"]=='October 5, 2000 Debate Transcript']
Oct5_2000_debate

#testing length
len(Oct5_2000_debate)

#October 17, 2000
Oct17_2000_debate = df[df["Debate"]=='October 17, 2000 Debate Transcript']
Oct17_2000_debate

#testing length
len(Oct17_2000_debate)

#October 3, 2000
Oct3_2000_debate = df[df["Debate"]=='October 3, 2000 Transcript']
Oct3_2000_debate

#testing length
len(Oct3_2000_debate)

#October 11, 2000
Oct11_2000_debate = df[df["Debate"]=='October 11, 2000 Debate Transcript']
Oct11_2000_debate

#testing length
len(Oct11_2000_debate)

#full debate transcript 2000, includes moderators and any other speakes in addition to candidates
debates_2000 = Oct5_2000_debate.append([Oct17_2000_debate, Oct3_2000_debate, Oct11_2000_debate])
debates_2000

#testing length
len(debates_2000)

#Gore2000
Gore00 = debates_2000[(debates_2000["Speaker_standardized"]=='Al Gore')]
Gore00

#testing length
len(Gore00)

#Lieberman2000
Lieberman00 = debates_2000[(debates_2000["Speaker_standardized"]=='Joseph Lieberman')]
Lieberman00

#testing length
len(Lieberman00)

#Bush2000
Bush00 = debates_2000[(debates_2000["Speaker_standardized"]=='George W. Bush')]
Bush00

#testing length
len(Bush00)

#Cheney2000
Cheney00 = debates_2000[(debates_2000["Speaker_standardized"]=='Dick Cheney')]
Cheney00

#testing length
len(Cheney00)

#GOP2000
GOP2000=Bush00.append([Cheney00])
GOP2000

#testing length
len(GOP2000)

#Dem2000
Dem2000=Gore00.append([Lieberman00])
Dem2000

#testing length
len(Dem2000)

#all candidates 2000
allcandidates00 = GOP2000.append([Dem2000]) 
allcandidates00

#testing length
len(allcandidates00)

#top 200 words
allcandidates00_freq= pd.Series(' '.join(allcandidates00.Transcript).split()).value_counts()[:200]
allcandidates00_freq

#analysis of particular words used by all candidates	
debates_2000_nuclear = debates_2000.Transcript.str.count('NUCLEAR')
debates_2000_nuclear.sum()	

debates_2000_atomic = debates_2000.Transcript.str.count('ATOMIC')
debates_2000_atomic.sum()	

debates_2000_proliferation = debates_2000.Transcript.str.count('PROLIFERATION')
debates_2000_proliferation.sum()	

debates_2000_nonproliferation = debates_2000.Transcript.str.count('NONPROLIFERATION')
debates_2000_nonproliferation.sum()	

debates_2000_non_proliferation = debates_2000.Transcript.str.count('NON-PROLIFERATION')
debates_2000_non_proliferation.sum()	

debates_2000_armsrace = debates_2000.Transcript.str.count('ARMS RACE')
debates_2000_armsrace.sum()	

debates_2000_women = debates_2000.Transcript.str.count('WOMEN')
debates_2000_women.sum()	

debates_2000_job = debates_2000.Transcript.str.count('JOB')
debates_2000_job.sum()	

debates_2000_jobs = debates_2000.Transcript.str.count('JOBS')
debates_2000_jobs.sum()	

debates_2000_unemployment = debates_2000.Transcript.str.count('UNEMPLOYMENT')
debates_2000_unemployment.sum()
	
debates_2000_world = debates_2000.Transcript.str.count('WORLD')
debates_2000_world.sum()	

debates_2000_foreign = debates_2000.Transcript.str.count('FOREIGN')
debates_2000_foreign.sum()
	
debates_2000_Iraq = debates_2000.Transcript.str.count('IRAQ')
debates_2000_Iraq.sum()
	
debates_2000_Soviet= debates_2000.Transcript.str.count('SOVIET')
debates_2000_Soviet.sum()	

debates_2000_Soviets= debates_2000.Transcript.str.count('SOVIETS')
debates_2000_Soviets.sum()
	
debates_2000_Russia= debates_2000.Transcript.str.count('RUSSIA')
debates_2000_Russia.sum()

#analysis of words by party

GOP2000_nuclear = GOP2000.Transcript.str.count('NUCLEAR')
GOP2000_nuclear.sum()

DEM2000_nuclear = DEM2000.Transcript.str.count('NUCLEAR')
DEM2000_nuclear.sum()

GOP2000_atomic = GOP2000.Transcript.str.count('ATOMIC')
GOP2000_atomic.sum()

DEM2000_atomic = DEM2000.Transcript.str.count('ATOMIC')
DEM2000_atomic.sum()

GOP2000_proliferation = GOP2000.Transcript.str.count('PROLIFERATION')
GOP2000_proliferation.sum()

DEM2000_proliferation = DEM2000.Transcript.str.count('PROLIFERATION')
DEM2000_proliferation.sum()

GOP2000_nonproliferation = GOP2000.Transcript.str.count('NONPROLIFERATION')
GOP2000_nonproliferation.sum()

DEM2000_nonproliferation = DEM2000.Transcript.str.count('NONPROLIFERATION')
DEM2000_nonproliferation.sum()

GOP2000_non_proliferation = GOP2000.Transcript.str.count('NON-PROLIFERATION')
GOP2000_non_proliferation.sum()

DEM2000_non_proliferation = DEM2000.Transcript.str.count('NON-PROLIFERATION')
DEM2000_non_proliferation.sum()

GOP2000_armsrace = GOP2000.Transcript.str.count('ARMS RACE')
GOP2000_armsrace.sum()

DEM00_armsrace = DEM00.Transcript.str.count('ARMS RACE')
DEM00_armsrace.sum()

GOP00_women = GOP00.Transcript.str.count('WOMEN')
GOP00_women.sum()

DEM2000_women = DEM2000.Transcript.str.count('WOMEN')
DEM2000_women.sum()

GOP2000_job = GOP2000.Transcript.str.count('JOB')
GOP2000_job.sum()

DEM2000_job = DEM2000.Transcript.str.count('JOB')
DEM2000_job.sum()

GOP2000_jobs = GOP2000.Transcript.str.count('JOBS')
GOP2000_jobs.sum()

DEM2000_jobs = DEM2000.Transcript.str.count('JOBS')
DEM2000_jobs.sum()

GOP2000_unemployment = GOP2000.Transcript.str.count('UNEMPLOYMENT')
GOP2000_unemployment.sum()

DEM2000_unemployment = DEM2000.Transcript.str.count('UNEMPLOYMENT')
DEM2000_unemployment.sum()

GOP2000_world = GOP2000.Transcript.str.count('WORLD')
GOP2000_world.sum()

DEM2000_world = DEM2000.Transcript.str.count('WORLD')
DEM2000_world.sum()

GOP2000_foreign = GOP2000.Transcript.str.count('FOREIGN')
GOP2000_foreign.sum()

DEM2000_foreign = DEM2000.Transcript.str.count('FOREIGN')
DEM2000_foreign.sum()

#Begin 2004
#October 8, 2004
Oct8_2004_debate = df[df["Debate"]=='October 8, 2004 Debate Transcript']
Oct8_2004_debate

#testing length
len(Oct8_2004_debate)

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

#analysis of words for all canidates in 2004
debates_2004_nuclear = debates_2004.Transcript.str.count('NUCLEAR')
debates_2004_nuclear.sum()	

debates_2004_atomic = debates_2004.Transcript.str.count('ATOMIC')
debates_2004_atomic.sum()	

debates_2004_proliferation = debates_2004.Transcript.str.count('PROLIFERATION')
debates_2004_proliferation.sum()	

debates_2004_nonproliferation = debates_2004.Transcript.str.count('NONPROLIFERATION')
debates_2004_nonproliferation.sum()	

debates_2004_non_proliferation = debates_2004.Transcript.str.count('NON-PROLIFERATION')
debates_2004_non_proliferation.sum()	

debates_2004_armsrace = debates_2004.Transcript.str.count('ARMS RACE')
debates_2004_armsrace.sum()	

debates_2004_women = debates_2004.Transcript.str.count('WOMEN')
debates_2004_women.sum()
	
debates_2004_job = debates_2004.Transcript.str.count('JOB')
debates_2004_job.sum()	

debates_2004_jobs = debates_2004.Transcript.str.count('JOBS')
debates_2004_jobs.sum()	

debates_2004_unemployment = debates_2004.Transcript.str.count('UNEMPLOYMENT')
debates_2004_unemployment.sum()	

debates_2004_world = debates_2004.Transcript.str.count('WORLD')
debates_2004_world.sum()	

debates_2004_foreign = debates_2004.Transcript.str.count('FOREIGN')
debates_2004_foreign.sum()	

debates_2004_Iraq = debates_2004.Transcript.str.count('IRAQ')
debates_2004_Iraq.sum()	

debates_2004_Soviet= debates_2004.Transcript.str.count('SOVIET')
debates_2004_Soviet.sum()	

debates_2004_Soviets= debates_2004.Transcript.str.count('SOVIETS')
debates_2004_Soviets.sum()	

debates_2004_Russia= debates_2004.Transcript.str.count('RUSSIA')
debates_2004_Russia.sum()

#analysis of words by party

GOP04_nuclear = GOP04.Transcript.str.count('NUCLEAR')
GOP04_nuclear.sum()

DEM04_nuclear = DEM04.Transcript.str.count('NUCLEAR')
DEM04_nuclear.sum()

GOP04_atomic = GOP04.Transcript.str.count('ATOMIC')
GOP04_atomic.sum()

DEM04_atomic = DEM04.Transcript.str.count('ATOMIC')
DEM04_atomic.sum()

GOP04_proliferation = GOP04.Transcript.str.count('PROLIFERATION')
GOP04_proliferation.sum()

DEM04_proliferation = DEM04.Transcript.str.count('PROLIFERATION')
DEM04_proliferation.sum()

GOP04_nonproliferation = GOP04.Transcript.str.count('NONPROLIFERATION')
GOP04_nonproliferation.sum()

DEM04_nonproliferation = DEM04.Transcript.str.count('NONPROLIFERATION')
DEM04_nonproliferation.sum()

GOP04_non_proliferation = GOP04.Transcript.str.count('NON-PROLIFERATION')
GOP04_non_proliferation.sum()

DEM04_non_proliferation = DEM04.Transcript.str.count('NON-PROLIFERATION')
DEM04_non_proliferation.sum()

GOP04_armsrace = GOP04.Transcript.str.count('ARMS RACE')
GOP04_armsrace.sum()

DEM04_armsrace = DEM04.Transcript.str.count('ARMS RACE')
DEM04_armsrace.sum()

GOP04_women = GOP04.Transcript.str.count('WOMEN')
GOP04_women.sum()

DEM04_women = DEM04.Transcript.str.count('WOMEN')
DEM04_women.sum()

GOP04_job = GOP04.Transcript.str.count('JOB')
GOP04_job.sum()

DEM04_job = DEM04.Transcript.str.count('JOB')
DEM04_job.sum()

GOP04_jobs = GOP04.Transcript.str.count('JOBS')
GOP04_jobs.sum()

DEM04_jobs = DEM04.Transcript.str.count('JOBS')
DEM04_jobs.sum()

GOP04_unemployment = GOP04.Transcript.str.count('UNEMPLOYMENT')
GOP04_unemployment.sum()

DEM04_unemployment = DEM04.Transcript.str.count('UNEMPLOYMENT')
DEM04_unemployment.sum()

GOP04_world = GOP04.Transcript.str.count('WORLD')
GOP04_world.sum()

DEM04_world = DEM04.Transcript.str.count('WORLD')
DEM04_world.sum()

GOP04_foreign = GOP04.Transcript.str.count('FOREIGN')
GOP04_foreign.sum()

DEM04_foreign = DEM04.Transcript.str.count('FOREIGN')
DEM04_foreign.sum()

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
	
#analysis of particular words used by candidates overall in 2008
debates_2008_nuclear = debates_2008.Transcript.str.count('NUCLEAR')
debates_2008_nuclear.sum()	

debates_2008_atomic = debates_2008.Transcript.str.count('ATOMIC')
debates_2008_atomic.sum()	

debates_2008_proliferation = debates_2008.Transcript.str.count('PROLIFERATION')
debates_2008_proliferation.sum()	

debates_2008_nonproliferation = debates_2008.Transcript.str.count('NONPROLIFERATION')
debates_2008_nonproliferation.sum()	

debates_2008_non_proliferation = debates_2008.Transcript.str.count('NON-PROLIFERATION')
debates_2008_non_proliferation.sum()	

debates_2008_armsrace = debates_2008.Transcript.str.count('ARMS RACE')
debates_2008_armsrace.sum()	

debates_2008_women = debates_2008.Transcript.str.count('WOMEN')
debates_2008_women.sum()	

debates_2008_job = debates_2008.Transcript.str.count('JOB')
debates_2008_job.sum()	

debates_2008_jobs = debates_2008.Transcript.str.count('JOBS')
debates_2008_jobs.sum()
	
debates_2008_unemployment = debates_2008.Transcript.str.count('UNEMPLOYMENT')
debates_2008_unemployment.sum()

debates_2008_world = debates_2008.Transcript.str.count('WORLD')
debates_2008_world.sum()
	
debates_2008_foreign = debates_2008.Transcript.str.count('FOREIGN')
debates_2008_foreign.sum()
	
debates_2008_Iraq = debates_2008.Transcript.str.count('IRAQ')
debates_2008_Iraq.sum()
	
debates_2008_Soviet= debates_2008.Transcript.str.count('SOVIET')
debates_2008_Soviet.sum()
	
debates_2008_Soviets= debates_2008.Transcript.str.count('SOVIETS')
debates_2008_Soviets.sum()
	
debates_2008_Russia= debates_2008.Transcript.str.count('RUSSIA')
debates_2008_Russia.sum()

#analysis of words by party

GOP08_nuclear = GOP08.Transcript.str.count('NUCLEAR')
GOP08_nuclear.sum()

DEM08_nuclear = DEM08.Transcript.str.count('NUCLEAR')
DEM08_nuclear.sum()

GOP08_atomic = GOP08.Transcript.str.count('ATOMIC')
GOP08_atomic.sum()

DEM08_atomic = DEM08.Transcript.str.count('ATOMIC')
DEM08_atomic.sum()

GOP08_proliferation = GOP08.Transcript.str.count('PROLIFERATION')
GOP08_proliferation.sum()

DEM08_proliferation = DEM08.Transcript.str.count('PROLIFERATION')
DEM08_proliferation.sum()

GOP08_nonproliferation = GOP08.Transcript.str.count('NONPROLIFERATION')
GOP08_nonproliferation.sum()

DEM08_nonproliferation = DEM08.Transcript.str.count('NONPROLIFERATION')
DEM08_nonproliferation.sum()

GOP08_non_proliferation = GOP08.Transcript.str.count('NON-PROLIFERATION')
GOP08_non_proliferation.sum()

DEM08_non_proliferation = DEM08.Transcript.str.count('NON-PROLIFERATION')
DEM08_non_proliferation.sum()

GOP08_armsrace = GOP08.Transcript.str.count('ARMS RACE')
GOP08_armsrace.sum()

DEM08_armsrace = DEM08.Transcript.str.count('ARMS RACE')
DEM08_armsrace.sum()

GOP08_women = GOP08.Transcript.str.count('WOMEN')
GOP08_women.sum()

DEM08_women = DEM08.Transcript.str.count('WOMEN')
DEM08_women.sum()

GOP08_job = GOP08.Transcript.str.count('JOB')
GOP08_job.sum()

DEM08_job = DEM08.Transcript.str.count('JOB')
DEM08_job.sum()

GOP08_jobs = GOP08.Transcript.str.count('JOBS')
GOP08_jobs.sum()

DEM08_jobs = DEM08.Transcript.str.count('JOBS')
DEM08_jobs.sum()

GOP08_unemployment = GOP08.Transcript.str.count('UNEMPLOYMENT')
GOP08_unemployment.sum()

DEM08_unemployment = DEM08.Transcript.str.count('UNEMPLOYMENT')
DEM08_unemployment.sum()

GOP08_world = GOP08.Transcript.str.count('WORLD')
GOP08_world.sum()

DEM08_world = DEM08.Transcript.str.count('WORLD')
DEM08_world.sum()

GOP08_foreign = GOP08.Transcript.str.count('FOREIGN')
GOP08_foreign.sum()

DEM08_foreign = DEM08.Transcript.str.count('FOREIGN')
DEM08_foreign.sum()

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

#analysis of particular words by candidates overall in 2012	
debates_2012_nuclear = debates_2012.Transcript.str.count('NUCLEAR')
debates_2012_nuclear.sum()	

debates_2012_atomic = debates_2012.Transcript.str.count('ATOMIC')
debates_2012_atomic.sum()	

debates_2012_proliferation = debates_2012.Transcript.str.count('PROLIFERATION')
debates_2012_proliferation.sum()
	
debates_2012_nonproliferation = debates_2012.Transcript.str.count('NONPROLIFERATION')
debates_2012_nonproliferation.sum()
	
debates_2012_non_proliferation = debates_2012.Transcript.str.count('NON-PROLIFERATION')
debates_2012_non_proliferation.sum()
	
debates_2012_armsrace = debates_2012.Transcript.str.count('ARMS RACE')
debates_2012_armsrace.sum()	

debates_2012_women = debates_2012.Transcript.str.count('WOMEN')
debates_2012_women.sum()
	
debates_2012_job = debates_2012.Transcript.str.count('JOB')
debates_2012_job.sum()
	
debates_2012_jobs = debates_2012.Transcript.str.count('JOBS')
debates_2012_jobs.sum()	

debates_2012_unemployment = debates_2012.Transcript.str.count('UNEMPLOYMENT')
debates_2012_unemployment.sum()	

debates_2012_world = debates_2012.Transcript.str.count('WORLD')
debates_2012_world.sum()
	
debates_2012_foreign = debates_2012.Transcript.str.count('FOREIGN')
debates_2012_foreign.sum()
	
debates_2012_Iraq = debates_2012.Transcript.str.count('IRAQ')
debates_2012_Iraq.sum()
	
debates_2012_Soviet= debates_2012.Transcript.str.count('SOVIET')
debates_2012_Soviet.sum()

debates_2012_Soviets= debates_2012.Transcript.str.count('SOVIETS')
debates_2012_Soviets.sum()

debates_2012_Russia= debates_2012.Transcript.str.count('RUSSIA')
debates_2012_Russia.sum()

#analysis of words by party

GOP12_nuclear = GOP12.Transcript.str.count('NUCLEAR')
GOP12_nuclear.sum()

DEM12_nuclear = DEM12.Transcript.str.count('NUCLEAR')
DEM12_nuclear.sum()

GOP12_atomic = GOP12.Transcript.str.count('ATOMIC')
GOP12_atomic.sum()

DEM12_atomic = DEM12.Transcript.str.count('ATOMIC')
DEM12_atomic.sum()

GOP12_proliferation = GOP12.Transcript.str.count('PROLIFERATION')
GOP12_proliferation.sum()

DEM12_proliferation = DEM12.Transcript.str.count('PROLIFERATION')
DEM12_proliferation.sum()

GOP12_nonproliferation = GOP12.Transcript.str.count('NONPROLIFERATION')
GOP12_nonproliferation.sum()

DEM12_nonproliferation = DEM12.Transcript.str.count('NONPROLIFERATION')
DEM12_nonproliferation.sum()

GOP12_non_proliferation = GOP12.Transcript.str.count('NON-PROLIFERATION')
GOP12_non_proliferation.sum()

DEM12_non_proliferation = DEM12.Transcript.str.count('NON-PROLIFERATION')
DEM12_non_proliferation.sum()

GOP12_armsrace = GOP12.Transcript.str.count('ARMS RACE')
GOP12_armsrace.sum()

DEM12_armsrace = DEM12.Transcript.str.count('ARMS RACE')
DEM12_armsrace.sum()

GOP12_women = GOP12.Transcript.str.count('WOMEN')
GOP12_women.sum()

DEM12_women = DEM12.Transcript.str.count('WOMEN')
DEM12_women.sum()

GOP12_job = GOP12.Transcript.str.count('JOB')
GOP12_job.sum()

DEM12_job = DEM12.Transcript.str.count('JOB')
DEM12_job.sum()

GOP12_jobs = GOP12.Transcript.str.count('JOBS')
GOP12_jobs.sum()

DEM12_jobs = DEM12.Transcript.str.count('JOBS')
DEM12_jobs.sum()

GOP12_unemployment = GOP12.Transcript.str.count('UNEMPLOYMENT')
GOP12_unemployment.sum()

DEM12_unemployment = DEM12.Transcript.str.count('UNEMPLOYMENT')
DEM12_unemployment.sum()

GOP12_world = GOP12.Transcript.str.count('WORLD')
GOP12_world.sum()

DEM12_world = DEM12.Transcript.str.count('WORLD')
DEM12_world.sum()

GOP12_foreign = GOP12.Transcript.str.count('FOREIGN')
GOP12_foreign.sum()

DEM12_foreign = DEM12.Transcript.str.count('FOREIGN')
DEM12_foreign.sum()

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

debates_2016_nuclear = debates_2016.Transcript.str.count('NUCLEAR')
debates_2016_nuclear.sum()
	
debates_2016_atomic = debates_2016.Transcript.str.count('ATOMIC')
debates_2016_atomic.sum()
	
debates_2016_proliferation = debates_2016.Transcript.str.count('PROLIFERATION')
debates_2016_proliferation.sum()
	
debates_2016_nonproliferation = debates_2016.Transcript.str.count('NONPROLIFERATION')
debates_2016_nonproliferation.sum()
	
debates_2016_non_proliferation = debates_2016.Transcript.str.count('NON-PROLIFERATION')
debates_2016_non_proliferation.sum()
		
debates_2016_armsrace = debates_2016.Transcript.str.count('ARMS RACE')
debates_2016_armsrace.sum()
	
debates_2016_women = debates_2016.Transcript.str.count('WOMEN')
debates_2016_women.sum()
	
debates_2016_job = debates_2016.Transcript.str.count('JOB')
debates_2016_job.sum()	

debates_2016_jobs = debates_2016.Transcript.str.count('JOBS')
debates_2016_jobs.sum()	

debates_2016_unemployment = debates_2016.Transcript.str.count('UNEMPLOYMENT')
debates_2016_unemployment.sum()
	
debates_2016_world = debates_2016.Transcript.str.count('WORLD')
debates_2016_world.sum()
	
debates_2016_foreign = debates_2016.Transcript.str.count('FOREIGN')
debates_2016_foreign.sum()
	
debates_2016_Iraq = debates_2016.Transcript.str.count('IRAQ')
debates_2016_Iraq.sum()
	
debates_2016_Soviet= debates_2016.Transcript.str.count('SOVIET')	
debates_2016_Soviet.sum()
	
debates_2016_Soviets= debates_2016.Transcript.str.count('SOVIETS')
debates_2016_Soviets.sum()
	
debates_2016_Russia= debates_2016.Transcript.str.count('RUSSIA')
debates_2016_Russia.sum()

#analysis of words by party

GOP16_nuclear = GOP16.Transcript.str.count('NUCLEAR')
GOP16_nuclear.sum()

DEM16_nuclear = DEM16.Transcript.str.count('NUCLEAR')
DEM16_nuclear.sum()

GOP16_atomic = GOP16.Transcript.str.count('ATOMIC')
GOP16_atomic.sum()

DEM16_atomic = DEM16.Transcript.str.count('ATOMIC')
DEM16_atomic.sum()

GOP16_proliferation = GOP16.Transcript.str.count('PROLIFERATION')
GOP16_proliferation.sum()

DEM16_proliferation = DEM16.Transcript.str.count('PROLIFERATION')
DEM16_proliferation.sum()

GOP16_nonproliferation = GOP16.Transcript.str.count('NONPROLIFERATION')
GOP16_nonproliferation.sum()

DEM16_nonproliferation = DEM16.Transcript.str.count('NONPROLIFERATION')
DEM16_nonproliferation.sum()

GOP16_non_proliferation = GOP16.Transcript.str.count('NON-PROLIFERATION')
GOP16_non_proliferation.sum()

DEM16_non_proliferation = DEM16.Transcript.str.count('NON-PROLIFERATION')
DEM16_non_proliferation.sum()

GOP16_armsrace = GOP16.Transcript.str.count('ARMS RACE')
GOP16_armsrace.sum()

DEM16_armsrace = DEM16.Transcript.str.count('ARMS RACE')
DEM16_armsrace.sum()

GOP16_women = GOP16.Transcript.str.count('WOMEN')
GOP16_women.sum()

DEM16_women = DEM16.Transcript.str.count('WOMEN')
DEM16_women.sum()

GOP16_job = GOP16.Transcript.str.count('JOB')
GOP16_job.sum()

DEM16_job = DEM16.Transcript.str.count('JOB')
DEM16_job.sum()

GOP16_jobs = GOP16.Transcript.str.count('JOBS')
GOP16_jobs.sum()

DEM16_jobs = DEM16.Transcript.str.count('JOBS')
DEM16_jobs.sum()

GOP16_unemployment = GOP16.Transcript.str.count('UNEMPLOYMENT')
GOP16_unemployment.sum()

DEM16_unemployment = DEM16.Transcript.str.count('UNEMPLOYMENT')
DEM16_unemployment.sum()

GOP16_world = GOP16.Transcript.str.count('WORLD')
GOP16_world.sum()

DEM16_world = DEM16.Transcript.str.count('WORLD')
DEM16_world.sum()

GOP16_foreign = GOP16.Transcript.str.count('FOREIGN')
GOP16_foreign.sum()

DEM16_foreign = DEM16.Transcript.str.count('FOREIGN')
DEM16_foreign.sum()

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
