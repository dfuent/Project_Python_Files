# -*- coding: utf-8 -*-

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

GOP1984_nuclear = GOP1984.Transcript.str.count('NUCLEAR')
GOP1984_nuclear.sum()

Dem1984_nuclear = Dem1984.Transcript.str.count('NUCLEAR')
Dem1984_nuclear.sum()

GOP1984_atomic = GOP1984.Transcript.str.count('ATOMIC')
GOP1984_atomic.sum()

Dem1984_atomic = Dem1984.Transcript.str.count('ATOMIC')
Dem1984_atomic.sum()

GOP1984_proliferation = GOP1984.Transcript.str.count('PROLIFERATION')
GOP1984_proliferation.sum()

Dem1984_proliferation = Dem1984.Transcript.str.count('PROLIFERATION')
Dem1984_proliferation.sum()

GOP1984_nonproliferation = GOP1984.Transcript.str.count('NONPROLIFERATION')
GOP1984_nonproliferation.sum()

DEM1984_nonproliferation = DEM1984.Transcript.str.count('NONPROLIFERATION')
DEM1984_nonproliferation.sum()

GOP1984_non_proliferation = GOP1984.Transcript.str.count('NON-PROLIFERATION')
GOP1984_non_proliferation.sum()

Dem1984_non_proliferation = Dem1984.Transcript.str.count('NON-PROLIFERATION')
Dem1984_non_proliferation.sum()

GOP1984_armsrace = GOP1984.Transcript.str.count('ARMS RACE')
GOP1984_armsrace.sum()

Dem1984_armsrace = Dem1984.Transcript.str.count('ARMS RACE')
Dem1984_armsrace.sum()

GOP1984_women = GOP1984.Transcript.str.count('WOMEN')
GOP1984_women.sum()

Dem1984_women = Dem8194.Transcript.str.count('WOMEN')
Dem1984_women.sum()

GOP1984_job = GOP1984.Transcript.str.count('JOB')
GOP1984_job.sum()

Dem1984_job = Dem1984.Transcript.str.count('JOB')
Dem1984_job.sum()

GOP1984_jobs = GOP1984.Transcript.str.count('JOBS')
GOP1984_jobs.sum()

Dem1984_jobs = Dem1984.Transcript.str.count('JOBS')
Dem1984_jobs.sum()

GOP1984_unemployment = GOP1984.Transcript.str.count('UNEMPLOYMENT')
GOP1984_unemployment.sum()

Dem1984_unemployment = Dem1984.Transcript.str.count('UNEMPLOYMENT')
Dem1984_unemployment.sum()

GOP1984_world = GOP1984.Transcript.str.count('WORLD')
GOP1984_world.sum()

Dem1984_world = Dem1984.Transcript.str.count('WORLD')
Dem1984_world.sum()

GOP1984_foreign = GOP1984.Transcript.str.count('FOREIGN')
GOP1984_foreign.sum()

Dem1984_foreign = Dem1984.Transcript.str.count('FOREIGN')
Dem1984_foreign.sum()



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

Dem1996_nuclear = Dem1996.Transcript.str.count('NUCLEAR')
Dem1996_nuclear.sum()

GOP1996_atomic = GOP1996.Transcript.str.count('ATOMIC')
GOP1996_atomic.sum()

Dem1996_atomic = Dem1996.Transcript.str.count('ATOMIC')
Dem1996_atomic.sum()

GOP1996_proliferation = GOP1996.Transcript.str.count('PROLIFERATION')
GOP1996_proliferation.sum()

Dem1996_proliferation = Dem1996.Transcript.str.count('PROLIFERATION')
Dem1996_proliferation.sum()

GOP1996_nonproliferation = GOP1996.Transcript.str.count('NONPROLIFERATION')
GOP1996_nonproliferation.sum()

Dem1996_nonproliferation = Dem1996.Transcript.str.count('NONPROLIFERATION')
Dem1996_nonproliferation.sum()

GOP1996_non_proliferation = GOP1996.Transcript.str.count('NON-PROLIFERATION')
GOP1996_non_proliferation.sum()

Dem1996_non_proliferation = Dem1996.Transcript.str.count('NON-PROLIFERATION')
Dem1996_non_proliferation.sum()

GOP1996_armsrace = GOP1996.Transcript.str.count('ARMS RACE')
GO19P96_armsrace.sum()

Dem1996_armsrace = Dem1996.Transcript.str.count('ARMS RACE')
Dem1996_armsrace.sum()

GOP1996_women = GOP1996.Transcript.str.count('WOMEN')
GOP1996_women.sum()

Dem1996_women = Dem1996.Transcript.str.count('WOMEN')
Dem1996_women.sum()

GOP1996_job = GOP1996.Transcript.str.count('JOB')
GOP1996_job.sum()

Dem1996_job = Dem1996.Transcript.str.count('JOB')
Dem1996_job.sum()

GOP1996_jobs = GOP1996.Transcript.str.count('JOBS')
GOP1996_jobs.sum()

Dem1996_jobs = Dem1996.Transcript.str.count('JOBS')
Dem1996_jobs.sum()

GOP1996_unemployment = GOP1996.Transcript.str.count('UNEMPLOYMENT')
GOP1996_unemployment.sum()

Dem1996_unemployment = Dem1996.Transcript.str.count('UNEMPLOYMENT')
Dem1996_unemployment.sum()

GOP1996_world = GOP1996.Transcript.str.count('WORLD')
GOP1996_world.sum()

Dem1996_world = Dem1996.Transcript.str.count('WORLD')
Dem1996_world.sum()

GOP1996_foreign = GOP1996.Transcript.str.count('FOREIGN')
GOP1996_foreign.sum()

Dem1996_foreign = Dem1996.Transcript.str.count('FOREIGN')
Dem1996_foreign.sum()


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

Dem2000_nuclear = Dem2000.Transcript.str.count('NUCLEAR')
Dem2000_nuclear.sum()

GOP2000_atomic = GOP2000.Transcript.str.count('ATOMIC')
GOP2000_atomic.sum()

Dem2000_atomic = Dem2000.Transcript.str.count('ATOMIC')
Dem2000_atomic.sum()

GOP2000_proliferation = GOP2000.Transcript.str.count('PROLIFERATION')
GOP2000_proliferation.sum()

Dem2000_proliferation = Dem2000.Transcript.str.count('PROLIFERATION')
Dem2000_proliferation.sum()

GOP2000_nonproliferation = GOP2000.Transcript.str.count('NONPROLIFERATION')
GOP2000_nonproliferation.sum()

Dem2000_nonproliferation = Dem2000.Transcript.str.count('NONPROLIFERATION')
Dem2000_nonproliferation.sum()

GOP2000_non_proliferation = GOP2000.Transcript.str.count('NON-PROLIFERATION')
GOP2000_non_proliferation.sum()

Dem2000_non_proliferation = Dem2000.Transcript.str.count('NON-PROLIFERATION')
Dem2000_non_proliferation.sum()

GOP2000_armsrace = GOP2000.Transcript.str.count('ARMS RACE')
GOP2000_armsrace.sum()

Dem2000_armsrace = DEM2000.Transcript.str.count('ARMS RACE')
Dem2000_armsrace.sum()

GOP00_women = GOP00.Transcript.str.count('WOMEN')
GOP00_women.sum()

Dem2000_women = Dem2000.Transcript.str.count('WOMEN')
Dem2000_women.sum()

GOP2000_job = GOP2000.Transcript.str.count('JOB')
GOP2000_job.sum()

Dem2000_job = Dem2000.Transcript.str.count('JOB')
Dem2000_job.sum()

GOP2000_jobs = GOP2000.Transcript.str.count('JOBS')
GOP2000_jobs.sum()

Dem2000_jobs = Dem2000.Transcript.str.count('JOBS')
Dem2000_jobs.sum()

GOP2000_unemployment = GOP2000.Transcript.str.count('UNEMPLOYMENT')
GOP2000_unemployment.sum()

Dem2000_unemployment = Dem2000.Transcript.str.count('UNEMPLOYMENT')
Dem2000_unemployment.sum()

GOP2000_world = GOP2000.Transcript.str.count('WORLD')
GOP2000_world.sum()

Dem2000_world = Dem2000.Transcript.str.count('WORLD')
Dem2000_world.sum()

GOP2000_foreign = GOP2000.Transcript.str.count('FOREIGN')
GOP2000_foreign.sum()

Dem2000_foreign = Dem2000.Transcript.str.count('FOREIGN')
Dem2000_foreign.sum()

#Begin 2004
#October 8, 2004
Oct8_2004_debate = df[df["Debate"]=='October 8, 2004 Debate Transcript']
Oct8_2004_debate

#testing length
len(Oct8_2004_debate)

#September 30, 2004
Sept30_2004_debate = df[df["Debate"]=='September 30. 2004 Debate Transcript']
Sept30_2004_debate

#testing length
len(Sept30_2004_debate)

#October 5, 2004
Oct5_2004_debate = df[df["Debate"]=='October 5, 2004 Transcript']
Oct5_2004_debate

#testing length
len(Oct5_2004_debate)

#October 13, 2004
Oct13_2004_debate = df[df["Debate"]=='October 13, 2004 Debate Transcript']
Oct13_2004_debate

#testing length
len(Oct13_2004_debate)

#full debate transcript 2004, includes moderators and any other speakes in addition to candidates
debates_2004 = Oct8_2004_debate.append([Sept30_2004_debate, Oct5_2004_debate, Oct13_2004_debate])
debates_2004

#testing length
len(debates_2004)

#Bush2004
Bush04 = debates_2004[(debates_2004["Speaker_standardized"]=='George W. Bush')]
Bush04

#testing length
len(Bush04)

#Cheney2004
Cheney04 = debates_2004[(debates_2004["Speaker_standardized"]=='Dick Cheney')]
Cheney04

#testing length
len(Cheney04)

#Kerry2004
Kerry04 = debates_2004[(debates_2004["Speaker_standardized"]=='John Kerry')]
Kerry04

#testing length
len(Kerry04)

#Bush2004
Bush04 = debates_2004[(debates_2004["Speaker_standardized"]=='George W. Bush')]
Bush04

#testing length
len(Bush04)

#Edwards2004
Edwards04 = debates_2004[(debates_2004["Speaker_standardized"]=='John Edwards')]
Edwards04

#testing length
len(Edwards04)

#GOP2004
GOP2004=Bush04.append([Cheney04])
GOP2004

#testing length
len(GOP2004)

#Dem2004
Dem2004=Kerry04.append([Edwards04])
Dem2004

#testing length
len(Dem2004)

#all candidates 2004
allcandidates04 = GOP2004.append([Dem2004]) 
allcandidates04

#testing length
len(allcandidates04)

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

GOP2004_nuclear = GOP2004.Transcript.str.count('NUCLEAR')
GOP2004_nuclear.sum()

Dem2004_nuclear = Dem2004.Transcript.str.count('NUCLEAR')
Dem2004_nuclear.sum()

GOP2004_atomic = GOP2004.Transcript.str.count('ATOMIC')
GOP2004_atomic.sum()

Dem2004_atomic = Dem2004.Transcript.str.count('ATOMIC')
Dem2004_atomic.sum()

GOP2004_proliferation = GOP2004.Transcript.str.count('PROLIFERATION')
GOP2004_proliferation.sum()

Dem2004_proliferation = Dem2004.Transcript.str.count('PROLIFERATION')
Dem2004_proliferation.sum()

GOP2004_nonproliferation = GOP2004.Transcript.str.count('NONPROLIFERATION')
GOP2004_nonproliferation.sum()

Dem2004_nonproliferation = Dem2004.Transcript.str.count('NONPROLIFERATION')
Dem2004_nonproliferation.sum()

GOP2004_non_proliferation = GOP2004.Transcript.str.count('NON-PROLIFERATION')
GOP2004_non_proliferation.sum()

Dem2004_non_proliferation = Dem2004.Transcript.str.count('NON-PROLIFERATION')
Dem2004_non_proliferation.sum()

GOP2004_armsrace = GOP2004.Transcript.str.count('ARMS RACE')
GOP2004_armsrace.sum()

Dem2004_armsrace = Dem2004.Transcript.str.count('ARMS RACE')
Dem2004_armsrace.sum()

GOP2004_women = GOP2004.Transcript.str.count('WOMEN')
GOP2004_women.sum()

Dem2004_women = Dem2004.Transcript.str.count('WOMEN')
Dem2004_women.sum()

GOP2004_job = GOP2004.Transcript.str.count('JOB')
GOP2004_job.sum()

Dem2004_job = Dem2004.Transcript.str.count('JOB')
Dem2004_job.sum()

GOP2004_jobs = GOP2004.Transcript.str.count('JOBS')
GOP2004_jobs.sum()

Dem2004_jobs = Dem2004.Transcript.str.count('JOBS')
Dem2004_jobs.sum()

GOP2004_unemployment = GOP2004.Transcript.str.count('UNEMPLOYMENT')
GOP2004_unemployment.sum()

Dem2004_unemployment = Dem2004.Transcript.str.count('UNEMPLOYMENT')
Dem2004_unemployment.sum()

GOP2004_world = GOP2004.Transcript.str.count('WORLD')
GOP2004_world.sum()

Dem2004_world = Dem204.Transcript.str.count('WORLD')
Dem2004_world.sum()

GOP2004_foreign = GOP2004.Transcript.str.count('FOREIGN')
GOP2004_foreign.sum()

Dem2004_foreign = Dem2004.Transcript.str.count('FOREIGN')
Dem2004_foreign.sum()

#Begin 2008
#October 15, 2008
Oct15_2008_debate = df[df["Debate"]=='October 15, 2008 Debate Transcript']
Oct15_2008_debate

#testing length
len(Oct15_2008_debate)

#October 2, 2008
Oct2_2008_debate = df[df["Debate"]=='October 2, 2008 Debate Transcript']
Oct2_2008_debate

#testing length
len(Oct2_2008_debate)

#Septmber 26, 2008
Sept26_2008_debate = df[df["Debate"]=='September 26, 2008 Debate Transcript']
Sept26_2008_debate

#testing length
len(Sept26_2008_debate)

#October 7, 2008
Oct7_2008_debate = df[df["Debate"]=='October 7, 2008 Debate Transcript']
Oct7_2008_debate

#testing length
len(Oct7_2008_debate)

#full debate transcript 2008, includes moderators and any other speakes in addition to candidates
debates_2008 = Oct15_2008_debate.append([Oct2_2008_debate, Sept26_2008_debate, Oct7_2008_debate])
debates_2008

#testing length
len(debates2008)

#McCain2008
McCain08 = debates_2008[(debates_2008["Speaker_standardized"]=='John McCain')]
McCain08

#testing length
len(McCain08)

#Palin2008
Palin08 = debates_2008[(debates_2008["Speaker_standardized"]=='Sarah Palin')]
Palin08

#testing length
len(Palin08)

#Obama2008
Obama08 = debates_2008[(debates_2008["Speaker_standardized"]=='Barack Obama')]
Obama08

#testing length
len(Obama08)

#Biden2008
Biden08 = debates_2008[(debates_2008["Speaker_standardized"]=='Joe Biden')]
Biden08

#testing length
len(Biden08)

#GOP2008
GOP2008=McCain08.append([Palin08])
GOP2008

#testing length
len(GOP2008)

#Dem2008

Dem2008=Obama08.append([Biden08])
Dem2008

#testing length
len(Dem2008)

#allcandidates2008
allcandidates08 = GOP2008.append([Dem2008]) 
allcandidates08

#testing length 
len(allcandidates08)

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

GOP2008_nuclear = GOP2008.Transcript.str.count('NUCLEAR')
GOP2008_nuclear.sum()

Dem2008_nuclear = Dem2008.Transcript.str.count('NUCLEAR')
Dem2008_nuclear.sum()

GOP2008_atomic = GOP2008.Transcript.str.count('ATOMIC')
GOP2008_atomic.sum()

Dem2008_atomic = Dem2008.Transcript.str.count('ATOMIC')
Dem2008_atomic.sum()

GOP2008_proliferation = GOP2008.Transcript.str.count('PROLIFERATION')
GOP2008_proliferation.sum()

Dem2008_proliferation = Dem2008.Transcript.str.count('PROLIFERATION')
Dem2008_proliferation.sum()

GOP2008_nonproliferation = GOP2008.Transcript.str.count('NONPROLIFERATION')
GOP2008_nonproliferation.sum()

Dem2008_nonproliferation = Dem2008.Transcript.str.count('NONPROLIFERATION')
Dem2008_nonproliferation.sum()

GOP2008_non_proliferation = GOP2008.Transcript.str.count('NON-PROLIFERATION')
GOP2008_non_proliferation.sum()

DEM2008_non_proliferation = Dem2008.Transcript.str.count('NON-PROLIFERATION')
DEM2008_non_proliferation.sum()

GOP2008_armsrace = GOP2008.Transcript.str.count('ARMS RACE')
GOP2008_armsrace.sum()

Dem2008_armsrace = Dem2008.Transcript.str.count('ARMS RACE')
Dem2008_armsrace.sum()

GOP2008_women = GOP2008.Transcript.str.count('WOMEN')
GOP2008_women.sum()

Dem2008_women = Dem2008.Transcript.str.count('WOMEN')
Dem2008_women.sum()

GOP2008_job = GOP2008.Transcript.str.count('JOB')
GOP2008_job.sum()

Dem2008_job = Dem2008.Transcript.str.count('JOB')
Dem2008_job.sum()

GOP2008_jobs = GOP2008.Transcript.str.count('JOBS')
GOP2008_jobs.sum()

Dem2008_jobs = Dem2008.Transcript.str.count('JOBS')
Dem2008_jobs.sum()

GOP2008_unemployment = GOP2008.Transcript.str.count('UNEMPLOYMENT')
GOP2008_unemployment.sum()

Dem2008_unemployment = Dem2008.Transcript.str.count('UNEMPLOYMENT')
Dem2008_unemployment.sum()

GOP2008_world = GOP2008.Transcript.str.count('WORLD')
GOP2008_world.sum()

Dem2008_world = Dem2008.Transcript.str.count('WORLD')
Dem2008_world.sum()

GOP2008_foreign = GOP2008.Transcript.str.count('FOREIGN')
GOP2008_foreign.sum()

Dem2008_foreign = Dem2008.Transcript.str.count('FOREIGN')
Dem2008_foreign.sum()

#Begin 2012
#October 16, 2012
Oct16_2012_debate = df[df["Debate"]=='October 16, 2012 Debate Transcript']
Oct16_2012_debate

#testing length
len(Oct16_2012_debate)

#October 3, 2012
Oct3_2012_debate = df[df["Debate"]=='October 3, 2012 Debate Transcript']
Oct3_2012_debate

#testing length
len(Oct3_2012_debate)

#October 22, 2012
Oct22_2012_debate = df[df["Debate"]=='October 22, 2012 Debate Transcript']
Oct22_2012_debate

#testing length
len(Oct22_2012_debate)

#October 11, 2012
Oct11_2012_debate = df[df["Debate"]=='October 11, 2012 Debate Transcript']
Oct11_2012_debate

#testing length
len(Oct11_2012_debate)

#full debate transcript 2012, includes moderators and any other speakes in addition to candidates
debates_2012 = Oct16_2012_debate.append([Oct3_2012_debate, Oct22_2012_debate, Oct11_2012_debate])
debates_2012

#testing length
len(debates_2012)

#Obama2012
Obama12 = debates_2012[(debates_2012["Speaker_standardized"]=='Barack Obama')]
Obama12

#testing length
len(Obama12)

#Biden2012
Biden12 = debates_2012[(debates_2012["Speaker_standardized"]=='Joe Biden')]
Biden12

#testing length
len(Biden12)

#Romney2012
Romney12 = debates_2012[(debates_2012["Speaker_standardized"]=='Mitt Romney')]
Romney12

#testing length
len(Romney12)

#Ryan2012
Ryan12 = debates_2012[(debates_2012["Speaker_standardized"]=='Paul Ryan')]
Ryan12

#testing length
len(Ryan12)

#GOP2012
GOP2012=Romney12.append([Ryan12])
GOP2012

#testing length
len(GOP2012)

#Dem2012
Dem2012=Obama12.append([Biden12])
Dem2012

#testing length
len(Dem2012)

#all candidates 2012
allcandidates12 = GOP2012.append([Dem2012]) 
allcandidates12

#testing length
len(allcandidates12)

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

GOP2012_nuclear = GOP2012.Transcript.str.count('NUCLEAR')
GOP2012_nuclear.sum()

Dem2012_nuclear = Dem2012.Transcript.str.count('NUCLEAR')
Dem2012_nuclear.sum()

GOP2012_atomic = GOP2012.Transcript.str.count('ATOMIC')
GOP2012_atomic.sum()

Dem2012_atomic = Dem2012.Transcript.str.count('ATOMIC')
Dem2012_atomic.sum()

GOP2012_proliferation = GOP2012.Transcript.str.count('PROLIFERATION')
GOP2012_proliferation.sum()

Dem2012_proliferation = Dem2012.Transcript.str.count('PROLIFERATION')
Dem2012_proliferation.sum()

GOP2012_nonproliferation = GOP2012.Transcript.str.count('NONPROLIFERATION')
GOP2012_nonproliferation.sum()

Dem2012_nonproliferation = Dem2012.Transcript.str.count('NONPROLIFERATION')
Dem2012_nonproliferation.sum()

GOP2012_non_proliferation = GOP2012.Transcript.str.count('NON-PROLIFERATION')
GOP2012_non_proliferation.sum()

Dem2012_non_proliferation = Dem2012.Transcript.str.count('NON-PROLIFERATION')
Dem2012_non_proliferation.sum()

GOP2012_armsrace = GOP2012.Transcript.str.count('ARMS RACE')
GOP2012_armsrace.sum()

Dem2012_armsrace = Dem2012.Transcript.str.count('ARMS RACE')
Dem2012_armsrace.sum()

GOP2012_women = GOP2012.Transcript.str.count('WOMEN')
GOP2012_women.sum()

Dem2012_women = Dem2012.Transcript.str.count('WOMEN')
Dem2012_women.sum()

GOP2012_job = GOP2012.Transcript.str.count('JOB')
GOP2012_job.sum()

Dem2012_job = Dem2012.Transcript.str.count('JOB')
Dem2012_job.sum()

GOP2012_jobs = GOP2012.Transcript.str.count('JOBS')
GOP2012_jobs.sum()

Dem2012_jobs = Dem2012.Transcript.str.count('JOBS')
Dem2012_jobs.sum()

GOP2012_unemployment = GOP2012.Transcript.str.count('UNEMPLOYMENT')
GOP2012_unemployment.sum()

Dem2012_unemployment = Dem2012.Transcript.str.count('UNEMPLOYMENT')
Dem2012_unemployment.sum()

GOP2012_world = GOP2012.Transcript.str.count('WORLD')
GOP2012_world.sum()

Dem2012_world = Dem2012.Transcript.str.count('WORLD')
Dem2012_world.sum()

GOP2012_foreign = GOP2012.Transcript.str.count('FOREIGN')
GOP2012_foreign.sum()

Dem2012_foreign = Dem2012.Transcript.str.count('FOREIGN')
Dem2012_foreign.sum()

#Begin2016
#October 4, 2016
Oct4_2016_debate = df[df["Debate"]=='October 4, 2016 Debate Transcript']
Oct4_2016_debate

#length test
len(Oct4_2016_debate)

#September 26, 2016
Sept26_2016_debate = df[df["Debate"]=='September 26, 2016 Debate Transcript']
Sept26_2016_debate

#length test
len(Sept26_2016_debate)

#October 19, 2016
Oct19_2016_debate = df[df["Debate"]=='October 19, 2016 Debate Transcript']
Oct19_2016_debate

#length test
len(Oct19_2016_debate)

#October 9, 2016
Oct9_2016_debate = df[df["Debate"]=='October 9, 2016 Debate Transcript']
Oct9_2016_debate

#length test
len(Oct9_2016_debate)

#full debate transcript 2016, includes moderators and any other speakes in addition to candidates
debates_2016 = Oct4_2016_debate.append([Sept26_2016_debate, Oct19_2016_debate, Oct9_2016_debate])
debates_2016

#length test
len(debates_2016)

#Trump2016
Trump16 = debates_2016[(debates_2016["Speaker_standardized"]=='Donald Trump')]
Trump16

#length test
len(Trump16)

#Pence16
Pence16 = debates_2016[(debates_2016["Speaker_standardized"]=='Mike Pence')]
Pence16

#length test
len(Pence16)

#Hillary Clinton16 --labled as Bill in data
Clinton16 = debates_2016[(debates_2016["Speaker_standardized"]=='Bill Clinton')]
Clinton16

#length test
len(Clinton16)

#Kaine16
Kaine16 = debates_2016[(debates_2016["Speaker_standardized"]=='Tim Kaine')]
Kaine16

#length test
len(Kaine16)

#GOP 2016
GOP2016 = Trump16.append([Pence16]) 
GOP2016

#length test
len(GOP2016)

#Dem 2016
Dem2016 = Clinton16.append([Kaine16]) 
Dem2016

#length test
len(Dem2016)

#allcandidates 2016
allcandidates16 = GOP2016.append([Dem2016]) 
allcandidates16

#length test
len(allcandidates16)

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

GOP2016_nuclear = GOP2016.Transcript.str.count('NUCLEAR')
GOP2016_nuclear.sum()

Dem2016_nuclear = Dem2016.Transcript.str.count('NUCLEAR')
Dem2016_nuclear.sum()

GOP2016_atomic = GOP2016.Transcript.str.count('ATOMIC')
GOP2016_atomic.sum()

Dem2016_atomic = Dem2016.Transcript.str.count('ATOMIC')
Dem2016_atomic.sum()

GOP2016_proliferation = GOP2016.Transcript.str.count('PROLIFERATION')
GOP2016_proliferation.sum()

Dem2016_proliferation = Dem2016.Transcript.str.count('PROLIFERATION')
Dem2016_proliferation.sum()

GOP2016_nonproliferation = GOP2016.Transcript.str.count('NONPROLIFERATION')
GOP2016_nonproliferation.sum()

Dem2016_nonproliferation = Dem2016.Transcript.str.count('NONPROLIFERATION')
Dem2016_nonproliferation.sum()

GOP2016_non_proliferation = GOP2016.Transcript.str.count('NON-PROLIFERATION')
GOP2016_non_proliferation.sum()

Dem2016_non_proliferation = Dem2016.Transcript.str.count('NON-PROLIFERATION')
Dem2016_non_proliferation.sum()

GOP2016_armsrace = GOP2016.Transcript.str.count('ARMS RACE')
GOP2016_armsrace.sum()

Dem2016_armsrace = Dem2016.Transcript.str.count('ARMS RACE')
Dem2016_armsrace.sum()

GOP2016_women = GOP2016.Transcript.str.count('WOMEN')
GOP2016_women.sum()

Dem2016_women = Dem2016.Transcript.str.count('WOMEN')
Dem2016_women.sum()

GOP2016_job = GOP2016.Transcript.str.count('JOB')
GOP2016_job.sum()

Dem2016_job = Dem2016.Transcript.str.count('JOB')
Dem2016_job.sum()

GOP2016_jobs = GOP2016.Transcript.str.count('JOBS')
GOP2016_jobs.sum()

Dem2016_jobs = Dem2016.Transcript.str.count('JOBS')
Dem2016_jobs.sum()

GOP2016_unemployment = GOP2016.Transcript.str.count('UNEMPLOYMENT')
GOP2016_unemployment.sum()

Dem2016_unemployment = Dem2016.Transcript.str.count('UNEMPLOYMENT')
Dem2016_unemployment.sum()

GOP2016_world = GOP2016.Transcript.str.count('WORLD')
GOP2016_world.sum()

Dem2016_world = Dem2016.Transcript.str.count('WORLD')
Dem2016_world.sum()

GOP2016_foreign = GOP2016.Transcript.str.count('FOREIGN')
GOP2016_foreign.sum()

Dem2016_foreign = Dem2016.Transcript.str.count('FOREIGN')
Dem2016_foreign.sum()

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
