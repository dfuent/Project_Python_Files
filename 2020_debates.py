# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 16:18:46 2020

@author: dfuentes
"""

import re
import pandas as pd

# File used to open 2020 debate text files and prepare them for integration w/
# the rest of the process

print('Start the process')

# file names
d = ['2020_debate1.txt', '2020_debate2.txt', '2020_VP_debate.txt']

# list to which transcripts will be added
t = []

# blank DF with transcript header
df = pd.DataFrame(columns = ['Transcript'])

# open file, split the text into a list by tab (4 spaces in the file means
# a transition to a new speaker)
with open('2020_debate1.txt','r') as f:    
    for l in f:
        t = re.split(r'\s{4,}', l)

print('First Presidential debate pulled')

# add list to DF
df['Transcript'] = t

# add extra info to DF, like which debate and the position (P or VP)
df['Debate'] = 'September 29, 2020 Debate'
df['Position'] = 'President'

# start same process for the second Prez debate
df2 = pd.DataFrame(columns = ['Transcript'])

t = []

with open('2020_debate2.txt','r') as f:    
    for l in f:
        t = re.split(r'\s{4,}', l)

print('Second Presidential debate pulled')        

df2['Transcript'] = t

df2['Debate'] = 'October 22, 2020 Debate'
df2['Position'] = 'President'


# start the process for the VP debate; process is the same as the two Prez
# debates
df3 = pd.DataFrame(columns = ['Transcript'])

with open('2020_VP_debate.txt','r') as f:    
    for l in f:
        t = re.split(r'\s{4,}', l)
 
print('Only Vice Presidential debate pulled')

df3['Transcript'] = t

df3['Debate'] = 'October 7, 2020 Debate'
df3['Position'] = 'Vice President'

df = df.append(df2)
df = df.append(df3)

# rev.com includes the time at which each candidate said a given statement
# take this out of the transcript and add it to another column
df['Time'] = df['Transcript'].apply(lambda st: st[st.find("(")+1:st.find(")")])

# need to clean up the () that were around the time. Replace with blanks
df['Transcript'] = df['Transcript'].replace("[\(\[].*?[\)\]]", "",regex=True)

# the speaker in the transcript is delimited with ':'; grab the name before
# the colon and put in the Speaker_Clean column
df['Speaker_Clean'] = df['Transcript'].apply(lambda st: st[:st.find(":")])

# put the purpose as Candidate (even though some of the transcript statements
# are from the moderators, these get cleaned up in the final files, so we can
# call everyone a candidate for ease)
df['Purpose'] = 'Candidate'

# we will add the Speaker_Standardized column, on which we join a mapping
# file later
df['Speaker_standardized'] = df['Speaker_Clean']

# used the following to make sure the name pull worked properly
#df['Transcript_OG'] = df['Transcript']

# Remove the name in the Transcript rows and use the rest of the text
# (index = 1) to clean up the Transcript. Checked to make sure candidates didn't
# have a colon in their transcript besides the one after their name
df['Transcript'] = df['Transcript'].str.split(':').str[1]

# remove any white space on the Transcript
df['Transcript'] = df['Transcript'].str.strip(' ')

# send the transcripts to a CSV to be used in the rest of the process
df.to_csv('2020debates.csv')

print('Process complete. File 2020debates.csv created')