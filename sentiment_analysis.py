# -*- coding: utf-8 -*-
"""
author: dfuentes 
Semester Project: Polisticians
Sentiment analysis with Plotly Express Charts
"""

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px, plotly.io as pio, matplotlib.pyplot as plt
import os, pandas as pd, numpy as np
import math
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

#%%




###### start sentiment analysis ######
# I will comment later #

pio.renderers.default='browser'

with open('Transcripts_full.csv', encoding = "utf-8") as f:
    raw = f.read()
    
df = pd.read_csv('Transcripts_full.csv') # call in for sentiment analysis

df['Transcript'] = df['Transcript'].fillna(value = 'xx')



sid = SentimentIntensityAnalyzer() 

t_list = df['Transcript'].tolist()

d_t = {}

for j in t_list:
    try:
        l = []

        ss = sid.polarity_scores(j)
        for i in sorted(ss):
            print('{0}: {1}, '.format(i, ss[i]), end='')
            l.append((i, ss[i]))

        d_t[j] = l
        print()
    except:
        pass
#print(d_t)

df['Sentiment'] = df['Transcript'].map(d_t)

print(df.head())

df = df[df.columns.dropna()]

print(df.dtypes)

df[['compound', 'neg', 'neu', 'pos']] = pd.DataFrame(df.Sentiment.values.tolist())



df[['compound', 'comp_val']] = pd.DataFrame(df['compound'].tolist(), index=df.index)
df[['neg', 'neg_val']] = pd.DataFrame(df['neg'].tolist(), index=df.index)   
df[['neu', 'neu_val']] = pd.DataFrame(df['neu'].tolist(), index=df.index) 
df[['pos', 'pos_val']] = pd.DataFrame(df['pos'].tolist(), index=df.index) 

#df_sent = pd.DataFrame(df['Sentiment'].tolist(), columns = ['compound', 'neg', 'neu', 'pos'])

df = df[df['Purpose'] == 'Candidate']
df = df[df['Transcript'] != 'xx']
df['cum_sentiment'] = df.groupby(['Debate', 'Actual Speaker'])['comp_val'].cumsum()
df['max_sentiment'] = df.groupby(['Debate', 'Actual Speaker'])['cum_sentiment'].tail(1)
df['max_sentiment'].fillna(method = 'backfill')
df['cum_wordcount'] = df.groupby(['Debate', 'Actual Speaker'])['word_count'].cumsum()
df['total_wordcount'] = df.groupby(['Debate', 'Actual Speaker'])['cum_wordcount'].transform('max')
df['sent per word'] = df['max_sentiment']/df['total_wordcount']

df = df.drop(['compound', 'neg', 'neu', 'pos'], axis=1)

df.to_csv('Transcripts_allyrs_and_candidate.csv')

# start graphing process



#%%

#debates_unique = df['Debate'].unique()

debates_unique = ['September 29, 2020 Debate', 'October 22, 2020 Debate', 'October 7, 2020 Debate'] #October 22, 2020 Debate'

print(debates_unique)

#%%

for d in debates_unique:
#    try:
    df_new = df[df['Debate'] == d]
    #df_new['Speaker_standardized'].fillna('xx', inplace = True)
    df_new.head()
    df_new = df_new.replace(r'^\s*$', np.nan, regex=True)
    df_new = df_new.fillna('xx')
    df_new = df_new[df_new['Purpose'] == 'Candidate']
    df_new = df_new[df_new['Actual Speaker'] != 'xx']
    df_new['ones'] = 1
    df_new['cum_linecount'] = df_new['ones'].cumsum()
    #df_new['cum_sentiment'] = df_new.groupby(['Affiliation'])['comp_val'].cumsum()
    #df_new['tot_linecount'] = df_new.groupby(['Affiliation'])['Line Count'].sum() 
    #df_new['cum_linecount'] = df_new.groupby(['Affiliation'])['Line Count'].cumsum()  
    #df_new['perc_linecount'] =   df_new['cum_linecount']/df_new['tot_linecount']
    #df_new['cum_wordcount'] = df_new.groupby(['Affiliation'])['word_count'].cumsum()
    df_new['aff_speaker'] = df_new['Party'] + ": " + df_new['Actual Speaker']
    cands = df_new['aff_speaker'].unique()
    col = {}
    for i in cands:
        if 'Democrat' in i:
            col[i] = 'blue'
        elif 'Republican' in i:
            col[i] = 'red'
        else:
            col[i] = 'green'
           
    #fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig = px.line(df_new, x = 'cum_linecount', y = 'cum_sentiment', title = d , color = 'aff_speaker', 
                  color_discrete_map=col)

    fig2 = px.line(df_new, x = 'cum_linecount', y = 'cum_wordcount', title = d , color = 'aff_speaker', 
                  color_discrete_map=col)
    
#    fig.add_trace(px.bar(df_new, x = 'Line Count', y = 'cumwordcount', title = d , color = 'aff_speaker', 
#                  color_discrete_map=col), secondary_y = False)        
        
    fig.show()
    fig2.show()
    
    df_new.to_csv('groupby.csv')
#    except:
#        pass

#%%

#box plots

col_box = {'Democrat': 'blue', 'Republican': 'red', 'Independent': 'green'}

#df['cum_sentiment'] = df.groupby(['Debate', 'Actual Speaker'])['comp_val'].cumsum()
#df['max_sentiment'] = df.groupby(['Debate', 'Actual Speaker'])['cum_sentiment'].transform('max')

df['Winner'].replace('', np.nan, inplace=True)
df.dropna(subset=['Winner'], inplace=True)

df.to_csv('check.csv')
fig = px.box(df, x = 'Winner', y = 'sent per word')
fig.show()

fig = px.box(df, x = 'Incumbent_president', y = 'sent per word', color = 'Position_map')
fig.show()

fig = px.box(df, x = 'Year', y = 'sent per word', color = 'Winner')
fig.show()

fig = px.box(df, x = 'Year', y = 'sent per word', color = 'Incumbent_president')
fig.show()

fig = px.box(df, x = 'Position_map', y = 'sent per word')
fig.show()

fig = px.box(df, x = 'Winner', y = 'sent per word', color = 'Party', color_discrete_map=col_box)
fig.show()

fig = px.box(df, x = 'Incumbent_president', y = 'sent per word', color = 'Party', color_discrete_map=col_box)
fig.show()

fig = px.box(df, x = 'Incumbent_president', y = 'total_wordcount', color = 'Party', color_discrete_map=col_box)
fig.show()

fig = px.box(df, x = 'Winner', y = 'total_wordcount', color = 'Party', color_discrete_map=col_box)
fig.show()

fig = px.box(df, x = 'Year', y = 'sent per word', color = 'Actual Speaker', color_discrete_map=col_box)
fig.show()

#%%

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
                df_y_spk.to_csv(abs_dir + '\Yearly_Files\Transcripts_' + j + '_' + str(int(i)) + '_wSentiments.csv')
            else:
                pass
    else:
        pass
