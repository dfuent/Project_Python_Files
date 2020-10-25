# -*- coding: utf-8 -*-
"""
@author: Lauren Neal
Semester Project: Polisticians
"""

## This section creates two new DF columns for word frequency distributions:
## one including stopwords and one without

## KINKS TO WORK OUT: 
## 1. This isn't the best or easiest way to try to get to cumulative word frequencies;
## have much more to do here

import nltk, re, pprint, pandas as pd, numpy as np
from nltk import word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords


# file_encoding = 'utf8'        # set file_encoding to the file encoding (utf8, latin1, etc.)
# input_fd = open('Transcripts_finalb.csv', encoding=file_encoding, errors = 'backslashreplace')

# df = pd.read_csv(input_fd)

# col_names = ["Line", "Unnamed", "Line_Count", "Debate", "Transcript", "Speaker_Clean", "Speaker_fin", "word_count", "0", "Speaker_standardized", "Affiliation", "Purpose", "Position", "Key", "Actual_Speaker", "Year", "Position_map", "Party", "Winner", "Incumbent_president"]
                 
# df = pd.read_csv('Transcripts_finalb.csv', encoding = 'utf-16') 

df = pd.read_table('Transcripts_finalb.csv', encoding = 'utf-16')
df = df.drop_duplicates(subset=["Transcript"])

# remove all speaker names contained in "Transcript" column (words with more than one consecutive capital letter)
df['Transcript']=df['Transcript'].str.replace(r'[A-Z]+\s?[A-Z]+', '', regex=True)

# ignore â€™,  â€', etc. (em dashes and apostrophes encoded strangely)
df['Transcript']=df['Transcript'].str.encode("ascii", "ignore")
df['Transcript']=df['Transcript'].str.decode(encoding = 'utf-8')


stop_words = set(stopwords.words('english'))

# convert Transcript column to list (where each row is a string-type list item)
t_list = df['Transcript'].tolist()

freq_dist_all = {}
freq_dist_no_stop = {}

# for every list item (each row in Transcript column a string object item in list)
for j in t_list:
        q = str(j)
        p = q.translate(str.maketrans('','', string.punctuation))
        line_words = list(p.split(" "))
        no_stop = [w.strip() for w in line_words if w.strip() not in stop_words]    
        freq_all = FreqDist(line_words)
        freq_no_stop = FreqDist(no_stop)
        
        freq_dist_all[j] = freq_all
        freq_dist_no_stop[j] = freq_no_stop
        print()
  
        
df['Frequency_All'] = df['Transcript'].map(freq_dist_all)
df['Freq_Sans_Stop'] = df['Transcript'].map(freq_dist_no_stop)


#%%
## Lauren
## cumulative word/frequency count by year & political party

## KINKS TO WORK OUT: 
## Experimenting to determine the best way to group DF by political party & year
## and then do cumulative word frequency count that I can animate or visualize in a cool way
## I've tried many iterations of this; nothing has quite done what I've wanted it to yet

import nltk, re, pprint, pandas as pd, numpy as np
from nltk import word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords
import textstat


df = pd.read_table('Transcripts_finalb.csv', encoding = 'utf-16')
df = df[df.Purpose.eq('Candidate')]
df['Transcript'] = df['Transcript'].fillna(value = 'xx')
df = df.drop_duplicates(subset=["Transcript", "Key"])


# remove all speaker names contained in "Transcript" column (words with more than one consecutive capital letter)
df['Transcript']=df['Transcript'].str.replace(r'[A-Z]+\s?[A-Z]+', '', regex=True)

# ignore â€™,  â€', etc. (em dashes and apostrophes encoded strangely)
df['Transcript']=df['Transcript'].str.encode("ascii", "ignore")
df['Transcript']=df['Transcript'].str.decode(encoding = 'utf-8')
df['Transcript']=df['Transcript'].str.lower()

df_party_year = df.groupby(['Affiliation', 'Year'])['Transcript'].apply(' '.join).reset_index()


stop_words = set(stopwords.words('english'))

#  threw MemoryError
# for index, row in df_party_year.iterrows():
#     if index == 0:
#         result = row['Transcript']
#         df_party_year.at[index, 'Transcript'] = result
#     elif 12 > index > 0:
#         result = row['Transcript']
#         df_party_year.at[index, 'Transcript'] = result.join(df_party_year.loc[index-1, 'Transcript'])
#     elif index == 12:
#         result = row['Transcript']
#         df_party_year.at[index, 'Transcript'] = result
#     elif index == 13:
#         result = row['Transcript']
#         df_party_year.at[index, 'Transcript'] = result.join(df_party_year.loc[index-1, 'Transcript'])
#     elif index == 14:
#         result = row['Transcript']
#         df_party_year.at[index, 'Transcript'] = result
#     else:
#         result = row['Transcript']
#         df_party_year.at[index, 'Transcript'] = result.join(df_party_year.loc[index-1, 'Transcript'])
    
    
# convert Transcript column to list (where each row is a string-type list item)
# tp_list = df_party_year['Transcript'].tolist()

freq_dist_all = {}
freq_dist_no_stop = {}

# def mergeDict(dict1, dict2):
#     dict3 = {**dict1, **dict2}
#     for key, value in dict3.items():
#         if key in dict1 and key in dict2:
#             dict3[key] = [dict1[key] + dict2[key]]
            
#     return dict3

# # for every list item (each row in Transcript column a string object item in list):
    
# for j in tp_list:
    
#          q = str(j).lower()
#          p = q.translate(str.maketrans(' ',' ', string.punctuation))
#          df_party_year['Words_All'] = list(p.split(" "))
#          df_party_year['Transcripts_Sans_Stop'] = [w.strip() for w in list(p.split(" ")) if w.strip() not in stop_words]    
#          # freq_all = FreqDist(line_words)
#          # freq_no_stop = FreqDist(no_stop)
        
#          # freq_dist_all[j] = freq_all
#          # freq_dist_no_stop[j] = freq_no_stop
       
        
# df_party_year['Words_All'] = line_words
# df_party_year['Words_Sans_Stop'] = no_stop

# for index, row in df_party_year.iterrows():
#     if index = 0:
#         result = list(row['Words_All'])
#         df_party_year.at[index, '']

# df_party_year['Frequency_All_Unique'] = cumulative_column
# df_party_year['Frequency_All_Cum'] = df_party_year['Frequency_All_Unique'].apply(len)   




#%%

## Lauren
## various readability scores & WordCloud (size contigent on word frequencies)
## by candidate

## KINKS TO WORK OUT: 
## 1. Same transcript lines still attributed to Hillary & Bill
## and George W. / H.W. -- is there a way to apply the mapping file to prevent this?
## 2. When I "join" Transcript rows, it actually places each row in infinitely many smaller chunks/columns
## so the readability scores and WordClouds are only based on one line's worth
## of text, as opposed to all of it

import nltk, re, pprint, pandas as pd, numpy as np
from nltk import word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords
import textstat
from wordcloud import WordCloud
from PIL import Image
import string
import plotly.express as px, plotly.io as pio, matplotlib.pyplot as plt

df = pd.read_table('Transcripts_finalb.csv', encoding = 'utf-16')
df = df.fillna('xx')
df = df.drop_duplicates(subset=["Actual Speaker", "Key", "Transcript", "Year"])

# remove all speaker names contained in "Transcript" column (words with more than one consecutive capital letter)
df['Transcript']=df['Transcript'].str.replace(r'[A-Z]+\s?[A-Z]+', '', regex=True)

# ignore â€™,  â€', etc. (em dashes and apostrophes encoded strangely)
df['Transcript']=df['Transcript'].str.encode("ascii", "ignore")
df['Transcript']=df['Transcript'].str.decode(encoding = 'utf-8')

df_cand = df[df.Purpose.eq('Candidate')]
#df_cand['Transcript Full'] = df_cand.groupby(['Actual Speaker'])['Transcript'].transform(lambda x: ' '.join(x))

#df_cand = df_cand.groupby(['Actual Speaker'])['Transcript'].apply(''.concat).reset_index()
#df_cand = df_cand.drop_duplicates(['Transcript Full'])
#df_party = df_2.groupby(['Affiliation'])['Transcript'].apply(' '.join).reset_index()

df_cand_2 = df_cand.groupby(['Actual Speaker'])['Transcript'].apply(' '.join).reset_index()

df_cand_2.to_csv('df_actual_speaker_2.csv')

cands_unique = df_cand_2['Actual Speaker'].unique()
#party_unique = df_party['Affiliation'].unique

for c in cands_unique:
    scores = {}
    
    df_new = df_cand_2[df_cand_2['Actual Speaker'] == c]
    c_string = df_new['Transcript'].to_string()
    
    scores['FK Reading Ease'] = textstat.flesch_reading_ease(c_string)
    scores['Smog Index'] = textstat.smog_index(c_string)
    scores['FK Grade Level'] = textstat.flesch_kincaid_grade(c_string)
    scores['Coleman Liau'] = textstat.coleman_liau_index(c_string)
    scores['Automated Readability'] = textstat.automated_readability_index(c_string)
    scores['Dale Chall Readability'] = textstat.dale_chall_readability_score(c_string)
    scores['Difficult Words'] = textstat.difficult_words(c_string)
    scores['Linsear Write Formula'] = textstat.linsear_write_formula(c_string)
    scores['Gunning Fog'] = textstat.gunning_fog(c_string)
    scores['Text Standard'] = textstat.text_standard(c_string)
    
    print(c, scores)
    print()
    
def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");

stop_words = set(stopwords.words('english'))

for d in cands_unique:
    punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
    
    df_new_2 = df_cand_2[df_cand_2['Actual Speaker'] == d]
    c_string_2 = df_new_2['Transcript'].to_string()
    
    for ele in c_string_2:
        if ele in punc:
            c_string_2 = c_string_2.replace(ele, " ")
            
    #c_string_2 = c_string.translate(str.maketrans('','', string.punctuation))
    words = nltk.tokenize.word_tokenize(c_string_2.lower())
    
    no_stop = [w.strip() for w in words if w.strip() not in stop_words]
    freq_all = FreqDist(words)
    freq_no_stop = FreqDist(no_stop)
    
    print(d, freq_all.most_common(10), freq_no_stop.most_common(10))
    
    wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, 
                         background_color='salmon', colormap='Pastel1', 
                         collocations=False).generate_from_frequencies(freq_no_stop)
    
    plot_cloud(wordcloud)
    print()
    
#df_freq = pd.DataFrame(list(freq.items()), columns = ["Word", "Frequency"])

# df = df.sort_values(by=['Year', 'Affiliation'])