# -*- coding: utf-8 -*-
"""
@author: Lauren Neal
Semester Project: Polisticians
Preliminary NLTK Tests / Experimentation
"""


import nltk, re, pprint, pandas as pd, numpy as np
from nltk import word_tokenize
from nltk import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#nltk.download('vader_lexicon')

# read csv in
with open('Transcripts_final.csv', encoding = "utf-8") as f:
    raw = f.read()
    
df = pd.read_csv('Transcripts_final.csv') # call in for sentiment analysis

df['Transcript'] = df['Transcript'].fillna(value = 'xx')


# separate all language into 'tokens' (break up string into words and punctuation)    
tokens = word_tokenize(raw)

# printing length of "tokens" gives us total word count [INCLUDING duplicates and punctuation]
print(len(tokens)) # =1009388

# create NLTK text out of 'tokens' (allows us to perform all NLTK functions)
transcripts_all = nltk.Text(tokens)
fd = FreqDist(transcripts_all)
sorted(fd)

# most commonly appearing 'tokens' -- displays cumuluative sum up to 26%
cumulative = 0.0
most_common_words = [word for (word, count) in fd.most_common()]
for rank, word in enumerate(most_common_words):
    cumulative += fd.freq(word)
    print("%3d %6.2f%% %s" % (rank + 1, cumulative * 100, word))
    if cumulative > 0.25:
        break

# prints 50 most common words and the number of times they appear
print(fd.most_common(50))

# occurrences of word "China"
print(fd['China']) # =172

# count of unique tokens (still includes punctuation)
print(len(set(transcripts_all))) # =19488

#same as above, after converting all to lowercase      
print(len(set(word.lower() for word in transcripts_all))) # =17616

#same as above, after converting all to lowercase and without numbers or punctuation    
print(len(set(word.lower() for word in transcripts_all if word.isalpha()))) # =13529

# list of all words (no duplicates, no punctuation, no numbers)
transcripts_sans_punct = set(word.lower() for word in transcripts_all if word.isalpha)

# list of all words (no punctuation, no numbers)
transcripts_sans_punct_with_dups = list(word.lower() for word in transcripts_all if word.isalpha)

# create and sort list of all words longer than 12 letters
lengthy = [w for w in transcripts_sans_punct if len(w) > 12]
print(sorted(lengthy))

# dispersion plot shows uses of specific words throughout the transcripts file
# x-axis represents the entire length of the transcripts file, markers indicate
# when words appear in the text
print(transcripts_all.dispersion_plot(["citizens", "democracy", "freedom", "Democrat", "Republican", "America"]))

# collocations functions displays commonly appearing 'bigrams' (word pairings)
print(transcripts_all.collocations())

# processes a sequence of words, and attaches a part of speech tag to each word
# commented out because it takes a lot of computing power!
# debates_tagged = nltk.pos_tag(transcripts_sans_punct)
# print(debates_tagged)

# import pre-existing / pre-built set of the most commonly appearing words in 
# the English language to compare against
from nltk.corpus import stopwords
# print(stopwords.words('english'))

# function to determine percentage of words in transcripts that AREN'T 'stopwords'
def nonstop_percent(text):
    stops = nltk.corpus.stopwords.words('english')
    nonstop = [w for w in text if w.lower() not in stops]
    return (len(nonstop) / len(text) * 100)

print(nonstop_percent(transcripts_sans_punct_with_dups)) # =63.93%

# to find a pair of words
#print(transcripts_all.findall(r"<I> (<.*>) <am>"))

#%%

###### start sentiment analysis ######
# I will comment later #

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

df[['compound', 'neg', 'neu', 'pos']] = pd.DataFrame(df.Sentiment.values.tolist(), index= df.index)


#df_sent = pd.DataFrame(df['Sentiment'].tolist(), columns = ['compound', 'neg', 'neu', 'pos'])

df.to_csv('Transcripts_final_sent.csv')

