# -*- coding: utf-8 -*-
"""
@author: Lauren Neal
Semester Project: Polisticians
WordCloud
"""

import nltk
import numpy as np
from nltk import word_tokenize
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from wordcloud import WordCloud
from PIL import Image

# read csv in
with open('debate_test.txt', encoding = "utf-8") as f:
    raw = f.read()

# separate all language into 'tokens' (break up string into words and punctuation)    
tokens = word_tokenize(raw)
tokens = [word.lower() for word in tokens]
# create NLTK text out of 'tokens' (allows us to perform all NLTK functions)
transcripts_all = nltk.Text(tokens)

# remove punctuation
import string
table = str.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in transcripts_all]

# list of all words (no punctuation, no numbers)
word_list = list(word for word in stripped if word.isalpha)

#word_tokenize accepts a string as an input, not a file. 
stop_words = set(stopwords.words('english'))

# strip stopwords from word list
words_list = [w.strip() for w in word_list if w.strip() not in stop_words]
 
def listToString(list):
    wordstring = " "
    return (wordstring.join(list))

# count word frequency
wordfreq = [words_list.count(w) for w in words_list]

# create dictionary of [words:counts]
zip_iterator = zip(words_list, wordfreq)
d = dict(zip_iterator)    

word_string = listToString(words_list)
       
def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");

wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, 
                       background_color='salmon', colormap='Pastel1', 
                       collocations=False).generate_from_frequencies(d)

plot_cloud(wordcloud)

# # Import image to np.array
# mask = np.array(Image.open('obama_mask.png'))

# wordcloud2 = WordCloud(width = 3000, height = 2000, random_state=1, 
#                        background_color='white', colormap='Greys', 
#                        collocations=False, mask=mask).generate(word_string)
# # Plot
# plot_cloud(wordcloud2)
# wordcloud2.to_file("wordcloud-obama.png")