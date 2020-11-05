# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 12:26:15 2020

@author: Jill Bennett
"""

import nltk, re, pprint, pandas as pd, numpy as np
from nltk import word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords
import textstat
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import string
import plotly.express as px, plotly.io as pio, matplotlib.pyplot as plt
# from gensim.summarization import summarize, keywords
# from pprint import pprint

pio.renderers.default='browser'
    
df = pd.read_csv('Transcripts_allyrs_and_candidate.csv')

df_cand = df[df.Purpose.eq('Candidate')]

cands_unique = df_cand['Actual Speaker'].unique()
party_unique = df_cand['Party'].unique()

def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");

for c in cands_unique:
    if type(c) is float:
        pass
    else:
        scores = {}
        df_new = df_cand[df_cand['Actual Speaker'] == c]
        trans = pd.Series(' '.join(df_new.Transcript).split())
        tokens = word_tokenize(trans.to_string(index=False))
    
        from nltk.corpus import stopwords
        stops = nltk.corpus.stopwords.words('english')
        name = c.lower().split()
        
        newStopWords = name + ['applause', 'laughter', 'anderson','crosstalk']
        for i in newStopWords:
            stops.append(i)
        
        token_alph = [token.lower() for token in tokens if token.isalpha()]
        transcripts = nltk.Text(token_alph)
        transcripts_all = nltk.Text(w for w in transcripts if w not in stops)

        fd = FreqDist(transcripts_all)
        sorted(fd)
        
        
        tagged = nltk.pos_tag(transcripts_all)
        
        nouns = [] 
        nouns = [word for (word, pos) in tagged if (pos == 'NN' or pos == 'NNS')]
        fd_nouns = FreqDist(nouns)
        
        adjectives = []
        adjectives = [word for (word, pos) in tagged if (pos == 'JJ')]
        fd_adj = FreqDist(adjectives)
        
        verbs = []
        verbs = [word for (word, pos) in tagged if (pos == 'VB' or pos == 'VBD' or pos == 'VBG' or pos == 'VBN' or pos == 'VBP' or pos == 'VBZ')]
        fd_verbs = FreqDist(verbs)
        
        scores['FK Reading Ease'] = textstat.flesch_reading_ease(trans.to_string(index=False))
        scores['Smog Index'] = textstat.smog_index(trans.to_string(index=False))
        scores['FK Grade Level'] = textstat.flesch_kincaid_grade(trans.to_string(index=False))
        scores['Coleman Liau'] = textstat.coleman_liau_index(trans.to_string(index=False))
        scores['Automated Readability'] = textstat.automated_readability_index(trans.to_string(index=False))
        scores['Dale Chall Readability'] = textstat.dale_chall_readability_score(trans.to_string(index=False))
        scores['Difficult Words'] = textstat.difficult_words(trans.to_string(index=False))
        scores['Linsear Write Formula'] = textstat.linsear_write_formula(trans.to_string(index=False))
        scores['Gunning Fog'] = textstat.gunning_fog(trans.to_string(index=False))
        scores['Text Standard'] = textstat.text_standard(trans.to_string(index=False))
    
        print(c, scores)
        print(c, 'Most Commonly Used Words: ', fd.most_common(10))
        print("Common Bigrams: ")
        print(transcripts.collocations())
        print(transcripts.concordance('nuclear'))
        # wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, 
        #                  background_color='salmon', colormap='Pastel1', 
        #                  collocations=False).generate_from_frequencies(fd)
    
        # plot_cloud(wordcloud)
        
        #pprint(summarize(tokens, word_count=20))
        # filename = "%s.jpg" % c
        # mask = np.array(Image.open(filename))
        
        noun_cloud = WordCloud(max_font_size=100,colormap="hsv").generate_from_frequencies(fd_nouns)
        plt.rcParams["figure.figsize"] = (16,12)
        plt.imshow(noun_cloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
        
        adj_cloud = WordCloud(max_font_size=100,colormap="hsv").generate_from_frequencies(fd_adj)
        plt.rcParams["figure.figsize"] = (16,12)
        plt.imshow(adj_cloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
        
        verb_cloud = WordCloud(max_font_size=100,colormap="hsv").generate_from_frequencies(fd_verbs)
        plt.rcParams["figure.figsize"] = (16,12)
        plt.imshow(verb_cloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
        
        print()
        
#%%

for p in party_unique:
    if type(p) is float:
        pass
    else:
        df_party = df_cand[df_cand['Party'] == p]
        party_trans = pd.Series(' '.join(df_party.Transcript).split())
        party_tokens = word_tokenize(party_trans.to_string(index=False))
        
        party_token_alph = [token.lower() for token in party_tokens if token.isalpha()]
        party_transcripts = nltk.Text(party_token_alph)
        party_transcripts_all = nltk.Text(w for w in party_transcripts if w not in stops)

        party_fd = FreqDist(party_transcripts_all)
        sorted(party_fd)
        
        
        party_tagged = nltk.pos_tag(party_transcripts_all)
        
        party_nouns = [] 
        party_nouns = [word for (word, pos) in party_tagged if (pos == 'NN' or pos == 'NNS')]
        party_fd_nouns = FreqDist(party_nouns)
        
        party_adjectives = []
        party_adjectives = [word for (word, pos) in party_tagged if (pos == 'JJ')]
        party_fd_adj = FreqDist(party_adjectives)
        
        party_verbs = []
        party_verbs = [word for (word, pos) in party_tagged if (pos == 'VB' or pos == 'VBD' or pos == 'VBG' or pos == 'VBN' or pos == 'VBP' or pos == 'VBZ')]
        party_fd_verbs = FreqDist(party_verbs)
        
        print(p, 'Most Commonly Used Words: ', party_fd.most_common(10))
        print("Common Bigrams: ")
        print(party_transcripts.collocations())
        
        from nltk.collocations import *
        bigram_measures = nltk.collocations.BigramAssocMeasures()
        trigram_measures = nltk.collocations.TrigramAssocMeasures()
        bigram_fd = nltk.FreqDist(nltk.bigrams(party_transcripts_all))
        trigram_fd = nltk.FreqDist(nltk.trigrams(party_transcripts_all))
        finder = BigramCollocationFinder.from_words(party_token_alph)
        finder_tri = TrigramCollocationFinder.from_words(party_token_alph)
        finder.apply_freq_filter(3)
        finder_tri.apply_freq_filter(3)
        print(finder.nbest(bigram_measures.pmi, 10))
        print(finder_tri.nbest(trigram_measures.pmi, 10))
       
        
        print(party_transcripts.concordance('nuclear'))
        # wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, 
        #                  background_color='salmon', colormap='Pastel1', 
        #                  collocations=False).generate_from_frequencies(fd)
    
        # plot_cloud(wordcloud)
        
        
        # filename = "%s.jpg" % c
        mask = np.array(Image.open('donkey.png'))
        mask_colors = ImageColorGenerator(mask)
        
        party_noun_cloud = WordCloud(max_font_size=150,colormap="hsv", mask=mask, background_color="white",
                                     random_state=42, width=mask.shape[1],
                                     height=mask.shape[0], color_func=mask_colors).generate_from_frequencies(party_fd_nouns)
        plt.rcParams["figure.figsize"] = (16,12)
        plt.imshow(party_noun_cloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
        
        party_adj_cloud = WordCloud(max_font_size=100,colormap="hsv").generate_from_frequencies(party_fd_adj)
        plt.rcParams["figure.figsize"] = (16,12)
        plt.imshow(party_adj_cloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
        
        party_verb_cloud = WordCloud(max_font_size=100,colormap="hsv").generate_from_frequencies(party_fd_verbs)
        plt.rcParams["figure.figsize"] = (16,12)
        plt.imshow(party_verb_cloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
        
        # fig = px.scatter(bigram_fd, x='tsne_1', y='tsne_2', hover_name='bigram', text='bigram', size='count', color='words', size_max=45
        #          , template='plotly_white', title='Bigram similarity and frequency', labels={'words': 'Avg. Length<BR>(words)'}
        #          , color_continuous_scale=px.colors.sequential.Sunsetdark)
        # fig.update_traces(marker=dict(line=dict(width=1, color='Gray')))
        # fig.update_xaxes(visible=False)
        # fig.update_yaxes(visible=False)
        # fig.show()

        # party_trigrams = WordCloud(max_font_size=100,colormap="hsv", mask=mask, background_color="white",
        #                              random_state=42, width=mask.shape[1],
        #                              height=mask.shape[0], color_func=mask_colors).generate_from_frequencies(trigram_fd)
        # plt.rcParams["figure.figsize"] = (16,12)
        # plt.imshow(party_trigrams, interpolation='bilinear')
        # plt.axis('off')
        # plt.show()
        
        print()
        
#%%
years_unique = df_cand['Year'].unique()

for y in years_unique:
        
        df_year = df_cand[df_cand['Year'] == y]
        year_trans = pd.Series(' '.join(df_year.Transcript).split())
        year_tokens = word_tokenize(year_trans.to_string(index=False))
        
        year_stops = nltk.corpus.stopwords.words('english')
        year_newStopWords = ['applause', 'crosstalk']
        for i in year_newStopWords:
            year_stops.append(i)
    
        year_transcripts = [token.lower() for token in year_tokens if token.isalpha()]
        year_transcripts_all = nltk.Text(w for w in year_transcripts if w not in year_stops)

        fd_year = FreqDist(year_transcripts_all)
        sorted(fd_year)
        print(fd_year.most_common(50))
        print()
        print(year_transcripts_all.dispersion_plot(["citizens", "democracy", "freedom", "Democrat", "Republican", "America"]))
        
#%%        

import nltk, re, pprint, pandas as pd, numpy as np
from nltk import word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords
import textstat
from wordcloud import WordCloud
from PIL import Image
import string
import plotly.express as px, plotly.io as pio, matplotlib.pyplot as plt

from nltk.text import Text

df = pd.read_csv('Transcripts_allyrs_and_candidate.csv')
test = pd.Series(' '.join(df.Transcript).split())
con_tokens = word_tokenize(test.to_string(index=False))
    
t = Text(con_tokens)
print(t.dispersion_plot(['women', 'nuclear', 'diversity', 'racist']))  
print(t.concordance('nuclear', width=100))
print(t.similar('nuclear'))

from nltk.corpus import stopwords
full_stops = nltk.corpus.stopwords.words('english')
    
full_transcripts = [token.lower() for token in con_tokens if token.isalpha()]
full_transcripts_all = nltk.Text(w for w in full_transcripts if w not in full_stops)

full_fd = nltk.FreqDist(full_transcripts_all)
sorted(full_fd)

print(full_fd)
print(full_transcripts_all.dispersion_plot(['women', 'nuclear', 'diversity', 'racist']))     

plt.figure(figsize=(16,5))
full_fd.plot(50)

#%%   


import scattertext as st
import re, io
from pprint import pprint
import pandas as pd
import numpy as np
from scipy.stats import rankdata, hmean, norm
import pytextrank, spacy
import os, pkgutil, json, urllib
from urllib.request import urlopen
from IPython.display import IFrame
from IPython.core.display import display, HTML
from scattertext import CorpusFromPandas, produce_scattertext_explorer
# display(HTML("<style>.container { width:98% !important; }</style>"))
nlp = spacy.load('en')

con_df = pd.read_csv('Transcripts_allyrs_and_candidates.csv')

c_df = con_df[con_df.Purpose.eq('Candidate')]

convention_df = c_df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Affiliation', 'Debate', 'Line Count',
       'Position', 'Purpose', 'Speaker_Clean', 'Speaker_standardized', 'word_count', 'Debate_date', 'Key',
       'Speaker_standardized_map', 'Year', 'Position_map',
       'Winner', 'Incumbent_president', 'Sentiment', 'comp_val',
       'neg_val', 'neu_val', 'pos_val', 'cum_sentiment', 'max_sentiment',
       'cum_wordcount', 'total_wordcount', 'sent per word',])

convention_df = convention_df.dropna(how='all')
convention_df.dropna(inplace=True)

convention_df = convention_df.assign(
    parse=lambda df: convention_df.Transcript.apply(nlp),
    party=lambda df: convention_df.Party.apply(
        {'Democrat': 'Democratic', 
         'Republican': 'Republican'}.get
    )
)

corpus = st.CorpusFromParsedDocuments(
    convention_df,
    category_col='Party',
    parsed_col='parse',
    feats_from_spacy_doc=st.PyTextRankPhrases()
).build(
).compact(
    st.AssociationCompactor(2000, use_non_text_features=True)
)
    
term_category_scores = corpus.get_metadata_freq_df('')
print(term_category_scores)    

term_ranks = np.argsort(
    np.argsort(-term_category_scores, axis=0), axis=0) + 1
metadata_descriptions = {
    term: '<br/>' + '<br/>'.join(
        '<b>%s</b> TextRank score rank: %s/%s' % (
            cat, 
            term_ranks.loc[term, cat], 
            corpus.get_num_metadata()
        )
        for cat in corpus.get_categories()
    )
    for term in corpus.get_metadata()
}

category_specific_prominence = term_category_scores.apply(
    lambda row: (row.Democrat 
                 if row.Democrat > row.Republican 
                 else -row.Republican),
    axis=1
)

html = st.produce_scattertext_explorer(
    corpus,
    category='Party',
    minimum_term_frequency=0,
    pmi_threshold_coefficient=0,
    width_in_pixels=1000,
    transform=st.dense_rank,
    metadata=corpus.get_df()['Actual Speaker'],
    scores=category_specific_prominence,
    sort_by_dist=False,
    use_non_text_features=True,
    topic_model_term_lists={term: [term] for term in         
                            corpus.get_metadata()},
    topic_model_preview_size=0,
    metadata_descriptions=metadata_descriptions,
    use_full_doc=True
)

file_name = 'first.html'
with open(file_name, 'w') as outf: outf.write(html)
rel_report_path = os.path.relpath(file_name)
display(IFrame(file_name, width=900, height=650))

html_2 = st.produce_scattertext_explorer(
    corpus,
    category='Democrat',
    category_name='Democratic',
    not_category_name='Republican',
    minimum_term_frequency=2,
    pmi_threshold_coefficient=0,
    width_in_pixels=1000,
    transform=st.dense_rank,
    use_non_text_features=True,
    metadata=corpus.get_df()['Actual Speaker'],
    term_scorer=st.RankDifference(),
    sort_by_dist=False,
    topic_model_term_lists={term: [term] for term in 
                            corpus.get_metadata()},
    topic_model_preview_size=0, 
    metadata_descriptions=metadata_descriptions,
    use_full_doc=True
)

file_name_2 = 'second.html'
with open(file_name_2, 'w') as outf: outf.write(html_2)
rel_report_path = os.path.relpath(file_name_2)
display(IFrame(file_name_2, width=900, height=650))


from scattertext import SampleCorpora, PhraseMachinePhrases, dense_rank, RankDifference, AssociationCompactor, produce_scattertext_explorer
from scattertext.CorpusFromPandas import CorpusFromPandas

corpus_2 = st.CorpusFromPandas(convention_df,
                             category_col='Party',
                             text_col='Transcript',
                             nlp=nlp).build()

html_3 = produce_scattertext_explorer(
    corpus_2,
    category='Democrat',
    category_name='Democratic',
    not_category_name='Republican',
    minimum_term_frequency=0, 
    pmi_threshold_coefficient=0,
    transform=st.dense_rank,
    metadata=corpus.get_df()['Actual Speaker'],
    term_scorer=st.RankDifference(),
    width_in_pixels=1000
)

file_name_3 = 'third.html'
with open(file_name_3, 'w') as outf: outf.write(html_3)
rel_report_path = os.path.relpath(file_name_3)
display(IFrame(file_name_3, width=900, height=650))

#%%     
  
fig = px.bar(long_bigram_df_tidy, title='Comparision: ' + ngrams_list[0] + ' | ' + ngrams_list[1], x='ngram', y='value'
             , color='variable', template='plotly_white', color_discrete_sequence=px.colors.qualitative.Bold
             , labels={'variable': 'Company:', 'ngram': 'N-Gram'})
fig.update_layout(legend_orientation="h")
fig.update_layout(legend=dict(x=0.1, y=1.1))
fig.update_yaxes(title='', showticklabels=False)
fig.show()