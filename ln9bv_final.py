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
        
        scores['FK Reading Ease'] = textstat.flesch_reading_ease(df_new.Transcript.to_string(index=False))
        scores['Smog Index'] = textstat.smog_index(df_new.Transcript.to_string(index=False))
        scores['FK Grade Level'] = textstat.flesch_kincaid_grade(df_new.Transcript.to_string(index=False))
        scores['Coleman Liau'] = textstat.coleman_liau_index(df_new.Transcript.to_string(index=False))
        scores['Automated Readability'] = textstat.automated_readability_index(df_new.Transcript.to_string(index=False))
        scores['Dale Chall Readability'] = textstat.dale_chall_readability_score(df_new.Transcript.to_string(index=False))
        scores['Difficult Words'] = textstat.difficult_words(df_new.Transcript.to_string(index=False))
        scores['Linsear Write Formula'] = textstat.linsear_write_formula(df_new.Transcript.to_string(index=False))
        scores['Gunning Fog'] = textstat.gunning_fog(df_new.Transcript.to_string(index=False))
        scores['Text Standard'] = textstat.text_standard(df_new.Transcript.to_string(index=False))
    
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
        # mask = np.array(Image.open("donald.png"))
        
        # noun_cloud = WordCloud(max_font_size=100,colormap="hot", mask=mask, background_color="black",
        #                              random_state=42, width=mask.shape[1],
        #                              height=mask.shape[0]).generate_from_frequencies(fd_nouns)
        # plt.rcParams["figure.figsize"] = (16,12)
        # plt.imshow(noun_cloud, interpolation='bilinear')
        # plt.axis('off')
        # plt.show()
        
        # adj_cloud = WordCloud(max_font_size=100,colormap="Oranges", mask=mask, background_color="black",
        #                              random_state=42, width=mask.shape[1],
        #                              height=mask.shape[0]).generate_from_frequencies(fd_adj)
        # plt.rcParams["figure.figsize"] = (16,12)
        # plt.imshow(adj_cloud, interpolation='bilinear')
        # plt.axis('off')
        # plt.show()
        
        # verb_cloud = WordCloud(max_font_size=100,colormap="OrRd", mask=mask, background_color="black",
        #                              random_state=42, width=mask.shape[1],
        #                              height=mask.shape[0]).generate_from_frequencies(fd_verbs)
        # plt.rcParams["figure.figsize"] = (16,12)
        # plt.imshow(verb_cloud, interpolation='bilinear')
        # plt.axis('off')
        # plt.show()
        
        # print()
        
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
        mask = np.array(Image.open('elephant2.jpg'))
        mask_colors = ImageColorGenerator(mask)
        
        party_noun_cloud = WordCloud(max_font_size=150,colormap="hsv", mask=mask, background_color="white",
                                     random_state=42, width=mask.shape[1],
                                     height=mask.shape[0], color_func=mask_colors).generate_from_frequencies(party_fd_nouns)
        plt.rcParams["figure.figsize"] = (16,12)
        plt.imshow(party_noun_cloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
        
        party_adj_cloud = WordCloud(max_font_size=100,colormap="Reds").generate_from_frequencies(party_fd_adj)
        plt.rcParams["figure.figsize"] = (16,12)
        plt.imshow(party_adj_cloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
        
        party_verb_cloud = WordCloud(max_font_size=100,colormap="Reds").generate_from_frequencies(party_fd_verbs)
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

remove_words = ['bush: ', 'obama: ', 'romney: ', 'mr. nixon: ', 'mr. kennedy: ', 'mr. reagan: ',
 'mccain: ', 'dole: ', 'ryan: ', 'harris: ', 'pence: ', 'trump: ', 'clinton: ',
 'carter: ', 'cheney: ', 'palin: ', 'perot: ', 'biden: ', 'dukakis: ', 'gore: ',
 'kerry: ', 'edwards: ', 'kaine: ', 'quayle: ', 'anderson: ', 'mr. mondale: ', 
 'lieberman: ', 'ferraro: ', 'mr. carter: ', 'kemp: ', 'bentsen: ', 'reagan: ',
 'mondale: ', 'mr. ford: ', 'ford: ']

replace_with = ['','','','','','','','','','','','','','','','','','','','',
                '','','','','','','','','','','','','','','',]

remove_words_up = [w.upper() for w in remove_words]
con_df = pd.read_csv('Transcripts_allyrs_and_candidates.csv')

c_df = con_df[con_df.Purpose.eq('Candidate')]

term_df = c_df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Affiliation', 'Debate', 'Line Count',
       'Position', 'Purpose', 'Speaker_Clean', 'Speaker_standardized', 'word_count', 'Debate_date', 'Key',
       'Speaker_standardized_map', 'Year', 'Position_map',
       'Winner', 'Incumbent_president', 'Sentiment', 'comp_val',
       'neg_val', 'neu_val', 'pos_val', 'cum_sentiment', 'max_sentiment',
       'cum_wordcount', 'total_wordcount', 'sent per word',])

term_df = term_df.dropna(how='all')
term_df.dropna(inplace=True)
# term_df['Transcript'] = term_df['Transcript'].replace(to_replace=remove_words_up, value=replace_with, inplace=True, regex=False)
term_df['Transcript'] = term_df['Transcript'].str.replace('MR. NIXON:', '', regex=False)
term_df['Transcript'] = term_df['Transcript'].str.replace('MR. KENNEDY:', '', regex=False)
term_df['Transcript'] = term_df['Transcript'].str.replace('MR. REAGAN:', '', regex=False)
term_df['Transcript'] = term_df['Transcript'].str.replace('MR. MONDALE:', '', regex=False)
term_df['Transcript'] = term_df['Transcript'].str.replace('MR. CARTER:', '', regex=False)
term_df['Transcript'] = term_df['Transcript'].str.replace('MR. FORD:', '', regex=False)
term_df['Transcript'] = term_df['Transcript'].str.replace('BUSH:', '', regex=False)
term_df['Transcript'] = term_df['Transcript'].str.replace('OBAMA:', '', regex=False)
term_df['Transcript'] = term_df['Transcript'].str.replace('MCCAIN:', '', regex=False)
term_df['Transcript'] = term_df['Transcript'].str.replace('ROMNEY:', '', regex=False)
term_df['Transcript'] = term_df['Transcript'].str.replace('DOLE:', '', regex=False)
term_df['Transcript'] = term_df['Transcript'].str.replace('RYAN:', '', regex=False)
term_df['Transcript'] = term_df['Transcript'].str.replace('HARRIS:', '', regex=False)
term_df['Transcript'] = term_df['Transcript'].str.replace('PENCE:', '', regex=False)
term_df['Transcript'] = term_df['Transcript'].str.replace('CHENEY:', '', regex=False)
term_df['Transcript'] = term_df['Transcript'].str.replace('TRUMP:', '', regex=False)
term_df['Transcript'] = term_df['Transcript'].str.replace('CLINTON:', '', regex=False)
term_df['Transcript'] = term_df['Transcript'].str.replace('CARTER:', '', regex=False)
term_df['Transcript'] = term_df['Transcript'].str.replace('KERRY:', '', regex=False)
term_df['Transcript'] = term_df['Transcript'].str.replace('EDWARDS:', '', regex=False)
term_df['Transcript'] = term_df['Transcript'].str.replace('GORE:', '', regex=False)
term_df['Transcript'] = term_df['Transcript'].str.replace('PALIN:', '', regex=False)
term_df['Transcript'] = term_df['Transcript'].str.replace('BIDEN:', '', regex=False)
term_df['Transcript'] = term_df['Transcript'].str.replace('DUKAKIS:', '', regex=False)
term_df['Transcript'] = term_df['Transcript'].str.replace('KAINE:', '', regex=False)
term_df['Transcript'] = term_df['Transcript'].str.replace('QUAYLE:', '', regex=False)
term_df['Transcript'] = term_df['Transcript'].str.replace('ANDERSON:', '', regex=False)
term_df['Transcript'] = term_df['Transcript'].str.replace('LIEBERMAN:', '', regex=False)
term_df['Transcript'] = term_df['Transcript'].str.replace('FERRARO:', '', regex=False)
term_df['Transcript'] = term_df['Transcript'].str.replace('KEMP:', '', regex=False)
term_df['Transcript'] = term_df['Transcript'].str.replace('BENTSEN:', '', regex=False)
term_df['Transcript'] = term_df['Transcript'].str.replace('REAGAN:', '', regex=False)
term_df['Transcript'] = term_df['Transcript'].str.replace('MONDALE:', '', regex=False)
term_df['Transcript'] = term_df['Transcript'].str.replace('FORD:', '', regex=False)


term_df.to_csv('term_df.csv')

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

terms_df = pd.read_csv('term_df.csv')
term_df = terms_df.drop(columns='Unnamed: 0')
term_df = term_df.dropna(how='all')
term_df.dropna(inplace=True)
term_df['parsed'] = term_df.Transcript.apply(nlp)
term_df = term_df[term_df.Party != 'Independent']

# term_df = term_df.assign(
#     parse=lambda df: term_df.Transcript.apply(nlp),
#     party=lambda df: term_df.Party.apply(
#         {'Democrat': 'Democratic', 
#          'Republican': 'Republican'}.get
#     )
# )

corpus = (CorpusFromPandas(term_df,
                           category_col='Party',
                           text_col='Transcript',
                           feats_from_spacy_doc=PhraseMachinePhrases(),
                           nlp=spacy.load('en', parser=False))
          .build().compact(AssociationCompactor(4000)))

# from bokeh.io import push_notebook, show, output_notebook
# from bokeh.layouts import row 
# from bokeh.plotting import figure

# sc = ScatterChartBokeh(corpus_2)

# chart_dict = sc.to_dict(category='Democrat', category_name='Democratic', not_category_name='Republican')

html_12 = produce_scattertext_explorer(corpus,
                                    category='Democrat',
                                    category_name='Democratic',
                                    not_category_name='Republican',
                                    minimum_term_frequency=0,
                                    pmi_threshold_coefficient=0,
                                    transform=dense_rank,
                                    metadata=corpus.get_df()['Actual Speaker'],
                                    term_scorer=RankDifference(),
                                    width_in_pixels=1000)
open('./rank_noun-phrases_final.html', 'wb').write(html_12.encode('utf-8'))

corpus = st.CorpusFromParsedDocuments(
    term_df,
    category_col='Party',
    parsed_col='parsed',
    feats_from_spacy_doc=st.PyTextRankPhrases()
).build(
).compact(st.AssociationCompactor(2000, use_non_text_features=True)
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

open('./rank_2_final.html', 'wb').write(html_2.encode('utf-8'))

from scattertext import SampleCorpora, PhraseMachinePhrases, dense_rank, RankDifference, AssociationCompactor, produce_scattertext_explorer
from scattertext.CorpusFromPandas import CorpusFromPandas
from gensim.models import word2vec
from scattertext import SampleCorpora, word_similarity_explorer_gensim, Word2VecFromParsedCorpus
from scattertext.CorpusFromParsedDocuments import CorpusFromParsedDocuments

topic_corpus = CorpusFromParsedDocuments(term_df, category_col='Party', parsed_col='parsed').build()
model = word2vec.Word2Vec(size=300,
                          alpha=0.025,
                          window=5,
                          min_count=10,
                          max_vocab_size=None,
                          sample=0,
                          seed=1,
                          workers=1,
                          min_alpha=0.0001,
                          sg=1,
                          hs=1,
                          negative=0,
                          cbow_mean=0,
                          iter=1,
                          null_word=0,
                          trim_rule=None,
                          sorted_vocab=1)
html_4 = word_similarity_explorer_gensim(topic_corpus,
                                       category='Democrat',
                                       category_name='Democratic',
                                       not_category_name='Republican',
                                       target_term='nuclear',
                                       minimum_term_frequency=5,
                                       pmi_threshold_coefficient=4,
                                       width_in_pixels=1000,
                                       metadata=term_df['Actual Speaker'],
                                       word2vec=Word2VecFromParsedCorpus(topic_corpus, model).train(),
                                       max_p_val=0.05,
                                       save_svg_button=True)
open('./nuclear_final.html', 'wb').write(html_4.encode('utf-8'))


html_5 = word_similarity_explorer_gensim(topic_corpus,
                                       category='Democrat',
                                       category_name='Democratic',
                                       not_category_name='Republican',
                                       target_term='economy',
                                       minimum_term_frequency=5,
                                       pmi_threshold_coefficient=4,
                                       width_in_pixels=1000,
                                       metadata=term_df['Actual Speaker'],
                                       word2vec=Word2VecFromParsedCorpus(topic_corpus, model).train(),
                                       max_p_val=0.05,
                                       save_svg_button=True)
open('./economy_final.html', 'wb').write(html_5.encode('utf-8'))



corpus_2 = st.CorpusFromPandas(term_df,
                             category_col='Party',
                             text_col='Transcript',
                             nlp=nlp).build()

html_3 = produce_scattertext_explorer(
    corpus_2,
    category='Democrat',
    category_name='Democratic',
    not_category_name='Republican',
    minimum_term_frequency=10, 
    pmi_threshold_coefficient=1,
    transform=st.dense_rank,
    metadata=corpus_2.get_df()['Actual Speaker'],
    term_scorer=st.RankDifference(),
    width_in_pixels=1000
)

open('./rank_final.html', 'wb').write(html_3.encode('utf-8'))

term_freq_df = corpus_2.get_term_freq_df()
term_freq_df['Democratic Score'] = corpus_2.get_scaled_f_scores('Democrat')
pprint(list(term_freq_df.sort_values(by='Democratic Score', ascending=False).index[:10]))

term_freq_df['Republican Score'] = corpus_2.get_scaled_f_scores('Republican')
pprint(list(term_freq_df.sort_values(by='Republican Score', ascending=False).index[:10]))

html_11 = produce_scattertext_explorer(
    corpus_2,
    category='Democrat',
    category_name='Democratic',
    not_category_name='Republican',
    metadata=term_df['Actual Speaker'],
    term_scorer=st.RankDifference(),
    width_in_pixels=1000
)

open('./rank_3_final.html', 'wb').write(html_11.encode('utf-8'))



feat_builder = st.FeatsFromOnlyEmpath()
empath_corpus = st.CorpusFromParsedDocuments(term_df,
                                              category_col='Party',
                                              feats_from_spacy_doc=feat_builder,
                                              parsed_col='Transcript').build()
html_6 = st.produce_scattertext_explorer(empath_corpus,
                                        category='Democrat',
                                        category_name='Democratic',
                                        not_category_name='Republican',
                                        width_in_pixels=1000,
                                        metadata=empath_corpus.get_df()['Actual Speaker'],
                                        use_non_text_features=True,
                                        use_full_doc=True,
                                        topic_model_term_lists=feat_builder.get_top_model_term_lists())
open("empath_final.html", 'wb').write(html_6.encode('utf-8'))

#%%     
  
fig = px.bar(long_bigram_df_tidy, title='Comparision: ' + ngrams_list[0] + ' | ' + ngrams_list[1], x='ngram', y='value'
             , color='variable', template='plotly_white', color_discrete_sequence=px.colors.qualitative.Bold
             , labels={'variable': 'Company:', 'ngram': 'N-Gram'})
fig.update_layout(legend_orientation="h")
fig.update_layout(legend=dict(x=0.1, y=1.1))
fig.update_yaxes(title='', showticklabels=False)
fig.show()

#%%     
  
import spacy
import scattertext as st
import pandas as pd, numpy as np
from gensim.models import word2vec
from scattertext import SampleCorpora, word_similarity_explorer_gensim, Word2VecFromParsedCorpus
from scattertext.CorpusFromParsedDocuments import CorpusFromParsedDocuments
nlp = spacy.load('en')

con_df = pd.read_csv('Transcripts_allyrs_and_candidates.csv')
c_df = con_df[con_df.Purpose.eq('Candidate')]

topic_df = c_df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Affiliation', 'Debate', 'Line Count',
       'Position', 'Purpose', 'Speaker_Clean', 'Speaker_standardized', 'word_count', 'Debate_date', 'Key',
       'Speaker_standardized_map', 'Year', 'Position_map',
       'Winner', 'Incumbent_president', 'Sentiment', 'comp_val',
       'neg_val', 'neu_val', 'pos_val', 'cum_sentiment', 'max_sentiment',
       'cum_wordcount', 'total_wordcount', 'sent per word',])

topic_df = topic_df.dropna(how='all')
topic_df.dropna(inplace=True)
topic_df['parsed'] = topic_df.Transcript.apply(nlp)
topic_df_2 = topic_df[topic_df.Party != 'Independent']
topic_corpus = CorpusFromParsedDocuments(topic_df, category_col='Party', parsed_col='parsed').build()
model = word2vec.Word2Vec(size=200,
                          alpha=0.025,
                          window=5,
                          min_count=5,
                          max_vocab_size=None,
                          sample=0,
                          seed=1,
                          workers=1,
                          min_alpha=0.0001,
                          sg=1,
                          hs=1,
                          negative=0,
                          cbow_mean=0,
                          iter=1,
                          null_word=0,
                          trim_rule=None,
                          sorted_vocab=1)
html_4 = word_similarity_explorer_gensim(topic_corpus,
                                       category='Democrat',
                                       category_name='Democratic',
                                       not_category_name='Republican',
                                       target_term='nuclear',
                                       minimum_term_frequency=5,
                                       pmi_threshold_coefficient=4,
                                       width_in_pixels=1000,
                                       metadata=topic_df['Actual Speaker'],
                                       word2vec=Word2VecFromParsedCorpus(topic_corpus, model).train(),
                                       max_p_val=0.05,
                                       save_svg_button=True)
open('./nuclear.html', 'wb').write(html_4.encode('utf-8'))


html_5 = word_similarity_explorer_gensim(topic_corpus,
                                       category='Democrat',
                                       category_name='Democratic',
                                       not_category_name='Republican',
                                       target_term='economy',
                                       minimum_term_frequency=5,
                                       pmi_threshold_coefficient=4,
                                       width_in_pixels=1000,
                                       metadata=topic_df['Actual Speaker'],
                                       word2vec=Word2VecFromParsedCorpus(topic_corpus, model).train(),
                                       max_p_val=0.05,
                                       save_svg_button=True)
open('./economy.html', 'wb').write(html_5.encode('utf-8'))

# np.seterr(divide='ignore', invalid='ignore')

# feat_builder = st.FeatsFromOnlyEmpath()
# empath_corpus = st.CorpusFromParsedDocuments(topic_df_2,
#                                               category_col='Party',
#                                               feats_from_spacy_doc=feat_builder,
#                                               parsed_col='Transcript').build()
# html_6 = st.produce_scattertext_explorer(empath_corpus,
#                                         category='Democrat',
#                                         category_name='Democratic',
#                                         not_category_name='Republican',
#                                         width_in_pixels=1000,
#                                         metadata=topic_df_2['Actual Speaker'],
#                                         use_non_text_features=True,
#                                         use_full_doc=True,
#                                         topic_model_term_lists=feat_builder.get_top_model_term_lists())
# open("empath_final.html", 'wb').write(html_6.encode('utf-8'))


general_inquirer_feature_builder = st.FeatsFromGeneralInquirer()
gen_corpus = st.CorpusFromPandas(topic_df_2,
                                  category_col='Party',
                                  text_col='Transcript',
                                  nlp=st.whitespace_nlp_with_sentences,
                                  feats_from_spacy_doc=general_inquirer_feature_builder).build()

html_7 = st.produce_frequency_explorer(gen_corpus,
                                      category='Democrat',
                                      category_name='Democratic',
                                      not_category_name='Republican',
                                      metadata=topic_df_2['Actual Speaker'],
                                      use_non_text_features=True,
                                      use_full_doc=True,
                                      term_scorer=st.LogOddsRatioUninformativeDirichletPrior(),
                                      grey_threshold=1.96,
                                      width_in_pixels=1000,
                                      topic_model_term_lists=general_inquirer_feature_builder.get_top_model_term_lists(),
                                      metadata_descriptions=general_inquirer_feature_builder.get_definitions())
open("general_inquiry_final.html", 'wb').write(html_7.encode('utf-8'))

moral_foundations_feats = st.FeatsFromMoralFoundationsDictionary()
moral_corpus = st.CorpusFromPandas(topic_df_2,
                             category_col='Party',
                             text_col='Transcript',
                             nlp=st.whitespace_nlp_with_sentences,
                             feats_from_spacy_doc=moral_foundations_feats).build()
cohens_d_scorer = st.CohensD(moral_corpus).use_metadata()
term_scorer = cohens_d_scorer.set_categories('Democrat', ['Republican']).get_score_df()

html_8 = st.produce_frequency_explorer(
    moral_corpus,
    category='Democrat',
    category_name='Democratic',
    not_category_name='Republican',
    metadata=topic_df_2['Actual Speaker'],
    use_non_text_features=True,
    use_full_doc=True,
    term_scorer=st.CohensD(moral_corpus).use_metadata(),
    grey_threshold=0,
    width_in_pixels=1000,
    topic_model_term_lists=moral_foundations_feats.get_top_model_term_lists(),                
    metadata_descriptions=moral_foundations_feats.get_definitions()
)
open("moral_foundations_final.html", 'wb').write(html_8.encode('utf-8'))
#%%     
  
import spacy
import scattertext as st
import pandas as pd, numpy as np
from gensim.models import word2vec
from scattertext import SampleCorpora, word_similarity_explorer_gensim, Word2VecFromParsedCorpus
from scattertext.CorpusFromParsedDocuments import CorpusFromParsedDocuments
nlp = spacy.load('en')

con_df = pd.read_csv('Transcripts_allyrs_and_candidates.csv')
c_df = con_df[con_df.Purpose.eq('Candidate')]

sent_df = c_df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Affiliation', 'Debate', 'Line Count',
       'Position', 'Purpose', 'Speaker_Clean', 'Speaker_standardized', 'word_count', 'Debate_date', 'Key',
       'Speaker_standardized_map', 'Year', 'Position_map',
       'Winner', 'Incumbent_president', 'Sentiment', 'comp_val',
       'neg_val', 'neu_val', 'pos_val', 'cum_sentiment', 'max_sentiment',
       'cum_wordcount', 'total_wordcount', 'sent per word',])

sent_df = sent_df.dropna(how='all')
sent_df.dropna(inplace=True)
sent_df['parsed'] = sent_df.Transcript.apply(nlp)

sent_corpus = (st.CorpusFromPandas(term_df,
                               category_col='Party',
                               text_col='Transcript',
                               nlp=st.whitespace_nlp_with_sentences)
           .build()
           .get_unigram_corpus())

from scipy.stats import hmean 

term_freq_df = sent_corpus.get_unigram_corpus().get_term_freq_df()
term_freq_df = term_freq_df.drop(columns=['Independent freq'])
term_freq_df = term_freq_df[term_freq_df.sum(axis=1) > 0]

term_freq_df['pos_precision'] = (term_freq_df['Democrat freq'] * 1./
                                 (term_freq_df['Democrat freq'] + term_freq_df['Republican freq']))

term_freq_df['pos_freq_pct'] = (term_freq_df['Democrat freq'] * 1.
                                /term_freq_df['Democrat freq'].sum())

term_freq_df['pos_hmean'] = (term_freq_df
                             .apply(lambda x: (hmean([x['pos_precision'], x['pos_freq_pct']])
                                               if x['pos_precision'] > 0 and x['pos_freq_pct'] > 0 
                                               else 0), axis=1))
term_freq_df.sort_values(by='pos_hmean', ascending=False).iloc[:10]

from scipy.stats import norm

def normcdf(x):
    return norm.cdf(x, x.mean(), x.std ())

term_freq_df['pos_precision_normcdf'] = normcdf(term_freq_df.pos_precision)

term_freq_df['pos_freq_pct_normcdf'] = normcdf(term_freq_df.pos_freq_pct.values)

term_freq_df['pos_scaled_f_score'] = hmean([term_freq_df['pos_precision_normcdf'], term_freq_df['pos_freq_pct_normcdf']])

term_freq_df.sort_values(by='pos_scaled_f_score', ascending=False).iloc[:10]

term_freq_df['neg_precision_normcdf'] = normcdf((term_freq_df['Republican freq'] * 1./
                                 (term_freq_df['Republican freq'] + term_freq_df['Democrat freq'])))

term_freq_df['neg_freq_pct_normcdf'] = normcdf((term_freq_df['Republican freq'] * 1.
                                /term_freq_df['Republican freq'].sum()))

term_freq_df['neg_scaled_f_score'] = hmean([term_freq_df['neg_precision_normcdf'],  term_freq_df['neg_freq_pct_normcdf']])

term_freq_df['scaled_f_score'] = 0
term_freq_df.loc[term_freq_df['pos_scaled_f_score'] > term_freq_df['neg_scaled_f_score'], 
                 'scaled_f_score'] = term_freq_df['pos_scaled_f_score']
term_freq_df.loc[term_freq_df['pos_scaled_f_score'] < term_freq_df['neg_scaled_f_score'], 
                 'scaled_f_score'] = 1-term_freq_df['neg_scaled_f_score']
term_freq_df['scaled_f_score'] = 2 * (term_freq_df['scaled_f_score'] - 0.5)
term_freq_df.sort_values(by='scaled_f_score', ascending=True).iloc[:10]

is_pos = term_freq_df.pos_scaled_f_score > term_freq_df.neg_scaled_f_score
freq = term_freq_df.pos_freq_pct_normcdf*is_pos - term_freq_df.neg_freq_pct_normcdf*~is_pos
prec = term_freq_df.pos_precision_normcdf*is_pos - term_freq_df.neg_precision_normcdf*~is_pos
def scale(ar): 
    return (ar - ar.min())/(ar.max() - ar.min())
def close_gap(ar): 
    ar[ar > 0] -= ar[ar > 0].min()
    ar[ar < 0] -= ar[ar < 0].max()
    return ar

html_9 = st.produce_scattertext_explorer(
    sent_corpus.remove_terms(set(sent_corpus.get_terms()) - set(term_freq_df.index)),
    category='Democrat',
    not_category_name='Republican',
    not_categories=['Republican'],
    
    x_label = 'Frequency',
    original_x = freq,
    x_coords = scale(close_gap(freq)),
    x_axis_labels = ['Frequent in Rep', 
                     'Not Frequent', 
                     'Frequent in Dem'],
    
    y_label = 'Precision',    
    original_y = prec,
    y_coords = scale(close_gap(prec)),
    y_axis_labels = ['Rep Precise', 
                     'Imprecise', 
                     'Dem Precise'],
    
    
    scores = (term_freq_df.scaled_f_score.values + 1)/2,
    sort_by_dist=False,
    show_characteristic=False
)
open("freq_final.html", 'wb').write(html_9.encode('utf-8'))

#%%


corpus_sq = st.CorpusFromPandas(
	term_df,
	category_col='Party',
	text_col='Transcript',
	nlp=st.whitespace_nlp_with_sentences
).build().get_unigram_corpus()

term_df.Party = term_df.Party.apply\
	(lambda x: {'Democrat': 'Democrat', 'Republican': 'Republican', 'Independent': 'Independent'}[x])

semiotic_square = st.SemioticSquare(
	corpus_sq,
	category_a='Democrat',
	category_b='Republican',
	neutral_categories=['Independent'],
	scorer=st.RankDifference(),
	labels={'not_a_and_not_b': 'Independent', 'a_and_b': 'Across All Parties'}
)

html_10 = st.produce_semiotic_square_explorer(semiotic_square,
                                           category_name='Democrat',
                                           not_category_name='Republican',
                                           x_label='Dem',
                                           y_label='Repub',
                                           neutral_category_name='Independent',
                                           metadata=term_df['Actual Speaker'])
open("sem_sq.html", 'wb').write(html_10.encode('utf-8'))

#%%

from scattertext import SampleCorpora, PhraseMachinePhrases, dense_rank, RankDifference, AssociationCompactor, produce_scattertext_explorer
from scattertext.CorpusFromPandas import CorpusFromPandas
from gensim.models import word2vec
from scattertext import SampleCorpora, word_similarity_explorer_gensim, Word2VecFromParsedCorpus
from scattertext.CorpusFromParsedDocuments import CorpusFromParsedDocuments
import scattertext as st
import re, io
from pprint import pprint
import pandas as pd
import numpy as np
from scipy.stats import rankdata, hmean, norm
import pytextrank, spacy
import os, pkgutil, json, urllib

nlp = spacy.load('en')

terms_df_gender = pd.read_csv('term_df_gender.csv')
term_df_gender = terms_df_gender.dropna(how='all')
term_df_gender.dropna(inplace=True)
term_df_gender['parsed'] = term_df_gender.Transcript.apply(nlp)
term_df_gender = term_df_gender[term_df_gender.Party != 'Independent']

corpus_gender = st.CorpusFromParsedDocuments(term_df_gender, 
                                             category_col='Gender', 
                                             parsed_col='parsed').build()
html_13 = st.produce_scattertext_explorer(corpus_gender,
                                       category='Female',
                                       category_name='Female',
                                       not_category_name='Male',
                                       minimum_term_frequency=5,
                                       width_in_pixels=1000,
                                       metadata=term_df_gender['Party'])

open("gender_final.html", 'wb').write(html_13.encode('utf-8'))

female_scores = corpus_gender.get_scaled_f_scores('Female')
democratic_scores = (st.CorpusFromParsedDocuments(term_df_gender, 
                                                  category_col='Party', 
                                                  parsed_col='parsed')
                     .build()
                     .get_scaled_f_scores('Democrat'))

html_14 = st.produce_scattertext_explorer(corpus_gender,
                                       category='Female',
                                       category_name='Female',
                                       not_category_name='Male',
                                       minimum_term_frequency=5,
                                       pmi_filter_thresold=4,
                                       width_in_pixels=1000,
                                       scores=female_scores,
                                       sort_by_dist=False,
                                       x_coords=democratic_scores,
                                       y_coords=female_scores,
                                       show_characteristic=False,
                                       metadata=(term_df_gender['Actual Speaker'] 
                                                 + ' (' 
                                                 + term_df_gender['Party'].apply(lambda x: x.upper()[0]) 
                                                 + ')'),
                                       x_label='More Democratic',
                                       y_label='More Female')
open("gender-by-party_final.html", 'wb').write(html_14.encode('utf-8'))

# empath_corpus = st.CorpusFromParsedDocuments(term_df_gender,
#                                              category_col='Gender',
#                                              feats_from_spacy_doc=st.FeatsFromOnlyEmpath(),
#                                              parsed_col='Transcript').build()
# html_15 = st.produce_scattertext_explorer(empath_corpus,
#                                        category='Female',
#                                        category_name='Female',
#                                        not_category_name='Male',
#                                        width_in_pixels=1000,
#                                        metadata=term_df_gender['Party'],
#                                        use_non_text_features=True,
#                                        use_full_doc=True)

# open("gender-empath_final.html", 'wb').write(html_15.encode('utf-8'))
#%%
import nltk, re, pprint, pandas as pd, numpy as np
from nltk import word_tokenize
from nltk import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px, plotly.io as pio, matplotlib.pyplot as plt


pio.renderers.default='browser'

#nltk.download('vader_lexicon')

# read csv in
with open('Transcripts_allyrs_and_candidate.csv', encoding = "utf-8") as f:
    raw = f.read()
    
df = pd.read_csv('Transcripts_allyrs_and_candidate.csv')

df_cand = df[df.Purpose.eq('Candidate')]

cands_unique = df_cand['Actual Speaker'].unique()
party_unique = df_cand['Party'].unique()

trans = pd.Series(' '.join(df.Transcript).split())
tokens = word_tokenize(trans.to_string(index=False))
    
from nltk.corpus import stopwords
stops = nltk.corpus.stopwords.words('english')

newStopWords = ['applause', 'laughter', 'anderson','crosstalk']
for i in newStopWords:
    stops.append(i)
        
token_alph = [token.lower() for token in tokens if token.isalpha()]
transcripts = nltk.Text(token_alph)
transcripts_all = nltk.Text(w for w in transcripts if w not in stops)

fd = FreqDist(transcripts_all)
sorted(fd)

# separate all language into 'tokens' (break up string into words and punctuation)    

# printing length of "tokens" gives us total word count [INCLUDING duplicates and punctuation]
print(len(tokens)) # =1009388


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
print(fd['china']) # =172

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
lengthy = [w for w in transcripts_sans_punct if len(w) > 15]
print(sorted(lengthy))

# dispersion plot shows uses of specific words throughout the transcripts file
# x-axis represents the entire length of the transcripts file, markers indicate
# when words appear in the text
print(transcripts_all.dispersion_plot(["nuclear", "race", "freedom", "women", "pro-choice", 
                                       'hispanic', 'socialism', 'communism', 'china', 'fascism', 'russia', "guns", "america"]))

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