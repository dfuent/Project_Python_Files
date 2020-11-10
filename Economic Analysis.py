import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# bring in data from csv files and store in dataframes for analysis
text_df = pd.read_csv('Transcripts_clean.csv') # debate transcript text data
econ_df = pd.read_csv('US_economic_data_annual.csv') # economic data 
map_df = pd.read_csv('map_file_final.csv') # mapping file showing party, winner/loser etc

# create a column to show when an incumbent won the current election to be used later for analysis
text_df['incumbent_win'] = np.where(np.logical_and(text_df['Winner']=='Yes',text_df['Incumbent_president']=='Yes'),1,0)
# add the economic data to the transcript dataframe 
text_df = pd.merge(text_df,econ_df,on='Year',how='left')
# group debates by whether an incumbent won or not and store the unemployment rate for each as a new column
text_df['unemployment_win'] = text_df.groupby(['Debate', 'incumbent_win'])['Unemployment'].transform('max')

# create new columns in text dataframe to count number of word mentions in each row of the transcripts 
text_df['Jobs Count'] = text_df.Transcript.str.count('JOBS')
text_df['Job Count'] = text_df.Transcript.str.count('JOB')
text_df['Economy Count'] = text_df.Transcript.str.count('ECONOMY')
text_df['Economic Count'] = text_df.Transcript.str.count('ECONOMIC')
text_df['Growth Count'] = text_df.Transcript.str.count('GROWTH')
text_df['Inflation Count'] = text_df.Transcript.str.count('INFLATION')
# text_df['Unemployment Count'] = text_df.Transcript.str.count('UNEMPLOYMENT') # checked but doesn't add much to analysis
# text_df['Work Count'] = text_df.Transcript.str.count('WORK') # difficult word to test as has many meanings so removed for now
text_df['Macro Count'] = text_df['Jobs Count'] + text_df['Economy Count'] + text_df['Growth Count'] # create a total econ word count
text_df['Macro Count + Inflation'] = text_df['Jobs Count'] + text_df['Economy Count'] + text_df['Growth Count'] + text_df['Inflation Count'] # create a total econ word count + inflation

# group sum of counts (and other data in dataframe for now) by Year to see total word counts in any given election year
yearGroup = text_df.groupby('Year', as_index=False).sum()
# add economic data to newly created dataframe with annual word counts sums
yearGroup = pd.merge(yearGroup,econ_df,on='Year',how='left')

# yearGroup.to_csv('somanytestfiles.csv')
# print(yearGroup)

# unemployment rate when incumbent wins or not
ax = sns.boxplot(x=text_df['incumbent_win'],y=text_df['Unemployment']) # boxplot shows range of unemployment rates when an uncumbent wins or not
ax.set(xlabel='Incumbent Victory. 1 is yes, 0 is no', ylabel='Unemployment rate %', title='Does a low unemployment rate help an incumbent candidate') # set axes and chart titles
plt.show()

# count of specific economic words (stacked) vs unemployment rate
ax1 = yearGroup[['Jobs Count','Job Count','Economy Count','Economic Count','Growth Count']].plot(kind='bar',stacked=True) # set LHS axis as stacked bar chart of word counts
ax2 = yearGroup['Unemployment_y'].plot(secondary_y=True, style='g') # set LHS as unemployment rate %
ax1.set_xlabel('Year') # x-axis label is Year
plt.xticks(yearGroup.index,yearGroup['Year'].values.astype(int)) # make sure Year shows in proper format and not as a float
ax1.set_ylabel('Number of economic words mentioned') # set LHS y-axis label
ax2.set_ylabel('Unemployment rate %') # set RHS y-axis label
plt.title('How often are economic words mentioned as compared to unemployment') # set plot title
plt.show()

# create new dataframe that groups word counts by Year and Party
partyGroup = text_df.groupby(['Year','Party'], as_index=False).sum()

# partyGroup.to_csv('Novembermillion.csv')

# economic words mentioned by party and vs unemployment 
ax3 = sns.barplot(x='Year', y='Macro Count', hue='Party', data=partyGroup, orient='v') # bar chart with each party as bar. use total Macro count instead of each word separately
ax3.set(xlabel='Year', ylabel='Number of economic words mentioned', title='Does Party influence economic word count') # set x-axis as year, LHS y-axis as word count and plot title
ax3.set_xticklabels(yearGroup['Year'].astype(int)) # make sure Year shows in proper format and not as a float
ax2_2 = yearGroup['Unemployment_y'].plot(secondary_y=True, style='g') # add secondary axis RHS as unemployment rate %
ax2_2.set_ylabel('Unemployment rate %') # set RHS y-axis label
plt.show()

# yearGroup.plot.bar(x='Year',y='Jobs Count',stacked=True)
ax1 = yearGroup[['Jobs Count','Job Count','Economy Count','Economic Count','Growth Count','Inflation Count']].plot(kind='bar',stacked=True) # show same stacked chart as before but add word inflation to count
ax2 = yearGroup['Unemployment_y'].plot(secondary_y=True, style='g') # set secondary axis RHS as unemployment rate %
plt.xticks(yearGroup.index,yearGroup['Year'].values.astype(int)) # make sure Year shows in proper format and not as a float
ax1.set_xlabel('Year') # x-axis label is Year
ax1.set_ylabel('Number of economic words mentioned') # set LHS y-axis label
ax2.set_ylabel('Unemployment rate %') # set RHS y-axis label
plt.title('How often are economic words mentioned now including "inflation"') # set plot title

plt.show()

# words as a % of total words
yearGroup['scale'] = yearGroup['Macro Count']/yearGroup['word_count'] # use total words in a year to scale total economic words
yearGroup['scale w inflation'] = yearGroup['Macro Count + Inflation']/yearGroup['word_count'] # use total words in a year to scale total economic words + inflation

# economic words as a % of total words vs unemployment
ax1_1 = yearGroup['scale'].plot(kind='bar') # bar chart shows scaled economic word usage as % of total words
ax2_1 = yearGroup['Unemployment_y'].plot(secondary_y=True, style='g') # set secondary axis RHS as unemployment rate %
plt.xticks(yearGroup.index,yearGroup['Year'].values.astype(int)) # make sure Year shows in proper format and not as a float
ax1_1.set(xlabel='Year', ylabel='Number of economic as % of total words', title='Does it appear words are used more frequently now because of longer debates') # set x-axis label, LHS y-axis label and chart title
ax2_1.set(xlabel='Year', ylabel='Unemployment rate %') # set RHS y-axis label
plt.show()

# economic words as a % of total words vs unemployment
ax1_1 = yearGroup['scale w inflation'].plot(kind='bar') # bar chart shows scaled economic word + inflation usage as % of total words
ax2_1 = yearGroup['Unemployment_y'].plot(secondary_y=True, style='g') # set secondary axis RHS as unemployment rate %
plt.xticks(yearGroup.index,yearGroup['Year'].values.astype(int)) # make sure Year shows in proper format and not as a float
ax1_1.set(xlabel='Year', ylabel='Number of economic as % of total words including "inflation"', title='Does it appear words now because of longer debates (with "inflation")') # set x-axis label, LHS y-axis label and chart title
ax2_1.set(xlabel='Year', ylabel='Unemployment rate %') # set RHS y-axis label
plt.show()
