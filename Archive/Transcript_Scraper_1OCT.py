# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 10:12:57 2020

Presidential and Vice Presidential Debate scraper

"""
from bs4 import BeautifulSoup, SoupStrainer
from urllib.request import Request, urlopen
import pandas as pd
import time
import io
from selenium import webdriver
from ftfy import fix_encoding

def transcript_scraper():
    
    #Option so that selenium doesn't open a new Chrome window
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    
    t_0 = time.time()
    
    # input headers to bypass issue loading transcript site
    
    hd = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
           'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
           'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
           'Accept-Encoding': 'none',
           'Accept-Language': 'en-US,en;q=0.8',
           'Connection': 'keep-alive'}
    
    # URL base
    root = 'https://www.debates.org/voter-education/debate-transcripts'
    
    # send request to site with headers to bypass Forbidden issue
    req = Request(root, headers = hd)
    
    # read site
    
    html_page = urlopen(req).read()
    
    # create HTML "soup"
    
    soup = BeautifulSoup(html_page, "lxml")
    
    # initiate web driver for Chrome
    driver = webdriver.Chrome(options=options)
    
    #use driver to open url
    driver.get(root)
    
    
    links = []
    for link in soup.findAll('a'):
        links.append(str(link.get('href')))
    
    
    t = [i for i in links if 'transcript' in i]
    
    t = list(set(t))
      
    
    fin_list = []
    
    for i in t:
        
        loop_time = time.time()

        url = 'https://www.debates.org/' + str(i)
        #print(url)
    
        
        #Option so that selenium doesn't open a new Chrome window
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        
        #initiate web driver
        driver = webdriver.Chrome(options=options)
        
        #use driver to open url
        driver.get(url)
        
        #wait three seconds to load page (probably not necessary)
        time.sleep(3)
        
        #extract page HTML and parse with BeautifulSoup
        html=driver.page_source
        soup=BeautifulSoup(html,'html.parser')
        
        #f = io.open('debate_final.txt', 'a', encoding = 'utf-8') # open file for appending ('a')
        
        h = soup('h1')
        h = str(h)[1:-1].replace('<h1>', '').replace('</h1>', '')
        print(h)
          
        tr = str(soup('p'))
        spl_tr = tr.split('</p>')
        
        l = 1
        speaker = ''
        for j in spl_tr:
            fix_encoding(j)
            j = j.replace('<p>', '')
            j = j.replace('</p>', '')
            j = j[2:].strip()

            if j.split(' ', 1)[0].strip() in['MR.', 'MS.', 'MRS.' ]:  
                temp = j.split(' ', 1)[1].strip()
                first_word = temp.split(' ', 1)[0]
                #print(type(first_word), first_word)
            else:
                
                first_word = j.split(' ', 1)[0].strip()
                
            try: 
                    
                last_char = first_word[-1]
                
            except:
                last_char = ''

            #print(last_char)
            try:
            
                if last_char == ':' and first_word.upper() == first_word:
                    #print(True)
                    speaker = first_word.replace(':', '')
                    #print(speaker)
            except:
                
                continue
                    
            fin_list.append((l, h, speaker, j)) 
            #f.write(str(l) +',' + h + ',' + speaker + ',' +  j)
            
            l += 1
    
        print('{0} loops for {1} took {2: .2f} seconds.'.format(l, h, time.time()-loop_time))

     
        ######################################     
    
    df = pd.DataFrame(fin_list, columns = ['Line Count', 'Debate', 'Speaker', 'Transcript'])
    df.to_csv('Transcripts_df.csv', index = False)    
        
    print('Finished in {0: .2f} seconds'.format(time.time()-t_0))
    return fin_list





#%%

transcript_scraper()



#%%
