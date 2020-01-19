import json
import sys
import os

try:
    import lda_helper
except ImportError:
    sys.path.insert(0, "C:\\dev\\ken_personal_projects\\lda_nlp")
finally:
    import LDA_wiki.wiki as wiki_rand



'''
A script to parse all of the the BBC news articles from Ken's U drive folder into a dictionary structure
then pickle the dictionary

This script will probably never be used again to be honest.
'''

dict_bbc = {}
bbc_topics = ["business", "entertainment", "politics", "sport", "tech"]

for topic in bbc_topics:
    count = 0
    title = topic
    dir_bbc_topic = dir_bbc_pages.format(topic)
    for file in os.listdir(dir_bbc_topic):
        try:
            f = open(os.path.join(dir_bbc_topic,file), encoding="utf8")
            dict_key_name = topic + "_" + str(count)
            dict_bbc[dict_key_name] = ""
            for line in f:
                dict_bbc[dict_key_name] = dict_bbc[dict_key_name] + line
                print(topic + "_" + str(count))
                print(topic + line)
        except UnicodeDecodeError:
            print("UnicodeDecodeError")
        finally:
            count = count + 1
            #dict_bbc[dict_key_name] = lda_helper.preprocess(dict_bbc[dict_key_name])
            dict_bbc[dict_key_name] = dict_bbc[dict_key_name]


lda_helper.save_pickle(dict_bbc, output_pickled)