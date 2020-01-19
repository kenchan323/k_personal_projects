import sys
import os

try:
    from nlp_helper import save_pickle, preprocess
except ImportError:
    sys.path.insert(0, "C:\\dev\\k_personal_projects\\lda_nlp")
finally:
    from nlp_helper import save_pickle, preprocess

'''
A script to parse all of the the BBC news articles from Ken's U drive folder into a dictionary structure
then pickle the dictionary

This script will probably never be used again to be honest.
'''

dict_bbc = {}
bbc_topics = ["business", "entertainment", "politics", "sport", "tech"]
dir_bbc_pages = "C:\\dev\\k_personal_projects\\lda_nlp\\bbc_raw_articles"
path_output_pickle_raw = "C:\\dev\\k_personal_projects\\lda_nlp\\bbc_raw_articles\\pickled_bbc_tokenised_articles.obj"
path_output_pickle_tokenise = "C:\\dev\\k_personal_projects\\lda_nlp\\bbc_raw_articles\\pickled_bbc_raw_articles.obj"

load_raw_article = True

if load_raw_article:
    path_output_pickle = path_output_pickle_raw
else:
    path_output_pickle = path_output_pickle_tokenise

for topic in bbc_topics:
    count = 0
    title = topic
    dir_bbc_topic = os.path.join(dir_bbc_pages, topic)
    for file in os.listdir(dir_bbc_topic):
        try:
            f = open(os.path.join(dir_bbc_topic, file), encoding="utf8")
            dict_key_name = topic + "_" + file
            dict_bbc[dict_key_name] = ""
            for line in f:
                dict_bbc[dict_key_name] = dict_bbc[dict_key_name] + line
                print(topic + "_" + str(count))
                print(topic + line)
        except UnicodeDecodeError:
            print("UnicodeDecodeError")
        finally:
            count = count + 1
            if load_raw_article:
                # We don't process at all, just keep the original content of the article
                dict_bbc[dict_key_name] = dict_bbc[dict_key_name]
            else:
                # We tokenise the article
                dict_bbc[dict_key_name] = preprocess(dict_bbc[dict_key_name])

# Now pickle the file
save_pickle(dict_bbc, path_output_pickle)