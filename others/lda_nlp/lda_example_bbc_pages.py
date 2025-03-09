import sys
import os
import pandas as pd

# NLP packages
import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from gensim.models import Phrases
stemmer = SnowballStemmer("english")
import nltk
nltk.download('wordnet')


try:
    import nlp_helper as helper
except ImportError:
    sys.path.insert(0, "/others/lda_nlp")
finally:
    import nlp_helper as helper


dir_lda = "/others/lda_nlp"
'''
Very basic script to do apply LDA (Latent Dirichlet allocation) on either:
Around 2500 BBC news article between 2004 and 2005 from the BBC Datasets website
    Source: http://mlg.ucd.ie/datasets/bbc.html
    - 5 default categories (business, tehcnology, entertainment, sport, politics)
    - The known, true distribution of news articles is as below:
    - bbc_topic_apriori = {Entertainment:0.229213483, 
                           Tech: 0.173483146, 
                           Politics: 0.229662921, 
                           Business: 0.18741573, 
                           Sport:0.180224719}
                           
The script then outputs to a spreadsheet which shows the results of the topic modelling. An article 
is classified as the ONE topic that has the highest probability over the distribution

'''

'''
User specified parameters below
'''
output_path_xlsx_pre = "C:\\dev\\ken_personal_projects\\lda_nlp\\results\\lda_results_bbc_pages_res.xlsx"

# If use_pre_processed = True, we start with the pre-processed corpus, do POS tag and retain only
# NOUN and PROPN (proper noun). This gives you the freedom to attempt different methods of pre-processing
# If use_pre_processed = False, then we use a pickled post-process (by Ken) corpus, which is already
# lemmatised, meaning we have lost the information needed to do POS tagging
use_unprocessed_corpus = True
# If you want to check for bi-gram in the text
do_bigram = True
# Number of topics to trained the LDA model for
k = 5
# In pre-processing the corpus and building dictionary, ignore words that appear in less than 50 docs
no_word_from_less_than_doc = 50
# In preprocessing the corpus and building dictionary, ignore words that appear in l> 40% of docs
no_word_from_more_than_pct_doc = 0.4
'''
User specified parameters above
'''

'''
The path to the pickled dictionary where key = article_category_id, value = post-processed 
list of (stemmed and lemmatised) word strings
'''
pickled_raw_bbc_pages = os.path.join(dir_lda, "bbc_raw_articles/pickled_bbc_raw_articles.obj")
pickled_stem_token_bbc_pages = os.path.join(dir_lda, "bbc_raw_articles/pickled_bbc_tokenised_articles.obj")
pickled_bbc_noun_stemmed_w_bigram = os.path.join(dir_lda,
                                                 "bbc_raw_articles/pickled_bbc_article_dict_stemmed_w_bigram.obj")

if use_unprocessed_corpus:
    # Load a dictionary containing the raw articles (un-tokenised)
    dict_bbc_articles = helper.load_pickle(pickled_raw_bbc_pages)
    # Load a temporary tokenised copy for doing bigram
    dict_bbc_articles_processed = helper.load_pickle(pickled_stem_token_bbc_pages)

    if os.path.exists(pickled_bbc_noun_stemmed_w_bigram):
        # ....or just load a pickled dictionary that Ken had previously done
        dict_bbc_articles_stemmed = helper.load_pickle(pickled_bbc_noun_stemmed_w_bigram)
    else:
        # or we build it!
        dict_bbc_articles_stemmed = {}
        dict_extra_bigram = {}
        for tag, article in dict_bbc_articles.items():
            dict_extra_bigram[tag] = []
            dict_bbc_articles_stemmed[tag] = []

        # For each article, find bigram
        bigram = Phrases(dict_bbc_articles_processed.values(), min_count=5)
        for id, content in dict_bbc_articles_processed.items():
            for token in bigram[content]:
                if '_' in token:
                    dict_extra_bigram[id].append(token)

        # Then we do POS tagging on the original pre-processed text and keep only NOUN and PROPNOUN
        for id, content in dict_bbc_articles.items():
            # Keep only nouns and pronouns
            list_nouns = helper.pos_tag_filter_for_noun_pronoun(content)
            for noun in list_nouns:
                # Lemmatise each noun/pronoun
                dict_bbc_articles_stemmed[id].append(stemmer.stem(WordNetLemmatizer().lemmatize(noun, pos='n')))
            dict_bbc_articles_stemmed[id].extend(dict_extra_bigram[id])
        # Let's pickle what we have done
        helper.save_pickle(dict_bbc_articles_stemmed, pickled_bbc_noun_stemmed_w_bigram)


    # Remove all words that only have 1 character
    for title, content in dict_bbc_articles_stemmed.items():
        dict_bbc_articles_stemmed[title] = [word for word in content if len(word) > 1]

    # list_of_content is a list of lists
    list_of_content = [v for k, v in dict_bbc_articles_stemmed.items()]


else:
    # otherwise if use_unprocessed_corpus == False
    # So we just use a processed (stemmed and tokenised) corpus
    # Dictionary of lists with already stemmed words
    dict_bbc_articles = helper.load_pickle(pickled_stem_token_bbc_pages)

    # list_of_content is a list of list of words
    list_of_content = [v for k, v in dict_bbc_articles.items()]
    if do_bigram:
        # Add bigrams and trigrams to docs (only ones that appear 5 times or more).
        bigram = Phrases(list_of_content, min_count=5)
        for idx in range(len(list_of_content)):
            for token in bigram[list_of_content[idx]]:
                if '_' in token:
                    # Token is a bigram, append to document.
                    list_of_content[idx].append(token)



# df_param is a DataFrame to keep track of the parameter we set. This will be a sheet in the output spreadsheet
df_param = pd.DataFrame(columns=["k",
                                 "rm_words_seen_in_less_than_n_doc",
                                 "rm_words_seen_in_more_than_n_pct_doc",
                                 "added_bigram",
                                 "use_pre_processed_pos_noun"])
df_param.loc[0] = [k, no_word_from_less_than_doc, no_word_from_more_than_pct_doc, do_bigram, use_unprocessed_corpus]

'''
for idx in range(len(list_of_content)):
    list_of_content[idx] = bigram[list_of_content[idx]]
'''

# Bag of Words on the data set (get all the UNIQUE words in the corpus)
dictionary = gensim.corpora.Dictionary(list_of_content)


# Filter out tokens in the dictionary that appear:
#     - less than {no_word_from_less_than_doc} documents (absolute number) or
#     - more than {no_word_from_more_than_pct_doc} documents (fraction of total corpus size, not absolute number).
#     - after the above two steps, keep only the first 100000 most frequent tokens.
dictionary.filter_extremes(no_below=no_word_from_less_than_doc, no_above=no_word_from_more_than_pct_doc)

# Training LDA using Bag of Words
# LDA model learning
#   BOW representation of each document in the corpus
print("Creating BOW representation of each document in the corpus...")
# bow_corpus == [...[(178,1),(2,5)],[(13,3),(198,1)]...]
bow_corpus = [dictionary.doc2bow(doc) for doc in list_of_content]
print("Creating BOW representation of each document in the corpus...DONE")
#  It's been suggested on the gensim page that workers is best set to #cpu_cores minus 1
# Politics, Business, Tech, Sport, Entertainment,
# Random state==1 for reproducible results
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=k,
                                       id2word=dictionary, passes=2,
                                       workers=3,
                                       random_state=1,
                                       eta="auto",
                                       alpha="asymmetric",
                                       per_word_topics=True)


# Printing out the topics (and top 10 words for each) for illustration
for idx, topic in lda_model.print_topics(num_words=10):
    print('Topic: {} \nWords: {}'.format(idx, topic))

# Now classify the original doc
dict_bbc_articles = helper.load_pickle(pickled_raw_bbc_pages)
# With hide_article_idx=True, in the output xlsx we only see each document's category, but not ID
result = helper.classify_bbc_dict(dict_bbc_articles, lda_model, dictionary, hide_article_idx=False)
helper.out_to_excel(result, output_path_xlsx_pre, df_param)

print("Results successfully output to {}".format(output_path_xlsx_pre))


'''
Below is attempt to applying a-priori hyper-parameter but not much difference seen in the results 
(because the priori probability of each genre are too similar, we have a really even sample)

# bbc_topic_apriori = Entertainment, Tech, Politics, Business, Sport
bbc_topic_apriori = [0.18741573,0.229213483,0.180224719,0.229662921,0.173483146]
# You can see they all genres basically take up about 20% each within the corpus ! Hence why the 
# below might be rather pointless/ineffective

# Specifying alpha (apriori belief) didnt do much as they are almost all 20% anyway so
lda_model.init_dir_prior(bbc_topic_apriori, name="alpha")
result = helper.classify_wiki_dict(dict_bbc_articles, lda_model, dictionary)
helper.out_to_csv(result, output_path_csv_post)
'''

'''
output_path_csv_alpah = "C:\\dev\\ken_personal_projects\\lda_nlp\\bbc_pages\\lda_results_bbc_pages_{}_applied.csv"
output_path_validation_alpah = "C:\\dev\\ken_personal_projects\\lda_nlp\\bbc_pages\\validation_{}_applied.xlsx"
for alp_value in np.arange(0, 1, 0.2):
    alp = 3*[0.1]
    alp.append(alp_value)
    lda_model.init_dir_prior(alp, name="alpha")

    result = helper.classify_wiki_dict(dict_bbc_articles, lda_model, dictionary)
    helper.out_to_csv(result, output_path_csv_alpah.format(alp))

    col_actual = ["entertainment", "tech", "politics", "business", "sport"]
    evaluate(output_path_csv_alpah.format(alp), output_path_validation_alpah.format(alp),col_actual)

    df_out.to_excel(validation_output)
'''
