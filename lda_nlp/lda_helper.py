from six.moves import cPickle as pickle
import gensim
import os
import pandas as pd
from gensim.models import Phrases
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import matplotlib.pyplot as plt
stemmer = SnowballStemmer("english")
import nltk
nltk.download('wordnet')
import spacy
# Load spacy EN model
nlp = spacy.load('en_core_web_sm')

'''
Some snippets borrowed from: 
https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24

These are some helper functions to support Ken's scripts on LDA (Latent Dirichlet allocation)
demonstrations.

 
'''

def lemmatize_stemming(text):
    '''
    Stemming of string
    '''
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# Tokenisation and wrapper function

def preprocess(text):
    '''
    Tokenization: Split the text into sentences and the sentences into words. Lowercase the words and
    remove punctuation.
        •Words that have fewer than 2 characters are removed.
        •All stopwords are removed.
        •Words are lemmatized — words in third person are changed to first person and verbs in past and
        future tenses are changed into present.
        •Words are stemmed — words are reduced to their root form.
    '''
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2:
            result.append(lemmatize_stemming(token))
    return result



def process_bigram(list_of_lists_of_stems, add_bigram=True):
    '''
    Detecting bi-gram in a list of lists of tokenised words, then either:
     - Add the bigram or (if add_birgram == True)
     - Replace the original unigram's with bigram's (if add_birgram == False)
    '''
    bigram = Phrases(list_of_lists_of_stems, min_count=5)
    if add_bigram:
        # Append the extra bigram
        for idx in range(len(list_of_lists_of_stems)):
            for token in bigram[list_of_lists_of_stems[idx]]:
                if '_' in token:
                    # Token is a bigram, append to document.
                    list_of_lists_of_stems[idx].append(token)
                    print(token)
    else:
        # Else we replace two uni-grams with bigrams
        list_of_lists_of_stems = [bigram[x] for x in list_of_lists_of_stems]
    return list_of_lists_of_stems



def print_detect_topic(text, model, dictionary, top_words=5):
    '''
    For a given text (in BoW representation) and a LDA model, print the topic distribution of it and
    the top n words by weight of those topics
    '''
    # Retrieving bag of words info (word index:word count) of headline
    bow_vector = dictionary.doc2bow(preprocess(text))
    for index, score in sorted(model[bow_vector], key=lambda tup: -1 * tup[1]):
        print("Score: {}\t Topic: {}".format(score, model.print_topic(index, top_words)))

def load_pickle(file):
    '''
    Loading a pickled file and return the underlying object
    '''
    with open(file, 'rb') as file:
        object_file = pickle.load(file)
    return object_file

def save_pickle(obj,file):
    '''
    Saving an object as a pickled file
    '''
    with open(file, 'wb') as file:
        pickle.dump(obj, file)


def classify(list_headlines, model, dict):
    '''
    Return a dictionary with keys being topic index (arbitrary), and values being lists of headlines
    that have been classified to be of that topic (soft classification, here I accept the topic
    that has the highest probability)


    list_headlines = a list of headlines
    model = trained LDA model
    dict = Bag of Worlds dictionary
    '''
    n_topic = len(model.get_topics())
    result = {}
    for n in range(n_topic):
        result[str(n)] = []
    for hl in list_headlines:
        bow_vector = dict.doc2bow(preprocess(hl))
        for index, score in sorted(model[bow_vector], key=lambda tup: -1 * tup[1]):
            #print("classed as " + str(index))
            result[str(index)].append(hl)
            break
    # Now we ensure all lists (values of the dictionary) are of the same length
    max_index = max(list(map(lambda x : len(result[x]), list(result.keys()))))
    print("max_index  " + str(max_index))
    for topic_idx in list(result.keys()):
        initial_len = len(result[topic_idx])
        if len(result[topic_idx]) < max_index:
            for x in range(initial_len, max_index, 1):
                result[topic_idx].append(" ")
    return result


def classify_bbc_dict(dict_bbc_articles, lda_model, dictionary, hide_article_idx=True):
    '''
    Return a dictionary with keys being topic index (arbitrary), and values being lists of headlines
    that have been classified to be of that topic (soft classification, here I accept the topic
    that has the highest probability)

    Parameters:
    bbc_dict = a dict structure with key = title, and val = list of strings
    model = trained LDA model (gensim)
    dict = Bag of Worlds dictionary
    '''
    n_topic = len(lda_model.get_topics())
    result = {}
    for n in range(n_topic):
        result[str(n)] = []
    for title in dict_bbc_articles.keys():
        content = dict_bbc_articles[title]
        bow_vector = dictionary.doc2bow(content)
        # lda_model[bow_vector][0] is pairs of set of the document level topic distribution.
        # Here we classify the document by whichever topic has the highest probability
        for index, score in sorted(lda_model[bow_vector][0], key=lambda tup: -1 * tup[1]):
            if hide_article_idx:
                result[str(index)].append(title.split("_")[0])
            else:
                result[str(index)].append(title)
            break
    # Now we ensure all lists (values of the dictionary) are of the same length
    max_index = max(list(map(lambda x : len(result[x]), list(result.keys()))))
    print("max_index  " + str(max_index))
    for topic_idx in list(result.keys()):
        initial_len = len(result[topic_idx])
        if len(result[topic_idx]) < max_index:
            for x in range(initial_len, max_index, 1):
                result[topic_idx].append(0)
    return result



def out_to_csv(dict_res, out_path):
    '''
    Ouput the returned result from classify() as a csv file
    '''
    df = pd.DataFrame(columns=list(dict_res.keys()))
    for topic_idx in list(dict_res.keys()):
        print(topic_idx)
        df[topic_idx] = dict_res[topic_idx]

    df.to_csv(out_path, index=False)


def out_to_excel(dict_res, out_path, df_param=None):
    '''
    Ouput the returned result from classify() as a xlsx file
    '''
    df = pd.DataFrame(columns=list(dict_res.keys()))
    for topic_idx in list(dict_res.keys()):
        print(topic_idx)
        df[topic_idx] = dict_res[topic_idx]

    writer = pd.ExcelWriter(out_path, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name="classified")
    if df_param is not None:
        df_param.to_excel(writer, sheet_name="param", index=False)
    writer.save()


def pos_tag_filter_for_noun_pronoun(content):
    '''
    Run POS tagging on a string and return tokens that are either classified as a NOUN or a
    PROPERNOUN
    '''
    doc = nlp(content)
    pos_result = [(token.text, token.pos_) for token in doc]
    return [token[0] for token in pos_result if (token[1] == "NOUN") or (token[1] == "PROPN")]






def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        print("Running model for k = {}".format(num_topics))
        model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        model_list.append(model)
        coherencemodel = gensim.models.CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')

        # coherencemodel = gensim.models.CoherenceModel(model=model, texts=texts,
        #                                              dictionary=dictionary, coherence='u_mass')
        coherence_values.append(coherencemodel.get_coherence())
        print("Running model for k = {} .... Done".format(num_topics))

    return model_list, coherence_values



def word_frequency(dictionary, stemmed_tokens, abs_count=False):
    '''
    If abs_count == False, we get document frequency for each word (as a percentage of corpus size)
    This function returns a DataFrame which contains the word frequency of each word in the vocab
    '''
    set_unique_words = set(dictionary.values())
    dict_word_freq = {}
    corpus_size = len(stemmed_tokens)
    for unique_word in set_unique_words:
        # Iniitialise count to 0 for each word
        dict_word_freq[unique_word] = 0

    for doc_content in stemmed_tokens:
        counted_words = set()
        for token in doc_content:
            if abs_count:
                # Hence we add to count for each word depending on the number of occurrence in doc
                dict_word_freq[token] = dict_word_freq[token] + 1
            else:
                if token not in counted_words:
                    # Only count unique occurrence in each doc (disregard actual count in a doc)
                    dict_word_freq[token] = dict_word_freq[token] + 1
                    counted_words.add(token)
    if not abs_count:
        for word in dict_word_freq.keys():
            # If interested in doc frequency, we normalise by corpus size
            dict_word_freq[word] = dict_word_freq[word] / corpus_size
    if abs_count:
        column_name = "count"
    else:
        column_name = "doc_frequency"
    # Return a dataframe with row index being each word
    return pd.DataFrame.from_dict(dict_word_freq, orient="index", columns=[column_name]).sort_values(column_name,
                                                                                                     ascending=False)







def coherence_scores_plot(dictionary, bow_corpus, list_of_content, start=10, step=1, limit=30):
    '''
    Plot the topic coherence score over a range of k (number of topics)
    '''
    model_list, coherence_values = compute_coherence_values(dictionary=dictionary,
                                                            corpus=bow_corpus,
                                                            texts=list_of_content, limit=limit,
                                                            start=start, step=step)

    x = range(start, limit, step)
    plt.figure()
    plt.plot(x, coherence_values)
    plt.xticks(x)
    plt.title("Topic Coherence Score (Section 7a Documents)", fontsize=15)
    plt.xlabel("Num Topics (k)")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()


def remove_month_string(list_of_content):
    '''
    Remove any token that is the (stemmed) name of a month
    '''

    month_stems = ["jan", "februari", "march", "april", "june", "juli", "august", "septemb",
                   "octob", "novem", "decemb"]
    for content_id in range(len(list_of_content)):
        for token in list_of_content[content_id]:
            if token in month_stems:
                list_of_content[content_id].remove(token)
    return list_of_content






