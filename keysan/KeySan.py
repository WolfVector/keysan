import inflect, re
from nltk import download
from nltk import data
from functools import lru_cache
from sklearn.feature_extraction.text import CountVectorizer

try:
    data.find("corpora/stopwords")
except LookupError:
    download("stopwords")

try:
    data.find("corpora/wordnet.zip")
except LookupError:
    download("wordnet")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def replace_phrase(string, src, dst):
    return string.replace(src, dst)

class KeySan:
    def __init__(self, cache=False, cache_size=5000):
        self.pipeline_transformation = []
            
        if cache:
            self.wnl = WordNetLemmatizer()
            self.p = inflect.engine()
            self.lemmatize = lru_cache(maxsize=cache_size)(self.wnl.lemmatize) # Cache for lemmatize 
            self.singular_noun = lru_cache(maxsize=cache_size)(self.p.singular_noun) # Cache for transforming words to singulars

    def add_pipe_transformation(self, callback, *args):
        self.pipeline_transformation.append([callback, args])   

    def get_keyphrases_by_regex(self, regex, textlist, stop_words="english", lower=True, transform_verbs=False, transform_plurals=False):
        phrases_dict = {}
        textlist_len = len(textlist)

        regex_list = regex
        if type(regex) == str:
            regex_list = [regex]
        
        # Load the stopwords from nltk if the argument is a string
        # Otherwise assume is a list with the stopwords
        if type(stop_words) == str:
            stop_words = stopwords.words(stop_words)
    
        for line in textlist:
            if lower: 
                line = line.lower() # To lowercase
            line = re.sub(r'[^\w\s]', ' ', line) # Remove punctuations
    
            if stopwords:
                line = line.split() # Convert the string to a list
                line = [word for word in line if word not in stop_words] # Remove stopwords
                line = " ".join(line)
    
            if transform_verbs:
                line = self.verbs_transformation(line) # Transform to simple tense
            if transform_plurals:
                line = self.plurals_transformation(line) # Transform to singular

            # Execute all the pipes added by the user
            for pipe in self.pipeline_transformation:
                line = pipe[0](line, *pipe[1])
    
            for regex_string in regex_list:
                match = re.search(regex_string, line) # Look for the substring using regex
        
                if match:
                    sub_string = match.group() # Get the substring
        
                    if sub_string in phrases_dict:
                        phrases_dict[sub_string] += 1
                    else:
                        phrases_dict[sub_string] = 1
    
                    break
        for key in phrases_dict:
            phrases_dict[key] = (phrases_dict[key] / textlist_len) * 100 # Get the percentage
    
        phrases_dict = sorted(phrases_dict.items(), key=lambda x:x[1], reverse=True) # Sort the results
        return phrases_dict

    def get_keyphrases_infront(self, string, textlist, number_of_words, stop_words="english", lower=True, transform_verbs=False, transform_plurals=False):
        return self.get_keyphrases_by_regex(f'{string}(\s+\w+){{{number_of_words}}}', textlist, stop_words, lower=lower, transform_verbs=transform_verbs, transform_plurals=transform_plurals)

    def get_keyphrases_behind(self, string, textlist, number_of_words, stop_words="english", lower=True, transform_verbs=False, transform_plurals=False):
        return self.get_keyphrases_by_regex(f'(\w+\s+){{{number_of_words}}}{string}', textlist, stop_words, lower=lower, transform_verbs=transform_verbs, transform_plurals=transform_plurals)

    def get_keyphrases_around(self, string, textlist, number_of_words, stop_words="english", lower=True, transform_verbs=False, transform_plurals=False):
        return self.get_keyphrases_by_regex(f'(\w+\s+){{{number_of_words}}}{string}(\s+\w+){{{number_of_words}}}', textlist, stop_words, lower=lower, transform_verbs=transform_verbs, transform_plurals=transform_plurals)

    def plurals_transformation(self, string):
        text = []
        string_splitted = string.split() # Convert the string to an array
        for word in string_splitted:
            text.append(self.singular_noun(word) or word) # Transform the words to singulars
    
        return " ".join(text)

    def verbs_transformation(self, string):
        text = []
        string_splitted = string.split() # Convert the string to an array
        for word in string_splitted:
            text.append(self.lemmatize(word, "v")) # Transform the words to the base verb
    
        return " ".join(text)

    def count_keyphrases(keyphrases_list):
        count = 0
        for keyphrase in keyphrases_list:
            count += keyphrase[1]
    
        return count

    def get_ngram(self, textlist, stop_words="english", ngram_range=(1,1)):
        coun_vect = CountVectorizer(stop_words=stop_words, binary=True, ngram_range=ngram_range)
        count_matrix = coun_vect.fit_transform(textlist)
        sum_words = count_matrix.sum(axis = 0) 
        
        words_freq = [(word, sum_words[0, i]) for word, i in coun_vect.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)

        return words_freq