
# coding: utf-8

# In[5]:

import os
import sys
import numpy as np
import nltk
import re
from bs4 import BeautifulSoup
from gensim import models
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords


dirname = './data/lyrics'
# Needed only if you want to remove stop words.
# nltk.download()

# Create genre mapping.
genre_tag = {
    'classical':1, 
    'folk':2, 
    'jazz and blues':3, 
    'pop':4, 
    'soul and reggae':5,
    'pink':6,
    'metal':7,
    'hip-hop':8,
    'dance and electronica':9,
    'classic pop and rock':10
}


def process_song(song, remove_stopwords = True):
    # Function to convert raw song lyrics to a sequence of words,
    # optionally removing stop words. Returns a list of words.
    #
    # 1. Remove HTML
    song_text = BeautifulSoup(song).get_text()
    #  
    # 2. Remove \\n, separate out comman and ! symbols from words \
    # and remove rest of the characters.
    song_text = re.sub(r"\\n"," ", song_text)
    # TODO: Should we keep comman and ! ??
    song_text = re.sub("(,|!)",r" \1", song_text)
    song_text = re.sub("[^a-zA-Z',!]"," ", song_text)
    #
    # 3. Convert words to lower case and split them
    words = song_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    #    return(words)
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join(words)) 


def write_features(train_features,test_features,train_file,test_file):
    with open(train_file, 'wb') as f:
        for i in enumerate(train_features):
            feature = features[id_]
            features_mat[:len(feature) , i, :] = np.asarray(feature)
            f.write(str(id_) + ' ' + str(labels[id_]) + ' ' + str(songs_len[id_]) + ' ' + str(features_len[id_]) + '\n')
    np.save(features_file, features_mat)


def prepare_data(test_train_split, train_features_file, test_features_file):
    vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 1000) 
    song_id = 0
    clean_train_songs=[]
    clean_test_songs=[]
    clean_train_labels=[]
    clean_test_labels=[]
    for genre in os.listdir(dirname):
        print("\nProcessing Genre: " + genre)
        songs = os.listdir(os.path.join(dirname, genre))
        num_songs = len(songs)
        song_index = 0 # Index of song within this genre
        for song in songs:
	    print(song)
            with open(os.path.join(dirname, genre, song), 'r') as song_lyrics:
                lyrics = song_lyrics.read()
                words = process_song(lyrics)
		print 'NumSongs: %d SongIndex %d SongId %d' % (num_songs, song_index, song_id)
                if (song_index  + 1) <= test_train_split * num_songs:
			clean_train_songs.append(words)
			clean_train_labels.append(genre)
		else:
			clean_test_songs.append(words)
			clean_test_labels.append(genre)
                    
            song_index = song_index + 1
            song_id = song_id + 1
    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of 
    # strings.
    train_data_features = vectorizer.fit_transform(clean_train_songs)
    test_data_features = vectorizer.fit_transform(clean_test_songs)

    # Numpy arrays are easy to work with, so convert the result to an 
    # array
    train_data_features = train_data_features.toarray()
    test_data_features = test_data_features.toarray()
    print train_data_features.shape
    print test_data_features.shape

    # Initialize a Random Forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators = 100) 

    # Fit the forest to the training set, using the bag of words as 
    # features and the sentiment labels as the response variable
    #
    # This may take a few minutes to run
    forest = forest.fit( train_data_features, clean_train_labels )
    
    # Use the random forest to make sentiment label predictions
    result = forest.predict(test_data_features)
    iteration=0
    acc=0
    for x in result:
	if (x==clean_test_labels[iteration]):
		acc=acc+1
	iteration=iteration+1
    print 'Accuracy=%f' % (acc*1.0/len(result))
            
            
if __name__ == '__main__':
    #MODEL_BINARY = './models/GoogleNews-vectors-negative300.bin'
    TEST_TRAIN_SPLIT = 0.9
    TRAIN_FEATURES_FILE = './features/train_bow_features'
    #TRAIN_LABELS_FILE = './features/train_labels'
    TEST_FEATURES_FILE = './features/test_bow_features'
    #TEST_LABELS_FILE = './features/test_labels'
    prepare_data(TEST_TRAIN_SPLIT, TRAIN_FEATURES_FILE,  TEST_FEATURES_FILE)

