
# coding: utf-8

# In[5]:

import os
import sys
import numpy as np
import nltk
import re
import random
from bs4 import BeautifulSoup
from gensim import models
# from nltk.corpus import stopwords


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
    'punk':6,
    'metal':7,
    'hip-hop':8,
    'dance and electronica':9,
    'classic pop and rock':10
}


def process_song(song, remove_stopwords = False):
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
    return(words)


def write_features(features, labels, features_len, songs_len, features_file, labels_file):
    max_seq_length = 100
    feature_dim = 300
    features_mat = np.zeros((max_seq_length, len(features), feature_dim))
    with open(labels_file, 'wb') as f:
        for i, id_ in enumerate(features):
            feature = features[id_]
	    length = min(len(feature), max_seq_length)
	    features_mat[:length , i, :] = np.asarray(feature[:length])
            f.write(str(id_) + ' ' + str(labels[id_]) + ' ' + str(songs_len[id_]) + ' ' + str(features_len[id_]) + '\n')
    np.save(features_file, features_mat)


def prepare_data(model_binary, test_train_split, train_features_file, train_labels_file, test_features_file, test_labels_file):
    
    model = models.Word2Vec.load_word2vec_format(model_binary, binary=True)
    
    train_feats = {}; test_feats = {}
    train_labels = {}; test_labels = {}
    train_feat_len = {}; test_feat_len = {}
    train_song_len = {}; test_song_len = {}
    
    song_id = 0
    num_empty_songs = 0
    for genre in os.listdir(dirname):
        print("\nProcessing Genre: " + genre)
        songs = os.listdir(os.path.join(dirname, genre))
        num_songs = len(songs)
        song_index = 0 # Index of song within this genre
        for song in songs:
	    print "Processing song %d of %d" % (song_index + 1, num_songs) 
           
            with open(os.path.join(dirname, genre, song), 'r') as song_lyrics:
                lyrics = song_lyrics.read()
                words = process_song(lyrics)
		
		# Ignore songs with no lyrics.
		if len(words) == 0:
		    num_empty_songs = num_empty_songs + 1
		    song_index = song_index + 1
		    song_id = song_id + 1
		    print 'Song Index %d in genre %s is empty' % (song_index, genre)
		    continue

                # Compute song features from features of words in the song.
                song_feat = []
                for word in words:
                    if word in model.vocab:
                        word_feat = model[word]
                        song_feat.append(word_feat)
                    
                # Add to training set.
                if (song_index  + 1) <= test_train_split * num_songs:
		# if random.uniform(0,1) < 0.9:  
		    train_feats[song_id] = song_feat
                    train_labels[song_id] = genre_tag[genre]
                    train_feat_len[song_id] = len(song_feat)
                    train_song_len[song_id] = len(words)
                # Add to test set.
                else:
		    test_feats[song_id] = song_feat
                    test_labels[song_id] = genre_tag[genre]
                    test_feat_len[song_id] = len(song_feat)
                    test_song_len[song_id] = len(words)
                    
            song_index = song_index + 1
            song_id = song_id + 1
            

    print('Number of empty songs  = ' + str(num_empty_songs))
    print('Number of training examples  = ' + str(len(train_feats)))
    print('Number of test examples  = ' + str(len(test_feats)))
    print(np.min([train_feat_len[id_] for id_ in train_feats]))
    print(np.min([test_feat_len[id_] for id_ in test_feats]))
        
    # Save features and labels
    write_features(train_feats, train_labels, train_feat_len, train_song_len, train_features_file, train_labels_file)
    write_features(test_feats, test_labels, test_feat_len, test_song_len, test_features_file, test_labels_file)
            
            
if __name__ == '__main__':
    MODEL_BINARY = './models/GoogleNews-vectors-negative300.bin'
    TEST_TRAIN_SPLIT = 0.9
    TRAIN_FEATURES_FILE = './features/train_features'
    TRAIN_LABELS_FILE = './features/train_labels'
    TEST_FEATURES_FILE = './features/test_features'
    TEST_LABELS_FILE = './features/test_labels'
    prepare_data(MODEL_BINARY, TEST_TRAIN_SPLIT, TRAIN_FEATURES_FILE, TRAIN_LABELS_FILE, TEST_FEATURES_FILE, TEST_LABELS_FILE)

