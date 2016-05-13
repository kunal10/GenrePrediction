
# coding: utf-8

# In[5]:

import os
import sys
import numpy as np
import nltk
import re
import random
from bs4 import BeautifulSoup
# from nltk.corpus import stopwords


dirname = './data/lyrics'
# Needed only if you want to remove stop words.
# nltk.download()

# Create genre mapping.
genre_freqs = {
    'classical':0, 
    'folk':0, 
    'jazz and blues':0, 
    'pop':0, 
    'soul and reggae':0,
    'punk':0,
    'metal':0,
    'hip-hop':0,
    'dance and electronica':0,
    'classic pop and rock':0
}

genre_weights = {
    'classical':0, 
    'folk':0, 
    'jazz and blues':0, 
    'pop':0, 
    'soul and reggae':0,
    'punk':0,
    'metal':0,
    'hip-hop':0,
    'dance and electronica':0,
    'classic pop and rock':0
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


def compute_genre_weights(test_train_split):
    song_id = 0
    num_non_empty_songs = 0
    num_empty_songs = 0
    for genre in os.listdir(dirname):
        print("\nProcessing Genre: " + genre)
        songs = os.listdir(os.path.join(dirname, genre))
        num_songs = len(songs)
        song_index = 0 # Index of song within this genre
        for song in songs:
	#    print "Processing song %d of %d" % (song_index + 1, num_songs) 
           
            with open(os.path.join(dirname, genre, song), 'r') as song_lyrics:
                lyrics = song_lyrics.read()
                words = process_song(lyrics)
		
		# Ignore songs with no lyrics.
		if len(words) == 0:
		    num_empty_songs = num_empty_songs + 1
		    song_index = song_index + 1
		    song_id = song_id + 1
		    # print 'Song Index %d in genre %s is empty' % (song_index, genre)
		    continue

                # Add to training set.
                if (song_index  + 1) > test_train_split * num_songs:
		    genre_freqs[genre] = genre_freqs[genre] + 1
                    num_non_empty_songs = num_non_empty_songs + 1

            song_index = song_index + 1
            song_id = song_id + 1
        print(genre_freqs[genre])
    
    total_weight = 0
    for genre, freq in genre_freqs.iteritems():
	    genre_weight = (1.0 * num_non_empty_songs) / freq 
	    genre_weights[genre] = genre_weight
	    print genre, freq, genre_weight
            total_weight = total_weight + genre_weight
    
    # Normalize the weights
    for genre in genre_weights:
	    genre_weights[genre] = (1.0 * genre_weights[genre]) / total_weight
	    print genre, genre_freqs[genre], genre_weights[genre]

if __name__ == '__main__':
    TEST_TRAIN_SPLIT = 0.9
    compute_genre_weights(TEST_TRAIN_SPLIT) 

