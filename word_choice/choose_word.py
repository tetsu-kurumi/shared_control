import config
import math
import pandas as pd
import numpy as np
import pickle

def csv_to_filtered_dict(csv_file):
    df = pd.read_csv(csv_file, header=None, names=['word', 'value'], low_memory=False)
    df = df[df['word'].apply(lambda x: isinstance(x, str))]
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df.dropna(subset=['value'], inplace=True)
    df['word'] = df['word'].astype(str)
    df['word'] = df['word'].str.upper()
    df_filtered = df[df['word'].apply(lambda x: len(x) == 5 and len(set(x)) == 5)]
    return dict(zip(df_filtered['word'], df_filtered['value']))

def save_obj(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def load_words():
    with open('words.txt', 'r') as words_file:
        word_list = [line.strip().upper() for line in words_file]
        filtered_word_list = [word for word in word_list if len(word) == len(set(word))]
        return filtered_word_list

def calculate_word_spread(word_list):
    distances = np.zeros(len(word_list))
    word_to_index = {word: i for i, word in enumerate(word_list)}
    
    for i, word in enumerate(word_list):
        distance = 0
        for letter in word:
            distance += math.sqrt(config.SLOT_POS[letter][0]**2 + config.SLOT_POS[letter][1]**2)
        distances[i] = distance
    
    return distances, word_to_index

try:
    word_frequency = load_obj('word_frequency.pkl')
except FileNotFoundError:
    word_frequency = csv_to_filtered_dict('unigram_freq.csv')
    save_obj(word_frequency, 'word_frequency.pkl')

try:
    keyboard_distance = load_obj('keyboard_distance.pkl')
    word_to_index = load_obj('word_to_index.pkl')
except FileNotFoundError:
    word_list = load_words()
    distances, word_to_index = calculate_word_spread(word_list)
    keyboard_distance = dict(zip(word_list, distances))
    save_obj(keyboard_distance, 'keyboard_distance.pkl')
    save_obj(word_to_index, 'word_to_index.pkl')

keyboard_distance_df = pd.DataFrame.from_dict(keyboard_distance, orient='index', columns=['Value_dist'])
word_freq_df = pd.DataFrame.from_dict(word_frequency, orient='index', columns=['Value_freq'])

merged_df = pd.merge(keyboard_distance_df, word_freq_df, left_index=True, right_index=True)
merged_df.reset_index(inplace=True)
merged_df.rename(columns={'index': 'Word'}, inplace=True)

def calculate_similarity(df):
    num_words = len(df)
    similarities = []
    
    for i in range(num_words):
        for j in range(i + 1, num_words):
            word1 = df.iloc[i]['Word']
            word2 = df.iloc[j]['Word']
            distance1, freq1 = df.iloc[i]['Value_dist'], df.iloc[i]['Value_freq']
            distance2, freq2 = df.iloc[j]['Value_dist'], df.iloc[j]['Value_freq']
            
            distance_diff = abs(distance1 - distance2)
            freq_diff = abs(freq1 - freq2)
            
            distance_threshold = 50
            freq_threshold = 1000
            
            if distance_diff < distance_threshold and freq_diff < freq_threshold:
                similarities.append(((word1, word2), (distance_diff, freq_diff)))
    
    return similarities

similar_pairs = calculate_similarity(merged_df)

for pair, (distance_diff, freq_diff) in similar_pairs:
    print(f"Words: {pair[0]} and {pair[1]} - Distance Diff: {distance_diff}, Frequency Diff: {freq_diff}")
