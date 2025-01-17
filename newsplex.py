#!/usr/bin/env python
"""
@author: metalcorebear
"""

"""
Newsplex uses topological data analysis to identify topological invariance in news discourse surrounding geopolitical events.
"""

import requests
import hashlib
import os
import json
import numpy as np
import pandas as pd
import re
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import itertools
from itertools import cycle

from gensim.models import Word2Vec

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.chunk import RegexpParser
from nltk.stem import WordNetLemmatizer

from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform

from gudhi import SimplexTree
from gudhi.wasserstein import wasserstein_distance


"""
Helper Functions
"""
# Get API key from json file
def get_key(key_location, key='api_key'):
    with open(key_location, 'r') as json_file:
        api_dict = json.load(json_file)
    api_key = api_dict[key]
    return api_key

def make_hash(string):
    s = str(string)
    m = hashlib.sha256()
    encoded = s.encode('utf-8')
    m.update(encoded)
    return m.hexdigest()

"""
Functions to build NewsAPI query.
"""
def construct_query(api_key, from_date, to_date, source_str, page=1):
    # from_date format : YYYY-MM-DD
    # limit per page: 100
    end_point = f'https://newsapi.org/v2/everything?from={from_date}&to={to_date}&language=en&page={str(page)}&sources={source_str}&apiKey={api_key}'
    return end_point

def run_query(url):
    response = requests.get(url)
    return response

def run_multi_query(api_key, from_date, to_date, source_str, total_pages=1):
    page_list = list(range(1,total_pages+1))
    responses = []
    for page in page_list:
        url = construct_query(api_key, from_date, to_date, source_str, page=page)
        response = run_query(url)
        responses.append(response)
    return responses

def agg_responses(responses):
    output = []
    for r in responses:
        o = filter_response(r)
        output.extend(o)
    return output

"""
Functions to filter and aggragate responses.
"""
def filter_response(response):
    if response.status_code == 200:
        json_ = response.json()
        totalResults = json_['totalResults']
        output = []
        for s in json_['articles']:
            output.append({'author':s['author'], 'title':s['title'], 
                           'date':s['publishedAt'], 'text':s['content']})
        return output
    else:
        print('ERROR: Response failed.')
        return []

def date_convert(date_string):
    date_object = datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%SZ')
    return date_object

def convert_all_dates(csv_file, metadata):
    df = pd.read_csv(csv_file)
    response_output = df.to_dict(orient='records')
    start_date = date_convert(metadata['start'])
    for a in response_output:
        date_ = date_convert(a['date'])
        delta = date_ - start_date
        a.update({'day':delta.days, 'hour':delta.total_seconds()/3600.0})
    return response_output

"""
Functions to process and save responses.
"""
def gen_date_list(start_date, end_date, d_delta=1):
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    date_delta = end - start
    days = date_delta.days
    d_ = timedelta(days=d_delta)
    date_list = []
    for d in range(days):
        s = start.strftime('%Y-%m-%d')
        s_ = start + d_
        s1 = s_.strftime('%Y-%m-%d')
        date_list.append((s,s1))
        start = s_
    return date_list

def run_news(start_date, end_date, sources, api_key, filepath='news_data', d_delta=1, total_pages=10):
    date_list = gen_date_list(start_date, end_date, d_delta=d_delta)
    for l in range(len(sources)):
        print(f'Running source list {l}.')
        for d in tqdm(date_list):
            responses = run_multi_query(api_key, d[0], d[1], sources[l], total_pages=total_pages)
            aggregated = agg_responses(responses)
            if len(aggregated) != 0:
                filename = os.path.join(filepath, f'date_range_' + d[0] + f'_' + d[1] + f'_{l}.csv')
                df = pd.DataFrame(aggregated)
                df.to_csv(filename)

def gen_metadata(directory, event_date):
    files = os.listdir(directory)
    csv_files = [f for f in files if '.csv' in f]
    dates_list = []
    for f in tqdm(csv_files):
        df = pd.read_csv(os.path.join(directory, f))
        dates_list.extend(list(df['date']))
    dates_list = [d.split('T')[0] for d in dates_list]
    dates_list = list(set(dates_list))
    date_objects = [datetime.strptime(date_string, '%Y-%m-%d') for date_string in dates_list]
    sorted_date_objects = sorted(date_objects)
    date_strings = [d.strftime('%Y-%m-%d') for d in sorted_date_objects]
    try:
        event_date_index = date_strings.index(event_date)
    except:
        event_date_index = len(date_strings)/2
    min_date = min(date_objects)
    max_date = max(date_objects)
    out_dict = {'start':min_date.strftime('%Y-%m-%d'), 'end':max_date.strftime('%Y-%m-%d'), 'event_date':event_date, 'event_index':event_date_index}
    with open(os.path.join(directory,'metadata.json'), 'w') as json_file:
        json.dump(out_dict, json_file)

"""
Functions to preprocess and tokenize article text.
"""
# Text preprocessing function
def preprocess_text(text):
    if isinstance(text, str) == False:
        tokens = []
    else:
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation and special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# Generate JSON file containing all preprocessed text
def process_csv_files(directory):
    files = os.listdir(directory)
    csv_files = [f for f in files if '.csv' in f]
    json_file = [f for f in files if '.json' in f]
    if not os.path.isdir(os.path.join(directory, 'pre_json')):
        os.mkdir(os.path.join(directory, 'pre_json'))
    output_json = {}
    with open(os.path.join(directory, json_file[0]), 'r') as json_file:
        metadata = json.load(json_file)
    for f in tqdm(csv_files):
        data = convert_all_dates(os.path.join(directory, f), metadata)
        for d in data:
            tokens = preprocess_text(d['text'])
            output_json.update({make_hash(d['title']):{'date':d['date'], 'day':d['day'], 'hour':d['hour'], 'tokens':tokens}})
    with open(os.path.join(os.path.join(directory, 'pre_json'), 'pre_json.json'), 'w') as json_out:
        json.dump(output_json, json_out)
    print(f'Total articles processed: {len(list(output_json.keys()))}')
    print(f'Token JSON file saved in {directory}/pre_json.')

def find_item_length(dataset, scale='day'):
    times = [v[scale] for (k,v) in dataset.items()]
    time_set = list(set(times))
    return len(time_set)

def time_selector(dataset, directory_path, scale='day'):
    times = [v[scale] for (k,v) in dataset.items()]
    time_set = list(set(times))
    for t in tqdm(time_set):
        token_set = []
        for v in dataset.values():
            if v[scale] == t:
                token_set.extend(v['tokens'])
        output = {t:token_set}
        filepath = scale + '_agg_' + str(t) + '.json'
        with open(os.path.join(os.path.join(directory_path, 'pre_json'), filepath), 'w') as json_file:
            json.dump(output, json_file)

"""
Word2Vec Model Functions.
"""
# Function to compute the mean vector for an article
def get_article_vector(article, model):
    vectors = [model.wv[word] for word in article if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)  # Handle empty articles
    return np.mean(vectors, axis=0)

"""
TDA Functions.
"""
def window_generator(seq, n=5):
    output = []
    it = iter(seq)
    result = tuple(itertools.islice(it, n))
    if len(result) == n:
        output.append(result)
    for elem in it:
        result = result[1:] + (elem,)
        output.append(result)
    return output

def calc_persistence(sorted_keys, article_embeddings, window, distance_threshold=1.0, n_components=2):
    normalized_dates = [float(a/max(sorted_keys)) for a in sorted_keys]
    tsne = TSNE(n_components=n_components, random_state=42)
    reduced_embeddings = tsne.fit_transform(article_embeddings)# Compute pairwise distances (e.g., Euclidean)
    pairwise_distances = squareform(pdist(reduced_embeddings))
    # Create a simplex tree with weights based on publication dates
    simplex_tree = SimplexTree()
    for i in window:
        # Add vertices with weights (normalized publication dates)
        simplex_tree.insert([i], filtration=normalized_dates[i])
    # Add edges with weights based on max(vertex1_weight, vertex2_weight)
    for i in window:
        for j in range(i + 1, len(reduced_embeddings)):
            weight = max(normalized_dates[i], normalized_dates[j])  # Use max weight of vertices
            if pairwise_distances[i, j] < distance_threshold:  # Optional distance threshold
                simplex_tree.insert([i, j], filtration=weight)
    # Expand to higher-dimensional simplices (optional)
    simplex_tree.expansion(2)  # Max dimension = 2
    # Compute Persistent Homology
    persistence = simplex_tree.persistence()
    return persistence

def clean_persistence(persistence):
    out = []
    for p in persistence:
        out.append(p[1])
    return np.array(out)

# Compute Wasserstein Distances
def compute_w_distances(persistences):
    p_vec = []
    output = dict()
    for k in persistences.keys():
        p_vec.append(clean_persistence(persistences[k]))
    for p in range(len(p_vec)-1):
        d = wasserstein_distance(p_vec[p], p_vec[p+1], keep_essential_parts=False)
        output.update({p:d})
    return output

def plot_distances(distances, labels, date=54, window_size=5, dirpath=''):
    lines = ["-","--","-.",":"]
    linecycler = cycle(lines)
    x = list(distances[0].keys())
    for i in range(len(distances)):
        plt.plot(x, distances[i].values(), next(linecycler), label=labels[i], linewidth=0.5, color='black')
    plt.suptitle('Moving Wasserstein Distances')
    plt.axvline(x=date, color='black', linestyle='--')
    plt.title(f'Window Size = {window_size}')
    plt.legend()
    plt.xlabel('Date Index')
    plt.ylabel('Wasserstein Distance')
    plt.savefig(os.path.join(dirpath, 'wasserstein_distances.png'), dpi=300)


class newsaggregator():
    """
    News aggregator downloads news articles from the NewsAPI service and preprocesses them for topological analysis.
    """
    def __init__(self, key_location, **kwargs):
        key_location = kwargs.pop('key_location', 'api_key.json')
        key = kwargs.pop('key', 'api_key')
        self.filepath = kwargs.pop('filepath', 'news_data')
        self.api_key = get_key(key_location, key=key)
        sources_1 = 'abc-news,al-jazeera-english,associated-press,axios,bloomberg,business-insider,cbs-news,cnn,espn,fox-news,google-news,hacker-news,ign,msnbc,national-geographic,national-review,nbc-news,new-scientist,newsweek'
        sources_2 = 'new-york-magazine,next-big-future,politico,recode,reddit-r-all,reuters,techcrunch,techradar,the-american-conservative,the-hill,the-huffington-post,the-next-web,the-wall-street-journal,the-washington-post,the-washington-times,time,usa-today,vice-news,wired'
        self.sources = [sources_1, sources_2]

    
    def run_query(self, event_date, **kwargs):
        """
        event date format: YYYY-MM-DD
        """
        d_delta = kwargs.pop('d_delta', 1)
        total_pages = kwargs.pop('pages', 10)
        date_bracket = kwargs.pop('date_bracket', 50)
        self.filepath = kwargs.pop('filepath', self.filepath)
        
        if not os.path.isdir(self.filepath):
            os.mkdir(self.filepath)

        event_date_object = datetime.strptime(event_date, '%Y-%m-%d')
        # get start and end dates
        delta = timedelta(days=date_bracket)
        start = event_date_object - delta
        end = event_date_object + delta
        start_date = start.strftime('%Y-%m-%d')
        end_date = end.strftime('%Y-%m-%d')

        # run query
        print('Running News Query...')
        run_news(start_date, end_date, self.sources, self.api_key, filepath=self.filepath, d_delta=d_delta, total_pages=total_pages)
        print('News API Query Complete')

        print('Generating Metadata...')
        gen_metadata(self.filepath, event_date)
        print(f'News Data Saved in {self.filepath}.')


    def generate_json(self, **kwargs):
        """
        Generates JSON file from downloaded news csv files. Aggregates on day and tokenizes text.
        """
        self.filepath = kwargs.pop('filepath', self.filepath)
        if os.path.isdir(self.filepath):
            print('Generating Master JSON File.')
            process_csv_files(self.filepath)

            json_path = os.path.join(self.filepath, 'pre_json')
            json_file_path = os.path.join(json_path, 'pre_json.json')

            with open(json_file_path, 'r') as json_file:
                dataset = json.load(json_file)

            print('Generating Day Aggregated JSON Files.')
            item_length = find_item_length(dataset, scale='day')
            print(f'Day Vector Size: {item_length}')
            time_selector(dataset, self.filepath, scale='day')
            print('Day Aggregation Complete.')
            
        else:
            print('News directory does not exist. Ensure query has been run or pass "filepath" kwarg.')



class newsplex():
    """
    Generates simplicial complex from filtration of news data, weighted by date. Generates persistence distributions using a moving window and computes the Wasserstein distances between persistence diagrams to identify regions of topological change.
    """
    def __init__(self, **kwargs):
        self.filepath = kwargs.pop('filepath', 'news_data')

    
    def fit(self, **kwargs):
        """
        Fits news data to Word2Vec model and generates embeddings.
        """
        vector_size = kwargs.pop('vector_size', 100)
        window = kwargs.pop('window', 5)
        min_count = kwargs.pop('min_count', 2)
        workers = kwargs.pop('workers', 4)
        
        day_json_path = os.path.join(self.filepath, 'pre_json')
        files = os.listdir(day_json_path)
        file_list = [os.path.join(day_json_path, f) for f in files if 'day_agg' in f]
        
        agg_json = {}

        print('Initializing...')
        for f in tqdm(file_list):
            with open(f, 'r') as json_file:
                d = json.load(json_file)
            agg_json.update(d)
        
        keys = [int(a) for a in agg_json.keys()]
        sorted_keys = [int(a) for a in sorted(keys)]
        tokenized_articles = [agg_json[str(k)] for k in sorted_keys]

        keys_dict = {'keys':sorted_keys}

        with open(os.path.join(self.filepath, 'sorted_keys.json'), 'w') as keys_file:
            json.dump(keys_dict, keys_file)
        
        # Train Word2Vec model
        print('Training Word2Vec Model...')
        model = Word2Vec(
            sentences=tokenized_articles,  # Tokenized articles
            vector_size=vector_size,              # Dimensionality of word embeddings
            window=window,                     # Context window size
            min_count=min_count,                  # Minimum word frequency
            workers=workers                     # Number of threads for training
        )
        
        # Save the model for future use
        model.save(os.path.join(self.filepath, 'word2vec_model.model'))
        print('Model Training Complete.')

        print('Generating Article Embeddings...')
        # Generate embeddings for all articles
        article_embeddings = np.array([get_article_vector(article, model) for article in tokenized_articles])
        
        # Save tokenized articles
        with open(os.path.join(self.filepath, 'tokenized_articles.pkl'), "wb") as f:
            pickle.dump(tokenized_articles, f)
        
        # Save embeddings
        np.save(os.path.join(self.filepath, 'article_embeddings.npy'), article_embeddings)

        print(f'Shape of Article Embeddings: {article_embeddings.shape}')
        print(f'Size of Tokenized Articles Vector: {len(tokenized_articles)}')
        print('Action Complete.')

    
    def plot_embeddings(self, **kwargs):
        """
        Gzenerates scatter plot of dimensionally reduced embeddings.
        """
        self.filepath = kwargs.pop('filepath', self.filepath)
        n_components = kwargs.pop('n_components', 2)
        
        with open(os.path.join(self.filepath, 'metadata.json'), 'r') as json_file:
            metadata = json.load(json_file)

        with open(os.path.join(self.filepath, 'sorted_keys.json'), 'r') as keys_file:
            keys_dict = json.load(keys_file)
        sorted_keys = keys_dict['keys']
        
        article_embeddings = np.load(os.path.join(self.filepath,'article_embeddings.npy'))
        event_index = metadata['event_index']
        
        c = ['black' if a<int(event_index) else 'tab:gray' for a in sorted_keys]
        
        tsne = TSNE(n_components=n_components, random_state=42)
        reduced_embeddings = tsne.fit_transform(article_embeddings)
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=c)
        event_date = metadata['event_date']
        plt.title(f'Before and After {event_date}')
        plt.suptitle('t-SNE Visualization of News Embeddings')
        plt.savefig(os.path.join(self.filepath,'embeddings.png'), dpi=300)
        plt.clf()
        print(f'Embeddings Plot Saved in {self.filepath}')
        print('Action Complete.')


    def calculate_persistences(self, **kwargs):
        """
        Calculates moving persistence distributions and plots Wasserstein distances.
        """
        self.filepath = kwargs.pop('filepath', self.filepath)
        window_size = kwargs.pop('window_size', 5)
        thresholds = kwargs.pop('thresholds', [0.3, 0.5, 0.7, 0.9])
        n_components = kwargs.pop('n_components', 2)

        article_embeddings = np.load(os.path.join(self.filepath,'article_embeddings.npy'))
        
        with open(os.path.join(self.filepath, 'sorted_keys.json'), 'r') as keys_file:
            keys_dict = json.load(keys_file)
        sorted_keys = keys_dict['keys']

        with open(os.path.join(self.filepath, 'metadata.json'), 'r') as json_file:
            metadata = json.load(json_file)
        
        windows = window_generator(sorted_keys, n=window_size)
        persistence_list = []

        print('Computing Persistences...')
        for t in tqdm(thresholds):
            persistences = dict()

            for i in range(len(windows)):
                p = calc_persistence(sorted_keys, article_embeddings, windows[i], distance_threshold=t, n_components=n_components)
                persistences.update({i:p})
            persistence_list.append(persistences)

        distances = []

        print('Computing Wasserstein Distances...')
        for p in tqdm(persistence_list):
            d = compute_w_distances(p)
            distances.append(d)

        plot_distances(distances, thresholds, date=int(metadata['event_index']), window_size=window_size, dirpath=self.filepath)
        plt.clf()
        print(f'Distance Plot Saved at {self.filepath}')
        print('Action Complete.')

        


    