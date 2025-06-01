# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 17:17:14 2021

@author: tsche

utils.py
Functions: get_grid, read_dataset
"""
import pandas as pd
import ast

def get_grid(name, metric):
    if metric == 'ndcg':
        grids = pd.read_excel('Grids.xls')
    elif metric == 'rmse':
        grids = pd.read_excel('Grids_rmse.xls')
    grids = grids.set_index('Algo')
    
    grid = grids[name]

    return grid

def read_dataset(name, frac=None):
    
    """ loading of different pre-downloaded datasets"""
    
    if name == 'ML-100k':
        data = pd.read_table(r"..\Datasets\Movielens\ml-100k\u.data", 
                             sep='\t', header = 0, names=['user', 'item', 'rating', 'timestamp'], engine='python')
        data.timestamp = pd.to_datetime(data.timestamp, unit='s', origin='1970-01-01')
        start = 1995
        end = 1998
        
    elif name == 'ML-1M':
        data = pd.read_table(r"..\Datasets\Movielens\ml-1m\ratings.dat", 
                             sep='::', header = 0, names=['user', 'item', 'rating', 'timestamp'], engine='python')
        data.timestamp = pd.to_datetime(data.timestamp, unit='s', origin='1970-01-01')
        start = 2000
        end = 2003
                
    elif name == 'ML-10M':
        data = pd.read_table(r"..\Datasets\Movielens\ml-10M100K\ratings.dat", 
                             sep='::', header = 0, names=['user', 'item', 'rating', 'timestamp'], engine='python')
        data.timestamp = pd.to_datetime(data.timestamp, unit='s', origin='1970-01-01')
        start = 1996
        end = 2009
        
        
    elif name == 'ML-100k-latest':
        data = pd.read_table(r"..\Datasets\Movielens\ml-latest-small\ratings.csv", 
                             sep='::', header = 0, names=['user', 'item', 'rating', 'timestamp'], engine='python')
        data.timestamp = pd.to_datetime(data.timestamp, unit='s', origin='1970-01-01')
        start = 1995
        end = 2017
               
    elif name == 'amazon-instantvideo':
        data = pd.read_table(r"..\Datasets\Amazon\ratings_Amazon_Instant_Video.csv", #windows: r"..\Datasets\Amazon\ratings_Amazon_Instant_Video.csv", linux: r"../Datasets/Amazon/ratings_Amazon_Instant_Video.csv"
                     sep=',', header = 0, names=['user', 'item', 'rating', 'timestamp'], engine='python') 
        data['user'] = data.groupby(['user']).ngroup()
        data['item'] = data.groupby(['item']).ngroup()
        data.timestamp = pd.to_datetime(data.timestamp, unit='s', origin='1970-01-01')
        start = 2007
        end = 2014
        
    elif name == 'amazon-books':
        data = pd.read_table(r"..\Datasets\Amazon\ratings_Books.csv",
                     sep=',', header = 0, names=['user', 'item', 'rating', 'timestamp'], engine='python') 
        data['user'] = data.groupby(['user']).ngroup()
        data['item'] = data.groupby(['item']).ngroup()
        data.timestamp = pd.to_datetime(data.timestamp, unit='s', origin='1970-01-01')
        start = 1997
        end = 2013
        
    elif name == 'amazon-toys':
        data = pd.read_table(r"..\Datasets\Amazon\ratings_Toys_and_Games.csv",
                     sep=',', header = 0, names=['user', 'item', 'rating', 'timestamp'], engine='python') 
        data['user'] = data.groupby(['user']).ngroup()
        data['item'] = data.groupby(['item']).ngroup()
        data.timestamp = pd.to_datetime(data.timestamp, unit='s', origin='1970-01-01')
        start = 2001
        end = 2014
    
    elif name == 'amazon-electronics':
        data = pd.read_table(r"..\Datasets\Amazon\ratings_Amazon_Electronics.csv",
                     sep=',', header = 0, names=['user', 'item', 'rating', 'timestamp'], engine='python') 
        data['user'] = data.groupby(['user']).ngroup()
        data['item'] = data.groupby(['item']).ngroup()
        data.timestamp = pd.to_datetime(data.timestamp, unit='s', origin='1970-01-01')
        start = 2000
        end = 2014
            
    elif name == 'amazon-music':
        data = pd.read_table(r"..\Datasets\Amazon\ratings_Digital_Music.csv",
                     sep=',', header = 0, names=['item', 'user', 'rating', 'timestamp'], engine='python') 
        data = data[['user', 'item', 'rating', 'timestamp']]
        data['user'] = data.groupby(['user']).ngroup()
        data['item'] = data.groupby(['item']).ngroup()
        data.timestamp = pd.to_datetime(data.timestamp, unit='s', origin='1970-01-01')
        start = 1998
        end = 2014
        
    elif name == 'netflix':
        data = pd.read_table(r"..\Datasets\netflix\NetflixRatings.csv", sep=",", names = ['item','user', 'rating', 'timestamp'])
        data = data[['user', 'item', 'rating', 'timestamp']]
        data.timestamp = pd.to_datetime(data.timestamp)
        start = 1998
        end = 2005
        
    elif name == 'yelp':
        data = pd.read_json(r"..\Datasets\yelp_training_set\yelp_training_set_review.json", lines=True)
        data = data.rename(columns={"user_id": "user", "business_id": "item", "stars": "rating", "date": "timestamp"})
        data = data[['user', 'item', 'rating', 'timestamp']]
        data.timestamp = pd.to_datetime(data.timestamp)
        data['user'] = data.groupby(['user']).ngroup()
        data['item'] = data.groupby(['item']).ngroup()
        start = 2006
        end = 2013
                
    elif name == 'epinions':
        data = pd.read_table(r"..\Datasets\epinions\rating_with_timestamp.txt", 
                             delim_whitespace=True, names = ['user','item','category','rating','helpfulness', 'timestamp'])
        data = data[['user', 'item', 'rating', 'timestamp']]
        data.timestamp = pd.to_datetime(data.timestamp, unit='s', origin='1970-01-01')
        start = 1999
        end = 2011

    elif name == 'movie-tweetings': #movies
        data = pd.read_table(r"..\Datasets\MovieTweetings\movie-tweetings-ratings.dat", #linux: r"../Datasets/MovieTweetings/movie-tweetings-ratings.dat", windows: r"..\Datasets\MovieTweetings\movie-tweetings-ratings.dat"
                             sep='::', engine='python', header=None,
                             names=['user', 'item', 'rating', 'timestamp'])
        data.timestamp = pd.to_datetime(data.timestamp, unit='s', origin='1970-01-01')
        start = 2013
        end = 2021


    elif name == 'librarything':  # Python-Dictionary als String  #Books

        reviews = {}
        reviews_file_path = r"..\Datasets\Amazon\lthing_data\reviews.txt" #linux: "../Datasets/Amazon/lthing_data/reviews.txt", windows: r"..\Datasets\Amazon\lthing_data\reviews.txt"

        # Process the file line by line to build the dictionary
        try:
            with open(reviews_file_path, 'r', encoding='utf-8') as f:
                exec_globals = {'reviews': reviews}
                for line in f:
                    if line.strip():
                        exec(line, exec_globals)
        except Exception as e:  # Catch any exception during file loading/parsing
            print(f"Error loading or parsing {reviews_file_path}: {e}")
            raise  # Re-raise the exception to stop execution

        rows = []

        skipped_reviews_count = 0

        for (item, user), rev in reviews.items():
            if 'stars' in rev:
                rows.append({
                    'user': user,
                    'item': item,
                    'rating': rev['stars'],
                    'timestamp': pd.to_datetime(rev['unixtime'], unit='s', origin='1970-01-01')
                })
            else:
                skipped_reviews_count += 1

        data = pd.DataFrame(rows)

        if skipped_reviews_count > 0:
            print(f"Skipped {skipped_reviews_count} reviews due to missing 'stars' key")

        # ID-Normalisierung wie gewohnt
        data['user'] = data.groupby(['user']).ngroup()
        data['item'] = data.groupby(['item']).ngroup()

        min_librarything_year = 2005

        initial_rows = len(data)
        data = data[data['timestamp'].dt.year >= min_librarything_year]

        start = data.timestamp.dt.year.min()
        end = data.timestamp.dt.year.max()


    elif name == 'modcloth':  #clothing
        file_path = r"..\Datasets\Modcloth\df_modcloth.csv" #linux: "../Datasets/Modcloth/df_modcloth.csv", windows: r"..\Datasets\Modcloth\df_modcloth.csv"
        data = pd.read_csv(file_path, sep=',')
        data = data.rename(columns={'user_id': 'user', 'item_id': 'item'})[['user', 'item', 'rating', 'timestamp']]
        data['timestamp'] = pd.to_datetime(data['timestamp'], format='ISO8601')
        # ID-Normalisierung
        data['user'] = data.groupby(['user']).ngroup()
        data['item'] = data.groupby(['item']).ngroup()
        # Start- und Endjahr automatisch aus Daten bestimmen
        start = data.timestamp.dt.year.min()
        end = data.timestamp.dt.year.max()
        print(f"Loaded {len(data)} reviews for ModCloth, year range: {start} to {end}")

    elif name == 'amazon-magazine':
        data = pd.read_table(r"..\Datasets\Amazon\amazon_magazine.csv",
                             sep=',', header=None, names=['user', 'item', 'rating', 'timestamp'], engine='python')
        data['user'] = data.groupby(['user']).ngroup()
        data['item'] = data.groupby(['item']).ngroup()
        data.timestamp = pd.to_datetime(data.timestamp, unit='s', origin='1970-01-01')
        start = data.timestamp.min().year
        end = data.timestamp.max().year

    elif name == 'amazon-beauty':
        data = pd.read_csv(r"..\Datasets\Amazon\All_Beauty.csv",
                           # windows: r"..\Datasets\Amazon\All_Beauty.csv", linux: r"../Datasets/Amazon/All_Beauty.csv"
                           sep=',', header=None, names=['user', 'item', 'rating', 'timestamp'], engine='python')
        data['user'] = data.groupby(['user']).ngroup()
        data['item'] = data.groupby(['item']).ngroup()
        data.timestamp = pd.to_datetime(data.timestamp, unit='s', origin='1970-01-01')
        start = data.timestamp.dt.year.min()
        end = data.timestamp.dt.year.max()
        print(f"Loaded {len(data)} reviews for Amazon Beauty, year range: {start} to {end}")



    elif name == 'amazon-giftcards': #warning: dataset has too little variance
        data = pd.read_csv(r"../Datasets/Amazon/Gift_Cards.csv",
                           # windows: r"..\Datasets\Amazon\Gift_Cards.csv", linux: r"../Datasets/Amazon/Gift_Cards.csv"
                           sep=',', header=None, names=['user', 'item', 'rating', 'timestamp'], engine='python')
        data['user'] = data.groupby(['user']).ngroup()
        data['item'] = data.groupby(['item']).ngroup()
        data.timestamp = pd.to_datetime(data.timestamp, unit='s', origin='1970-01-01')
        start = data.timestamp.dt.year.min()
        end = data.timestamp.dt.year.max()
        print(f"Loaded {len(data)} reviews for Amazon Giftcards, year range: {start} to {end}")



    else:
        raise ValueError('Dataset not implemented')

    data = data.groupby("user").filter(lambda grp: len(grp) > 2)

    if frac is not None:
        data = data.sample(frac=frac)

    data = data.drop_duplicates(subset=['user', 'item'], keep='last') # entfernen doppelter user items kombis
    return data, start, end

