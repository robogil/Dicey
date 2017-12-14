from django.db import models

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import pickle

# Set up dataset
bgg = pd.read_csv("./data/Cleaned_BGG")
bgg.drop('Unnamed: 0',axis = 1, inplace = True)
bgg.drop_duplicates(subset= 'names', inplace= True)
bgg['names'].value_counts().head()

# Load user choices
user_choices= pickle.load(open("./data/gamelist.p", "rb"))
user_names= pickle.load(open("./data/usernames.p", "rb"))
testlist = []
for name in user_choices.keys():
    for game in user_choices[name]:
        testlist.append(name+'~'+game)
df_user= []
df_game = []
for _ in testlist:
    user, game= _.split('~')
    df_user.append(user)
    df_game.append(game)

# Create User listing
rec_user = pd.DataFrame({'User':df_user, 'Game': df_game})
rec_user = rec_user.reindex(columns = ['User','Game'])
rec_user.head()

#Combine user data with games from main bgg dataframe
comp = pd.merge(rec_user, bgg, how= 'inner', left_on= 'Game', right_on='names')

def rec(ui, num):
    # Set up df index
    alg_rec = pd.DataFrame([])
    alg_rec['name'] = bgg['names']
    alg_rec = alg_rec.set_index('name')

    # User Score
    users = comp[comp['Game'] == ui ]
    users = list(users['User'])
    high_ranked= []

    for _ in users:
        temp = comp[comp['User'] == _]
        high_ranked.extend(temp.iloc[:, 1])
    high_ranked = pd.Series(high_ranked)
    high_ranked = high_ranked.value_counts()
    high_ranked = high_ranked * 0.04545453
    alg_rec['user_score'] = high_ranked
    alg_rec = alg_rec.fillna(0)

    # Category Score
    cat = pd.DataFrame(bgg.iloc[:,3])
    cat = cat.join(bgg.iloc[:,69:])
    cat.iloc[215:225,:]
    target= cat[cat['names'] == ui]
    cat_scores = []

    for _ in cat['names']:
        test = cat[cat['names'] == _]
        temp = cosine_similarity(target.drop('names', axis = 1), test.drop('names', axis = 1))[0]
        cat_scores.extend(temp)

    alg_rec['cat_scores'] = cat_scores

    # Mechanic Score
    mech = pd.DataFrame(bgg.iloc[:,3])
    mech = mech.join(bgg.iloc[:,20:69])

    target= mech[mech['names'] == ui]
    mech_scores = []

    for _ in mech['names']:
        test = mech[mech['names'] == _]
        temp = cosine_similarity(target.drop('names', axis = 1), test.drop('names', axis = 1))[0]
        mech_scores.extend(temp)

    alg_rec['mech_scores'] = mech_scores

    # The Great Decider
    alg_rec['mech_scores'] = alg_rec['mech_scores'] * 2.0
    alg_rec['cat_scores'] = alg_rec['cat_scores'] * 1.5
    alg_rec['total'] = alg_rec ['user_score'] + alg_rec['cat_scores'] + alg_rec['mech_scores']
    rec = alg_rec.sort_values('total', ascending= False).head(num)
    for _ in rec.index[1:num]:
        print(_)
