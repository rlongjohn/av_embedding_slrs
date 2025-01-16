from glob import glob
from glob import escape
import pandas as pd
import json
import random
from math import floor
import itertools

def create_json(text_data, users, split = "train"):
    fname = "./datasets/processed_amazon/amazon_" + split + ".jsonl"
    
    data_dict_list = []

    problem_id = 0

    # same source pairs
    for user in users:
        user_texts = text_data.text[text_data.user_id == user].to_list()
        for combo in itertools.combinations(user_texts, 2):
            data_dict = {
                'id': str(problem_id),
                'same': True,
                'authors': [user, user],
                'pair': list(combo)
            }
            problem_id+=1
            data_dict_list.append(data_dict)
    
    n_ss = problem_id

    # diff source pairs
    poss_ds = list(itertools.combinations(users, 2))
    ds_pairs = random.sample(poss_ds, n_ss)
    for pair in ds_pairs:
        user0 = pair[0]
        user1 = pair[1]
        user0_text = random.sample(text_data.text[text_data.user_id == user0].to_list(), 1)
        user1_text = random.sample(text_data.text[text_data.user_id == user1].to_list(), 1)
        data_dict = {
            'id': str(problem_id),
            'same': False,
            'authors': [user0, user1],
            'pair': [user0_text[0], user1_text[0]]
        }
        problem_id+=1
        data_dict_list.append(data_dict)

    # write to file
    with open(fname, 'w') as f:
        for data_dict in data_dict_list:
            json.dump(data_dict, f)
            f.write("\n")

def main():
    data_dir = './datasets/amazon/*/'

    user_ids = []
    text_ids = []
    texts = []

    for prob_dir in glob(data_dir):
        
        text_counter = 0
        files = glob(escape(prob_dir) + '/*')

        for file in files:
            user_id = file.split('/')[4].split(' - ')[1]
            user_ids.append(user_id)
            text_ids.append(text_counter)
            text_counter += 1 
            fp = open(file)
            texts.append(fp.read())
            fp.close()

    # put into dataframe
    text_data = pd.DataFrame({'user_id': user_ids, 'text_id': text_ids, 'text': texts})

    users = text_data.user_id.unique()
    n_train_users = floor(0.9*len(users))
    train_users = users[0:n_train_users]
    test_users = users[n_train_users:len(users)]

    create_json(text_data, train_users, "train")
    create_json(text_data, test_users, "test")

if __name__ == "__main__":
    main()