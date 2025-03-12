from glob import glob
from glob import escape
import pandas as pd
import json
import numpy as np
from math import floor
import itertools

def create_json(text_data, users, split, rng):
    """Process the text data into json files in the same format as the VeriDark datasets

    Processes the text data into document pairs for AV. Resulting pairs are written to JSON

    Same author pairs are all the possible combinations of user documents

    We sample the same number of different author pairs as there are same author pairs by:
    - sample two authors from all the possible pairs of authors
    - sample a document from each of these authors
    - do this until there are the same number of different author pairs

    Arguments:
        text_data: dataframe
            user_id, text_id (this is where this text was located in its respective directory), text
        users: list
            list of users to pull from the text data for generating AV pairs
        split: str
            "train" or "test", used for naming the file that is written out
        rng: numpy random number generator
            used for the sampling, seeded in main code for reproducibility

    """
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
    ds_pairs = rng.choice(poss_ds, size=n_ss, replace=False)
    for pair in ds_pairs:
        user0 = pair[0]
        user1 = pair[1]
        user0_text = rng.choice(text_data.text[text_data.user_id == user0].to_list(), size=1)
        user1_text = rng.choice(text_data.text[text_data.user_id == user1].to_list(), size=1)
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

    # seed random number generator for reproducibility
    rng = np.random.default_rng(315558056615664205347275756463660878572)

    data_dir = './datasets/amazon/*/'

    user_ids = []
    texts = []

    # sort directories for reproducibility
    for prob_dir in sorted(glob(data_dir)):
        
        # sort files for reproducibility
        files = sorted(glob(escape(prob_dir) + '/*'))

        for file in files:
            user_id = file.split('/')[4].split(' - ')[1]
            user_ids.append(user_id)
            fp = open(file)
            texts.append(fp.read())
            fp.close()

    # put into dataframe
    text_data = pd.DataFrame({'user_id': user_ids, 'text': texts}).drop_duplicates()

    users = text_data.user_id.unique() # note pandas unique returns in order
    n_train_users = floor(0.9*len(users))
    train_users = users[0:n_train_users]
    test_users = users[n_train_users:len(users)]

    create_json(text_data, train_users, "train", rng)
    create_json(text_data, test_users, "test", rng)

if __name__ == "__main__":
    main()