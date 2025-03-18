from base_logger import logger
import argparse
import json
import pandas as pd
import numpy as np
import string
import csv
import re
from collections import Counter
from scipy import special
from models.models import NeuralScoreLR, ManualScoreLR
import utils.lexical as lex
import nltk

def process_json_to_df(fdir, stop_words, punct_list, english_vocab):
    """Process the json of text data into a pandas dataframe.

    For counting punctuation, takes all characters in the input string
    and checks against string.punctuation (this is what's passed to punct_list).

    For counting stop words, use the regex from the nltk wordpunct tokenizer
    https://www.nltk.org/api/nltk.tokenize.regexp.html#nltk.tokenize.regexp.WordPunctTokenizer
    except keep apostrophes (') together to be able to find those stop words (e.g., "it's").

    stop_words are passed as an argument but what's input are the words in
    the quanteda-stopwords.txt file.

    To avoid processing the english_vocab every time, it is passed as an argument.
    However, it's generally calculated via set(w.lower() for w in nltk.corpus.words.words())
    based on https://www.nltk.org/book/ch02.html#code-unusual

    Lexical features are implemented via the lexical.py file in utils/

    We exclude document pairs for which at least one of the texts has an undefined lexical feature.

    Arguments:
        fdir: str
        stop_words: list
        punct_list: list
        english_vocab: set

    Returns:
        text_data: dataframe
            problem id, same_source, user_id, text_id, text
        stop_punct_data: dataframe
            problem id, same_source, user_id, text_id, text, [punct cols], [stop cols]
        lexical_data: dataframe
            problem id, same_source, user_id, text_id, text, [lexical cols]
    """
    with open(fdir, 'r') as json_file:
        json_list = list(json_file)

    data_dicts = []

    for json_str in json_list:
        data_dicts.append(json.loads(json_str))

    problem_ids = []
    same_sources = []
    user_ids = []
    text_ids = []
    texts = []
    stop_punct_counts = []
    yulesi = []
    ttrs = []
    honorers = []
    unusuals = []
    uppers = []
    digits = []
    avgchars = []
    puncts = []

    for d in data_dicts:
        # first text
        text0 = d['pair'][0]
        text0_punct = [a for a in text0 if a in punct_list]
        text0_punct_count = Counter(text0_punct)
        text0_stop = [s.lower() for s in re.findall(r"[\w']+|[^\w\s]+", text0)]
        text0_stop_count = Counter(text0_stop)
        text0_yulei = lex.calc_yulei(text0)
        text0_ttr = lex.calc_ttr(text0)
        text0_honorer = lex.calc_honorer(text0)
        text0_unusual = lex.calc_unusual_word_ratio(text0, english_vocab)
        text0_upper = lex.calc_upper_case_ratio(text0)
        text0_digit = lex.calc_digit_ratio(text0)
        text0_avgchar = lex.calc_avg_chars(text0)
        text0_punct = lex.calc_punct_ratio(text0)
        # second text
        text1 = d['pair'][1]
        text1_punct = [a for a in text1 if a in punct_list]
        text1_punct_count = Counter(text1_punct)
        text1_stop = [s.lower() for s in re.findall(r"[\w']+|[^\w\s]+", text1)]
        text1_stop_count = Counter(text1_stop)
        text1_yulei = lex.calc_yulei(text1)
        text1_ttr = lex.calc_ttr(text1)
        text1_honorer = lex.calc_honorer(text1)
        text1_unusual = lex.calc_unusual_word_ratio(text1, english_vocab)
        text1_upper = lex.calc_upper_case_ratio(text1)
        text1_digit = lex.calc_digit_ratio(text1)
        text1_avgchar = lex.calc_avg_chars(text1)
        text1_punct = lex.calc_punct_ratio(text1)

        if not any([np.isinf(text0_yulei), np.isinf(text1_yulei), 
                    np.isinf(text0_unusual), np.isinf(text1_unusual), 
                    np.isinf(text0_honorer), np.isinf(text1_honorer)]):
            problem_ids.append(d['id'])
            same_sources.append(d['same'])
            user_ids.append(d['authors'][0])
            text_ids.append(0)
            texts.append(text0)
            stop_punct_counts.append([text0_punct_count[c] for c in punct_list] + [text0_stop_count[c] for c in stop_words])
            yulesi.append(text0_yulei)
            ttrs.append(text0_ttr)
            honorers.append(text0_honorer)
            unusuals.append(text0_unusual)
            uppers.append(text0_upper)
            digits.append(text0_digit)
            avgchars.append(text0_avgchar)
            puncts.append(text0_punct)

            problem_ids.append(d['id'])
            same_sources.append(d['same'])
            user_ids.append(d['authors'][1])
            text_ids.append(1)
            texts.append(text1)
            stop_punct_counts.append([text1_punct_count[c] for c in punct_list] + [text1_stop_count[c] for c in stop_words])
            yulesi.append(text1_yulei)
            ttrs.append(text1_ttr)
            honorers.append(text1_honorer)
            unusuals.append(text1_unusual)
            uppers.append(text1_upper)
            digits.append(text1_digit)
            avgchars.append(text1_avgchar)
            puncts.append(text1_punct)

    text_data = pd.DataFrame({'problem_id': problem_ids, 
                            'same_source': same_sources, 
                            'user_id': user_ids,
                            'text_id': text_ids,
                            'text': texts})

    stop_punct_data = pd.DataFrame(stop_punct_counts, columns=punct_list + stop_words)
    stop_punct_data = stop_punct_data.assign(problem_id = problem_ids, same_source = same_sources, user_id = user_ids, text_id = text_ids, text = texts)
    cols = ['problem_id', 'same_source', 'user_id', 'text_id', 'text'] + punct_list + stop_words
    stop_punct_data = stop_punct_data[cols]

    lexical_data = pd.DataFrame({'problem_id': problem_ids,
                                'same_source': same_sources,
                                'user_id': user_ids,
                                'text_id': text_ids,
                                'text': texts,
                                'yulei': yulesi,
                                'ttr': ttrs,
                                'honorer': honorers,
                                'unusual': unusuals,
                                'upper': uppers,
                                'digit': digits,
                                'avgchar': avgchars,
                                'punct': puncts})

    return text_data, stop_punct_data, lexical_data

def sample_data(text_data, stop_punct_data, znorm_count_data, znorm_lexical_data, n, rng):
    """Samples n same-author and n different-author exs from the relevant dataframes.

    Arguments:
    
    Returns:
        Samples of the data frames with n*4 rows (n*2 for each SA problem; n*2 for each DA)
        text_data: dataframe 
            problem_id, same_source, user_id, text_id, text
        stop_punct_data: dataframe
            problem id, same_source, user_id, text_id, text, [punct cols], [stop cols]
        znorm_count_data: dataframe (normalized)
            problem id, same_source, user_id, text_id, text, [punct cols], [stop cols]
        znorm_lexical_data: dataframe (normalized)
            problem id, same_source, user_id, text_id, text, [lexical cols]
    """
    ss_probs = rng.choice(text_data['problem_id'][text_data['same_source'] == True].unique(), size=n, replace=False).tolist()
    ds_probs = rng.choice(text_data['problem_id'][text_data['same_source'] == False].unique(), size=n, replace=False).tolist()

    text_data = text_data.loc[text_data['problem_id'].isin(ss_probs + ds_probs), :]
    stop_punct_data = stop_punct_data.loc[stop_punct_data['problem_id'].isin(ss_probs + ds_probs), :]
    znorm_count_data = znorm_count_data.loc[znorm_count_data['problem_id'].isin(ss_probs + ds_probs), :]
    znorm_lexical_data = znorm_lexical_data.loc[znorm_lexical_data['problem_id'].isin(ss_probs + ds_probs), :]

    return text_data, stop_punct_data, znorm_count_data, znorm_lexical_data

def z_transform(count_data, count_means, count_stds):
    """Z-score standardizes the data. Used in apply function on rows of the df.

    Arguments:
        count_data: arraylike
            vector of counts (row in the count df)
        count_means: arraylike
            vector of means for each word/punct
        count_stds: float
            vector of std devs for each word/punct
    
    Returns:
        standardized data row: arraylike (row in df)
    """
    return (count_data - count_means) / count_stds

def get_lnlr(r1, r2, alpha):
    """Calculates the natural log of the multinomial dirichlet LR.

    Returns ln{[B(r1 + r2 + alpha)*B(alpha)] / [B(r1 + alpha)*B(r2 + alpha)]}
    where B(.) is the multivariate beta function.

    The multivariate beta function B(x) is
    [Gamma(x_1)*Gamma(x_2)*...*Gamma(x_n)]/ [Gamma(x_1 +..._x_n)]
    where Gamma(.) is the standard gamma function.

    Use the natural log for numerical stability (values can be extreme).

    Arguments:
        r1: arraylike
            vector of counts
        r2: arraylike
            vector of counts
        alpha: arraylike
            dirichlet prior parameters

    Returns:
        lnlr: float
            natural log of the likelihood ratio
    """
    lnlr = (np.sum(special.loggamma(alpha + r1 + r2)) - special.loggamma(np.sum(alpha + r1 + r2))
           - np.sum(special.loggamma(alpha + r1)) + special.loggamma(np.sum(alpha + r1))
           - np.sum(special.loggamma(alpha + r2)) + special.loggamma(np.sum(alpha + r2))
           + np.sum(special.loggamma(alpha)) - special.loggamma(np.sum(alpha)))
    return lnlr

def main():
    logger.info("Starting setup")

    # set seed for reproducibility
    rng = np.random.default_rng(110926552114378630015775287249569522156)

    # set up command line parsing to get list of datasets to run experiments on
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", "--datasets", nargs='+', help="list of datasets to run experiments for")
    parser.add_argument("-ntr", "--ntrain", type=int, help="Number of instances to train on for each of same source and diff source")
    parser.add_argument("-nte", "--ntest", type=int, help="Number of instances to test on for each of same source and diff source")
    parser.add_argument("-nrep", "--nrepeat", type=int, help="Number of times to run experiment")
    args = parser.parse_args()

    # load in the stopword list from quanteda
    stop_words = []
    with open('quanteda-stopwords.txt', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            stop_words.append(row[0])
    # get list of punctuation
    punct_list = list(string.punctuation)
    # english vocab for unusual word ratio
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())

    logger.info("Setup finished. Starting experiments")
    for dataset in args.datasets:
        logger.info("Starting experiments for " + dataset)
        
        # data setup
        if dataset == "darkreddit":
            train_dir = "./datasets/darkreddit_authorship_verification_anon/darkreddit_authorship_verification_train_nodupe_anon.jsonl"
            test_dir = "./datasets/darkreddit_authorship_verification_anon/darkreddit_authorship_verification_test_nodupe_anon.jsonl"
        elif dataset == "silkroad":
            train_dir = "./datasets/darknet_authorship_verification_silkroad1_anon/darknet_authorship_verification_train_nodupe_anon.jsonl"
            test_dir = "./datasets/darknet_authorship_verification_silkroad1_anon/darknet_authorship_verification_test_nodupe_anon.jsonl"
        elif dataset == "agora":
            train_dir = "./datasets/darknet_authorship_verification_agora_anon/darknet_authorship_verification_train_nodupe_anon.jsonl"
            test_dir = "./datasets/darknet_authorship_verification_agora_anon/darknet_authorship_verification_test_nodupe_anon.jsonl"
        elif dataset == "amazon":
            train_dir = "./datasets/processed_amazon/amazon_train.jsonl"
            test_dir = "./datasets/processed_amazon/amazon_test.jsonl"
        
        # data processing
        logger.info("Processing data")
        train_text_data_full, train_stop_punct_data_full, train_lexical_data_full = process_json_to_df(fdir=train_dir, stop_words=stop_words, punct_list=punct_list, english_vocab=english_vocab)
        stoppunct_cols = train_stop_punct_data_full.columns[5:]
        lex_cols = train_lexical_data_full.columns[5:]
        
        test_text_data_full, test_stop_punct_data_full, test_lexical_data_full = process_json_to_df(fdir=test_dir, stop_words=stop_words, punct_list=punct_list, english_vocab=english_vocab)

        # z-score normalize count data
        # deduplicate first - same user/text combo might be in ss and ds
        z_data = train_stop_punct_data_full[['user_id', 'text'] + stoppunct_cols.to_list()].drop_duplicates()
        train_count_means = z_data[stoppunct_cols].mean()
        train_count_stds = z_data[stoppunct_cols].std()
        train_znorm_counts_full = train_stop_punct_data_full.copy()
        train_znorm_counts_full[stoppunct_cols] = train_znorm_counts_full[stoppunct_cols].apply(z_transform, axis=1, args=(train_count_means, train_count_stds))
        z_lex_data = train_lexical_data_full[['user_id', 'text'] + lex_cols.to_list()].drop_duplicates()
        train_lex_means = z_lex_data[lex_cols].mean()
        train_lex_stds = z_lex_data[lex_cols].std()
        train_znorm_lex_full = train_lexical_data_full.copy()
        train_znorm_lex_full[lex_cols] = train_znorm_lex_full[lex_cols].apply(z_transform, axis=1, args=(train_lex_means, train_lex_stds))

        # z transform test data
        test_znorm_counts_full = test_stop_punct_data_full.copy()
        test_znorm_counts_full[stoppunct_cols] = test_znorm_counts_full[stoppunct_cols].apply(z_transform, axis=1, args=(train_count_means, train_count_stds))
        test_znorm_lex_full = test_lexical_data_full.copy()
        test_znorm_lex_full[lex_cols] = test_znorm_lex_full[lex_cols].apply(z_transform, axis=1, args=(train_lex_means, train_lex_stds))

        # remove columns with 0 counts for standardized vals going into distances (since they're undefined)
        # they also would not change the cosine distance (since all 0s)
        znorm_stoppunct_cols = stoppunct_cols.drop(train_count_stds[(train_count_stds == 0)].index.tolist())
        train_znorm_counts_full = train_znorm_counts_full.loc[:, ['problem_id', 'same_source', 'user_id', 'text_id', 'text'] + znorm_stoppunct_cols.tolist()]
        test_znorm_counts_full = test_znorm_counts_full.loc[:, ['problem_id', 'same_source', 'user_id', 'text_id', 'text'] + znorm_stoppunct_cols.tolist()]

        for nrep in range(args.nrepeat):

            logger.info("Starting new repeat of experiment")

            # subsample data
            train_text_data, train_stop_punct_data, train_znorm_counts, train_znorm_lex = sample_data(
                train_text_data_full, 
                train_stop_punct_data_full,
                train_znorm_counts_full,
                train_znorm_lex_full, 
                args.ntrain,
                rng
            )
            test_text_data, test_stop_punct_data, test_znorm_counts, test_znorm_lex = sample_data(
                test_text_data_full, 
                test_stop_punct_data_full,
                test_znorm_counts_full,
                test_znorm_lex_full, 
                args.ntest,
                rng
            )

            # split data into same source, different source
            ss_train_data = train_text_data.loc[train_text_data['same_source'] == True, ['user_id', 'problem_id', 'text_id', 'text']]
            ds_train_data = train_text_data.loc[train_text_data['same_source'] == False, ['user_id', 'problem_id', 'text_id', 'text']]
            ss_train_znorm = train_znorm_counts.loc[train_znorm_counts['same_source'] == True, ['user_id', 'problem_id', 'text_id'] + znorm_stoppunct_cols.to_list()]
            ds_train_znorm = train_znorm_counts.loc[train_znorm_counts['same_source'] == False, ['user_id', 'problem_id', 'text_id'] + znorm_stoppunct_cols.to_list()]
            ss_train_lex = train_znorm_lex.loc[train_znorm_lex['same_source'] == True, ['user_id', 'problem_id', 'text_id'] + lex_cols.to_list()]
            ds_train_lex = train_znorm_lex.loc[train_znorm_lex['same_source'] == False, ['user_id', 'problem_id', 'text_id'] + lex_cols.to_list()]
            train_counts = train_stop_punct_data.loc[:, ['user_id', 'text'] + stoppunct_cols.to_list()].drop_duplicates()

            logger.info("Train text data nrows: " + str(train_text_data.shape[0]))
            logger.info("Test text data nrows: " + str(test_text_data.shape[0]))
            
            ss_test_data = test_text_data.loc[test_text_data['same_source'] == True, ['user_id', 'problem_id', 'text_id', 'text']]
            ds_test_data = test_text_data.loc[test_text_data['same_source'] == False, ['user_id', 'problem_id', 'text_id', 'text']]
            ss_test_znorm = test_znorm_counts.loc[test_znorm_counts['same_source'] == True, ['user_id', 'problem_id', 'text_id'] + znorm_stoppunct_cols.to_list()]
            ds_test_znorm = test_znorm_counts.loc[test_znorm_counts['same_source'] == False, ['user_id', 'problem_id', 'text_id'] + znorm_stoppunct_cols.to_list()]
            ss_test_counts = test_stop_punct_data.loc[test_stop_punct_data['same_source'] == True, ['user_id', 'problem_id', 'text_id'] + stoppunct_cols.to_list()]
            ds_test_counts = test_stop_punct_data.loc[test_stop_punct_data['same_source'] == False, ['user_id', 'problem_id', 'text_id'] + stoppunct_cols.to_list()]
            ss_test_lex = test_znorm_lex.loc[test_znorm_lex['same_source'] == True, ['user_id', 'problem_id', 'text_id'] + lex_cols.to_list()]
            ds_test_lex = test_znorm_lex.loc[test_znorm_lex['same_source'] == False, ['user_id', 'problem_id', 'text_id'] + lex_cols.to_list()]

            logger.info("ss train text data nrows: " + str(ss_train_data.shape[0]))
            logger.info("ds train text data nrows: " + str(ds_train_data.shape[0]))
            logger.info("ss test text data nrows: " + str(ss_test_data.shape[0]))
            logger.info("ds test text data nrows: " + str(ds_test_data.shape[0]))
            
            # set prior for multinomial dirichlet model
            col_counts = np.array(train_counts.iloc[:, 2:].sum(axis = 0)) + 1
            alpha = col_counts/sum(col_counts)*len(stoppunct_cols)

            logger.info("Starting training of neural slr methods")

            # instantiate lr models
            luar_slr_avg = NeuralScoreLR(embedding_model="luar", handle_long="avg", dist="cosine")
            luar_slr_tru = NeuralScoreLR(embedding_model="luar", handle_long="truncate", dist="cosine")
            luar_slr_win = NeuralScoreLR(embedding_model="luar", handle_long="window", dist="cosine")
            cisr_slr_avg = NeuralScoreLR(embedding_model="cisr", handle_long="avg", dist="cosine")
            cisr_slr_tru = NeuralScoreLR(embedding_model="cisr", handle_long="truncate", dist="cosine")
            znorm_slr = ManualScoreLR(feature_cols=znorm_stoppunct_cols, dist="cosine")
            lex_slr = ManualScoreLR(feature_cols=lex_cols, dist="cosine")

            # train slr methods
            logger.info("---------- Training LUAR AVG ----------")
            luar_slr_avg.train(ss_train_data, ds_train_data)
            logger.info("---------- Training LUAR TRU ----------")
            luar_slr_tru.train(ss_train_data, ds_train_data)
            logger.info("---------- Training LUAR WIN ----------")
            luar_slr_win.train(ss_train_data, ds_train_data)
            logger.info("---------- Training CISR AVG ----------")
            cisr_slr_avg.train(ss_train_data, ds_train_data)
            logger.info("---------- Training CISR TRU ----------")
            cisr_slr_tru.train(ss_train_data, ds_train_data)
            logger.info("---------- Training ZNORM ----------")
            znorm_slr.train(ss_train_znorm, ds_train_znorm)
            logger.info("---------- Training LEXICAL ----------")
            lex_slr.train(ss_train_lex, ds_train_lex)

            logger.info("Evaluating SLR methods on the test data")
            
            # evaluate on the test data
            logger.info("---------- Testing LUAR AVG ----------")
            ss_test_lrs_luar_avg, ds_test_lrs_luar_avg = luar_slr_avg.test(ss_test_data, ds_test_data)
            logger.info("---------- Testing LUAR TRU ----------")
            ss_test_lrs_luar_tru, ds_test_lrs_luar_tru = luar_slr_tru.test(ss_test_data, ds_test_data)
            logger.info("---------- Testing LUAR WIN ----------")
            ss_test_lrs_luar_win, ds_test_lrs_luar_win = luar_slr_win.test(ss_test_data, ds_test_data)
            logger.info("---------- Testing CISR AVG ----------")
            ss_test_lrs_cisr_avg, ds_test_lrs_cisr_avg = cisr_slr_avg.test(ss_test_data, ds_test_data)
            logger.info("---------- Testing CISR TRU ----------")
            ss_test_lrs_cisr_tru, ds_test_lrs_cisr_tru = cisr_slr_tru.test(ss_test_data, ds_test_data)
            logger.info("---------- Testing ZNORM ----------")
            ss_test_lrs_znorm, ds_test_lrs_znorm = znorm_slr.test(ss_test_znorm, ds_test_znorm)
            logger.info("---------- Testing LEXICAL ----------")
            ss_test_lrs_lex, ds_test_lrs_lex = lex_slr.test(ss_test_lex, ds_test_lex)

            logger.info("Evaluating multinomial dirichlet method on the test data")

            # multinomial evaluate on test data
            ss_test_lrs_counts = []
            for prob in ss_test_data['problem_id'].unique():
                r1 = ss_test_counts.loc[(ss_test_counts['problem_id'] == prob) & (ss_test_counts['text_id'] == 0), stoppunct_cols].values.flatten()
                r2 = ss_test_counts.loc[(ss_test_counts['problem_id'] == prob) & (ss_test_counts['text_id'] == 1), stoppunct_cols].values.flatten()
                ss_test_lrs_counts.append(get_lnlr(r1, r2, alpha))

            ds_test_lrs_counts = []
            for prob in ds_test_data['problem_id'].unique():
                r1 = ds_test_counts.loc[(ds_test_counts['problem_id'] == prob) & (ds_test_counts['text_id'] == 0), stoppunct_cols].values.flatten()
                r2 = ds_test_counts.loc[(ds_test_counts['problem_id'] == prob) & (ds_test_counts['text_id'] == 1), stoppunct_cols].values.flatten()
                ds_test_lrs_counts.append(get_lnlr(r1, r2, alpha))

            logger.info("Writing to CSV")

            # write training scores to csv for plots
            train_labels = [[True]*args.ntrain, [False]*args.ntrain]
            train_labels = [j for i in train_labels for j in i]
            train_scores_df = pd.DataFrame({'problem_id': ss_train_data['problem_id'].unique().tolist() + ds_train_data['problem_id'].unique().tolist(),
                                            'same_source': train_labels,
                                            'luar_avg_score': getattr(luar_slr_avg, 'ss_scores') + getattr(luar_slr_avg, 'ds_scores'),
                                            'luar_tru_score': getattr(luar_slr_tru, 'ss_scores') + getattr(luar_slr_tru, 'ds_scores'),
                                            'luar_win_score': getattr(luar_slr_win, 'ss_scores') + getattr(luar_slr_win, 'ds_scores'),
                                            'cisr_avg_score': getattr(cisr_slr_avg, 'ss_scores') + getattr(cisr_slr_avg, 'ds_scores'),
                                            'cisr_tru_score': getattr(cisr_slr_tru, 'ss_scores') + getattr(cisr_slr_tru, 'ds_scores'),
                                            'znorm_score': getattr(znorm_slr, 'ss_scores') + getattr(znorm_slr, 'ds_scores'),
                                            'lex_score': getattr(lex_slr, 'ss_scores') + getattr(lex_slr, 'ds_scores')})
            train_scores_df.to_csv('./results/' + dataset + '_train' + str(nrep) + '.csv', index=False)

            # write test scores to csv for analysis
            test_labels = [[True]*args.ntest, [False]*args.ntest]
            test_labels = [j for i in test_labels for j in i]
            test_lr_results = pd.DataFrame({'problem_id': ss_test_data['problem_id'].unique().tolist() + ds_test_data['problem_id'].unique().tolist(),
                                    'same_source': test_labels,
                                    'luar_avg_lr': np.concatenate(ss_test_lrs_luar_avg).tolist() + np.concatenate(ds_test_lrs_luar_avg).tolist(),
                                    'luar_tru_lr': np.concatenate(ss_test_lrs_luar_tru).tolist() + np.concatenate(ds_test_lrs_luar_tru).tolist(),
                                    'luar_win_lr': np.concatenate(ss_test_lrs_luar_win).tolist() + np.concatenate(ds_test_lrs_luar_win).tolist(),
                                    'cisr_avg_lr': np.concatenate(ss_test_lrs_cisr_avg).tolist() + np.concatenate(ds_test_lrs_cisr_avg).tolist(),
                                    'cisr_tru_lr': np.concatenate(ss_test_lrs_cisr_tru).tolist() + np.concatenate(ds_test_lrs_cisr_tru).tolist(),
                                    'znorm_lr': np.concatenate(ss_test_lrs_znorm).tolist() + np.concatenate(ds_test_lrs_znorm).tolist(),
                                    'lex_lr': np.concatenate(ss_test_lrs_lex).tolist() + np.concatenate(ds_test_lrs_lex).tolist(),
                                    'count_lr': ss_test_lrs_counts + ds_test_lrs_counts})
            
            test_lr_results.to_csv('./results/' + dataset + '_test' + str(nrep) + '.csv', index=False)

if __name__ == "__main__":
    main()