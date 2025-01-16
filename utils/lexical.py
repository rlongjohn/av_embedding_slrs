from lexicalrichness import LexicalRichness
import pandas as pd
import numpy as np
from collections import Counter
import re
import string


def get_wordfrequency_table(bow):
    """Get table of i frequency and number of terms that appear i times in text of length N.

    For Yule's I, Yule's K, and Simpson's D.
    
    In the returned table, freq column indicates the number of frequency of appearance in
    the text. fv_i_N column indicates the number of terms in the text of length N that
    appears freq number of times.

    (Rachel note: copied from lexicalrichness package - they have a bug in their yule's i)

    Parameters
    ----------
    bow: array-like
        List of words

    Returns
    -------
    pandas.core.frame.DataFrame
    """
    term_freq_dict = Counter(bow)

    freq_i_N = (pd.DataFrame.from_dict(term_freq_dict, orient='index')
                .reset_index()
                .rename(columns={0: 'freq'})
                .groupby('freq').size().reset_index()
                .rename(columns={0: 'fv_i_N'})
                .assign(sum_element=lambda df: df.fv_i_N * np.square(df.freq))
                )

    return freq_i_N

def calc_yulei(text):
    """Yule's I (Yule 1944).

    (Rachel note - modified from the package; they have a bug)

    See Also
    --------
    frequency_wordfrequency_table:
        Get table of i frequency and number of terms that appear i times in text of length N.

    Returns
    -------
    Float
        Yule's I
    """

    lexr = LexicalRichness(text)
    freq_i_N = get_wordfrequency_table(lexr.wordlist)
    total_sum = freq_i_N.sum_element.sum()
    i = lexr.words**2 / (total_sum - lexr.words)
    return i

def calc_ttr(text):
    lexr = LexicalRichness(text)
    return lexr.ttr

def calc_honorer(text):
    lexr = LexicalRichness(text)
    freq_i_N = get_wordfrequency_table(lexr.wordlist)
    if sum(freq_i_N.freq == 1) == 0:
        v1 = 0
    else:
        v1 = freq_i_N.loc[freq_i_N['freq'] == 1, 'fv_i_N'].item()
    return 100*np.log(lexr.words)/(1-v1/lexr.terms)

def calc_unusual_word_ratio(text, english_vocab):
    text_sep = text.split()
    text_vocab = set(w.lower() for w in text_sep if w.isalpha())
    unusual = text_vocab - english_vocab
    if len(text_vocab) == 0:
        return np.inf
    else:
        return len(unusual)/len(text_vocab)

def calc_upper_case_ratio(text):
    n_char = sum(not chr.isspace() for chr in text)
    return len(re.findall(r'[A-Z]', text)) / n_char

def calc_digit_ratio(text):
    n_char = sum(not chr.isspace() for chr in text)
    n_digit = sum(chr in '0123456789' for chr in text)
    return n_digit / n_char

def calc_avg_chars(text):
    lexr = LexicalRichness(text)
    return sum( map(len, lexr.wordlist) ) / len(lexr.wordlist)

def calc_punct_ratio(text):
    n_char = sum(not chr.isspace() for chr in text)
    n_punct = sum(chr in string.punctuation for chr in text)
    return n_punct / n_char

def apply_yulei(row):
    return calc_yulei(row['text'])

def apply_ttr(row):
    return calc_ttr(row['text'])

def apply_honorer(row):
    return calc_honorer(row['text'])

def apply_unusual(row, vocab):
    return calc_unusual_word_ratio(row['text'], vocab)

def apply_upper(row):
    return calc_upper_case_ratio(row['text'])

def apply_digit(row):
    return calc_digit_ratio(row['text'])

def apply_avgchar(row):
    return calc_avg_chars(row['text'])

def apply_punct(row):
    return calc_punct_ratio(row['text'])