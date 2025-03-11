import pandas as pd
import numpy as np
from collections import Counter
import re
import string

def get_wordfrequency_table(words):
    """Get table of i frequency and number of terms that appear i times in text of length N.

    For Yule's I and Honore's R.
    
    In the returned table, freq column indicates the number of frequency of appearance in
    the text. fv_i_N column indicates the number of terms in the text of length N that
    appears freq number of times.

    (Rachel note: copied from lexicalrichness package 
    https://github.com/LSYS/LexicalRichness/blob/master/lexicalrichness/lexicalrichness.py)

    Parameters
    ----------
    words: array-like
        List of words

    Returns
    -------
    pandas.core.frame.DataFrame with columns
        freq: int
            number of word occurrences
        fv_i_N: int
            number of words in the text that appear "freq" times
        sum_element: int
            sum element used in yule's i. equal to freq^2*fv_i_N
    """
    term_freq_dict = Counter(words)

    freq_i_N = (pd.DataFrame.from_dict(term_freq_dict, orient='index')
                .reset_index()
                .rename(columns={0: 'freq'})
                .groupby('freq').size().reset_index()
                .rename(columns={0: 'fv_i_N'})
                .assign(sum_element=lambda df: df.fv_i_N * np.square(df.freq))
                )

    return freq_i_N

def calc_ttr(text):
    """Calculate the type-token ratio

    Calculates the type-token ratio, which is equal to
    (# unique words)/(# total words)

    The text is split into words with string split() which is equivalent to the nltk whitespace tokenizer, see
    https://www.nltk.org/api/nltk.tokenize.regexp.html#nltk.tokenize.regexp.WhitespaceTokenizer

    Note that this only splits on whitespace, so e.g., trailing punctuation is considered part
    of the preceding word.
    Hello, my name is Rachel. --> ["Hello,", "my", "name", "is", "Rachel."]

    Arguments:
    text: str
        Text to calculate the type-token ratio for

    Returns:
    float
        Type-token ratio for the text.
    """
    words = text.split()
    unique_words = set([w.lower() for w in words])
    return len(unique_words) / len(words)

def calc_honorer(text):
    """Calculate Honore's R

    Calculates Honore's R, which is equal to
    100 ∗ (log # total words )/ (1 − (# words appearing once)/(# unique words))

    The text is split into words with string split() which is equivalent to the nltk whitespace tokenizer, see
    https://www.nltk.org/api/nltk.tokenize.regexp.html#nltk.tokenize.regexp.WhitespaceTokenizer

    Note that this only splits on whitespace, so e.g., trailing punctuation is considered part
    of the preceding word.
    Hello, my name is Rachel. --> ["Hello,", "my", "name", "is", "Rachel."]

    Arguments:
    text: str
        Text to calculate Honore's R for

    Returns:
    float
        Honore's R for the text.
        inf if all the words are only used once.
    """
    words = text.split()
    words = [w.lower() for w in words]
    freq_i_N = get_wordfrequency_table(words)
    if sum(freq_i_N.freq == 1) == 0:
        v1 = 0
    else:
        v1 = freq_i_N.loc[freq_i_N['freq'] == 1, 'fv_i_N'].item()
    
    unique_words = set([w.lower() for w in words])
    if len(unique_words) == v1:
        return np.inf
    else:
        return 100*np.log(len(words))/(1-(v1/len(unique_words)))
    
def calc_yulei(text):
    """Calculate Yule's I

    Calculates Yule's I, which is equal to
    (# total words)^2 / [\sum_l l^2*(# words occurring l times) - # total words]
    based on the formula in https://doi.org/10.1558/ijsll.30305

    The text is split into words with string split() which is equivalent to the nltk whitespace tokenizer, see
    https://www.nltk.org/api/nltk.tokenize.regexp.html#nltk.tokenize.regexp.WhitespaceTokenizer

    Note that this only splits on whitespace, so e.g., trailing punctuation is considered part
    of the preceding word.
    Hello, my name is Rachel. --> ["Hello,", "my", "name", "is", "Rachel."]

    Arguments:
    text: str
        Text to calculate Yule's I for

    Returns:
    float
        Yule's I for the text.
    """

    words = text.split()
    words = [w.lower() for w in words]
    freq_i_N = get_wordfrequency_table(words)
    total_sum = freq_i_N.sum_element.sum()
    return len(words)**2 / (total_sum - len(words))

def calc_unusual_word_ratio(text, english_vocab):
    """Calculate the unusual word ratio
    
    Calculates the unusual word ratio using the implementation in https://www.nltk.org/book/ch02.html#code-unusual.

    To avoid processing the english_vocab every time, it is passed as an argument.
    However, it's generally calculated via set(w.lower() for w in nltk.corpus.words.words()) per the link above.

    The text is split with string split() which is equivalent to the nltk whitespace tokenizer, see
    https://www.nltk.org/api/nltk.tokenize.regexp.html#nltk.tokenize.regexp.WhitespaceTokenizer

    Arguments:
    text: str
        Text to calculate the unusual word ratio for
    english_vocab: set
        Set of usual English words

    Returns:
    float
        Unusual word ratio. Returns inf if there are no words in the text that are strictly alphabetic.

    """
    text_sep = text.split()
    text_vocab = set(w.lower() for w in text_sep if w.isalpha())
    unusual = text_vocab - english_vocab
    if len(text_vocab) == 0:
        return np.inf
    else:
        return len(unusual)/len(text_vocab)

def calc_upper_case_ratio(text):
    """Calculate the upper case ratio

    Calculates the ratio of upper case characters to non-whitespace characters.

    Arguments:
    text: str
        Text to calculate the upper case ratio for
    
    Returns:
    float
        Upper case ratio.
    """
    n_char = sum(not chr.isspace() for chr in text)
    return len(re.findall(r'[A-Z]', text)) / n_char

def calc_digit_ratio(text):
    """Calculate the digit ratio

    Calculates the ratio of digit characters to non-whitespace characters.

    Arguments:
    text: str
        Text to calculate the digit ratio for

    Returns:
    float
        Digit ratio.
    """
    n_char = sum(not chr.isspace() for chr in text)
    n_digit = sum(chr in '0123456789' for chr in text)
    return n_digit / n_char

def calc_avg_chars(text):
    """Calculate the average characters per word

    Calculates the average number of characters per word.

    The text is split with string split() which is equivalent to the nltk whitespace tokenizer, see
    https://www.nltk.org/api/nltk.tokenize.regexp.html#nltk.tokenize.regexp.WhitespaceTokenizer

    Note that this only splits on whitespace, so e.g., trailing punctuation is considered part
    of the preceding word.
    Hello, my name is Rachel. --> ["Hello,", "my", "name", "is", "Rachel."]

    Arguments:
    text: str
        Text to calculate the average number of characters for

    Returns:
    float
        Average number of characters per word in the text.
    """
    words = text.split()
    return sum(map(len, words)) / len(words)

def calc_punct_ratio(text):
    """Calculate the punctuation ratio

    Calculates the ratio of punctuation characters to non-whitespace characters.
    Uses the list of punctuation in python's string.punctuation.

    Arguments:
    text: str
        Text to calculate the punctuation ratio for

    Returns:
    float
        Punctuation ratio.
    """
    n_char = sum(not chr.isspace() for chr in text)
    n_punct = sum(chr in string.punctuation for chr in text)
    return n_punct / n_char
    