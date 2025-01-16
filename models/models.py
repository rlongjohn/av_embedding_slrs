from sklearn.neighbors import KernelDensity
import numpy as np
from models.embeddings import LuarModel, CisrModel
from scipy.spatial import distance
from base_logger import logger

class ScoreLR:
    """Defines the score-based likelihood ratio model.
    """
    def __init__(self, kernel = "gaussian", dist = "cosine"):
        self.kernel = kernel
        self.dist = dist
        self.ss_scores = []
        self.ds_scores = []

    def score(self, x, y):
        if self.dist == "cosine":
            return distance.cosine(x,y)
        else:
            return distance.euclidean(x,y)

    def fit_densities(self):
        n = len(self.ss_scores)
        iqr_ss = np.quantile(self.ss_scores, 0.75) - np.quantile(self.ss_scores, 0.25)
        b_ss = min(np.std(self.ss_scores), iqr_ss/1.34)
        iqr_ds = np.quantile(self.ds_scores, 0.75) - np.quantile(self.ds_scores, 0.25)
        b_ds = min(np.std(self.ds_scores), iqr_ds/1.34)
        self.ss_bw = 0.9*b_ss*n**(-1/5)
        self.ds_bw = 0.9*b_ds*n**(-1/5)
        self.ss_kde = KernelDensity(kernel=self.kernel, bandwidth=self.ss_bw).fit(np.array(self.ss_scores)[:, np.newaxis])
        self.ds_kde = KernelDensity(kernel=self.kernel, bandwidth=self.ds_bw).fit(np.array(self.ds_scores)[:, np.newaxis])

    def slr_from_score(self, scores):
        scores = np.array(scores)[:, np.newaxis]
        return np.exp(self.ss_kde.score_samples(scores)) / np.exp(self.ds_kde.score_samples(scores))
    
class ManualScoreLR(ScoreLR):
    """Defines the score-based likelihood ratio model based on manually-defined features.
    """ 
    def __init__(self, feature_cols, kernel = "gaussian", dist = "cosine"):
        super().__init__(kernel, dist)
        self.feature_cols = feature_cols

    def train(self, ss_train_data, ds_train_data):
        self.ss_scores = []
        self.ds_scores = []
        logger.info("Training same source")
        ss_probs = list(set(ss_train_data['problem_id']))
        counter = 0
        for prob in ss_probs:
            if counter % 100 == 0:
                logger.info("On row: " + str(counter))
            counter += 1
            
            znorm0 = ss_train_data.loc[(ss_train_data['problem_id'] == prob) & (ss_train_data['text_id'] == 0), self.feature_cols].values.flatten().tolist()
            znorm1 = ss_train_data.loc[(ss_train_data['problem_id'] == prob) & (ss_train_data['text_id'] == 1), self.feature_cols].values.flatten().tolist()

            self.ss_scores.append(self.score(znorm0, znorm1))

        logger.info("Training different source")
        ds_probs = list(set(ds_train_data['problem_id']))
        counter = 0
        for prob in ds_probs:
            if counter % 100 == 0:
                logger.info("On row: " + str(counter))
            counter += 1

            znorm0 = ds_train_data.loc[(ds_train_data['problem_id'] == prob) & (ds_train_data['text_id'] == 0), self.feature_cols].values.flatten().tolist()
            znorm1 = ds_train_data.loc[(ds_train_data['problem_id'] == prob) & (ds_train_data['text_id'] == 1), self.feature_cols].values.flatten().tolist()

            self.ds_scores.append(self.score(znorm0, znorm1))

        logger.info("Fitting densities")
        self.fit_densities()

    def test(self, ss_test_data, ds_test_data):
        self.ss_test_scores = []
        ss_test_probs = list(set(ss_test_data['problem_id']))
        ss_test_slrs = []

        counter = 0
        logger.info("Testing same source")
        for prob in ss_test_probs:

            if counter % 100 == 0:
                logger.info("On row: " + str(counter))
            counter += 1

            znorm0 = ss_test_data.loc[(ss_test_data['problem_id'] == prob) & (ss_test_data['text_id'] == 0), self.feature_cols].values.flatten().tolist()
            znorm1 = ss_test_data.loc[(ss_test_data['problem_id'] == prob) & (ss_test_data['text_id'] == 1), self.feature_cols].values.flatten().tolist()
            score = self.score(znorm0, znorm1)

            self.ss_test_scores.append(score)
            ss_test_slrs.append(self.slr_from_score([score]))
    
        self.ds_test_scores = []
        ds_test_probs = list(set(ds_test_data['problem_id']))

        ds_test_slrs = []

        counter = 0
        logger.info("Testing different source")
        for prob in ds_test_probs:

            if counter % 100 == 0:
                logger.info("On row: " + str(counter))
            counter += 1

            znorm0 = ds_test_data.loc[(ds_test_data['problem_id'] == prob) & (ds_test_data['text_id'] == 0), self.feature_cols].values.flatten().tolist()
            znorm1 = ds_test_data.loc[(ds_test_data['problem_id'] == prob) & (ds_test_data['text_id'] == 1), self.feature_cols].values.flatten().tolist()
            score = self.score(znorm0, znorm1)

            self.ds_test_scores.append(score)
            ds_test_slrs.append(self.slr_from_score([score]))

        return ss_test_slrs, ds_test_slrs

class NeuralScoreLR(ScoreLR):
    """Defines the score-based likelihood ratio model based on neural network features.
    """ 
    def __init__(self, embedding_model, handle_long = "truncate", kernel = "gaussian", dist="cosine"):
        super().__init__(kernel, dist)
        self.handle_long = handle_long
        
        if embedding_model == "luar":
            self.embedding_model = LuarModel()
        elif embedding_model == "cisr":
            self.embedding_model = CisrModel()
        else:
            self.embedding_model = LuarModel()

    def score(self, x, y):
        if self.handle_long == "avg":
            emb_x = self.embedding_model.calc_embedding_avg(x)
            emb_y = self.embedding_model.calc_embedding_avg(y)
        else:
            emb_x = self.embedding_model.calc_embedding_truncated(x)
            emb_y = self.embedding_model.calc_embedding_truncated(y)
        
        if self.dist == "cosine":
            return distance.cosine(emb_x, emb_y)
        else:
            return distance.euclidean(emb_x, emb_y)
    
    def train(self, ss_train_data, ds_train_data):
        self.ss_scores = []
        self.ds_scores = []
        logger.info("Training same source")
        ss_probs = list(set(ss_train_data['problem_id']))
        counter = 0
        for prob in ss_probs:
            if counter % 100 == 0:
                logger.info("On row: " + str(counter))
            counter += 1
            
            text0 = ss_train_data['text'][(ss_train_data['problem_id'] == prob) & (ss_train_data['text_id'] == 0)].tolist()[0]
            text1 = ss_train_data['text'][(ss_train_data['problem_id'] == prob) & (ss_train_data['text_id'] == 1)].tolist()[0]

            self.ss_scores.append(self.score(text0, text1))

        logger.info("Training different source")
        ds_probs = list(set(ds_train_data['problem_id']))
        counter = 0
        for prob in ds_probs:
            if counter % 100 == 0:
                logger.info("On row: " + str(counter))
            counter += 1
            
            text0 = ds_train_data['text'][(ds_train_data['problem_id'] == prob) & (ds_train_data['text_id'] == 0)].tolist()[0]
            text1 = ds_train_data['text'][(ds_train_data['problem_id'] == prob) & (ds_train_data['text_id'] == 1)].tolist()[0]

            self.ds_scores.append(self.score(text0, text1))


        logger.info("Fitting densities")
        self.fit_densities()

    def test(self, ss_test_data, ds_test_data):
        self.ss_test_scores = []
        ss_test_probs = list(set(ss_test_data['problem_id']))

        ss_test_slrs = []

        counter = 0
        logger.info("Testing same source")
        for prob in ss_test_probs:
            if counter % 100 == 0:
                logger.info("On row: " + str(counter))
            counter += 1

            text0 = ss_test_data['text'][(ss_test_data['problem_id'] == prob) & (ss_test_data['text_id'] == 0)].tolist()[0]
            text1 = ss_test_data['text'][(ss_test_data['problem_id'] == prob) & (ss_test_data['text_id'] == 1)].tolist()[0]
            score = self.score(text0, text1)

            self.ss_test_scores.append(score)
            ss_test_slrs.append(self.slr_from_score([score]))

        self.ds_test_scores = []
        ds_test_probs = list(set(ds_test_data['problem_id']))

        ds_test_slrs = []

        counter = 0
        logger.info("Testing different source")
        for prob in ds_test_probs:
            if counter % 100 == 0:
                logger.info("On row: " + str(counter))
            counter += 1

            text0 = ds_test_data['text'][(ds_test_data['problem_id'] == prob) & (ds_test_data['text_id'] == 0)].tolist()[0]
            text1 = ds_test_data['text'][(ds_test_data['problem_id'] == prob) & (ds_test_data['text_id'] == 1)].tolist()[0]
            score = self.score(text0, text1)

            self.ds_test_scores.append(score)
            ds_test_slrs.append(self.slr_from_score([score]))

        return ss_test_slrs, ds_test_slrs