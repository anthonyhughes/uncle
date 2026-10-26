import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import tqdm
import spacy


class NLProcessor(object):
    def __init__(self, rng):
        self.rng = rng
        self.nlp = spacy.load("en_core_web_sm")
        self.vectorizer = CountVectorizer(min_df=5)
        self.word_vec_len = 300

    def process_spam(self, spam, ham):
        """
        Processes spam and ham emails to return lemmatized, cleaned documents and class labels.
        """
        # Combine spam and ham data
        all_docs = spam + ham
        n_spam = len(spam)

        # Process documents with spaCy
        docs = [
            ' '.join(
                [token.lemma_ for token in self.nlp(raw_doc)
                 if token.is_alpha and not (token.is_stop)]
            )
            for raw_doc in tqdm.tqdm(all_docs)
        ]

        # Create labels: spam = 1, ham = 0
        Y = np.concatenate((np.ones(n_spam), np.zeros(len(ham))))

        # Shuffle data and labels together
        indices = self.rng.permutation(len(docs))
        docs = [docs[i] for i in indices]
        Y = Y[indices]

        return docs, Y

    def process_newsgroups(self, newsgroups):
        """
        Processes newsgroups data to return lemmatized, cleaned documents and class labels.
        """
        # Process documents with spaCy
        docs = [
            ' '.join(
                [token.lemma_ for token in self.nlp(raw_doc)
                 if token.is_alpha and not (token.is_stop)]
            )
            for raw_doc in tqdm.tqdm(newsgroups.data)
        ]

        # Convert target to {+1, -1}
        Y = (np.array(newsgroups.target) * 2) - 1

        return docs, Y

    def learn_vocab(self, docs):
        """
        Learns a vocabulary from the given documents using CountVectorizer.
        """
        self.vectorizer.fit(docs)

    def get_bag_of_words(self, docs):
        """
        Converts documents into a bag-of-words representation.
        """
        return self.vectorizer.transform(docs)

    def get_mean_word_vector(self, docs):
        """
        Computes the mean word vector for each document using spaCy's word vectors.
        """
        n = len(docs)
        X = np.zeros((n, self.word_vec_len))

        for i, doc in enumerate(tqdm.tqdm(docs)):
            vectors = np.array([token.vector for token in self.nlp(doc)])
            if len(vectors) > 0:
                X[i, :] = vectors.mean(axis=0)

        return X
