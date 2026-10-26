from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

# Loads the enron spam dataset with the old python 2.7 / spacy 2.0.0
# used by Koh et al. 2019. This implementation is meant to reproduce their
# results as faithfully as possible, even where the best practices have changed.
# Run this with the following poetry install:

'''
[tool.poetry]
name = "group-influence"
version = "0.1.0"
description = "A project with legacy Python 2 dependencies"
authors = ["Your Name <your.email@example.com>"]

[tool.poetry.dependencies]
python = ">=2.7,<2.8"
jupyter = "1.0.0"
matplotlib = "2.2.4"
numpy = "1.16.5"
pandas = "0.24.2"
scikit-learn = "0.20.4"
scipy = "1.2.2"
spacy = "2.0.0"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
'''


import spacy
from sklearn.feature_extraction.text import CountVectorizer
import os
import numpy as np
import sys

import tqdm

class NLProcessor(object):
    def __init__(self, rng):
        # self.nlp = English()
        self.rng = rng
        self.nlp = spacy.load('en_core_web_sm')

        # spacy is amazing at figuring out if words in its
        # vocabulary are in its vocabulary
        for lexeme in self.nlp.vocab:
            lexeme.is_oov = False

        # self.nlp = spacy.load('en_core_web_sm-1.2.0')
        self.vectorizer = CountVectorizer(min_df=5)
        self.word_vec_len = 300

    def process_spam(self, spam, ham):
        """
        Takes in a list of spam emails and a list of ham emails
        and returns a tuple (docs, Y), where:
        - docs is a list of documents, with each document lemmatized
        and stripped of stop and OOV words.
        - Y is an array of classes {0, 1}. Each element is an example.
        +1 means spam, 0 means ham.
        """
        docs = []
        for raw_doc in tqdm.tqdm(spam + ham, desc="Preprocessing tokens..."):
            doc = self.nlp(raw_doc)
            docs.append(' '.join(
                [token.lemma_ for token in doc if (token.is_alpha and not (token.is_oov or token.is_stop))]))

        Y = np.zeros(len(spam) + len(ham))
        Y[:len(spam)] = 1
        Y[len(spam):] = 0

        docs_Y = zip(docs, Y)
        self.rng.shuffle(docs_Y)
        docs, Y = zip(*docs_Y)

        Y = np.array(Y)

        return docs, Y

    def process_newsgroups(self, newsgroups):
        """
        Takes in a newsgroups object returned by fetch_20newsgroups()
        and returns a tuple (docs, Y), where:
        - docs is a list of documents, with each document lemmatized
        and stripped of stop and OOV words.
        - Y is an array of classes {+1, -1}. Each element is an example.
        """
        docs = []
        for raw_doc in newsgroups.data:
            doc = self.nlp(raw_doc)
            docs.append(' '.join(
                [token.lemma_ for token in doc if (token.is_alpha and not (token.is_oov or token.is_stop))]))

        # Convert target to {+1, -1}. It is originally {+1, 0}.
        Y = (np.array(newsgroups.target) * 2) - 1

        return (docs, Y)

    def learn_vocab(self, docs):
        """
        Learns a vocabulary from docs.
        """
        self.vectorizer.fit(docs)

    def get_bag_of_words(self, docs):
        """
        Takes in a list of documents and converts it into a bag of words
        representation. Returns X, a sparse matrix where each row is an example
        and each column is a feature (word in the vocab).
        """
        X = self.vectorizer.transform(docs)
        return X

    def get_mean_word_vector(self, docs):
        """
        Takes in a list of documents and returns X, a matrix where each row
        is an example and each column is the mean word vector in that document.
        """
        n = len(docs)
        X = np.empty([n, self.word_vec_len])
        doc_vec = np.zeros(self.word_vec_len)
        for idx, doc in enumerate(tqdm.tqdm(docs, desc="Generating X matrix...")):
            doc_vec = reduce(lambda x, y: x + y, [token.vector for token in self.nlp(doc)])
            doc_vec /= n
            X[idx, :] = doc_vec
        return X


def init_lists(folder):
    a_list = []
    file_list = os.listdir(folder)
    for a_file in tqdm.tqdm(file_list, desc="init_lists..."):
        with open(os.path.join(folder, a_file), 'r') as f:
            a_list.append(f.read().decode("latin-1"))
    return a_list


def process_spam(dataset_dir, truncate=None):
    rng = np.random.RandomState(0)
    nlprocessor = NLProcessor(rng)

    spam = init_lists(os.path.join(dataset_dir, 'enron1', 'spam'))
    ham = init_lists(os.path.join(dataset_dir, 'enron1', 'ham'))

    docs, Y = nlprocessor.process_spam(spam[:truncate], ham[:truncate])
    num_examples = len(Y)

    train_fraction = 0.8
    valid_fraction = 0.0
    num_train_examples = int(train_fraction * num_examples)
    num_valid_examples = int(valid_fraction * num_examples)
    num_test_examples = num_examples - num_train_examples - num_valid_examples

    docs_train = docs[:num_train_examples]
    Y_train = Y[:num_train_examples]

    docs_valid = docs[num_train_examples: num_train_examples + num_valid_examples]
    Y_valid = Y[num_train_examples: num_train_examples + num_valid_examples]

    docs_test = docs[-num_test_examples:]
    Y_test = Y[-num_test_examples:]

    assert (len(docs_train) == len(Y_train))
    assert (len(docs_valid) == len(Y_valid))
    assert (len(docs_test) == len(Y_test))
    assert (len(Y_train) + len(Y_valid) + len(Y_test) == num_examples)

    nlprocessor.learn_vocab(docs_train)
    X_train = nlprocessor.get_bag_of_words(docs_train)
    X_valid = nlprocessor.get_bag_of_words(docs_valid)
    X_test = nlprocessor.get_bag_of_words(docs_test)

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


def load_spam(spam_path, output_path):
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = process_spam(spam_path, None)

    # Convert them to dense matrices
    X_train = X_train.toarray()
    X_valid = X_valid.toarray()
    X_test = X_test.toarray()

    np.savez(output_path,
             X_train=X_train,
             Y_train=Y_train,
             X_test=X_test,
             Y_test=Y_test,
             X_valid=X_valid,
             Y_valid=Y_valid)

if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] in ["-h", "--help"]:
        print("Usage: python load_spam.py <spam_path> <output_path>")
        print("\nArguments:")
        print("  <spam_path>   Path to the spam dataset file.")
        print("  <output_path> Path to the output .npz file.")
        sys.exit(1)

    spam_path = sys.argv[1]
    output_path = sys.argv[2]

    try:
        load_spam(spam_path, output_path)
        print("Spam data processed and saved to:", output_path)
    except Exception as e:
        print("An error occurred:", e)
        sys.exit(1)