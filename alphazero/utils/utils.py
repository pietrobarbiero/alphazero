import logging
import os
import pickle


def save_as_pickle(filename, data):
    completeName = os.path.join("./datasets/",
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)


def load_pickle(filename):
    completeName = os.path.join("./datasets/",
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data
