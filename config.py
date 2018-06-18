"""
All constants configuration for the project is here
"""
import os
#import hunspell

###################################################
#   file system configuration                     #
###################################################
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = "/data/SemEval2014/"
DOMAIN = "Restaurants"
DATA_FILE = "Restaurants_Trial.xml"
TRAIN_FILE = "rest_train_2014_processed.csv"
TEST_FILE = "rest_test_2014_processed.csv"
WORD2VEC_FILE = "glove.840B.300d.txt"
FOLDER = ROOT_PATH + DATA_PATH
EMBEDDING_PATH = FOLDER + WORD2VEC_FILE
#HUNSPELL_PATH = "/usr/share/hunspell/"
#HUNSPELL_DICT = ["en_US.dic", "en_US.aff"]

# loading dictionaries for hunspell is a bit wierd, you have to put the dictionaries
# in a root-derivative folder path e.g. a folder ~/some-other-path is not allowed
#hobj = hunspell.HunSpell(HUNSPELL_PATH + HUNSPELL_DICT[0],
#                         HUNSPELL_PATH + HUNSPELL_DICT[1])


def configure(year, domain, embedding, aim):
    """
    set file related parameters
    year: 2014 or 2016
    domain: rest or laptop
    embedding: glove or word2vec
    aim: trial, train or test
    """
    global DATA_FOLDER, DOMAIN, DATA_FILE, WORD2VEC_FILE
    DATA_FOLDER = "data/SemEval%s/" % year
    domainDict = {"rest": "Restaurants", "laptop": "Laptops"}
    DOMAIN = domainDict[domain]
    # fileDict = {"trial": "Restaurants_Trial.xml", "train": "%s_Train.xml" % DOMAIN, "test": "%s_Test_Data_phaseB.xml" % DOMAIN}
    fileDict = {"trial": "Restaurants_Trial.xml", "train": "%s_Train.xml" % DOMAIN, "test": "%s_Test.xml" % DOMAIN}
    DATA_FILE = fileDict[aim]
    embeddingDict = {"glove": "glove.840B.300d.txt", "word2vec": "GoogleNews-vectors-negative300.bin"}
    WORD2VEC_FILE = embeddingDict[embedding]
    EMBEDDING_PATH = FOLDER + WORD2VEC_FILE
