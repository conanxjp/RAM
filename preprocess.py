import xml.etree.ElementTree as et
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from nltk.metrics import edit_distance
import hunspell
import re
from tqdm import tqdm
import argparse
import sys
import config as cf


def parse2014(filepath, year, domain, aim, isTerm):
    """
    parse 2014 raw data in xml format
    only tested for restaurant data
    """
    print("parsing %s %s %s data ..." % (year, domain, aim))
    data = pd.DataFrame(columns = ['id', 'text', 'aspect', 'polarity'])
    tree = et.parse(filepath)
    root = tree.getroot()
    sentences = root.findall('sentence');
    i = 0
    for sentence in tqdm(sentences):
        id = sentence.attrib.get('id')
        text = sentence.find('text').text
        if isTerm:
            # use aspect term words/phrases
            aspectTerms = sentence.find('aspectTerms')
            if aspectTerms != None:
                for term in aspectTerms.findall('aspectTerm'):
                    if term.attrib.get('polarity') != 'conflict':
                        data.loc[i] = [id, text, term.attrib.get('term'), term.attrib.get('polarity')]
                        i = i + 1
        else:
            # use aspect categories
            for category in sentence.find('aspectCategories').findall('aspectCategory'):
                if category.attrib.get('polarity') != 'conflict':
                    data.loc[i] = [id, text, category.attrib.get('category'), category.attrib.get('polarity')]
                    i = i + 1
    writeCSV(data, cf.ROOT_PATH + cf.DATA_PATH + '%s_%s_%s_raw.csv' % (domain, aim, year))
    return data


def writeCSV(dataframe, filepath):
    print("write dataframe to csv file")
    dataframe.to_csv(filepath, index = False)


def tokenize(data):
    print("tokenize words ...")
    wordData = []
    for s in data:
        wordData.append([w for w in word_tokenize(s.lower())])
    return wordData


def cleanup(wordData):
    print("cleaning up words ...")
    dictionary = embeddingDict(cf.EMBEDDING_PATH)
    print("correct dashed words ...")
    wordData = cleanOp(wordData, re.compile(r'-'), dictionary, correctDashWord)
    wordData = cleanOp(wordData, re.compile(r'-'), dictionary, cleanDashWord)
    print("correct words with time, plus, numbers ...")
    wordData = cleanOp(wordData, re.compile(r':'), dictionary, parseTime)
    wordData = cleanOp(wordData, re.compile('\+'), dictionary, parsePlus)
    wordData = cleanOp(wordData, re.compile(r'\d+'), dictionary, parseNumber)
    print("correct spellings ...")
    wordData = cleanOp(wordData, re.compile(r''), dictionary, correctSpell)
    return wordData


def cleanOp(wordData, regex, dictionary, op):
    for i, sentence in tqdm(enumerate(wordData)):
        if bool(regex.search(sentence)):
            newSentence = ''
            for word in word_tokenize(sentence.lower()):
                if bool(regex.search(word)) and word not in dictionary:
                    word = op(word)
                newSentence = newSentence + ' ' + word
            wordData[i] = newSentence[1:]
    return wordData


def parseTime(word):
    time_re = re.compile(r'^(([01]?\d|2[0-3]):([0-5]\d)|24:00)(pm|am)?$')
    if not bool(time_re.match(word)):
        return word
    else:
        dawn_re = re.compile(r'0?[234]:(\d{2})(am)?$')
        earlyMorning_re = re.compile(r'0?[56]:(\d{2})(am)?$')
        morning_re = re.compile(r'((0?[789])|(10)):(\d{2})(am)?$')
        noon_re = re.compile(r'((11):(\d{2})(am)?)|(((0?[01])|(12)):(\d{2})pm)$')
        afternoon_re = re.compile(r'((0?[2345]):(\d{2})pm)|((1[4567]):(\d{2}))$')
        evening_re = re.compile(r'((0?[678]):(\d{2})pm)|(((1[89])|20):(\d{2}))$')
        night_re = re.compile(r'(((0?9)|10):(\d{2})pm)|((2[12]):(\d{2}))$')
        midnight_re = re.compile(r'(((0?[01])|12):(\d{2})am)|(0?[01]:(\d{2}))|(11:(\d{2})pm)|(2[34]:(\d{2}))$')
        if bool(noon_re.match(word)):
            return 'noon'
        elif bool(evening_re.match(word)):
            return 'evening'
        elif bool(morning_re.match(word)):
            return 'morning'
        elif bool(earlyMorning_re.match(word)):
            return 'early morning'
        elif bool(night_re.match(word)):
            return 'night'
        elif bool(midnight_re.match(word)):
            return 'midnight'
        elif bool(dawb_re.match(word)):
            return 'dawn'
        else:
            return word


def parsePlus(word):
    return re.sub('\+', ' +', word)


def parseNumber(word):
    if bool(re.search(r'\d+', word)):
        return word
    else:
        search = re.search(r'\d+', word)
        pos = search.start()
        num = search.group()
        return word[:pos] + ' %s ' % num + parseNumber(word[pos+len(num):])


def checkSpell(word):
    return cf.hobj.spell(word)


def correctSpell(word):
    suggestions = cf.hobj.suggest(word)
    if len(suggestions) != 0:
        distance = [edit_distance(word, s) for s in suggestions]
        return suggestions[distance.index(min(distance))]
    else:
        return word


def createTempVocabulary(wordData, aim):
    print("create temperary vocabulary ...")
    words = sorted(set([word for l in wordData for word in word_tokenize(l)]))
    vocabulary = filterWordEmbedding(words, cf.EMBEDDING_PATH, aim)
    return vocabulary


def splitDashWord(word):
    if '-' not in word:
        return [word]
    else:
        return word.split('-')


def cleanDashWord(word):
    return ''.join([s + ' ' for s in word.split('-')])


def correctDashWord(word):
    splittedWords = word.split('-')
    for i, word in enumerate(splittedWords):
        if not checkSpell(word):
            splittedWords[i] = correctSpell(word)
    return ''.join([s + '-' for s in splittedWords])[:-1]


def joinWord(words):
    return ''.join([s + ' ' for s in words])[:-1]


def embeddingDict(embeddingPath):
    print("load embedding dictionary ...")
    dictionary = []
    with open(embeddingPath) as f:
        for line in tqdm(f):
            values = line.split()
            word = joinWord(values[:-300])
            dictionary.append(word)
    f.close()
    return dictionary


def filterWordEmbedding(words, embeddingPath, aim):
    print("check against embedding dictionary ...")
    vocabulary = []
    filteredEmbeddingDict = []
    words = [word.lower() for word in words]
    with open(embeddingPath) as f:
        for line in tqdm(f):
            values = line.split()
            word = values[0]
            try:
                if word in words:
                    vocabulary.append(word)
                    filteredEmbeddingDict.append(line)
            except:
                print("stopping in filterWordEmbedding")
                print("word: ", word)

    f.close()
    unknownWords = [word for word in words if word not in vocabulary]
    with open(cf.FOLDER + '%s_filtered_%s.txt' % (cf.WORD2VEC_FILE[0:-4], aim), 'w+') as f:
        for line in filteredEmbeddingDict:
            f.write(line)
    with open('unknown.txt', 'w+') as f:
        for i, word in enumerate(unknownWords):
            f.write(word + '\n')
    return filteredEmbeddingDict


def createVocabulary(trainDictPath, testDictPath, gloveDictPath):
    print("create final vocabulary ...")
    dictionary = []
    with open(trainDictPath) as f:
        for line in f:
            dictionary.append(line)
    f.close()
    with open(testDictPath) as f:
        for line in f:
            dictionary.append(line)
    f.close()
    with open(gloveDictPath) as f:
        miscFlag = True
        anecFlag = True
        for line in f:
            if not (miscFlag or anecFlag):
                break
            word = line.split()[0]
            if miscFlag and word == 'miscellaneous':
                dictionary.append(line)
                miscFlag = False
            if anecFlag and word == 'anecdotes':
                dictionary.append(line)
                anecFlag = False
    f.close()
    dictionary = set(dictionary)
    dictionaryNP = np.zeros((len(dictionary) + 1, 300))
    with open(cf.FOLDER + '%s_filtered.txt' % cf.WORD2VEC_FILE[0:-4], 'w+') as f:
        for i, line in enumerate(dictionary):
            values = line.split()
            try:
                dictionaryNP[i] = np.asarray(values[-300:], dtype='float32')
            except ValueError:
                print(joinWord(values[:-300]))
            f.write(line)
    f.close()
    dictionaryNP[-1] = np.random.normal(0, 0.01, [1,300])
    np.save(cf.FOLDER + 'glove', dictionaryNP)


def splitData():
    """
    To randomly sample a small fraction from the processed train and test data,
    which will be used for testing the models
    """
    trainDataPath = cf.FOLDER + cf.TRAIN_FILE
    testDataPath = cf.FOLDER + cf.TEST_FILE
    train = pd.read_csv(trainDataPath)
    test = pd.read_csv(testDataPath)
    validData = train.sample(frac = 0.1, replace = False)
    trainData = train.ix[train.index.difference(validData.index)]
    testSample = test.sample(frac = 0.3, replace = False)
    writeCSV(validData, cf.FOLDER + 'rest_train_valid.csv')
    writeCSV(trainData, cf.FOLDER + 'rest_train_2014_processed.csv')
    writeCSV(testSample, cf.FOLDER + 'rest_test_sample.csv')


def getPositions():
    def locations(data, type):
        print(type)
        maxLength = 80
        wordPositions = np.zeros((len(data), maxLength))
        for i, aspect in enumerate(data['aspect']):
            aspects = word_tokenize(aspect.lower())
            texts = word_tokenize(data['text'].loc[i].lower())
            try:
                start = texts.index(aspects[0])
                end = texts.index(aspects[-1])
            except ValueError:
                start = 0
                end = 0
                print(aspect + " is not found in " + data['text'].loc[i])
            tmax = len(texts)
            positions = np.zeros(tmax)
            for j in range(tmax):
                if j < start:
                    positions[j] = 1 - abs(j - start) / tmax
                elif j > end:
                    positions[j] = 1 - abs(j - end) / tmax
                else:
                    positions[j] = 0
            positions = np.pad(positions, (0, maxLength - tmax), 'constant')
            wordPositions[i] = positions
        np.save(cf.FOLDER + 'word_positions_%s' % type, wordPositions)

    trainDataPath = cf.FOLDER + cf.TRAIN_FILE
    testDataPath = cf.FOLDER + cf.TEST_FILE
    validDataPath = cf.FOLDER + 'rest_train_valid.csv'
    testSamplePath = cf.FOLDER + 'rest_test_sample.csv'
    train = pd.read_csv(trainDataPath)
    test = pd.read_csv(testDataPath)
    valid = pd.read_csv(validDataPath)
    sample = pd.read_csv(testSamplePath)

    # calculate word positions
    locations(train, 'train')
    locations(test, 'test')
    locations(valid, 'valid')
    locations(sample, 'sample')


def prepare():
    def readyData(data, textEncode):
        dim_polarity = 3
        maxLength = 80
        polarities = data['polarity']
        polarity_indices = [[], [], []]
        for key, val in polarity_encode.items():
            polarity_indices[val] = polarities.loc[polarities == key].index
            data.loc[polarity_indices[val], 'polarity'] = val
        # one hot representation for y labels
        y = np.zeros([len(data), dim_polarity])
        data['polarity'] = [int(p) for p in data['polarity']]
        y[np.arange(len(data)), data['polarity']] = 1
        # get sentence word sequence length
        seqlen = [len(word_tokenize(texts)) for texts in data['text']]
        test = [len(encode) for encode in textEncode]
        # get the word encoding indices for each sentence
        X = [np.pad(textEncode[i], (0, maxLength - seqlen[i]), 'constant') for i in range(len(seqlen))]
        return X, y, seqlen

    trainDataPath = cf.FOLDER + cf.TRAIN_FILE
    testDataPath = cf.FOLDER + cf.TEST_FILE
    validDataPath = cf.FOLDER + 'rest_train_valid.csv'
    testSamplePath = cf.FOLDER + 'rest_test_sample.csv'
    train = pd.read_csv(trainDataPath)
    test = pd.read_csv(testDataPath)
    valid = pd.read_csv(validDataPath)
    sample = pd.read_csv(testSamplePath)
    polarity_encode = {'positive': 0, 'neutral': 1, 'negative': 2}
    trainEncode = np.load(cf.FOLDER + 'word_index_train.npy')
    testEncode = np.load(cf.FOLDER + 'word_index_test.npy')
    validEncode = np.load(cf.FOLDER + 'word_index_valid.npy')
    sampleEncode = np.load(cf.FOLDER + 'word_index_sample.npy')

    trainData = readyData(train, trainEncode)
    testData = readyData(test, testEncode)
    validData = readyData(valid, validEncode)
    sampleData = readyData(sample, sampleEncode)

    trainAspects = np.load(cf.FOLDER + 'aspect_encoding_train.npy')
    testAspects = np.load(cf.FOLDER + 'aspect_encoding_test.npy')
    validAspects = np.load(cf.FOLDER + 'aspect_encoding_valid.npy')
    sampleAspects = np.load(cf.FOLDER + 'aspect_encoding_sample.npy')

    return [trainData, trainAspects], [testData, testAspects], [validData, validAspects], [sampleData, sampleAspects]


def encodeAllData():
    """
    encode the process data into index array in the filtered glove dictionary,
    which will be used by the
    """

    print("encoding data ...")
    def encodeData(filePath, type):
        data = pd.read_csv(filePath)
        texts = data['text']
        sentences = [word_tokenize(text) for text in texts]
        textIndex = []
        encoding = pd.DataFrame(columns = ['id', 'text_encode', 'aspect', 'polarity'])
        # for counting the length of the longest sentence
        max = 0
        for i, words in enumerate(sentences):
            sentenceIndex = []
            for word in words:
                try:
                    idx = index.index(word)
                except ValueError:
                    idx = len(index)
                sentenceIndex.append(idx)
            if max < len(sentenceIndex):
                max = len(sentenceIndex)
            textIndex.append(sentenceIndex)
        # print(max)
        np.save(cf.FOLDER + 'word_index_%s' % type, np.array(textIndex))

        aspects = data['aspect']
        aspectEncoding = []
        aspectTerms = [word_tokenize(aspectTerm) for aspectTerm in aspects]
        for words in aspectTerms:
            encode = []
            for word in words:
                try:
                    idx = index.index(word)
                except ValueError:
                    idx = len(index)
                encode.append(glove[idx])
            encode = np.mean(encode, 0)
            aspectEncoding.append(encode)
        np.save(cf.FOLDER + 'aspect_encoding_%s' % type, np.array(aspectEncoding))

    dictionary = {}
    with open(cf.FOLDER + '%s_filtered.txt' % cf.WORD2VEC_FILE[0:-4], 'r') as f:
        for line in f:
            values = line.split()
            word = joinWord(values[:-300])
            vector = np.array(values[-300:], dtype='float32')
            dictionary[word] = vector
    f.close()
    index = list(dictionary.keys())
    glove = np.load(cf.FOLDER + 'glove.npy')
    trainDataPath = cf.FOLDER + cf.TRAIN_FILE
    testDataPath = cf.FOLDER + cf.TEST_FILE
    validDataPath = cf.FOLDER + 'rest_train_valid.csv'
    testSampleDataPath = cf.FOLDER + 'rest_test_sample.csv'
    encodeData(trainDataPath, 'train')
    encodeData(testDataPath, 'test')
    encodeData(validDataPath, 'valid')
    encodeData(testSampleDataPath, 'sample')



###############################################
# below are the two functions used by main    #
###############################################
def processData(year, domain, embedding):
    """
    entry point of the preprocessing functions
    """
    PARSER = {'2014': parse2014}
    parser = PARSER[year];
    # process train data
    aim = 'train'
    cf.configure(year, domain, embedding, aim)
    trainDataPath = cf.FOLDER + cf.DATA_FILE
    trainData = parser(trainDataPath, year, domain, aim, True)
    # use tokenize instead of clean up for baselines
    trainData['text'] = cleanup(trainData['text'])
    trainData['aspect'] = cleanup(trainData['aspect'])
    trainVocabulary = createTempVocabulary(trainData['text'], aim)
    writeCSV(trainData, cf.FOLDER + 'rest_train_2014_processed.csv')
    # # process test data
    aim = 'test'
    cf.configure(year, domain, embedding, aim)
    testDataPath = cf.FOLDER + cf.DATA_FILE
    testData = parser(testDataPath, year, domain, aim, True)
    testData['text'] = cleanup(testData['text'])
    testData['aspect'] = cleanup(testData['aspect'])
    testVocabulary = createTempVocabulary(testData['text'], aim)
    writeCSV(testData, cf.FOLDER + 'rest_test_2014_processed.csv')

    # export the final embedding dictionary by combining the dict from train and test data
    createVocabulary(cf.FOLDER + '%s_filtered_train.txt' % cf.WORD2VEC_FILE[0:-4], cf.FOLDER + '%s_filtered_test.txt' % cf.WORD2VEC_FILE[0:-4], cf.EMBEDDING_PATH)
    # sampling from the processed train and test data
    splitData()
    # calculate word positions respect to aspect term(s)
    getPositions()

def prepareData():
    """
    second function to run after processData function to
    prepare the data in the required format for model
    """
    # encode all data
    encodeAllData()
    # return prepared datasets
    return prepare()
