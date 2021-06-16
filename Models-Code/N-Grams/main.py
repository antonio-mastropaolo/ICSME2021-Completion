import random
from nltk import bigrams, trigrams, ngrams, FreqDist
from collections import Counter, defaultdict
from tqdm import tqdm
import pickle, re, json
import string, dill
from nltk.corpus import reuters
from operator import itemgetter
from Logger import Logger
import logging
import sys
from nltk.tokenize import RegexpTokenizer
import argparse
import os


########## GLOBAL VARIABLES ##########

RESULTS_FOLDER = 'results_5-GRAM'
THRESHOLD = 15
PREDICTIONS_FILE_JAVADOC = ''
TARGET_FILE_JAVADOC = ''
TARGET_FILE_INSIDE = ''
CONFIDENCE_JAVADOC = ''
CONFIDENCE_INSIDE = ''
PREDICTIONS_FILE_INSIDE = ''

#####################################


def createModel(corpus, model_name, n):
    model = defaultdict(lambda: defaultdict(lambda: 0))

    if n == 3:

        for sentence in tqdm(corpus):
            # splitted_sentence = sentence.split(' ')

            for w1, w2, w3 in trigrams(sentence, pad_right=False, pad_left=True):
                model[(w1, w2)][w3] += 1

    if n == 5:

        for sentence in tqdm(corpus):
            # splitted_sentence = sentence.split(' ')

            for w1, w2, w3, w4, w5 in ngrams(sentence, 5, pad_right=False, pad_left=True):
                model[(w1, w2, w3, w4)][w5] += 1

    if n == 7:

        for sentence in tqdm(corpus):
            # splitted_sentence = sentence.split(' ')

            for w1, w2, w3, w4, w5, w6, w7 in ngrams(sentence, 7, pad_right=False, pad_left=True):
                model[(w1, w2, w3, w4, w5, w6)][w7] += 1

    # Let's transform the counts to probabilities
    for w1_wn in tqdm(model):
        total_count = float(sum(model[w1_wn].values()))
        for w_m in model[w1_wn]:
            model[w1_wn][w_m] /= total_count

    print('Dumping the model!!')
    model_dir = os.path.join(RESULTS_FOLDER, model_name)
    with open(model_dir, 'wb') as file:
        dill.dump(model, file)

    return model

def getPrediction(model, context, input, output, logger=None):

    global PREDICTIONS_FILE_JAVADOC
    global PREDICTIONS_FILE_INSIDE
    global TARGET_FILE_INSIDE
    global TARGET_FILE_JAVADOC
    global CONFIDENCE_JAVADOC
    global CONFIDENCE_INSIDE

    predictions = []
    chain_of_words = ''
    label_list = output[0:-1]  # skipping </s>
    obj_pred = {}

    # print('output: ', output[0:-1])

    if THRESHOLD < len(label_list):
        label_list = label_list[0:THRESHOLD]

    confidence = ''
    for (idx, label) in enumerate(label_list):

        if idx + 1 == THRESHOLD:
            break

        # Get chain of predictions
        if len(context) == 2:
            test_dict = dict(model[context[0], context[1]])
        elif len(context) == 4:
            test_dict = dict(model[context[0], context[1], context[2], context[3]])
        else:
            test_dict = dict(model[context[0], context[1], context[2], context[3], context[4], context[5]])

        if len(test_dict) == 0:

            predicted_word = ''

            chain_of_words += predicted_word + ' '
            confidence += '<prob>0</prob>'
            chain_of_words = chain_of_words.replace('\n', '')


        else:

            predicted_word = list(test_dict.keys())[0]  # get the most likely word
            pred_key = 'prediction@{}'.format(idx + 1)

            if predicted_word == None: predicted_word = ''

            chain_of_words += predicted_word + ' '
            chain_of_words = chain_of_words.replace('\n','')
            confidence += ' <prob>{}</prob>'.format(test_dict[predicted_word])

            obj_pred[pred_key] = chain_of_words

        predictions.append(obj_pred)
        context.pop(0)
        context.append(predicted_word)

    if 'complete javadoc comment:' in input:
        TARGET_FILE_JAVADOC.write(' '.join(output[0:-1])+'\n')
        PREDICTIONS_FILE_JAVADOC.write(chain_of_words+'\n')
        CONFIDENCE_JAVADOC.write(confidence+'\n')
    else:
        TARGET_FILE_INSIDE.write(' '.join(output[0:-1])+'\n')
        PREDICTIONS_FILE_INSIDE.write(chain_of_words+'\n')
        CONFIDENCE_INSIDE.write(confidence + '\n')



    return predictions


def runTest(dataset_task, task_specific_model, n, logger=None):

    predictionsList = []

    input_list = dataset_task['input']
    output_list = dataset_task['output']

    for (x, y) in tqdm(zip(input_list, output_list)):

        only_comment = re.findall("<sep>([\s\S]*?)<sep>", x)[-1].lstrip().rstrip().lower().replace('<extra_id_0>', '')

        y = y.lower().split(' ')
        tokens_list = only_comment.split(' ')

        # Discard empty token
        tokens_list = [token for token in tokens_list if token != '']

        # Retrieving context for the model
        if len(tokens_list) < n - 1:

            # Padding if we cannot match the required n-grams
            for i in range(n - len(tokens_list) - 1):
                tokens_list.insert(i, None)

            context = tokens_list
            # print(context)

        else:
            context = tokens_list[len(tokens_list) - (n - 1): len(tokens_list)]

        _ = getPrediction(task_specific_model, context, x, y, logger=logger)

    return predictionsList


def loadData(dataset):

    filtered_corpora = []

    with open(dataset, 'r') as fread:
        items = fread.readlines()

        for (idx, item) in enumerate(items):
            filtered_corpora.append(item.split(' '))

    #
    tokens = [item for sublist in filtered_corpora for item in sublist]
    vocab = FreqDist(tokens)

    for sublist in filtered_corpora:
        for (idx, token) in enumerate(sublist):
            if vocab[token] <= 1: sublist[idx] = '<UNK>'

    return filtered_corpora


def main():
    parser = argparse.ArgumentParser("N-gram Language Model")

    parser.add_argument('--n', type=int,
                        required=True,
                        help='Order of N-gram model to create (i.e. 1 for unigram, 2 for bigram, etc.)')

    parser.add_argument('--save_file_name',
                        type=str,
                        default='N-gram.pkl',
                        help='N-Grams output model name(e.g: 3-grams.pkl'
                        )

    args = parser.parse_args()

    if not os.path.exists(RESULTS_FOLDER):
        os.mkdir(RESULTS_FOLDER)

    global CONFIDENCE_INSIDE
    global CONFIDENCE_JAVADOC
    global PREDICTIONS_FILE_JAVADOC
    global PREDICTIONS_FILE_INSIDE
    global TARGET_FILE_INSIDE
    global TARGET_FILE_JAVADOC
    THRESHOLD = 15

    #### LOADING THE TEST SETS ####
    test_javadoc_input = []
    test_single_comment_input = []

    test_javadoc_output = []
    test_single_comment_output = []

    path_to_source = 'test.source'
    path_to_target = 'test.target'

    with open(path_to_source, 'r') as fread:
        lines = fread.readlines()
        target = open(path_to_target, 'r')
        target_lines = target.readlines()
        for (line, target) in zip(lines, target_lines):

            if 'complete javadoc comment:' in line:
                test_javadoc_input.append(line)
                test_javadoc_output.append(target)

            else:
                test_single_comment_input.append(line)
                test_single_comment_output.append(target)


    test_set = {'input': test_javadoc_input + test_single_comment_input,
                'output': test_javadoc_output + test_single_comment_output}

    ##############################################

    base_path_dataset = 'NGrams'

    loaded_javadoc = loadData(os.path.join(base_path_dataset, 'javadoc_train_ngram.txt'))
    loaded_method_level = loadData(os.path.join(base_path_dataset, 'inside_train_ngram.txt'))
    loaded_items = loaded_javadoc + loaded_method_level

    print('****** MODEL CREATION IS ABOUT TO START! ******')

    n_gram_model = createModel(loaded_items, args.save_file_name, args.n)
    # with open('results_5-GRAM/5-Grams.pickle', 'rb') as file:
    #     n_gram_model = dill.load(file)


    print('****** MODEL CREATION ENDS! ******')

    print('\n ****** GENERATING PREDICTIONS ******')

    PREDICTIONS_FILE_JAVADOC = open(os.path.join(RESULTS_FOLDER, 'predictions_javadoc.txt'), 'a+')
    TARGET_FILE_JAVADOC = open(os.path.join(RESULTS_FOLDER, 'target_javadoc.txt'), 'a+')
    TARGET_FILE_INSIDE = open(os.path.join(RESULTS_FOLDER, 'target_inside.txt'), 'a+')
    PREDICTIONS_FILE_INSIDE = open(os.path.join(RESULTS_FOLDER, 'predictions_inside.txt'), 'a+')
    CONFIDENCE_INSIDE = open(os.path.join(RESULTS_FOLDER, 'confidence_inside.txt'), 'a+')
    CONFIDENCE_JAVADOC = open(os.path.join(RESULTS_FOLDER, 'confidence_javadoc.txt'), 'a+')

    logger = Logger('logger.log', 'log_object', logging.INFO)
    logger_info = logger.getLogger()
    _ = runTest(test_set, n_gram_model, args.n, logger=logger_info)

    PREDICTIONS_FILE_JAVADOC.close()
    TARGET_FILE_JAVADOC.close()
    TARGET_FILE_INSIDE.close()
    PREDICTIONS_FILE_INSIDE.close()
    CONFIDENCE_INSIDE.close()
    CONFIDENCE_JAVADOC.close()



if __name__ == '__main__':
    main()
