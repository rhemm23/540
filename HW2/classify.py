import os
import math

#These first two functions require os operations and so are completed for you
#Completed for you
def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    top_level = os.listdir(directory)
    dataset = []
    for d in top_level:
        if d.startswith('.'):
            # ignore hidden files
            continue
        if d[-1] == '/':
            label = d[:-1]
            subdir = d
        else:
            label = d
            subdir = d+"/"
        files = os.listdir(directory+subdir)
        for f in files:
            bow = create_bow(vocab, directory+subdir+f)
            dataset.append({'label': label, 'bow': bow})
    return dataset

#Completed for you
def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """

    top_level = os.listdir(directory)
    vocab = {}
    for d in top_level:
        if d.startswith('.'):
            # ignore hidden files
            continue
        subdir = d if d[-1] == '/' else d+'/'
        files = os.listdir(directory+subdir)
        for f in files:
            with open(directory+subdir+f,'r') as doc:
                for word in doc:
                    word = word.strip()
                    if not word in vocab and len(word) > 0:
                        vocab[word] = 1
                    elif len(word) > 0:
                        vocab[word] += 1
    return sorted([word for word in vocab if vocab[word] >= cutoff])

#The rest of the functions need modifications ------------------------------
#Needs modifications
def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = {}
    with open(filepath, 'r') as doc:
        for word in doc:
            word = word.strip()
            if word in vocab:
                if word in bow:
                    bow[word] += 1
                else:
                    bow[word] = 1
            else:
                if None in bow:
                    bow[None] += 1
                else:
                    bow[None] = 1
    return bow

#Needs modifications
def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """

    smooth = 1 # smoothing factor
    logprob = {}

    for label in label_list:
        count = sum(1 if dataset['label'] == label else 0 for dataset in training_data)
        logprob[label] = math.log((count + smooth) / (len(training_data) + smooth * len(label_list)))

    return logprob

#Needs modifications
def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """

    smooth = 1 # smoothing factor
    word_prob = {}

    tot_word_count = 0
    sum_word_count = {}

    for dataset in training_data:
        if dataset['label'] == label:
            for word, count in dataset['bow'].items():
                tot_word_count += count
                if word in sum_word_count:
                    sum_word_count[word] += count
                else:
                    sum_word_count[word] = count

    for word in (vocab + [None]):
        num = smooth
        if word in sum_word_count:
            num += sum_word_count[word]

        den = tot_word_count + smooth * (len(vocab) + 1)
        word_prob[word] = math.log(num / den)

    return word_prob


##################################################################################
#Needs modifications
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    label_list = [f for f in os.listdir(training_directory) if not f.startswith('.')] # ignore hidden files
    
    vocab = create_vocabulary(training_directory, cutoff)
    training_data = load_training_data(vocab, training_directory)

    retval['vocabulary'] = vocab
    retval['log prior'] = prior(training_data, label_list)
    retval['log p(w|y=2016)'] = p_word_given_label(vocab, training_data, '2016')
    retval['log p(w|y=2020)'] = p_word_given_label(vocab, training_data, '2020')

    return retval

#Needs modifications
def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    retval = {}
    doc_bow = create_bow(model['vocabulary'], filepath)
    
    p_2016_given_doc = model['log prior']['2016']
    p_2020_given_doc = model['log prior']['2020']

    for word in doc_bow:
        p_2016_given_doc += model['log p(w|y=2016)'][word] * doc_bow[word]
        p_2020_given_doc += model['log p(w|y=2020)'][word] * doc_bow[word]

    retval['log p(y=2016|x)'] = p_2016_given_doc
    retval['log p(y=2020|x)'] = p_2020_given_doc

    retval['predicted y'] = '2016' if p_2016_given_doc > p_2020_given_doc else '2020'

    return retval
