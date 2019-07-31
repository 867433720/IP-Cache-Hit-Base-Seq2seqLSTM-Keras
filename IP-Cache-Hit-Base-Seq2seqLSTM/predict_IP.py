from pickle import load
from numpy import array, argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
import pandas as pd
import numpy as np

def load_dataset(filename):
    return load(open(filename, 'rb'))
# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer
# max sentence length
def max_length(lines):
	return max(len(line.split()) for line in lines)
# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X
# Map an integer to a word
def map_int_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Predict the target sequence
def predict_sequence(model, tokenizer, source):
    pred = model.predict(source, verbose=0)[0]
    integers = [argmax(vector) for vector in pred]
    target = list()
    for i in integers:
        word = map_int_to_word(i, tokenizer)
        if word is None:
            break
        target.append(word)
    #return ' '.join(target)
    return target
def getlistnum_input(li):  # 这个函数就是要对列表的每个元素进行计数
    li = list(li)
    set1 = set(li)
    dict1 = {}
    for item in set1:
        dict1.update({item: li.count(item)})
    return dict1


# count element
def getlistnum_output(Input_count, temp):  # 这个函数就是要对列表的每个元素进行计数
    li = list(temp)
    dict1 = {}
    for item in Input_count:
        dict1.update({item: li.count(item)})
    return dict1


# Load the file to preprocess
def load_file(filename):
    file = open(filename, mode='rt', encoding='utf-8')
    text = file.read()
    file.close()
    return text


# Split the text into sentences
def to_predict(data):
    result = []
    predict_IP = []
    predict_Time = []
    dataset = load_dataset('Input_IP-Output_IP_both.pkl')
    # train = load_dataset('Input_IP-Output_IP-train.pkl')
    # test = load_dataset('Input_IP-Output_IP-test.pkl')
    # prepare input tokenizer
    input_tokenizer = create_tokenizer(dataset[:, 0])
    # input_vocab_size = len(input_tokenizer.word_index) + 1
    input_length = max_length(dataset[:, 0])
    # prepare output tokenizer
    output_tokenizer = create_tokenizer(dataset[:, 1])
    model = load_model('model_weight_test.h5')
    # output_vocab_size = len(output_tokenizer.word_index) + 1
    # output_length = max_length(dataset[:, 1])
    # Prepare data
    # trainX = encode_sequences(input_tokenizer, input_length, train[:, 0])
    for i in range(int(0.1 / 0.00001)):
        temp_input = data[((data['Time'].astype('float32') >= 0.00005 * i) & ((data['Time'].astype('float32') < 0.00005 * (i + 1))))]
        temp_output = data[((data['Time'].astype('float32') >= 0.00005 * (i + 1)) & ((data['Time'].astype('float32') < 0.00005 * (i + 2))))]
        if len(temp_input):
            Input_count = getlistnum_input(temp_input['Destination'])
            valuesI = list(Input_count.values())
            keys = list(Input_count.keys())
            stringI = " ".join('%s' % id for id in valuesI)
            result = stringI.strip().split(",")
            print("predict object",result)
            testX = encode_sequences(input_tokenizer, input_length, result)
            translation = predict_sequence(model, output_tokenizer, testX)
            index = 0
            if len(keys)==len(translation):
                for time in translation:
                    if int(time) > 0:
                        predict_IP.append(keys[index])
                        predict_Time.append(0.000005 * (i + 1))
                    index = index + 1
            Output_count = getlistnum_output(Input_count, temp_output['Destination'])
            valuesO = list(Output_count.values())
            print("predect IP_time",predict_Time)
            print("truth IP",valuesO)
            print("predect IP_ID",predict_IP)
        else:
            continue
    column1 = ['IP']
    IP = pd.DataFrame(columns=column1, data=predict_IP)
    IP.to_csv('./data/IP.csv')
    column2 = ['Predict_Start_Time']
    IP = pd.DataFrame(columns=column2, data=predict_Time)
    IP.to_csv('./data/Predict_Start_Time.csv')
    return 0
# Load datasets
filename = './data/route.csv'
file = pd.read_csv(filename)
to_predict(file)

