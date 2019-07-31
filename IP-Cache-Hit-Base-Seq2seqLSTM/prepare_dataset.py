import pickle as pkl
from numpy import random

def load_clean_data(filename):
    file = open(filename, 'rb')
    return pkl.load(file)

def save_clean_data(sentences, filename):
    pkl.dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)


raw_data = load_clean_data('Input_IP-Output_IP.pkl')
print(len(raw_data))
dataset = raw_data[:10000, :]
random.shuffle(dataset)
train_set = dataset[:9000, :]
test_set = raw_data[9000:10000, :]
save_clean_data(dataset, 'Input_IP-Output_IP_both.pkl')
save_clean_data(train_set, 'Input_IP-Output_IP-train.pkl')
save_clean_data(test_set, 'Input_IP-Output_IP-test.pkl')
