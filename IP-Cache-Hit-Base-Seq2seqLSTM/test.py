import pickle as pkl
import numpy as np
import pandas as pd
IP_list = []
# count element
def getlistnum_input(li):#这个函数就是要对列表的每个元素进行计数
    li = list(li)
    set1 = set(li)
    dict1 = {}
    for item in set1:
        dict1.update({item:li.count(item)})
    return dict1
# count element
def getlistnum_output(Input_count,temp):#这个函数就是要对列表的每个元素进行计数
    li = list(temp)
    dict1 = {}
    for item in Input_count:
        dict1.update({item:li.count(item)})
    return dict1

# Load the file to preprocess
def load_file(filename):
    file = open(filename, mode='rt', encoding='utf-8')
    text = file.read()
    file.close()
    return text

# Split the text into sentences
def to_pair(data):
    result = []
    for i in range(int(0.1/0.00001)):
        temp_input = data[((data['Time'].astype('float32') >=0.00005*i) & ((data['Time'].astype('float32') < 0.00005*(i+1))))]
        temp_output = data[((data['Time'].astype('float32') >=0.00005*(i+1)) & ((data['Time'].astype('float32') < 0.00005*(i+2))))]
        if len(temp_input):
            Input_count = getlistnum_input(temp_input['Destination'])
            valuesI = list(Input_count.values())
            keys = list(Input_count.keys())
            IP_list.append(keys)
            print(IP_list)
            stringI = " ".join('%s' %id for id in valuesI)
            #temp = getlistnum_input(temp_output['Destination'])
            Output_count = getlistnum_output(Input_count,temp_output['Destination'])
            valuesO = list(Output_count.values())
            stringO = " ".join('%s' %id for id in valuesO)
            str = stringI+","+stringO
            result.append(str.strip().split(","))
        else:
            continue
    return np.array(result)

# Save the cleaned data to the given filename
def save_data(sentences, filename):
    pkl.dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)

filename = './data/route.csv'
file = pd.read_csv(filename)
pairs = to_pair(file)
print("pairs",pairs)
print("length of pairs",len(pairs))
save_data(pairs, 'Input_IP-Output_IP.pkl')

# Checking the cleaned data
for i in range(100):
    print('[%s] => [%s]' % (pairs[i, 0], pairs[i, 1]))
