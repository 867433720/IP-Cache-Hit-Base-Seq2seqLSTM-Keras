import pickle as pkl
import numpy as np
import pandas as pd
# count element
dict1 = {}
dict2 = {}
def setDicValuesZero(dic):
    for dickey in dic.keys():
        dic.update({dickey:0})
    return dic
# max sentence length
def max_length(lines):
	return max(len(line.split()) for line in lines)

def getlistnum_input(li):#这个函数就是要对列表的每个元素进行计数
    li = list(li)
    set1 = set(li)
    for item in set1:
        setDicValuesZero(dict1)
        dict1.update({item:li.count(item)})
    print(dict1)
    return dict1
# count element
def getlistnum_output(Input_count,temp):#这个函数就是要对列表的每个元素进行计数
    li = list(temp)
    for item in Input_count:
        dict2.update({item:li.count(item)})
    return dict2

# Load the file to preprocess
def load_file(filename):
    file = open(filename, mode='rt', encoding='utf-8')
    text = file.read()
    file.close()
    return text

# Split the text into sentences
def to_pair(data):
    result = []
    for i in range(int(0.03/0.00001)):
        temp_input = data[((data['Time'].astype('float32') >=0.000005*i) & ((data['Time'].astype('float32') < 0.000005*(i+1))))]
        temp_output = data[((data['Time'].astype('float32') >=0.000005*(i+1)) & ((data['Time'].astype('float32') < 0.000005*(i+2))))]
        if len(temp_input):
            Input_count = getlistnum_input(temp_input['Destination'])
            valuesI = list(Input_count.values())
            stringI = " ".join('%s' %id for id in valuesI)
            Output_count = getlistnum_output(Input_count,temp_output['Destination'])
            valuesO = list(Output_count.values())
            stringO = " ".join('%s' %id for id in valuesO)
            str = stringI+","+stringO
            #print(str.strip().split(","))
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
Output_length = max_length(pairs[:, 1])
input_length = max_length(pairs[:, 0])
print("intput_length",input_length)
print("Output_length",Output_length)
save_data(pairs, 'Input_IP-Output_IP.pkl')

# Checking the cleaned data
for i in range(100):
    print('[%s] => [%s]' % (pairs[i, 0], pairs[i, 1]))
