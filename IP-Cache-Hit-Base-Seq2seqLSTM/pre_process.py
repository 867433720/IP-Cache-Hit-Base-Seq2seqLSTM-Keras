import pickle as pkl
import numpy as np
import pandas as pd

# count element
def getlistnum_input(li):#这个函数就是要对列表的每个元素进行计数
    li = list(li)
    set1 = set(li)
    dict1 = {}
    a = 2  # 出现次数阈值，剔除掉小于等于这个阈值之下的出现次数IP对象，不做预测
    for item in set1:
        if li.count(item) > a:  #
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
    window_size = 0.00003#这里step size 等于窗口大小
    step_size = 0.00001
    deadline_Time = 0.08#截止时间戳是1.7多，一百万多条记录，这里由于训练时间问题，选取部分
    for i in range(int(deadline_Time/step_size)):
        temp_input = data[((data['Time'].astype('float32') >=step_size*i) & ((data['Time'].astype('float32') < step_size*i+window_size)))]
        temp_output = data[((data['Time'].astype('float32') >=step_size*i+window_size) & ((data['Time'].astype('float32') < step_size*i+window_size+window_size*0.5)))]
        if len(temp_input):
            Input_count = getlistnum_input(temp_input['Destination'])
            valuesI = list(Input_count.values())
            if len(valuesI) > 0:
                stringI = " ".join('%s' %id for id in valuesI)
                Output_count = getlistnum_output(Input_count,temp_output['Destination'])
                valuesO = list(Output_count.values())
                stringO = " ".join('%s' %id for id in valuesO)
                str = stringI+","+stringO
                #print(str.strip().split(","))
                result.append(str.strip().split(","))
            else:
                continue
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
