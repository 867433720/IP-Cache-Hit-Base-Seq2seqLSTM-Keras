# Training Encoder-Decoder model to represent word embeddings and finally
# save the trained model as 'model.h5'

from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint


# load a clean dataset
def load_clean_sentences(filename):
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


# one hot encode target sequence
def encode_output(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = array(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y


# define NMT model
def define_model(input_vocab, output_vocab, input_timesteps, output_timesteps, n_units):
	model = Sequential()
	model.add(Embedding(input_vocab, n_units, input_length=input_timesteps, mask_zero=True))
	model.add(LSTM(n_units))
	model.add(RepeatVector(output_timesteps))
	model.add(LSTM(n_units, return_sequences=True))
	model.add(TimeDistributed(Dense(output_vocab, activation='softmax')))
	return model


# load datasets
dataset = load_clean_sentences('Input_IP-Output_IP_both.pkl')
train = load_clean_sentences('Input_IP-Output_IP-train.pkl')
test = load_clean_sentences('Input_IP-Output_IP-test.pkl')
# prepare english tokenizer
Output_tokenizer = create_tokenizer(dataset[:, 1])
Output_vocab_size = len(Output_tokenizer.word_index) + 1
Output_length = max_length(dataset[:, 1])
print('OutputIP Vocabulary Size: %d' % Output_vocab_size)
print('OutputIP Max Length: %d' % (Output_length))

# prepare german tokenizer
Input_tokenizer = create_tokenizer(dataset[:, 0])
Input_vocab_size = len(Input_tokenizer.word_index) + 1
Input_length = max_length(dataset[:, 0])
print('InputIP Vocabulary Size: %d' % Input_vocab_size)
print('InputIP Max Length: %d' % (Input_length))

# prepare training data
trainX = encode_sequences(Input_tokenizer, Input_length, train[:, 0])
trainY = encode_sequences(Output_tokenizer, Output_length, train[:, 1])
trainY = encode_output(trainY, Output_vocab_size)

# prepare validation data
testX = encode_sequences(Input_tokenizer, Input_length, test[:, 0])
testY = encode_sequences(Output_tokenizer, Output_length, test[:, 1])
testY = encode_output(testY, Output_vocab_size)

# define model
model = define_model(Input_vocab_size, Output_vocab_size, Input_length, Output_length, 256)
model.compile(optimizer='adam', loss='categorical_crossentropy')
# summarize defined model
print(model.summary())
# fit model
filename = './data/model_test.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(trainX, trainY, epochs=30, batch_size=64, validation_data=(testX, testY), callbacks=[checkpoint], verbose=2)
model.save('./data/model_weight_test.h5')