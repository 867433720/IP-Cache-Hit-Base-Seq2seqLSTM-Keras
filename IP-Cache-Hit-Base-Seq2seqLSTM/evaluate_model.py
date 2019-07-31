from pickle import load
from numpy import array, argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu

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
    return ' '.join(target)

# Evaluate the model
def evaluate_model(model, tokenizer, source, raw_dataset):
    predicted, actual = list(), list()
    for i, source in enumerate(source):
        source = source.reshape((1, source.shape[0]))
        translation = predict_sequence(model, tokenizer, source)
        raw_source, raw_target = raw_dataset[i]
        print('src=[%s], target=[%s], predicted=[%s]' % (raw_source, raw_target, translation))
        actual.append(raw_target.split())
        predicted.append(translation.split())
    # Bleu Scores
    print('Bleu-10: %f' % corpus_bleu(actual, predicted, weights=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)))
    print('Bleu-20: %f' % corpus_bleu(actual, predicted, weights=(0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05)))
    print('Bleu-50: %f' % corpus_bleu(actual, predicted, weights=(0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02)))
# Load datasets
dataset = load_dataset('Input_IP-Output_IP_both.pkl')
train = load_dataset('Input_IP-Output_IP-train.pkl')
test = load_dataset('Input_IP-Output_IP-test.pkl')
# prepare input tokenizer
input_tokenizer = create_tokenizer(dataset[:, 0])
input_vocab_size = len(input_tokenizer.word_index) + 1
input_length = max_length(dataset[:, 0])
# prepare output tokenizer
output_tokenizer = create_tokenizer(dataset[:, 1])
output_vocab_size = len(output_tokenizer.word_index) + 1
output_length = max_length(dataset[:, 1])
# Prepare data
trainX = encode_sequences(input_tokenizer, input_length, train[:, 0])
print("test[:, 0]",test[:, 0])
testX = encode_sequences(input_tokenizer, input_length, test[:, 0])

model = load_model('model_weight_test.h5')

print('Testing on trained examples')
print(testX)
#evaluate_model(model, output_tokenizer, trainX, train)

print('Testing on test examples')
evaluate_model(model, output_tokenizer, testX, test)
