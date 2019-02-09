import numpy as np
from keras.preprocessing.text import one_hot
from keras_preprocessing import sequence
from keras import Sequential, callbacks
from keras.models import load_model
from keras.layers import Embedding, LSTM, Dense, GRU
import matplotlib.pyplot as plt
import json
import io
#import nltk
#from nltk.stem import WordNetLemmatizer, PorterStemmer
#nltk.download('wordnet')


def save_labels(data_dictionary):

    ratings = []
    for i in range(len(data_dictionary)):
        ratings.append(data_dictionary[i]['rating'])

    labels = np.array(ratings)

    # Assign 0 to negative and 1 to positive comments.
    # Decide according to the ratings (4 and 5 positive, below negative).
    labels[np.argwhere(labels <= 3)] = 0
    labels[np.argwhere(labels >= 4)] = 1
    labels = labels.astype(int)

    # Save labels as txt file.
    np.savetxt('labels.txt', labels.astype(int), fmt='%i')


def get_content(data_dictionary, data_type):
    """
    Extract content from dictionary.

    :param data_dictionary: dictionary
    :param data_type: str
    :return: list
    """

    content = []
    for i in range(len(data_dictionary)):
        content.append(data_dictionary[i][data_type])

    # Replace \n and \t with space.
    for i in range(len(content)):
        content[i] = content[i].replace('\n', ' ')
        content[i] = content[i].replace('\t', ' ')
        content[i] = content[i].lower()

    return content


#%% Dataset 2

with io.open('./data/json_data/all.json', 'r') as f:
    data_dict = json.load(f)

y2_test = np.loadtxt('./data/json_data/labels.txt').astype(int)

review_list = get_content(data_dict, 'comment')
stopwords_removal = get_content(data_dict, 'stopwords_removal')
lemmatized = get_content(data_dict, 'lemmatized_comment')

max_review_length = 350

vocab = len(sorted(set(review_list)))
encoded_reviews = [one_hot(line, vocab) for line in review_list]
x2_test = sequence.pad_sequences(encoded_reviews, maxlen=max_review_length)

vocab = len(sorted(set(lemmatized)))
encoded_lemma = [one_hot(line, vocab) for line in lemmatized]
xlemma_test = sequence.pad_sequences(encoded_lemma, maxlen=max_review_length)

vocab = len(sorted(set(stopwords_removal)))
encoded_stopwords = [one_hot(line, vocab) for line in stopwords_removal]
xstopword_test = sequence.pad_sequences(encoded_stopwords, maxlen=max_review_length)


#%% Dataset 1

#lemmatizer = WordNetLemmatizer()
#ps = PorterStemmer()

negative = open('./data/negative10kmod.txt', 'r').readlines()
positive = open('./data/positive10kmod.txt', 'r').readlines()

temp = []
for line in negative:
    temp.append(line.strip())
negative = temp.copy()

temp = []
for line in positive:
    temp.append(line.strip())
positive = temp.copy()

positive = [line.lower() for line in positive]
negative = [line.lower() for line in negative]



### Create word vectors ###
negative_vocab = len(sorted(set(negative)))
positive_vocab = len(sorted(set(positive)))
vocab = len(sorted(set(positive+negative)))

encoded_negative = [one_hot(line, vocab) for line in negative]
encoded_positive = [one_hot(line, vocab) for line in positive]

x_train = encoded_positive[:7000] + encoded_negative[:7000]
x_test = encoded_positive[7000:] + encoded_negative[7000:]
y_train = [1] * 7000 + [0] * 7000
y_test = [1] * 2926 + [0] * 2704

max_review_length = 350
x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)


#%% Train classifier.

gg = 0
if gg == 0:
    ### Build the model ###
    embedding_vector_length = 32
    model = Sequential()
    model.add(Embedding(vocab, embedding_vector_length, input_length=max_review_length))
    model.add(LSTM(100, dropout=0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, validation_split=0.15, epochs=5, batch_size=64)

    ### Model evaluation ###
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Testset Accuracy: %.2f%%" % (scores[1]*100))

    scores_2 = model.evaluate(x2_test, y2_test, verbose=0)
    print("Accuracy on dataset 2: %.2f%%" % (scores_2[1]*100))

    scores_stopword = model.evaluate(xstopword_test, y2_test, verbose=0)
    print("Accuracy on stopwords: %.2f%%" % (scores_stopword[1]*100))

    scores_lemma = model.evaluate(xlemma_test, y2_test, verbose=0)
    print("Accuracy on lemmatized: %.2f%%" % (scores_lemma[1]*100))

    model.save('model.h5')

elif gg == 1:
    ### Build the model ###
    embedding_vector_length = 32
    model = Sequential()
    model.add(Embedding(vocab, embedding_vector_length, input_length=max_review_length))
    model.add(GRU(100, dropout=0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, validation_split=0.15, epochs=5, batch_size=64)

    ### Model evaluation ###
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Testset Accuracy: %.2f%%" % (scores[1]*100))

    scores_2 = model.evaluate(x2_test, y2_test, verbose=0)
    print("Accuracy on dataset 2: %.2f%%" % (scores_2[1]*100))

    scores_stopword = model.evaluate(xstopword_test, y2_test, verbose=0)
    print("Accuracy on stopwords: %.2f%%" % (scores_stopword[1]*100))

    scores_lemma = model.evaluate(xlemma_test, y2_test, verbose=0)
    print("Accuracy on lemmatized: %.2f%%" % (scores_lemma[1]*100))

elif gg == 2:
    ### Build the model ###
    embedding_vector_length = 32
    model = Sequential()
    model.add(Embedding(vocab, embedding_vector_length, input_length=max_review_length))
    model.add(LSTM(100, dropout=0.3, return_sequences=True))
    model.add(LSTM(80, dropout=0.3, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, validation_split=0.15, epochs=5, batch_size=64)

    ### Model evaluation ###
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Testset Accuracy: %.2f%%" % (scores[1]*100))

    scores_2 = model.evaluate(x2_test, y2_test, verbose=0)
    print("Accuracy on dataset 2: %.2f%%" % (scores_2[1]*100))

    scores_stopword = model.evaluate(xstopword_test, y2_test, verbose=0)
    print("Accuracy on stopwords: %.2f%%" % (scores_stopword[1]*100))

    scores_lemma = model.evaluate(xlemma_test, y2_test, verbose=0)
    print("Accuracy on lemmatized: %.2f%%" % (scores_lemma[1]*100))



else:
    ### Load and evaluate previous model ###
    model = load_model('model.h5')
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    ### Predict sentiment from reviews ###
    bad = "The app is constantly freezing"
    good = "This game is really fun and addictive"
    bad_ = one_hot(bad, vocab)
    good_ = one_hot(good, vocab)
    bad_encoded = sequence.pad_sequences([bad_], maxlen=max_review_length)
    good_encoded = sequence.pad_sequences([good_], maxlen=max_review_length)
    print(bad, "\nSentiment: ", model.predict(np.array([bad_encoded][0]))[0][0])
    print(good, "\nSentiment: ", model.predict(np.array([good_encoded][0]))[0][0])


x = np.arange(1, 6)
# summarize history for accuracy
plt.plot(x, model.history.history['acc'])
# plt.plot(x, model.history.history['val_acc'])
plt.title('Embedding-LSTM model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(x, model.history.history['loss'])
# plt.plot(x, model.history.history['val_loss'])
plt.title('Embedding-LSTM model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
plt.show()

# from keras.utils import plot_model
# plot_model(model, to_file='model.png')

print('LIVE DEMONSTRATION')
# Demonstration
cont = True
while cont:
    print('\n--------------------------------------')

    inp = str(input('Enter your comment: '))
    if inp == 'x':
        cont = False
    else:
        input_onehot = one_hot(inp, vocab)
        guess_input = sequence.pad_sequences([input_onehot], maxlen=max_review_length)
        predicted = np.round(model.predict(np.array([guess_input][0]))[0][0], decimals=3)
        if predicted > 0.5:
            sentimental = 'This is a POSITIVE review'
        else:
            sentimental = 'This is a NEGATIVE review'
        print("\n{} with a sentiment score of: {}".format(sentimental, predicted))


