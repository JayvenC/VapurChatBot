import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

with open("intents.json") as file:
    data= json.load(file)

try:
    with open("data.pickle", "rb") as f: #rb as read bites
        words, labels, training, output = pickle.load(f) #saves all variables into picle file
except:
    words = []
    labels = []
    docs_x = [] #list of all different patterns
    docs_y = [] #words and tags for those words

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            existing_words = nltk.word_tokenize(pattern)
            words.extend(existing_words)
            docs_x.append(existing_words)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != '?']
    words = sorted(list(set(words))) #make sure there are no duplicate elements

    labels = sorted(labels)

    #neural networks only understand numbers, all of our code is in strings as of now

    training = []
    output =[]

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag =[]

        existing_words = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in existing_words:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training) #change into arrays
    output = numpy.array(output)

    with open("data.pickle", "wb") as f: 
        pickle.dump((words, labels, training, output), f)

tensorflow.compat.v1.reset_default_graph

net = tflearn.input_data(shape=[None, len(training[0])]) # define the input shape that we are expecting for our model
net = tflearn.fully_connected(net, 8) #two hidden layers with 8 neurons each that are fully connected to input
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") # softmax has 6 neurons; softmax activation gives us a probability for each neuron(tag) in the output layer
net = tflearn.regression(net)

model = tflearn.DNN(net) #DNN is a type of neural network; trains the model

try:
    model.load("model.tflearn")
    
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True) #n_epoch shows the data a "n" amount of times
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

def chat():
    print("Vapur bot is ready to chat! (type quit to terminate)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results) # will give us the index of the greatest number probabibility
        tag = labels[results_index]


        if(results[results_index] > 0.7):
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print(random.choice(responses))

        else:
            print("I'm not sure what happened. Please try again")

chat()