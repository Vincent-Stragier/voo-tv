import random
import json
import os
import pickle
import numpy as np

import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model


def near_script(filename: str):
    return os.path.join(os.path.dirname(__file__), filename)


lemmatizer = WordNetLemmatizer()
# intents = json.loads(open(near_script("intents.json"), "r").read())

# print(near_script("classes.pkl"))
words = pickle.load(open(near_script("words.pkl"), "rb"))
classes = pickle.load(open(near_script("classes.pkl"), "rb"))
model = load_model(near_script("chatbot_model.h5"))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word) for word in sentence_words]


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for sentence_word in sentence_words:
        for index, word in enumerate(words):
            if sentence_word == word:
                bag[index] = 1
    return np.array(bag, dtype=list).astype('float64')


def predict_class(sentence):
    bow = bag_of_words(sentence)
    results = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[index, result] for index, result
               in enumerate(results) if result > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for result in results:
        return_list.append(
            {"intent": classes[result[0]], "probability": str(result[1])})
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]["intent"]
    list_of_intents = intents_json["intents"]
    for intent in list_of_intents:
        if intent["tag"] == tag:
            result = random.choice(intent['responses'])
            break
    return result


if __name__ == '__main__':
    print("Chatbot is starting")

    while True:
        message = input("Human >> ")
        predicted_intents = predict_class(message)
        print(predicted_intents, "\n")
        response = get_response(predicted_intents, intents)
        print("Bot >>", response)
