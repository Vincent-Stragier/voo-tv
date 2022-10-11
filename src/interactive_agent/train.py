# Train the neural networks for the NLU
import pandas

import random
import json
import os
import pickle
import numpy as np
import re
import torch  # BERT
import torch.nn as nn  # BERT
import random
import transformers  # BERT
import matplotlib.pyplot as plt
import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

from transformers import AutoModel, BertTokenizerFast  # BERT


def near_script(filename: str):
    return os.path.join(os.path.dirname(__file__), filename)


intent = pandas.read_excel(near_script('intent.xlsx'))
intent_requests = list(intent[intent.columns[1]])
intent_classes = list(intent[intent.columns[0]])


# import GPU for training
device = torch.device("cuda")
lemmatizer = WordNetLemmatizer()
# intents = json.loads(open(near_script("intents.json"), "r").read())

words = []
classes = sorted(list(set(intent_classes)))
documents = []
ignore_letters = ["?", "!", ".", ",", ";", ":"]

for request, intent_class in zip(intent_requests, intent_classes):
    # for pattern in intent["patterns"]:
    word_list = nltk.word_tokenize(request)
    words.extend(word_list)
    documents.append((word_list, intent_class))

words = sorted(set([lemmatizer.lemmatize(word)
                    for word in words if word not in ignore_letters]))

pickle.dump(words, open(near_script("words.pkl"), "wb"))
pickle.dump(classes, open(near_script("classes.pkl"), "wb"))
# pickle.dump(labels, open(near_script("labels.pkl"), "wb"))

training = []
output_empty = [0] * len(classes)

for document in documents:
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower())
                     for word in word_patterns]
    bag = [1 if word in word_patterns else 0 for word in words]
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=list)


train_x, train_y = list(training[:, 0]), list(training[:, 1])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=["accuracy"])

hist = model.fit(np.array(train_x), np.array(train_y),
                 epochs=200, batch_size=5, verbose=1)
model.save(near_script("chatbot_model.h5"), hist)
print("Done (with custom network)")
exit('No BERT training')

"""Training with BERT"""
# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')  # BERT
# Import BERT-base pretrained model
bert = AutoModel.from_pretrained('bert-base-uncased')  # BERT

label_encoder = LabelEncoder()
intent_labels = label_encoder.fit_transform(intent_classes)

train_text = intent_requests
train_labels = intent_labels

seq_len = [len(i.split()) for i in train_text]
max_seq_len = max(seq_len)
print(f'{max_seq_len = }')

# tokenize and encode sequences in the training set
tokens_train = tokenizer(
    train_text,
    max_length = max_seq_len,
    # pad_to_max_length=True, # depreciated use padding='longest'
    padding='longest',
    truncation=True,
    return_token_type_ids=False
)

# for train set
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels)

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
#define a batch size
batch_size = 16
# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)
# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)
# DataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert

        # dropout layer
        self.dropout = nn.Dropout(0.2)

        # relu activation function
        self.relu = nn.ReLU()
        # dense layer
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, len(set(train_labels))) # _, Number of classes
        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)
        # define the forward pass

    def forward(self, sent_id, mask):
        # pass the inputs to the model
        cls_hs = self.bert(sent_id, attention_mask=mask)[0][:, 0]

        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        # output layer
        x = self.fc3(x)

        # apply softmax activation
        x = self.softmax(x)
        return x

# from transformers import BERT_Model
# freeze all the parameters. This will prevent updating of model weights during fine-tuning.
for param in bert.parameters():
      param.requires_grad = False
model = BERT_Arch(bert)
# push the model to GPU
model = model.to(device)
from torchinfo import summary
summary(model)

from transformers import AdamW
# define the optimizer
optimizer = AdamW(model.parameters(), lr = 1e-3)

# Compute class weigth
from sklearn.utils.class_weight import compute_class_weight
#compute the class weights
# class_wts = compute_class_weight('balanced', np.unique(train_labels), train_labels)
class_wts = compute_class_weight( 'balanced', classes=np.unique(train_labels),  y=train_labels)
print(f'{class_wts = }')

# convert class weights to tensor
weights = torch.tensor(class_wts, dtype=torch.float)
weights = weights.to(device)
# loss function
cross_entropy = nn.NLLLoss(weight=weights)
# exit()
# empty lists to store training and validation loss of each epoch
train_losses=[]
# number of training epochs
epochs = 200
# We can also use learning rate scheduler to achieve better results
# lr_sch = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

# function to train the model
def train():
    model.train()
    total_loss = 0

    # empty list to save model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(train_dataloader):
        # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            print(f'  Batch {step:>5,}  of  {len(train_dataloader):>5,}.')

        # push the batch to gpu
        batch = [r.to(device) for r in batch]
        # print(batch)
        sent_id, mask, labels = batch
        # get model predictions for the current batch
        preds = model(sent_id, mask)
        # print(preds)
        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)
        # add on to the total loss
        total_loss = total_loss + loss.item()
        # backward pass to calculate the gradients
        loss.backward()
        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # update parameters
        optimizer.step()
        # clear calculated gradients
        optimizer.zero_grad()

        # We are not using learning rate scheduler as of now
        # lr_sch.step()
        # model predictions are stored on GPU. So, push it to CPU
        preds = preds.detach().cpu().numpy()
        # append the model predictions
        total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)
    # returns the loss and predictions
    return avg_loss, total_preds

# Train the model
for epoch in range(epochs):
     
    print(f'\n Epoch {epoch + 1:} / {epochs:}')
    
    #train model
    train_loss, _ = train()
    
    # append training and validation loss
    train_losses.append(train_loss)
    # it can make your experiment reproducible, similar to set  random seed to all options where there needs a random seed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
print(f'\nTraining Loss: {train_loss:.3f}')

def get_prediction(input_string):
    input_string = re.sub(r'[ ^ a-zA-Z] +', '', input_string)
    test_text = [input_string]
    model.eval()

    tokens_test_data = tokenizer(
        test_text,
        max_length=max_seq_len,
        padding='longest',
        truncation=True,
        return_token_type_ids=False
    )
    test_seq = torch.tensor(tokens_test_data['input_ids'])
    test_mask = torch.tensor(tokens_test_data['attention_mask'])

    preds = None
    with torch.no_grad():
        preds = model(test_seq.to(device), test_mask.to(device))
    preds = preds.detach().cpu().numpy()
    print(preds)
    preds = np.argmax(preds, axis=1)
    print("Intent Identified: ", label_encoder.inverse_transform(preds)[0])
    return label_encoder.inverse_transform(preds)[0]


def get_response(message):
    intent = get_prediction(message)
    # for i in data['intents']:
    #     if i["tag"] == intent:
    #         result = random.choice(i["responses"])
    #         break
    # print(f"Response : {result}")
    # \nResponse: {result}
    return f"Intent: {intent}"

get_response("Quelle heure est-il ?")


"""
BERT is less performant... why?
Wrong implementation? (preprocessing =/=)
Not trained for French?
Not enough training data?
"""