import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, AdamW, GPT2LMHeadModel
import math, pickle
import datetime, re
from model import ContinuousPromptLearning, MLP
from utils import now_time, Dictionary, Batchify

def train(model, data, token_num):
    model.train()
    loss_fn = torch.nn.BCELoss()
    while True:
        user, item, seq, mask, aspect, _ = data.next_batch()
        user, item, seq, mask, aspect = user.to(device), item.to(device), seq.to(device), mask.to(device), aspect.to(device)
        aspect = nn.functional.one_hot(aspect, num_classes=token_num)
        aspect = aspect.sum(dim=1).float()
        optimizer.zero_grad()
        outputs = model.forward(user, item, seq, None)
        #lm_loss = outputs.loss
        predict_token = outputs.logits[:, -1, :]
        aspect_prob = torch.softmax(predict_token, dim=-1)
        aspect_loss = loss_fn(aspect_prob, aspect)
        #loss = lm_loss + aspect_loss
        aspect_loss.backward()
        optimizer.step()
        if data.step == data.total_step:
            break

def evaluate(model, data, token_num):
    model.eval()
    text_loss = 0.
    total_sample = 0
    loss_fn = torch.nn.BCELoss()
    with torch.no_grad():
        while True:
            user, item, seq, mask, aspect, _ = data.next_batch()
            user, item, seq, mask, aspect = user.to(device), item.to(device), seq.to(device), mask.to(device), aspect.to(device)
            aspect = nn.functional.one_hot(aspect, num_classes=token_num)
            aspect = aspect.sum(dim=1).float()
            outputs = model.forward(user, item, seq, mask)
            predict_token = outputs.logits[:, -1, :]
            aspect_prob = torch.softmax(predict_token, dim=-1)
            aspect_loss = loss_fn(aspect_prob, aspect)
            batch_size = user.size(0)
            text_loss += batch_size * aspect_loss.item()
            total_sample += batch_size
            if data.step == data.total_step:
                break
    return text_loss / total_sample

def generate(model, data):
    model.eval()
    idss_predict, rating_predict = [], []
    with torch.no_grad():
        while True:
            user, item, seq, _, _, _ = data.next_batch()
            user, item, seq = user.to(device), item.to(device), seq.to(device)
            outputs = model(user, item, seq, None)
            last_token = outputs.logits[:, -1, :]  # the last token, (batch_size, ntoken)
            word_prob = torch.softmax(last_token, dim=-1)
            token = torch.topk(word_prob, 3, dim=1).indices
            ids = token.tolist()
            idss_predict.extend(ids)
            aspect_embedding = model.transformer.wte(token)
            user_embedding = model.user_embeddings(user)
            item_embedding = model.item_embeddings(item)
            rating_pred = recsys_model(user_embedding, item_embedding, aspect_embedding)
            rating_pred = rating_pred.squeeze(1).tolist()
            rating_predict.extend(rating_pred)
            if data.step == data.total_step:
                break
    return idss_predict, rating_predict

def train_recommendation(model, recsys_model, data):
    model.train()
    recsys_model.train()
    loss_fn = torch.nn.MSELoss()
    while True:
        user, item, seq, _, _, rating = data.next_batch()
        user, item, seq, rating = user.to(device), item.to(device), seq.to(device), rating.to(device)
        outputs = model(user, item, seq, None)
        last_token = outputs.logits[:, -1, :]  # the last token, (batch_size, ntoken)
        word_prob = torch.softmax(last_token, dim=-1)
        token = torch.topk(word_prob, 3, dim=1).indices.tolist()
        token = torch.tensor(token, dtype=torch.int64).to(device)
        aspect_embedding = model.transformer.wte(token)
        user_embedding = model.user_embeddings(user)
        item_embedding = model.item_embeddings(item)
        rating_pred = recsys_model(user_embedding, item_embedding, aspect_embedding)
        loss = loss_fn(rating_pred.squeeze(1), rating)
        loss.backward()
        optimizer.step()
        if data.step == data.total_step:
            break

## Setting Data Path
data_path = 'Amazon/MoviesAndTV/reviews.pickle'
#data_path = 'TripAdvisor/reviews.pickle'
#data_path = 'Yelp/reviews.pickle'

## Setting Hyperparameters and Initializing Symbols
cuda = True
seq_len = 20
aspect_len = 3
batch_size = 128
epochs = 10
lr = 0.0001
data = []
user_dict = Dictionary()
item_dict = Dictionary()
bos = '<bos>'
eos = '<eos>'
pad = '<pad>'
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token=bos, eos_token=eos, pad_token=pad)
device = torch.device('cuda' if cuda else 'cpu')

## Loading Review Data
reviews = pickle.load(open(data_path, 'rb'))[:128000]
for review in reviews:
    (fea, adj, tem, sco) = review['template']
    predicted = review['predicted']
    aspects = fea + ' ' + adj + ' ' + predicted
    tokens = tokenizer(tem)['input_ids']
    aspect_tokens = tokenizer(aspects, padding='max_length', max_length=aspect_len)['input_ids']
    text = tokenizer.decode(tokens[:seq_len])
    #aspect = tokenizer.decode(aspect_tokens[:aspect_len])
    aspect = aspect_tokens[:aspect_len]
    user_dict.add_entity(review['user'])
    item_dict.add_entity(review['item'])
    data.append({'user': user_dict.term2idx[review['user']], 'item': item_dict.term2idx[review['item']], 'rating': review['rating'], 'text': text, 'aspect': aspect})
data_length = len(data)
print(data_length)
train_record = data[:4*data_length//5]
test_record = data[4*data_length//5:]
train_data = Batchify(train_record, tokenizer, bos, eos, batch_size, shuffle=True)
test_data = Batchify(test_record, tokenizer, bos, eos, batch_size)

## Initializing Learning Model
user_num = len(user_dict)
item_num = len(item_dict)
token_num = len(tokenizer)
model = ContinuousPromptLearning.from_pretrained('gpt2', user_num, item_num)
model.resize_token_embeddings(token_num)
model.to(device)
optimizer = AdamW(model.parameters(), lr=lr)
latent_dim = [5*768, 256, 128]
recsys_model = MLP(latent_dim)

## Prompt Tuning
for epoch in range(epochs):
    print(now_time() + 'epoch {}'.format(epoch))
    train(model, train_data, token_num)
    loss = evaluate(model, train_data, token_num)
    print(now_time() + ' prompt loss {:4.4f} on evaluation'.format(loss))

## Prompt and LM Tuning + Aspect-Based Recommendation
for param in model.parameters():
    param.requires_grad = True
optimizer = AdamW(model.parameters(), lr=lr)
for epoch in range(epochs):
    print(now_time() + 'epoch {}'.format(epoch))
    train(model, train_data, token_num)
    #train_recommendation(model, recsys_model, train_data)
    loss = evaluate(model, train_data, token_num)
    print(now_time() + 'prompt+LM loss {:4.4f} on evaluation'.format(loss))

## Model Testing
test_loss = evaluate(model, test_data, token_num)
print('=' * 50)
print(now_time() + 'loss in test set {:4.4f}'.format(test_loss))
idss_predict, rating_predict = generate(model, test_data)
idss_truth = test_data.aspect.tolist()
rating_truth = test_data.rating.tolist()
idss_predict = [x for y in idss_predict for x in y]
idss_truth = [x for y in idss_truth for x in y]

## Aspect Term Extraction Performance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(idss_truth, idss_predict)
precision = precision_score(idss_truth, idss_predict, average="micro")
recall = recall_score(idss_truth, idss_predict, average="micro")
f1 = f1_score(idss_truth, idss_predict, average="micro")
print('Accuracy Performance: '+str(accuracy))
print('Precision: '+str(precision))
print('Recall: '+str(recall))
print('F1-Score: '+str(f1))

## Aspect-Based Recommendation Peroformance
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
from math import sqrt

#rating_predict = [x[0] for y in rating_predict for x in y]
rmse = sqrt(mean_squared_error(rating_truth, rating_predict))
mae = mean_absolute_error(rating_truth, rating_predict)
rating_truth = [int(x>0.5) for x in rating_truth]
rating_predict = [int(x>0.5) for x in rating_predict]
auc = roc_auc_score(rating_truth, rating_predict)
print('RMSE: '+str(rmse))
print('MAE: '+str(mae))
print('AUC: '+str(auc))

'''
## Print Model Outputs

def postprocessing(string):
    string = re.sub('\'s', ' \'s', string)
    string = re.sub('\'m', ' \'m', string)
    string = re.sub('\'ve', ' \'ve', string)
    string = re.sub('n\'t', ' n\'t', string)
    string = re.sub('\'re', ' \'re', string)
    string = re.sub('\'d', ' \'d', string)
    string = re.sub('\'ll', ' \'ll', string)
    string = re.sub('\(', ' ( ', string)
    string = re.sub('\)', ' ) ', string)
    string = re.sub(',+', ' , ', string)
    string = re.sub(':+', ' , ', string)
    string = re.sub(';+', ' . ', string)
    string = re.sub('\.+', ' . ', string)
    string = re.sub('!+', ' ! ', string)
    string = re.sub('\?+', ' ? ', string)
    string = re.sub(' +', ' ', string).strip()
    return string

def ids2tokens(ids, tokenizer, eos):
    text = tokenizer.decode(ids)
    text = postprocessing(text)  # process punctuations: "good!" -> "good !"
    tokens = []
    for token in text.split():
        if token == eos:
            break
        tokens.append(token)
    return tokens

tokens_test = [ids2tokens(ids[1:], tokenizer, eos) for ids in test_data.seq.tolist()]
tokens_predict = [ids2tokens(ids, tokenizer, eos) for ids in idss_predict]
text_test = [' '.join(tokens) for tokens in tokens_test]
text_predict = [' '.join(tokens) for tokens in tokens_predict]
text_out = ''
for (real, fake) in zip(text_test, text_predict):
    text_out += '{}\n{}\n\n'.format(real, fake)
with open('record.txt', 'w', encoding='utf-8') as f:
    f.write(text_out)
'''
