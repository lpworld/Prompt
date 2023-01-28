import math
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, AdamW, GPT2LMHeadModel
import random

cuda = True
device = torch.device('cuda' if cuda else 'cpu')

class AspectPrompt:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, nuser, nitem, freezeLM=True, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # freeze pretrained model parameters
        if freezeLM:
            for param in model.parameters():
                param.requires_grad = False

        model.init_prompt(nuser, nitem)
        return model

    def init_prompt(self, nuser, nitem):
        self.src_len = 2
        emsize = self.transformer.wte.weight.size(1)  # 768
        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize)

        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, user, item, text, mask, ignore_index=-100):
        device = user.device
        batch_size = user.size(0)

        # embeddings
        u_src = self.user_embeddings(user)
        i_src = self.item_embeddings(item)
        w_src = self.transformer.wte(text)
        src = torch.cat([u_src.unsqueeze(1), i_src.unsqueeze(1), w_src], 1)

        if mask is None:
            # auto-regressive generation
            return super().forward(inputs_embeds=src)
        else:
            # training
            pad_left = torch.ones((batch_size, self.src_len), dtype=torch.int64).to(device)
            pad_input = torch.cat([pad_left, mask], 1)

            # prediction for training
            pred_left = torch.full((batch_size, self.src_len), ignore_index, dtype=torch.int64).to(device)  # (batch_size, src_len)
            pred_right = torch.where(mask == 1, text, torch.tensor(ignore_index).to(device))  # replace <pad> with ignore_index
            prediction = torch.cat([pred_left, pred_right], 1)  # (batch_size, total_len)

            return super().forward(attention_mask=pad_input, inputs_embeds=src, labels=prediction)


class ContinuousPromptLearning(AspectPrompt, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)

class MLP(torch.nn.Module):
    def __init__(self, latent_dim):
        super(MLP, self).__init__()
        self.latent_dim = latent_dim
        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(self.latent_dim[:-1], self.latent_dim[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
        self.affine_output = torch.nn.Linear(in_features=self.latent_dim[-1], out_features=1).to(device)
        self.logistic = torch.nn.Sigmoid()
        self.fc_layers = self.fc_layers.to(device)

    def forward(self, user_embedding, item_embedding, aspect_embedding):
        batch_size = user_embedding.size(0)
        aspect_num = 3
        hidden_dim = 768
        aspect_embedding = aspect_embedding.reshape([batch_size, aspect_num*hidden_dim])
        vector = torch.cat([user_embedding, item_embedding, aspect_embedding], dim=1)  # the concat latent vector
        #vector = vector.float()
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.Dropout(p=0.1)(vector)
            vector = torch.nn.ReLU()(vector)
            #vector = torch.nn.BatchNorm1d()(vector)
        rating = self.affine_output(vector)
        rating = self.logistic(rating)
        return rating
