import torch
import torch.nn as nn
import numpy as np
import os

class PrototypicalNet(nn.Module):
    def __init__(self, config):
        super(SiaGRU, self).__init__()
        self.config = config
        self.embeds_dim = config["embedding_size"]
        num_word = config["sequence_length"]
        self.embeds = nn.Embedding(num_word, self.embeds_dim)
        self.ln_embeds = nn.LayerNorm(self.embeds_dim)
        self.hidden_size = config["hidden_size"]
        self.num_layer = 2
        self.gru = nn.LSTM(self.embeds_dim, self.hidden_size, batch_first=True, bidirectional=False, num_layers=1)
        self.h0 = self.init_hidden((2 * self.num_layer, 1, self.hidden_size))

    def init_hidden(self, size):
        h0 = nn.Parameter(torch.randn(size))
        nn.init.xavier_normal_(h0)
        return h0

    def get_word_vectors(self, path):
        word_vectors = np.load(path)
        return word_vectors

    def forward_once(self, x):
        output, hidden = self.gru(x)
        return hidden[0].squeeze()

    def forward(self, input):
        support_set = input["support"]
        query_set = input["queries"]
        labels = input["labels"]

        word_vectors = self.get_word_vectors(os.path.join(self.config["output_path"], "word_vectors.npy"))
        support_embeds = [[word_vectors[i] for i in j] for j in support_set]
        query_embeds = [[word_vectors[i] for i in j] for j in query_set]

        support_embeds = torch.Tensor(support_embeds)
        query_embeds = torch.Tensor(query_embeds)

        support_embeds = support_embeds.reshape((-1, self.config["sequence_length"], self.config["embedding_size"]))

        support_encoding = self.forward_once(support_embeds)
        query_encoding = self.forward_once(query_embeds)

        support_encoding = support_encoding.reshape((self.config["num_classes"], self.config["num_support"], -1))
        support_mean = torch.mean(support_encoding, dim=1)
        support_mean_1 = support_mean[0].repeat((self.config["num_classes"] * self.config["num_queries"], 1))
        support_mean_2 = support_mean[1].repeat((self.config["num_classes"] * self.config["num_queries"], 1))
        support_mean = torch.cat((support_mean_1, support_mean_2), dim=0)

        # print(support_mean.size())
        query_encoding = query_encoding.repeat((self.config["num_classes"], 1))
        dis = -torch.norm(support_mean - query_encoding, p=2, dim=-1, keepdim=True)
        # print(dis.size())
        dis = dis.reshape((-1, self.config["num_classes"]))
        # dis = dis.reshape((self.config["num_classes"] * self.config["num_queries"]))
        predictions = torch.softmax(dis, dim=1)
        # print(predictions.size())

        return predictions

        # sent1 = input[0]
        # sent2 = input[1]
        #
        # x1 = self.ln_embeds(self.embeds(sent1).transpose(1, 2).contiguous()).transpose(1, 2)
        # x2 = self.ln_embeds(self.embeds(sent2).transpose(1, 2).contiguous()).transpose(1, 2)
        #
        # encoding1 = self.forward_once(x1)
        # encoding2 = self.forward_once(x2)
        #
        # # sim = torch.exp(-torch.norm(encoding1 - encoding2, p=2, dim=1, keepdim=True))
        # dis = -torch.norm(encoding1 - encoding2, p=2, dim=1, keepdim=True)

        # return self.fc(sim)
