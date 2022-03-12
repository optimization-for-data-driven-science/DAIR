import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence
import ipdb
import config


class Net(nn.Module):
    """ tweaking code to function as CNN_LSTM model now
    """

    def __init__(self, embedding_tokens):
        super(Net, self).__init__()
        question_features = 1024
        vision_features = config.output_features   ## 2048
        glimpses = 2

        self.text = TextProcessor(
            embedding_tokens=embedding_tokens,
            embedding_features=300,
            lstm_features=question_features,
            drop=0.5
        )

        self.cnn = build_cnn(
            v_features=vision_features,  #(2048, 14, 14)
            out_features=4096, #1024, 512, 128
            kernel_size=1,
            bias=True
        )

        self.classifier = Classifier(
            in_features=glimpses * vision_features + question_features,
            mid_features=1024,
            out_features=config.max_answers,  ## 3000
            drop=0.5
        )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, v, q, q_len):  ## dim of raw v   torch.Size([64, 2048, 14, 14])        q.size())  torch.Size([64, 23])       q_len.size()) torch.Size([64])

        #ipdb.set_trace()   ### here ipdb.set_trace works!!!
        q = self.text(q, list(q_len.data))  # torch.Size([64, 1024])
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)     #torch.Size([64, 2048, 14, 14])



        v = self.cnn(v)
        v_flat = flatten(v)  ## torch.Size([64, 4096]) => all these 4096 features- you do a weighted mean on it- so you finally have one number
        if config.vis_attention:
            print(v.shape)
            my_index = torch.argmax(v_flat, dim=1)  # then for attn map- i pick the one with highest value
            home = v[71][my_index[71]]
            plot_attn = torch.nn.functional.softmax(home)

        ipdb.set_trace()
        combined = torch.cat([v_flat, q], dim=1)
        answer = self.classifier(combined)  ## torch.Size([64, 3000])
        return answer


# def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#              padding=0, dilation=1, groups=1, bias=True):

class build_cnn(nn.Module):
    def __init__(self,  v_features, out_features, kernel_size=1, bias=True):
        super(build_cnn, self).__init__()
        self.v_conv = nn.Conv2d(v_features, out_features, kernel_size=kernel_size, bias=bias)    #kernel_size=1, bias=True (default)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, v):
        v = self.v_conv(v)         ## not using drop on v at all!!!
        x = self.relu(v)
        return x


def flatten(input):
    n, c = input.size()[:2]
    # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
    input = input.view(n, 1, c, -1) # [n, 1, c, s]
    weighted_mean = input.sum(dim=-1) # [n, g, v]
    return weighted_mean.view(n, -1)


class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('lin1', nn.Linear(in_features, mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('drop2', nn.Dropout(drop))
        self.add_module('lin2', nn.Linear(mid_features, out_features))


class TextProcessor(nn.Module):
    def __init__(self, embedding_tokens, embedding_features, lstm_features, drop=0.0):
        super(TextProcessor, self).__init__()
        self.embedding = nn.Embedding(embedding_tokens, embedding_features, padding_idx=0)
        self.drop = nn.Dropout(drop)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(input_size=embedding_features,
                            hidden_size=lstm_features,
                            num_layers=1)
        self.features = lstm_features

        self._init_lstm(self.lstm.weight_ih_l0)
        self._init_lstm(self.lstm.weight_hh_l0)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

        init.xavier_uniform_(self.embedding.weight)

    def _init_lstm(self, weight):
        for w in weight.chunk(4, 0):
            init.xavier_uniform_(w)

    def forward(self, q, q_len):
        embedded = self.embedding(q)
        tanhed = self.tanh(self.drop(embedded))
        packed = pack_padded_sequence(tanhed, q_len, batch_first=True)
        _, (_, c) = self.lstm(packed)
        return c.squeeze(0)

class Attention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(Attention, self).__init__()
        self.v_conv = nn.Conv2d(v_features, mid_features, 1, bias=False)  # let self.lin take care of bias
        self.q_lin = nn.Linear(q_features, mid_features)
        self.x_conv = nn.Conv2d(mid_features, glimpses, 1)

        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, v, q):
        v = self.v_conv(self.drop(v))
        q = self.q_lin(self.drop(q))
        q = tile_2d_over_nd(q, v)
        x = self.relu(v + q)
        x = self.x_conv(self.drop(x))
        return x

def tile_2d_over_nd(feature_vector, feature_map):
    """ Repeat the same feature vector over all spatial positions of a given feature map.
        The feature vector should have the same batch size and number of features as the feature map.
    """
    n, c = feature_vector.size()
    spatial_size = feature_map.dim() - 2
    tiled = feature_vector.view(n, c, *([1] * spatial_size)).expand_as(feature_map)
    return tiled
