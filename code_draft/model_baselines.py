import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()

        self.args = args
        
        self.model = nn.LSTM(args.regions, args.regions, args.lstm_layers)

        self.regressor = nn.Sequential(nn.Linear(args.regions, args.regions)
                                        , nn.ReLU()
                                        , nn.Dropout(p=0.1)
                                        , nn.Linear(args.regions, args.semantic))
        
        self.classifier = nn.Sequential(nn.Linear(args.regions, args.regions)
                                        , nn.ReLU()
                                        , nn.Dropout(p=0.1)
                                        , nn.Linear(args.regions, 1)
                                        , Transpose(-1, -2)
                                        , nn.Linear(args.time_points, args.classes))
        if args.task == 'all':
            pass
        elif args.task == 'reg':
            self.classifier = None
        elif args.task == 'cla':
            self.regressor = None

    def reset_parameters(self):
        self.model.reset_parameters()
        if self.args.task == 'all':
            for layer in self.regressor:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            for layer in self.classifier:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        elif self.args.task == 'reg':
            for layer in self.regressor:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        elif self.args.task == 'cla':
            for layer in self.classifier:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def forward(self, x):
        x, _ = self.model(x)
        if self.args.task == 'all':
            logits1 = self.regressor(x)
            logits2 = self.classifier(x)
            logits2 = F.softmax(logits2)
        elif self.args.task == 'reg':
            logits1 = self.regressor(x)
            logits2 = None
        elif self.args.task == 'cla':
            logits1 = None
            logits2 = self.classifier(x)
            logits2 = F.softmax(logits2)

        return logits2, logits1

class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()

        self.args = args
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=args.regions, nhead=args.nheads)
        self.model = nn.TransformerEncoder(self.encoder_layer, num_layers=args.lstm_layers)
        
        self.regressor = nn.Sequential(nn.Linear(args.regions, args.regions)
                                        , nn.ReLU()
                                        , nn.Dropout(p=0.1)
                                        , nn.Linear(args.regions, args.semantic))
        
        self.classifier = nn.Sequential(nn.Linear(args.regions, args.regions)
                                        , nn.ReLU()
                                        , nn.Dropout(p=0.1)
                                        , nn.Linear(args.regions, 1)
                                        , Transpose(-1, -2)
                                        , nn.Linear(args.time_points, args.classes))
        if args.task == 'all':
            pass
        elif args.task == 'reg':
            self.classifier = None
        elif args.task == 'cla':
            self.regressor = None

    def reset_parameters(self):
        for layer in self.model.layers:
          if hasattr(layer, 'reset_parameters'):
              layer.reset_parameters()
        if self.args.task == 'all':
            for layer in self.regressor:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            for layer in self.classifier:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        elif self.args.task == 'reg':
            for layer in self.regressor:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        elif self.args.task == 'cla':
            for layer in self.classifier:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def forward(self, x):
        x = self.model(x)
        if self.args.task == 'all':
            logits1 = self.regressor(x)
            logits2 = self.classifier(x)
            logits2 = F.softmax(logits2)
        elif self.args.task == 'reg':
            logits1 = self.regressor(x)
            logits2 = None
        elif self.args.task == 'cla':
            logits1 = None
            logits2 = self.classifier(x)
            logits2 = F.softmax(logits2)

        return logits2, logits1

class TS_Transformer(nn.Module):
    def __init__(self, args):
        super(TS_Transformer, self).__init__()

        self.config = TimeSeriesTransformerConfig(prediction_length=args.regions)

        self.model = TimeSeriesTransformerModel(self.config)

        self.regressor = (nn.Linear(args.regions, args.regions)
                                        , nn.ReLU()
                                        , nn.Dropout(p=0.1)
                                        , nn.Linear(args.regions, args.semantic))
        
        self.classifier = self.classifier = nn.Sequential(nn.Linear(args.regions, args.regions)
                                        , nn.ReLU()
                                        , nn.Dropout(p=0.1)
                                        , nn.Linear(args.regions, 1)
                                        , Transpose(-1, -2)
                                        , nn.Linear(args.time_points, args.classes))

        if args.task == 'all':
            pass
        elif args.task == 'reg':
            self.classifier = None
        elif args.task == 'cla':
            self.regressor = None

    def reset_parameters(self):
        self.model.reset_parameters()
        if self.args.task == 'all':
            for layer in self.regressor:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            for layer in self.classifier:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        elif self.args.task == 'reg':
            for layer in self.regressor:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        elif self.args.task == 'cla':
            for layer in self.classifier:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
    
    def forward(self, x):
        x_output = self.model(x)
        x = x_output.sequences
        if self.args.task == 'all':
            logits1 = self.regressor(x)
            logits2 = self.classifier(x)
            logits2 = F.softmax(logits2)
        elif self.args.task == 'reg':
            logits1 = self.regressor(x)
            logits2 = None
        elif self.args.task == 'cla':
            logits1 = None
            logits2 = self.classifier(x)
            logits2 = F.softmax(logits2)

        return logits2, logits1, x_output.loss

class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()

        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)
