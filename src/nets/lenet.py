import torch
import torch.nn as nn
import torch.nn.functional as F


class MIL(nn.Module):
    def __init__(self, in_dim=1, num_classes=2, **kwargs):
        super(MIL, self).__init__()
        self.num_classes = num_classes

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(in_dim, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, 500),
            nn.ReLU(),
        )
        self.num_features = 500
        self.classifier = nn.Linear(500, self.num_classes)

    def forward(self, x, only_feat=False):
        # TODO: check this
        batch_size = x.shape[0]
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(batch_size, -1)
        H = self.feature_extractor_part2(H)  # NxL
        
        if only_feat:
            return H
        
        Y_prob = self.classifier(H)
        # Y_hat = torch.ge(Y_prob, 0.5).float()

        # return Y_prob, Y_hat, A
        return Y_prob


class MILAttention(nn.Module):
    def __init__(self, in_dim=1, num_classes=2, **kwargs):
        super(MILAttention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1
        self.num_classes = num_classes

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(in_dim, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.classifier = nn.Linear(self.L*self.K, 2)

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        return Y_prob


class MILGatedAttention(nn.Module):
    def __init__(self, in_dim=1, num_classes=2, **kwargs):
        super(MILGatedAttention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(in_dim, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Linear(self.L*self.K, 2)

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        return Y_prob


def lenet5(**kwargs):
    model = MIL(**kwargs)
    return model 

def lenet5_c3(**kwargs):
    model = MIL(in_dim=3, **kwargs)
    return model 

def attn_lenet5(**kwargs):
    model = MILAttention(**kwargs)
    return model 


def gated_attn_lenet5(**kwargs):
    model = MILGatedAttention(**kwargs)
    return model


