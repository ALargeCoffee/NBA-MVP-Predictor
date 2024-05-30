from torch import nn

# Class for the predictor model
class MVPPredictorModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model_sequence = nn.Sequential(
            nn.Linear(15, 30),
            nn.ReLU(),
            nn.LayerNorm(30),
            nn.Linear(30, 60),
            nn.ReLU(),
            nn.LayerNorm(60),
            nn.Linear(60, 30),
            nn.ReLU(),
            nn.LayerNorm(30),
            nn.Linear(30, 15),
            nn.Softmax()
        )

    def forward(self, input):
        return self.model_sequence(input)
