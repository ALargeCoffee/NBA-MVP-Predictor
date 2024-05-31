from torch import nn

# Class for the predictor model
class MVPPredictorModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model_sequence = nn.Sequential(
            nn.Linear(15, 15),
            nn.ReLU(),
            nn.LayerNorm(15),
            nn.Linear(15, 10),
            nn.ReLU(),
            nn.LayerNorm(10),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.LayerNorm(5),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model_sequence(input)
