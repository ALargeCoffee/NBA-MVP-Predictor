import torch
from torch import optim
from torch import nn
import numpy as np
import data_fetcher
import prediction_model

if __name__ == '__main__':

    # Retrieve data and seasons to be randomly sampled
    data = data_fetcher.fetch_data()
    grouped_data = data.groupby('Season')
    seasons = data['Season'].unique()

    # Initialize model, optimizer, and loss function
    model = prediction_model.MVPPredictorModel()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    loss_function = nn.BCELoss()
    prev_loss = 1000
    curr_loss = 500
    iterations = 0

    # Iterate until convergence threshold
    while abs(prev_loss - curr_loss) > 1e-7:
        iterations += 1
        optimizer.zero_grad()

        # Randomly choose a season and slice important columns
        chosen_season = np.random.choice(seasons)
        season_x = torch.Tensor(grouped_data.get_group(chosen_season).iloc[:, 1:-2].to_numpy())
        season_y = torch.Tensor(grouped_data.get_group(chosen_season).loc[:, 'MVP'].to_numpy())
        outputs = model(season_x)

        # Need to softmax outputs since each player computes individually
        outputs = nn.Softmax(dim=0)(torch.squeeze(outputs))

        # Optimize and reset
        loss = loss_function(outputs, season_y)
        loss.backward()
        optimizer.step()
        prev_loss = curr_loss
        curr_loss = loss.item()
        if iterations % 100 == 0:
            print("Iteration " + str(iterations) + " loss: " + str(curr_loss))
    print("Final loss: " + str(curr_loss))