import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import time
import os
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'data', 'u.data')

data = pd.read_csv(file_path, sep="\t", header=None, 
                   names=['userId', 'movieId', 'rating', 'timestamp'], 
                   engine='python')

user_ids = data['userId'].unique()
movie_ids = data['movieId'].unique()

user_mapping = {user_id: idx for idx, user_id in enumerate(user_ids)}
movie_mapping = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

data['userId'] = data['userId'].map(user_mapping)
data['movieId'] = data['movieId'].map(movie_mapping)

num_users = len(user_ids)
num_movies = len(movie_ids)

train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

class MovieLensDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe[['userId', 'movieId']].values
        self.ratings = dataframe['rating'].values.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.LongTensor(self.data[idx]), torch.FloatTensor([self.ratings[idx]])

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_movies, embedding_size):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)
        
        self.global_bias = nn.Parameter(torch.tensor(0.0))
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_movies, 1)
        
        nn.init.constant_(self.user_bias.weight, 0)
        nn.init.constant_(self.item_bias.weight, 0)

    def forward(self, X):
        user_emb = self.user_embedding(X[:, 0])
        movie_emb = self.movie_embedding(X[:, 1])
        
        dot_product = torch.sum(user_emb * movie_emb, dim=1)
        
        user_b = self.user_bias(X[:, 0]).squeeze()
        item_b = self.item_bias(X[:, 1]).squeeze()
        
        prediction = self.global_bias + user_b + item_b + dot_product
        return prediction

def train_and_evaluate(embedding_size, learning_rate, regularization, num_epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 1024
    train_dataset = MovieLensDataset(train_data)
    test_dataset = MovieLensDataset(test_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = MatrixFactorization(num_users, num_movies, embedding_size)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=regularization)

    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            predictions = model(X)
            loss = criterion(predictions, y.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

    model.eval()
    squared_errors = []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            predictions = model(X)
            residuals = predictions - y.squeeze()
            squared_errors.extend((residuals ** 2).tolist())

    rmse = np.sqrt(np.mean(squared_errors))
    return rmse, train_losses

embedding_sizes = [16, 32]
learning_rates = [0.01, 0.05, 0.1]
regularizations = [0.0, 1e-4]

print(f"Total Number of Combinations: {len(embedding_sizes) * len(learning_rates) * len(regularizations)}")
print("-" * 60)
print(f"{'Emb Size':<10} | {'LR':<10} | {'Reg (Decay)':<12} | {'Test RMSE':<10}")
print("-" * 60)

results = []
best_rmse = float('inf')
best_params = {}
best_history = []

start_time = time.time()

for emb in embedding_sizes:
    for lr in learning_rates:
        for reg in regularizations:
            rmse, history = train_and_evaluate(emb, lr, reg, num_epochs=10)
            
            print(f"{emb:<10} | {lr:<10} | {reg:<12} | {rmse:.4f}")
            results.append((emb, lr, reg, rmse))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {'embedding_size': emb, 'lr': lr, 'regularization': reg}
                best_history = history

elapsed_time = time.time() - start_time
print("-" * 60)
print(f"Grid Search completed. Duration: {elapsed_time:.2f} seconds.")
print(f"BEST RESULT: RMSE = {best_rmse:.4f}")
print(f"BEST PARAMETERS: {best_params}")

plt.figure(figsize=(10, 5))
plt.plot(best_history, marker='o', label=f"Best Params: {best_params}")
plt.title("Training Loss of the Best Model")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid()
plt.show()

df_results = pd.DataFrame(results, columns=['Embedding', 'Learning Rate', 'Regularization', 'RMSE'])
print(df_results.sort_values(by='RMSE'))