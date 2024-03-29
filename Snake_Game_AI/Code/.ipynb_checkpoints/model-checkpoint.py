import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_DeepQ(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size).cuda()
        self.linear2 = nn.Linear(hidden_size, output_size).cuda()
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        torch.save(self.state_dict(), file_name)

class DeepQ_Trainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimer = optim.Adam(model.parameters(), lr = self.lr)    
        self.criterion = nn.MSELoss()
        for i in self.model.parameters():
            print(i.is_cuda)
    
    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype = torch.float).cuda()
        next_state = torch.tensor(next_state, dtype = torch.float).cuda()
        action = torch.tensor(action, dtype = torch.long).cuda()
        reward = torch.tensor(reward, dtype = torch.float).cuda()
        if(len(state.shape) == 1):
            state = torch.unsqueeze(state, 0).cuda()
            next_state = torch.unsqueeze(next_state, 0).cuda()
            action = torch.unsqueeze(action, 0).cuda()
            reward = torch.unsqueeze(reward, 0).cuda()
            done = (done, )
        pred = self.model(state).cuda()
        target = pred.clone().cuda()
        for idx in range(len(done)):
            cur_Q = reward[idx]
            if not done[idx]:
                cur_Q = reward[idx] + self.gamma * torch.max(self.model(next_state[idx])).cuda()
            target[idx][torch.argmax(action).item()] = cur_Q
        self.optimer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimer.step()