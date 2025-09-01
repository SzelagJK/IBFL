import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
	def __init__(self, input_size): 
		super(NeuralNetwork, self).__init__()
		self.fc1 = nn.Linear(input_size, 8)
		self.fc2 = nn.Linear(8, 16)
		self.fc3 = nn.Linear(16, 8)
		self.fc4 = nn.Linear(8, 1)
		self.sigmoid = nn.Sigmoid()
		self.dropoutMiddle = nn.Dropout(p=0.5)
		self.dropoutEnd = nn.Dropout(p=0.2)

	def forward(self, x): 
		x = self.sigmoid(self.fc1(x))
		x = self.dropoutMiddle(x)
		x = self.sigmoid(self.fc2(x))
		x = self.dropoutMiddle(x)
		x = self.sigmoid(self.fc3(x))
		x = self.dropoutEnd(x)
		return self.fc4(x)