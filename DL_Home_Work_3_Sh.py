from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='./runs')

train_indices = torch.arange(5000)

train_cifar100_dataset = datasets.CIFAR100(download=True, root='./', transform=
    transforms.ToTensor(), train=True
)
train_cifar100_dataset = data_utils.Subset(train_cifar100_dataset, train_indices)

test_indices = torch.arange(1500)
test_cifar100_dataset = datasets.CIFAR100(download=True, root='./', transform=
    transforms.ToTensor(), train=False
)
test_cifar100_dataset = data_utils.Subset(test_cifar100_dataset, test_indices)

train_cifar100_dataloader = DataLoader(dataset=train_cifar100_dataset, batch_size=1, shuffle=True)
test_cifar100_dataloader = DataLoader(dataset=test_cifar100_dataset, batch_size=1, shuffle=True)

class CIFAR100PredictorPerceptron(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fully_connected_layer = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.out_layer = torch.nn.Linear(hidden_size, output_size, bias=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fully_connected_layer(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out_layer(x)
        x = self.sigmoid(x)

        return x

model = CIFAR100PredictorPerceptron(input_size=3072, hidden_size=180, output_size=100)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()
bce = torch.nn.BCELoss()

num_epochs = 20

for epoch in range(num_epochs):
    error = 0
    correct_guess = 0

    for x, y in train_cifar100_dataloader:
        model.train()
        optimizer.zero_grad()
        prediction = model(x)
        zero_tensor = torch.zeros_like(prediction)
        loss = loss_fn(prediction, y)
        error += loss

        loss.backward()
        optimizer.step()

        predicted_indices = torch.argmax(prediction, dim=1)
        correct_guess += (predicted_indices == y).sum()

    writer.add_scalar('Training Accuracy', correct_guess.float()/len(train_indices) , epoch)
    writer.add_scalar('Training Loss', error/len(train_indices), epoch)

    correct_guess = 0
    for x, y in test_cifar100_dataloader:
        model.eval()
        prediction = model(x)
        zero_tensor = torch.zeros_like(prediction)
        loss = loss_fn(prediction, y)
        error += loss

        predicted_indices = torch.argmax(prediction, dim=1)
        correct_guess += (predicted_indices == y).sum()

    writer.add_scalar('Testing Loss', error/len(test_indices), epoch)
    writer.add_scalar('Testing Accuracy', correct_guess.float()/len(test_indices), epoch)
    
    print(f"Epoch {epoch} completed")

print("Finished")

writer.close()