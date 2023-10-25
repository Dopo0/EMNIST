import streamlit as st
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import pickle
import numpy as np

def main():
    st.title("EMNIST Classification")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1,), (0.3,))
    ])

    train_set = datasets.EMNIST('Data_emnist/', split='letters', download=True, train=True, transform=transform)
    trainLoader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    test_set = datasets.EMNIST('DATA_EMNIST/', split='letters', download=True, train=False, transform=transform)
    testLoader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

    training_data = enumerate(trainLoader)
    batch_idx, (images, labels) = next(training_data)
    print(type(images))
    print(images.shape)
    print(labels.shape)
    print(labels.unique())

    label_image_dict = get_random_image_for_each_label(trainLoader)

    fig = plt.figure(figsize=(13, 3))  # Adjusting the figure size for better visualization
    for i, label in enumerate(sorted(label_image_dict.keys())):
        image = label_image_dict[label]
        plt.subplot(2, 13, i + 1)
        plt.imshow(image[0].cpu().numpy(), cmap='inferno')
        plt.title(label)
        plt.yticks([])
        plt.xticks([])

    st.pyplot(fig)

    return trainLoader,testLoader


def get_random_image_for_each_label(loader):
    label_image_dict = {}

    # Traverse the dataset
    for images, labels in loader:
        for i, label in enumerate(labels):
            if label.item() not in label_image_dict:
                label_image_dict[label.item()] = images[i]
            if len(label_image_dict) == 26:
                break
        if len(label_image_dict) == 26:
            break

    return label_image_dict


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.convolutional_neural_network_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=24, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),  # Adding dropout
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(48 * 7 * 7, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=64, out_features=26)
        )
    def forward(self, x):
        x = self.convolutional_neural_network_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        x = F.log_softmax(x, dim=1)
        return x

def train(trainLoader,testLoader,epochs = 5):

    model = Network()
    model.to(device)
    print(model)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()


    train_losses = []
    test_losses = []
    accuracys = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for idx, (images, labels) in enumerate(trainLoader):

            images = images.to(device)
            labels = labels.to(device) - 1

            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        else:
            model.eval()
            test_loss = 0
            accuracy = 0

            with torch.no_grad():
                for images, labels in testLoader:
                    images = images.to(device)
                    labels = labels.to(device) - 1

                    log_probabilities = model(images)
                    test_loss += criterion(log_probabilities, labels)

                    probabilities = torch.exp(log_probabilities)
                    top_prob, top_class = probabilities.topk(1, dim=1)
                    predictions = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(predictions.type(torch.FloatTensor))

            train_losses.append(train_loss / len(trainLoader))
            test_losses.append(test_loss.cpu().numpy() / len(testLoader))
            accuracys.append(accuracy / len(testLoader))
            st.write("Epoch: {}/{}  ".format(epoch + 1, epoch),
                  "Training loss: {:.4f}  ".format(train_loss / len(trainLoader)),
                  "Testing loss: {:.4f}  ".format(test_loss / len(testLoader)),
                  "Test accuracy: {:.4f}  ".format(accuracy / len(testLoader)))
    # save the model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    model_stats = {"Training loss":train_losses,
                   "Testing loss":test_losses,
                   "accuracy":accuracys}
    with open('model_stats.pkl', 'wb') as f:
        pickle.dump(model_stats,f)

    fig = plt.figure(figsize=(5, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(test_losses, label="Testing Loss")
    plt.legend()
    plt.grid()
    st.pyplot(fig)

def test(trainLoader,device):
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model_stats.pkl', 'rb') as f:
        model_stats = pickle.load(f)
    st.subheader("Actual model")
    st.write(model)
    st.write(model_stats)
    training_data = enumerate(trainLoader)
    batch_idx, (images, labels) = next(training_data)

    img = images[2]
    img = img.to(device)
    img = img.view(-1, 1, 28, 28)
    with torch.no_grad():
        logits = model.forward(img)

    probabilities = F.softmax(logits, dim=1).detach().cpu().numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 8), ncols=2)
    ax1.imshow(img.view(1, 28, 28).detach().cpu().numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(1,27), probabilities, color='b')
    ax2.set_yticks(np.arange(1,27))
    ax2.set_title('Model Prediction')
    st.pyplot(fig)

if __name__=="__main__":
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    trainLoader,testLoader = main()
    col1, col2 = st.columns(2)
    with col1:
        epochs = st.sidebar.slider("Epochs",1,50,10)
        btn = st.sidebar.button("Train again")
        if btn:
            train(trainLoader,testLoader,epochs)
    with col2:
        btn_test = st.button("Random Test")

    if btn_test:
        test(trainLoader,device)
