import torch
from torch.optim import Adam
import torch.nn.functional as F
from models import DenseGCN3Layer
from preprocessing import load_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    time_steps = load_dataset('data/elliptic_bitcoin_dataset')
    for step in time_steps:
        time_steps[step] = time_steps[step].to(device)

    train = list(time_steps.values())[:34]
    test = list(time_steps.values())[34:]

    num_features = time_steps[1].x.shape[1]
    model = DenseGCN3Layer(num_features).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(1000):
        epoch_loss = 0
        for step in train:
            optimizer.zero_grad()
            out = model(step)
            loss = F.binary_cross_entropy(out[step.mask], step.y[step.mask])
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
        if epoch % 10 == 9:
            print('Epoch {}: loss {}'.format(epoch + 1, epoch_loss / len(train)))

    model.eval()
    corrects = 0
    sums = 0
    for step in test:
        pred = model(step).round()
        corrects += pred[step.mask].eq(step.y[step.mask]).sum().item()
        sums += step.mask.sum().item()
    acc = corrects / sums
    print('Test accuracy: {:.4f}'.format(acc))
