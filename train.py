import torch
from torch_geometric.datasets import Flickr
from torch.optim import Adam
import torch.nn.functional as F
from models import GCN3Layer

# ==== Model training example ====

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Flickr('data/Flickr')
data = dataset[0].to(device)

model = GCN3Layer(dataset.num_features).to(device)
optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(1000):
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    if epoch % 10 == 9:
        print('Epoch {}: loss {}'.format(epoch + 1, loss.item()))

model.eval()
_, pred = model(data).max(dim=1)
correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
acc = correct / data.test_mask.sum()
print('Test accuracy: {:.4f}'.format(acc))
