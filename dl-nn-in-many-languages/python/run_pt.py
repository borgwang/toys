import torch
import torch.nn.functional as F

n_classes = 4
batch_size = 4
hidden_dim = 50
x = torch.randn(batch_size, 100)
y = torch.randint(0, n_classes, (batch_size,))
w1 = torch.randn(100, hidden_dim, requires_grad=True) * 0.1
b1 = torch.randn(hidden_dim, requires_grad=True) * 0.1
w2 = torch.randn(hidden_dim, n_classes, requires_grad=True) * 0.1
b2 = torch.randn(n_classes, requires_grad=True) * 0.1

# forward pass
h1 = x @ w1 + b1
h1.retain_grad()
a1 = F.relu(h1)
h2 = a1 @ w2 + b2
h2.retain_grad()

# loss calculation
#loss = F.cross_entropy(h2, y, reduction='sum')
loss = F.cross_entropy(h2, y)

# backward pass
loss.backward()
with torch.no_grad():
  # manually calculate the gradient of loss w.r.t. h2
  d_h2 = (F.softmax(h2) - F.one_hot(y, num_classes=n_classes).float()) / batch_size
  d_w2 = a1.T @ d_h2
  d_b2 = d_h2.sum(0)
  d_a1 = d_h2 @ w2.T
  d_h1 = d_a1 * (h1 > 0).float()  # Ensure this is correct

  assert h2.grad is not None
  # print("h2.grad: \n", h2.grad)
  # print("Calculated d_h2: \n", d_h2)
  assert torch.allclose(h2.grad, d_h2, atol=1e-6)  # Allow for small numerical differences

  assert h1.grad is not None
  # print("h1.grad:", h1.grad)
  # print("Calculated d_h1:", d_h1)
  assert torch.allclose(h1.grad, d_h1, atol=1e-6)  # Allow for small numerical differences
