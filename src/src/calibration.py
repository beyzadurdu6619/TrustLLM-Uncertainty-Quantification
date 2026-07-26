import torch
import torch.nn as nn
import torch.optim as optim


class TemperatureScaler(nn.Module):
    """PyTorch module that calibrates logits using the Temperature scaling
    parameter (T).

    TR: Logitleri T parametresine bölerek kalibre eden PyTorch sınıfı.
    """

    def __init__(self):
        super(TemperatureScaler, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, valid_logits, valid_labels):
        """Finds optimal T parameter on validation set using L-BFGS optimizer.

        TR: Validation seti üzerinde en uygun T değerini optimizasyonla bulur.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(self.forward(valid_logits), valid_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        print(f"Optimal Temperature (T): {self.temperature.item():.4f}")