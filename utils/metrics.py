import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def quantile_losses(y_true, y_pred):
    quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]

    losses = []
    for i, q in enumerate(quantiles):
        errors = y_true - y_pred[..., i]
        losses.append(torch.max(( q - 1 ) * errors, q * errors).unsqueeze(-1))
    
    losses = torch.cat(losses, dim=2)

    return losses