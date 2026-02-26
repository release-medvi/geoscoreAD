import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyMLPFewShot(nn.Module):
    """
    Minimal MLP for token-level few-shot anomaly classification.
    Input:  token features [N, D]
    Output: anomaly logit  [N]
    """

    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_few_shot_classifier(
    X_train,
    y_train,
    device="cuda",
    lr=1e-2,
    epochs=300,
    lambda_dice=0.5,
):
    """
    Pure few-shot training:
    - No validation
    - No early stopping
    - Return last epoch model
    """

    X_train = X_train.to(device).float()
    y_train = y_train.to(device).float()

    model = TinyMLPFewShot(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

    # class imbalance handling
    num_pos = (y_train == 1).sum().float()
    num_neg = (y_train == 0).sum().float()
    pos_weight = num_neg / (num_pos + 1e-8)

    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(epochs):
        model.train()

        logits = model(X_train)
        probs = torch.sigmoid(logits)

        loss_bce = bce(logits, y_train)
        loss_dice = soft_dice_loss(probs, y_train)
        loss = loss_bce + lambda_dice * loss_dice

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def soft_dice_loss(probs, targets, eps=1e-6):
    """
    probs: sigmoid outputs, shape [N]
    targets: {0,1}, shape [N]
    """
    intersection = (probs * targets).sum()
    union = probs.sum() + targets.sum()
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice




