import torch

class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=-1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
    def forward(self, inputs, targets, mask=None):
        inputs = inputs.narrow(1, 0, targets.shape[1])
        outputs = torch.log_softmax(inputs, dim=1)
        labels = torch.softmax(targets * self.alpha, dim=1)
        loss = (outputs * labels).mean(dim=1)
        if mask is not None:
            loss = loss * mask.float()
        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss
        return outputs


a = torch.rand(32, 65)
KD_loss = KnowledgeDistillationLoss()

print(KD_loss(a,a))