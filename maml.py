import torch
import torch.nn as nn
from torch.nn.utils.stateless import functional_call

class MAML(nn.Module):
    def __init__(self, model, inner_lr=0.01, inner_steps=1):
        super(MAML, self).__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps

    def forward(self, support_x, support_y, query_x, query_y):
        # 复制一份模型参数作为内循环更新的起点
        fast_weights = {name: param.clone() for name, param in self.model.named_parameters()}
        # 内循环：在支持集上进行快速适应
        for _ in range(self.inner_steps):
            preds = functional_call(self.model, fast_weights, support_x)
            support_y = support_y.squeeze()
            # print(support_y.shape)
            loss = nn.CrossEntropyLoss()(preds, support_y)
            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            # 使用梯度下降更新fast_weights
            fast_weights = {name: param - self.inner_lr * grad
                            for ((name, param), grad) in zip(fast_weights.items(), grads)}
        # 查询集上计算更新后模型的损失
        query_preds = functional_call(self.model, fast_weights, query_x)
        query_y = query_y.squeeze()
        query_loss = nn.CrossEntropyLoss()(query_preds, query_y)
        return query_loss, query_preds