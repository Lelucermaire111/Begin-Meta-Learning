import torch
# 训练函数增加准确率打印
def train_maml(maml, train_loader, test_loader, meta_optimizer, epochs):
    for epoch in range(epochs):
        meta_loss_sum = 0.0
        acc_sum = 0.0
        num_batches = len(train_loader)
        for support_x, support_y, query_x, query_y in train_loader:
            # print(support_x.shape)
            # print(query_x.shape)
            # print(support_y.shape)
            # print(query_y.shape)
            support_x, support_y, query_x, query_y = support_x.cuda(), support_y.cuda(), query_x.cuda(), query_y.cuda()
            loss, query_preds = maml(support_x, support_y, query_x, query_y)
            meta_loss_sum += loss.item()
            # 计算预测准确率：假设 query_y 为1D张量且与 query_preds 的 batch size 匹配
            pred_labels = torch.argmax(query_preds, dim=1)
            batch_acc = (pred_labels == query_y).float().mean().item()
            acc_sum += batch_acc

            meta_optimizer.zero_grad()
            loss.backward()
            meta_optimizer.step()
        avg_loss = meta_loss_sum / num_batches
        avg_acc = acc_sum / num_batches
        if((epoch + 1) % 10 == 0):
            print(f"Epoch {epoch+1}: Meta Loss = {avg_loss:.4f}, Test Accuracy = {avg_acc*100:.2f}%")
        if((epoch + 1) % 100 == 0):
            avg_test_acc = evaluate_maml(maml, test_loader, epochs=5)
            print("=====================================")
            print(f"Test acc: {avg_test_acc*100:.4f}%")
            print("=====================================")
        torch.save(maml.model.state_dict(), 'maml_miniimagenet.pth')
        
# 评估函数：可以根据不同的内循环步数观察模型适应后的表现
def evaluate_maml(maml, test_loader, epochs):
    maml.eval()  # 设置模型为评估模式
    # 对于不同的内循环步数，评估适应后的查询集性能
    acc_sum = 0.0
    count = 0
    for epochs in range(0, epochs):
        # 遍历测试任务，每个 batch 返回一个任务
        for support_x, support_y, query_x, query_y in test_loader:
            support_x, support_y, query_x, query_y = support_x.cuda(), support_y.cuda(), query_x.cuda(), query_y.cuda()
            loss, query_preds = maml(support_x, support_y, query_x, query_y)
            pred_labels = torch.argmax(query_preds, dim=1)
            batch_acc = (pred_labels == query_y).float().mean().item()
            acc_sum += batch_acc
            count += 1
    avg_acc = acc_sum / count
    return avg_acc
