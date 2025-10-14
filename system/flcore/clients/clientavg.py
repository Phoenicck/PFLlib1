import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
from sklearn.preprocessing import label_binarize
from sklearn import metrics

class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
    
    # 给出一个函数可以计算OS*，UNK，HOS的函数
    # def unknown_test(self):
    #     testloader = self.load_test_data()
    #     self.model.eval()
    #     test_loss = 0.0
    #     correct = 0
    #     total = 0
    #     all_labels = []
    #     all_probs = []

    #     with torch.no_grad():
    #         for i, (x, y) in enumerate(testloader):
    #             if type(x) == type([]):
    #                 x[0] = x[0].to(self.device)
    #             else:
    #                 x = x.to(self.device)
    #             y = y.to(self.device)
    #             outputs = self.model(x)
    #             loss = self.loss(outputs, y)
    #             test_loss += loss.item() * y.size(0)
    #             _, predicted = torch.max(outputs.data, 1)
    #             total += y.size(0)
    #             correct += (predicted == y).sum().item()

    #             all_labels.extend(y.cpu().numpy())
    #             all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    #     test_acc = 100.0 * correct / total
    #     test_loss = test_loss / total

    #     # 计算AUC
    #     try:
    #         all_labels_np = np.array(all_labels)
    #         all_probs_np = np.array(all_probs)
    #         if self.num_classes == 2:
    #             # 二分类问题
    #             auc = metrics.roc_auc_score(all_labels_np, all_probs_np[:, 1])
    #         else:
    #             # 多分类问题，使用one-vs-rest方法
    #             all_labels_binarized = label_binarize(all_labels_np, classes=list(range(self.num_classes)))
    #             auc = metrics.roc_auc_score(all_labels_binarized, all_probs_np, average='macro', multi_class='ovr')
    #     except Exception as e:
    #         print(f"Error in AUC calculation: {e}")
    #         auc = float('nan')

    #     return test_acc, test_loss, auc

    def unknown_test(self):
        testloader = self.load_test_data()
        self.model.eval()
        test_loss = 0.0
        known_correct = 0
        known_total = 0
        unk_correct = 0
        unk_total = 0
        all_labels = []
        all_probs = []

        prob_threshold = 0.5  # 未知类检测阈值，可调

        with torch.no_grad():
            for i, (x, y) in enumerate(testloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                loss = self.loss(outputs, y)
                test_loss += loss.item() * y.size(0)

                # 区分已知类和未知类
                known_mask = (y >= 0) & (y < 6)
                unknown_mask = (y == 6)  # 假设未知类标签为-1

                # 已知类预测
                _, predicted = torch.max(outputs.data, 1)
                known_correct += ((predicted == y) & known_mask).sum().item()
                known_total += known_mask.sum().item()

                # 未知类检测
                max_probs, _ = torch.max(torch.softmax(outputs, dim=1), dim=1)
                unknown_pred = max_probs < prob_threshold
                unk_correct += (unknown_pred & unknown_mask).sum().item()
                unk_total += unknown_mask.sum().item()

                all_labels.extend(y.cpu().numpy())
                all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

        # 计算OS*和UNK
        os_star = 100.0 * known_correct / known_total if known_total > 0 else 0.0
        unk_acc = 100.0 * unk_correct / unk_total if unk_total > 0 else 0.0
        hos = 2 * (os_star * unk_acc) / (os_star + unk_acc + 1e-8) if (os_star + unk_acc) > 0 else 0.0
        test_loss = test_loss / (known_total + unk_total)
        print(f"Known total: {known_total}, Known correct: {known_correct}, OS*: {os_star:.2f}%")
        print(f"Unknown total: {unk_total}, Unknown correct: {unk_correct}, UNK: {unk_acc:.2f}%")
        print(f"HOS: {hos:.2f}%")
        # 计算AUC（仅已知类）
        try:
            all_labels_np = np.array(all_labels)
            all_probs_np = np.array(all_probs)
            known_indices = np.where((all_labels_np >= 0) & (all_labels_np < self.num_classes))[0]
            known_labels = all_labels_np[known_indices]
            known_probs = all_probs_np[known_indices]
            if len(np.unique(known_labels)) > 1:
                auc = metrics.roc_auc_score(
                    label_binarize(known_labels, classes=list(range(self.num_classes))),
                    known_probs,
                    average='macro', multi_class='ovr'
                )
            else:
                auc = float('nan')
        except Exception as e:
            print(f"Error in AUC calculation: {e}")
            auc = float('nan')
