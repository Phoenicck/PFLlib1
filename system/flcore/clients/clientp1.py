import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
from sklearn.preprocessing import label_binarize
from sklearn import metrics


class Clientp1(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        # p1
        self.p1_local_soft_labels = None  # 本地软标签
        #本地各个类别的数量
        self.p1_local_nums_per_class = None
        self.caculate_local_nums_per_class()
        print(f"Client {self.id} local nums per class: {self.p1_local_nums_per_class}")
        #print("\nClient p1 initialized.")

    def train(self,warmup=False):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)


        #预热阶段的训练
        if warmup:
            # print("warmup\n")
            for epoch in range(max_local_epochs):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    #print(y)
                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))
                    output = self.model(x)
                    # print("output before softmax:", output)
                    # print("y:", y)
                    loss = self.loss(output, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
        else:
            # print("normal train\n")
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
    

    def compute_local_soft_labels(self):
        """
        计算每个类别的本地软标签：对本地训练集每个类别，统计其 softmax 概率的均值
        """
        self.model.eval()
        trainloader = self.load_train_data()
        class_soft_labels = {}
        class_counts = {}

        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                probs = torch.softmax(outputs, dim=1)  # [batch, num_classes]
                for prob, label in zip(probs, y):
                    label = label.item()
                    if label not in class_soft_labels:
                        class_soft_labels[label] = prob.clone()
                        class_counts[label] = 1
                    else:
                        class_soft_labels[label] += prob
                        class_counts[label] += 1

        # 对每个类别求平均
        for label in class_soft_labels:
            class_soft_labels[label] /= class_counts[label]

        self.p1_local_soft_labels_per_class = class_soft_labels  # {label: mean_soft_label}
        #print(f"Client {self.id} computed local soft labels per class.")
        # for label, soft in class_soft_labels.items():
        #     print(f"Class {label}: {soft}")
        self.model.train()

    def caculate_local_nums_per_class(self):
        """
        计算本地每个类别的样本数量
        """
        trainloader = self.load_train_data()
        class_counts = {}

        for _, y in trainloader:
            for label in y:
                label = label.item()
                if label not in class_counts:
                    class_counts[label] = 1
                else:
                    class_counts[label] += 1
        self.p1_local_nums_per_class = class_counts 

    # p1
    # 根据argmax选择与对应类的软标签，计算其kl散度，太高的就是未知类 就是假已知类
    # 这样的逻辑是没有问题的，不用担心误判；原本就已经是未知类了
    # 只要不是未知类，就不会误判
    def test_metrics(self,warmup=False):
            
            testloaderfull = self.load_test_data()
            
            self.model.eval()

            test_acc = 0
            test_num = 0
            y_prob = []
            y_true = []
            
            with torch.no_grad():
                for x, y in testloaderfull:
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    # print("x type:", type(x), "x shape:", x.shape)
                    # print("y type:", type(y), "y shape:", y.shape)
                    y = y.to(self.device)
                    output = self.model(x)

                    if warmup:
                       test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                    else:
                        #print("Client p1 test metrics with unknown detection.")
                         # 需要插入计算kl散度的代码
                        kl_divs = self.compute_kl_divergence(output,y)
                        # print("kl_divs:", kl_divs)
                        # 设置一个阈值，假设是0.5，超过这个值的样本被认为是未知类
                        threshold = 0.3
                        # 将超过阈值的样本的预测类别设为一个新的类别，比如num_classes
                        preds = torch.argmax(output, dim=1)
                        preds[kl_divs > threshold] = 6  # 假设未知

                        test_acc += (torch.sum(preds == y)).item()
                    test_num += y.shape[0]
    
                    y_prob.append(output.detach().cpu().numpy())
                    nc = self.num_classes
                    if self.num_classes == 2:
                        nc += 1
                    lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                    if self.num_classes == 2:
                        lb = lb[:, :2]
                    y_true.append(lb)
                    

            # self.model.cpu()
            # self.save_model(self.model, 'model')

            y_prob = np.concatenate(y_prob, axis=0)
            y_true = np.concatenate(y_true, axis=0)

            auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
            if warmup:
                return test_acc, test_num, auc
            else:
                unknown_test_status=self.unknown_test()
                return test_acc, test_num, auc, unknown_test_status

    # 根据y的argmax去计算kl散度
    def compute_kl_divergence(self, outputs, y):
        """
        计算输出与对应类的软标签之间的 KL 散度
        outputs: 模型的原始输出 logits，形状为 [batch_size, num_classes] 
        返回：每个样本的 KL 散度，形状为 [batch_size]
        """
        probs = torch.softmax(outputs, dim=1)  # 转换为概率分布
        kl_divs = []

        for prob ,yy in zip(probs,y):
            label = torch.argmax(prob).item()
            
            if (self.p1_local_soft_labels_per_class is not None and 
                label in self.p1_local_soft_labels_per_class):
                soft_label = self.p1_local_soft_labels_per_class[label].to(self.device)
                kl_div = torch.sum(soft_label * (torch.log(soft_label + 1e-10) - torch.log(prob + 1e-10)))
                kl_divs.append(kl_div.item())
            else:
                # 如果该类别没有软标签，设定一个较高的 KL 散度值，表示不确定
                kl_divs.append(float('inf'))
            #print("Predicted label:", label, "true y", yy, "KL Divergence:", kl_div)
        

        return torch.tensor(kl_divs)


    def unknown_test(self):
        testloader = self.load_test_data()
        self.model.eval()
        
        known_correct = 0
        known_total = 0
        unk_correct = 0
        unk_total = 0
        all_labels = []
        all_probs = []

        kl_threshold = 0.5  # KL散度判未知的阈值

        with torch.no_grad():
            for i, (x, y) in enumerate(testloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)

                # 计算KL散度
                kl_divs = self.compute_kl_divergence(outputs, y)
                preds = torch.argmax(outputs, dim=1)
                preds[kl_divs > kl_threshold] = 6  # KL大于阈值的直接判为未知

                # 区分已知类和未知类
                known_mask = (y >= 0) & (y < 6)
                unknown_mask = (y == 6)

                # 已知类预测
                known_correct += ((preds == y) & known_mask).sum().item()
                known_total += known_mask.sum().item()

                # 未知类检测
                unk_correct += ((preds == 6) & unknown_mask).sum().item()
                unk_total += unknown_mask.sum().item()

                all_labels.extend(y.cpu().numpy())
                all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

        # 计算OS*和UNK
        os_star = 100.0 * known_correct / known_total if known_total > 0 else 0.0
        unk_acc = 100.0 * unk_correct / unk_total if unk_total > 0 else 0.0
        hos = 2 * (os_star * unk_acc) / (os_star + unk_acc + 1e-8) if (os_star + unk_acc) > 0 else 0.0

        return known_correct, known_total, unk_correct, unk_total, os_star, unk_acc, hos
    
