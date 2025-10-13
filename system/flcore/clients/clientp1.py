import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client


class Clientp1(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        # p1
        self.p1_local_soft_labels = None  # 本地软标签
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
                    print("output before softmax:", output)
                    print("y:", y)
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
    
    # 计算本地软标签
    def compute_local_soft_labels(self):
        """
        计算本地软标签：对本地训练集所有样本，记录模型输出的 softmax 概率
        """
        self.model.eval()
        trainloader = self.load_train_data()
        soft_labels = []
        with torch.no_grad():
            for x, _ in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                outputs = self.model(x)
                probs = torch.softmax(outputs, dim=1)
                soft_labels.append(probs.cpu())
        # 拼接所有 batch 的 soft label
        self.p1_local_soft_labels = torch.cat(soft_labels, dim=0)
        print(f"Client {self.id} computed local soft labels.")
        print(torch.mean(self.p1_local_soft_labels, dim=0))
        self.model.train()

