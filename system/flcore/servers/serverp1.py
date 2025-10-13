import time
import numpy as np
import os
import torch
import copy
import logging
import sys
from flcore.clients.clientp1 import Clientp1
from flcore.servers.serverbase import Server
from threading import Thread


class Fedp1(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(Clientp1)
        print("\n-------------FedP1 Algorithm-------------")
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        # p1
        self.p1_best_acc = 0
        self.p1_warmup=False # 是否预热
        self.p1_global_soft_labels = None  # 全局软标签
        self.p1_warmup_rounds =1 # 预热轮数，可调


    def train(self):
        # 配置 logging
        

        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            #预热客户端
            if i < self.p1_warmup_rounds:
                print(f"\n-------------Warmup Round number: {i}-------------")
                self.p1_warmup=True # 预热阶段

                for client in self.selected_clients:
                    #print(f"\n-------------Warmup Round number: {i} Client {client.id}-------------")
                    client.train(self.p1_warmup) # 预热阶段训练
                
                # 预热阶段仅用客户端自己的数据进行测试（看能不能开始加软标签）
                self.evaluate1()
                # 预热结束后，聚合模型
                if i == self.p1_warmup_rounds - 1:
                    self.p1_warmup=False # 结束预热阶段

                    self.receive_models()
                    if self.dlg_eval and i%self.dlg_gap == 0:
                        self.call_dlg(i)
                    self.aggregate_parameters() # 暂时按照FedAvg聚合
                    # 计算全局软标签
                    self.compute_global_soft_labels()
                    # 分发全局软标签
                    # self.distribute_global_soft_labels()
                    print(f"\n-------------Warmup Finished. Aggregate global model.-------------") 
                # 计算时间 
                self.Budget.append(time.time() - s_t)
                print('-'*25, 'time cost', '-'*25, self.Budget[-1])
            
            # 预热结束 继续训练   
            else:
                self.send_models() 
                self.distribute_global_soft_labels()
                if i%self.eval_gap == 0:
                    print(f"\n-------------Round number: {i}-------------")
                    print("\nEvaluate global model")
                    self.evaluate(epoch=i)
                # threads = [Thread(target=client.train)
                #            for client in self.selected_clients]
                # [t.start() for t in threads]
                # [t.join() for t in threads]
                for client in self.selected_clients:
                    client.train()

                self.receive_models()
                if self.dlg_eval and i%self.dlg_gap == 0:
                    self.call_dlg(i)
                self.aggregate_parameters()
                # 计算全局软标签
                self.compute_global_soft_labels()

                self.Budget.append(time.time() - s_t)
                print('-'*25, 'time cost', '-'*25, self.Budget[-1])

                if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                    break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(Clientp1)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

     # 1012 added
    # 正常的测试代码 
    def evaluate(self, acc=None, loss=None,epoch=None):
        
        # 多了个epoch参数，用于保存当前最好的模型
        stats = self.test_metrics()
        stats_train = self.train_metrics()
        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        # 检查是否传入了acc和loss参数
        #print(acc, loss)
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            print("loss is not None")  # 调试信息
            loss.append(train_loss)

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            print("acc is not None")  # 调试信息
            acc.append(test_acc)
        # 各个客户端的准确率和AUC
        for idx, (acc, auc, n) in enumerate(zip(stats[2], stats[3], stats[1])):
            print(f"Client{idx} test acc: {acc/n:.4f} test auc: {auc/n:.4f}")
        
        # 平均
        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))
        if epoch is not None:
            if test_acc > self.p1_best_acc:
                self.p1_best_acc = test_acc
                self.save_global_model(epoch)
                # 输出当前最好的准确率
                print("Current Best Test Accuracy: {:.4f}".format(self.p1_best_acc))

    #预热阶段的测试代码
    def evaluate1(self):
        stats = self.test_metrics()
        stats_train = self.train_metrics()
        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        # 各个客户端的准确率和AUC
        for idx, (acc, auc, n) in enumerate(zip(stats[2], stats[3], stats[1])):
            print(f"Client{idx} test acc: {acc/n:.4f} test auc: {auc/n:.4f}")
        # 平均
        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

    def save_global_model(self, epoch=None):
        # 多了个epoch参数，用于保存当前最好的模型
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if epoch is None:
            model_file = self.algorithm + "_server.pt"
        else:
            model_file = f"{self.algorithm}_server_best_epoch{epoch}.pt"
        model_path = os.path.join(model_path, model_file)
        torch.save(self.global_model, model_path)
    
    def test(self,epoch=None):
        # 增加测试功能
        print("\nTest Evaluate global model")
        self.load_model(epoch)
        s_t = time.time()
        self.selected_clients = self.select_clients()
        self.send_models()
        self.evaluate()
        print('-'*25, 'time cost', '-'*25, time.time() - s_t)
    
    def load_model(self,epoch=None):
        # 个性化加载模型
        if epoch is None:
            model_path = os.path.join("models", self.dataset)
            model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        else:
            model_path = os.path.join("models", self.dataset)
            model_file = f"{self.algorithm}_server_best_epoch{epoch}.pt"
            model_path = os.path.join(model_path, model_file)

        print(f"Load model from {model_path}")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    # 计算全局软标签
    def compute_global_soft_labels(self):
        """
        计算全局软标签：对所有客户端的本地软标签取平均
        """
        print("\nCompute global soft labels.")
        all_soft_labels = []
        for client in self.selected_clients:
            client.compute_local_soft_labels()
            if client.p1_local_soft_labels is None:
                print(f"Client {client.id} has no local soft labels. Skipping.")
                continue
            all_soft_labels.append(client.p1_local_soft_labels)
        
        if len(all_soft_labels) == 0:
            print("No local soft labels collected from clients.")
            return
        
        # 拼接所有客户端的 soft label
        all_soft_labels_cat = torch.cat(all_soft_labels, dim=0)
        self.p1_global_soft_labels = torch.mean(all_soft_labels_cat, dim=0)
        print("Computed global soft labels.")
        print(self.p1_global_soft_labels)
    # 分发全局软标签
    def distribute_global_soft_labels(self):
        """
        分发全局软标签：将全局软标签发送给所有客户端
        """
        print("\nDistribute global soft labels to clients.")
        if self.p1_global_soft_labels is None:
            print("Global soft labels have not been computed yet.")
            return
        for client in self.selected_clients:
            client.p1_global_soft_labels = copy.deepcopy(self.p1_global_soft_labels)
        print("Distributed global soft labels to clients.")
