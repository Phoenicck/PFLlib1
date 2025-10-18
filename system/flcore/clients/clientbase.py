import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.few_shot = args.few_shot

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay


    # def load_train_data(self, batch_size=None):
    #     if batch_size == None:
    #         batch_size = self.batch_size
    #     train_data = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
    #     return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False, few_shot=self.few_shot)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
        
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        # fedavg

        unknown_test_status=self.unknown_test()
        #####
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
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

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
 
                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                # print("测试auc的计算")
                # print("self.num_classes:", nc)
                # print("y:", y)

                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                # print("lb:", lb)
                # print("output:", output)
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc, unknown_test_status

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y


    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
    
    #changed in p1
    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def unknown_test(self):
        testloader = self.load_test_data()
        self.model.eval()
        
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
        
        # for auc calculation
        from sklearn.metrics import roc_auc_score, average_precision_score

        # 将 all_labels 转为 numpy
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Step 1: 二分类标签：未知=1, 已知=0
        is_unknown = (all_labels == 6).astype(int)

        # Step 2: 构造未知得分
        # 方法A: 直接取第7类概率
        p_unknown = all_probs[:, 6]

        # 方法B: 用 1 - max(已知类置信度)
        p_max_known = np.max(all_probs[:, :6], axis=1)
        score_1_minus_max = 1.0 - p_max_known

        # Step 3: 计算 AUROC / AUPR
        auroc_a = roc_auc_score(is_unknown, p_unknown)
        aupr_a = average_precision_score(is_unknown, p_unknown)

        auroc_b = roc_auc_score(is_unknown, score_1_minus_max)
        aupr_b = average_precision_score(is_unknown, score_1_minus_max)
        print("Client {} unknown test AUC results:".format(self.id))
        print(f"[p_unknown] AUROC={auroc_a:.4f}, AUPR={aupr_a:.4f}")
        print(f"[1-max_known] AUROC={auroc_b:.4f}, AUPR={aupr_b:.4f}")
        
        return known_correct, known_total, unk_correct, unk_total, os_star, unk_acc, hos
    