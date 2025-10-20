import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.mixture import BayesianGaussianMixture
import matplotlib.pyplot as plt

class Clientp1(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        # p1
        self.p1_local_soft_labels = None  # 本地软标签
        #本地各个类别的数量
        self.p1_local_nums_per_class = None
        self.kl_threshold=args.kl_threshold  # KL散度判未知的阈值
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

        self.p1_local_soft_labels = class_soft_labels  # {label: mean_soft_label}
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
                       
                        # 设置一个阈值，假设是0.5，超过这个值的样本被认为是未知类
                        
                        preds = torch.argmax(output, dim=1)
                        preds[kl_divs > self.kl_threshold] = 6  # 假设未知

                        test_acc += (torch.sum(preds == y)).item()
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
                    #lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
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

    # def test_metrics(self, warmup=False):
    #     testloaderfull = self.load_test_data()
    #     self.model.eval()

    #     test_acc = 0
    #     test_num = 0
    #     y_prob = []
    #     y_true = []
    #     all_kl = []
    #     all_preds = []
    #     all_labels = []

    #     with torch.no_grad():
    #         for x, y in testloaderfull:
    #             x = x.to(self.device) if not isinstance(x, list) else x[0].to(self.device)
    #             y = y.to(self.device)
    #             output = self.model(x)

    #             if warmup:
    #                 test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
    #             else:
    #                 # ---------- 替换固定阈值为 DPGMM 聚类 ----------
    #                 kl_divs = self.compute_kl_divergence(output, y).detach().cpu().numpy()
    #                 all_kl.append(kl_divs)

    #                 preds = torch.argmax(output, dim=1).detach().cpu().numpy()
    #                 all_preds.append(preds)
    #                 all_labels.append(y.detach().cpu().numpy())

    #             test_num += y.shape[0]
    #             y_prob.append(output.detach().cpu().numpy())

    #     # ---------- warmup 阶段不改动 ----------
    #     if warmup:
    #         y_prob = np.concatenate(y_prob, axis=0)
    #         y_true = np.concatenate(y_true, axis=0)
    #         auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
    #         return test_acc, test_num, auc

    #     # ---------- 用 DPGMM 自动区分已知/未知 ----------
    #     all_kl = np.concatenate(all_kl)
    #     all_preds = np.concatenate(all_preds)
    #     all_labels = np.concatenate(all_labels)

    #     # 使用变分贝叶斯高斯混合模型（DPGMM 近似）
    #     kl_input = all_kl.reshape(-1, 1)
    #     dpgmm = BayesianGaussianMixture(
    #         n_components=5,
    #         covariance_type='full',
    #         weight_concentration_prior_type='dirichlet_process',
    #         weight_concentration_prior=1e-2,
    #         max_iter=500,
    #         random_state=42
    #     )
    #     dpgmm.fit(kl_input)
    #     cluster_labels = dpgmm.predict(kl_input)

    #     # 选均值最小的簇为“已知类”，其余为“未知类”
    #     means = dpgmm.means_.flatten()
    #     known_cluster = np.argmin(means)
    #     # Q = self.dpgmm_cluster(all_kl, show_plot=True)
    #     Q = (cluster_labels == known_cluster).astype(int)  # 1=known, 0=unknown

    #     # 替换未知类预测标签
    #     all_preds[Q == 0] = 6  # 你的真实未知类label也是6，不变

    #     # ---------- 保留原有 AUC 计算逻辑 ----------
    #     test_acc = np.sum(all_preds == all_labels)
    #     y_prob = np.concatenate(y_prob, axis=0)
    #     y_true = label_binarize(all_labels, classes=np.arange(self.num_classes))
    #     auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

    #     unknown_test_status = self.unknown_test()
    #     return test_acc, test_num, auc, unknown_test_status


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
            
            if (self.p1_local_soft_labels is not None and 
                label in self.p1_local_soft_labels):
                soft_label = self.p1_local_soft_labels[label].to(self.device)
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
        all_kl = []

        # kl_threshold = 0.4 # KL散度判未知的阈值

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
                preds[kl_divs > self.kl_threshold] = 6  # KL大于阈值的直接判为未知

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
                all_kl.extend(kl_divs.cpu().numpy())

        # 计算OS*和UNK
        os_star = 100.0 * known_correct / known_total if known_total > 0 else 0.0
        unk_acc = 100.0 * unk_correct / unk_total if unk_total > 0 else 0.0
        hos = 2 * (os_star * unk_acc) / (os_star + unk_acc + 1e-8) if (os_star + unk_acc) > 0 else 0.0

        # ---- AUROC & AUPR 计算 ----
        all_labels = np.array(all_labels)
        all_kl = np.array(all_kl)

        # 未知类为1，已知类为0
        is_unknown = (all_labels == 6).astype(int)

        # KL越大越可能未知
        try:
            auroc_kl = roc_auc_score(is_unknown, all_kl)
            aupr_kl = average_precision_score(is_unknown, all_kl)
        except ValueError:
            print('Only one class present in y_true. AUROC and AUPR are undefined.')
            auroc_kl, aupr_kl = np.nan, np.nan
        print("-------------------------------------------")
        print(f"[KL-divergence] AUROC: {auroc_kl:.4f}, AUPR: {aupr_kl:.4f}")
        print("===========================================\n")

        return known_correct, known_total, unk_correct, unk_total, os_star, unk_acc, hos
    
    # def unknown_test(self):
    #     testloader = self.load_test_data()
    #     self.model.eval()
        
    #     known_correct = 0
    #     known_total = 0
    #     unk_correct = 0
    #     unk_total = 0
    #     all_labels = []
    #     all_probs = []
    #     all_kl = []

    #     with torch.no_grad():
    #         for i, (x, y) in enumerate(testloader):
    #             if isinstance(x, list):
    #                 x = x[0].to(self.device)
    #             else:
    #                 x = x.to(self.device)
    #             y = y.to(self.device)
    #             outputs = self.model(x)

    #             # 计算 KL 散度
    #             kl_divs = self.compute_kl_divergence(outputs, y)
    #             all_kl.extend(kl_divs.cpu().numpy())
    #             all_labels.extend(y.cpu().numpy())
    #             all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    #     # -------------------- 用 DPGMM 判定未知 --------------------
    #     all_kl = np.array(all_kl)
    #     all_labels = np.array(all_labels)
    #     kl_input = all_kl.reshape(-1, 1)

    #     dpgmm = BayesianGaussianMixture(
    #         n_components=2,
    #         covariance_type='full',
    #         weight_concentration_prior_type='dirichlet_process',
    #         weight_concentration_prior=1e-2,
    #         max_iter=500,
    #         random_state=42
    #     )
    #     dpgmm.fit(kl_input)
    #     cluster_labels = dpgmm.predict(kl_input)

    #     # 选 KL 均值最小的簇为已知，其他为未知
    #     means = dpgmm.means_.flatten()
    #     known_cluster = np.argmin(means)
    #     Q = (cluster_labels == known_cluster).astype(int)  # 1=known, 0=unknown

    #     # -------------------- 统计结果 --------------------
    #     preds_unknown = np.zeros_like(all_labels)
    #     preds_unknown[Q == 0] = 6  # 预测为未知类
    #     preds_unknown[Q == 1] = -1 # 非未知类，仅标记方便统计

    #     known_mask = (all_labels >= 0) & (all_labels < 6)
    #     unknown_mask = (all_labels == 6)

    #     # 已知类预测（只统计 DPGMM 判为已知的）
    #     known_correct = np.sum((Q == 1) & known_mask)
    #     known_total = np.sum(known_mask)

    #     # 未知类检测
    #     unk_correct = np.sum((Q == 0) & unknown_mask)
    #     unk_total = np.sum(unknown_mask)

    #     os_star = 100.0 * known_correct / known_total if known_total > 0 else 0.0
    #     unk_acc = 100.0 * unk_correct / unk_total if unk_total > 0 else 0.0
    #     hos = 2 * (os_star * unk_acc) / (os_star + unk_acc + 1e-8) if (os_star + unk_acc) > 0 else 0.0

    #     # ---- AUROC & AUPR 计算 ----
    #     is_unknown = (all_labels == 6).astype(int)
    #     try:
    #         auroc_kl = roc_auc_score(is_unknown, all_kl)
    #         aupr_kl = average_precision_score(is_unknown, all_kl)
    #     except ValueError:
    #         print('Only one class present in y_true. AUROC and AUPR are undefined.')
    #         auroc_kl, aupr_kl = np.nan, np.nan

        

    #     return known_correct, known_total, unk_correct, unk_total, os_star, unk_acc, hos
    # def unknown_test(self):
    #     testloader = self.load_test_data()
    #     self.model.eval()
        
    #     all_labels = []
    #     all_probs = []
    #     all_preds = []
    #     all_kl = []

    #     with torch.no_grad():
    #         for i, (x, y) in enumerate(testloader):
    #             if isinstance(x, list):
    #                 x = x[0].to(self.device)
    #             else:
    #                 x = x.to(self.device)
    #             y = y.to(self.device)
    #             outputs = self.model(x)

    #             # KL 散度
    #             kl_divs = self.compute_kl_divergence(outputs, y)
    #             all_kl.extend(kl_divs.cpu().numpy())
    #             all_labels.extend(y.cpu().numpy())
    #             all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
    #             all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    #     # -------------------- DPGMM 判定未知 --------------------
    #     all_kl = np.array(all_kl)
    #     all_labels = np.array(all_labels)
    #     all_preds = np.array(all_preds)
    #     kl_input = all_kl.reshape(-1, 1)

    #     dpgmm = BayesianGaussianMixture(
    #         n_components=5,
    #         covariance_type='full',
    #         weight_concentration_prior_type='dirichlet_process',
    #         weight_concentration_prior=1e-2,
    #         max_iter=500,
    #         random_state=42
    #     )
    #     dpgmm.fit(kl_input)
    #     cluster_labels = dpgmm.predict(kl_input)

    #     # KL 均值最小的簇为已知，其余为未知
    #     means = dpgmm.means_.flatten()
    #     known_cluster = np.argmin(means)
    #     Q = (cluster_labels == known_cluster).astype(int)  # 1=known, 0=unknown

    #     # 将 DPGMM 判未知的样本标记为 6
    #     preds_dpgmm = all_preds.copy()
    #     preds_dpgmm[Q == 0] = 6

    #     # -------------------- 统计指标 --------------------
    #     known_mask = (all_labels >= 0) & (all_labels < 6)
    #     unknown_mask = (all_labels == 6)

    #     # 已知类预测
    #     known_correct = np.sum((preds_dpgmm == all_labels) & known_mask)
    #     known_total = np.sum(known_mask)

    #     # 未知类检测
    #     unk_correct = np.sum((preds_dpgmm == 6) & unknown_mask)
    #     unk_total = np.sum(unknown_mask)

    #     # HOS 指标
    #     os_star = 100.0 * known_correct / known_total if known_total > 0 else 0.0
    #     unk_acc = 100.0 * unk_correct / unk_total if unk_total > 0 else 0.0
    #     hos = 2 * (os_star * unk_acc) / (os_star + unk_acc + 1e-8) if (os_star + unk_acc) > 0 else 0.0

    #     # -------------------- 每个已知类准确率 --------------------
    #     per_class_acc = {}
    #     for c in range(6):
    #         class_mask = (all_labels == c) & (Q == 1)  # 仅统计被判为已知的样本
    #         correct = np.sum(preds_dpgmm[class_mask] == all_labels[class_mask])
    #         total = np.sum(class_mask)
    #         per_class_acc[c] = 100.0 * correct / total if total > 0 else 0.0

    #     # -------------------- AUROC & AUPR（KL） --------------------
    #     is_unknown = (all_labels == 6).astype(int)
    #     try:
    #         auroc_kl = roc_auc_score(is_unknown, all_kl)
    #         aupr_kl = average_precision_score(is_unknown, all_kl)
    #     except ValueError:
    #         auroc_kl, aupr_kl = np.nan, np.nan

    #     # -------------------- 打印信息 --------------------
    #     print("-------------------------------------------")
    #     print(f"[DPGMM] KL means: {means}, known_cluster={known_cluster}")
    #     print(f"[DPGMM-based] AUROC: {auroc_kl:.4f}, AUPR: {aupr_kl:.4f}")
    #     print(f"OS*: {os_star:.2f}, UNK: {unk_acc:.2f}, HOS: {hos:.2f}")
    #     print(f"Per-class accuracy: {per_class_acc}")
    #     print("===========================================\n")

    #     return known_correct, known_total, unk_correct, unk_total, os_star, unk_acc, hos


    # 自动设置kl阈值 DPGMM
    def dpgmm_cluster(self, kl_values, show_plot=False):
        """
        使用 DPGMM 对 KL 散度进行聚类，并区分已知/未知类。
        返回：Q（0=未知，1=已知）
        """
        kl_values = kl_values.reshape(-1, 1)
        dpgmm = BayesianGaussianMixture(
            n_components=2,      # 最大混合数量，模型会自动调节
            covariance_type='full',
            weight_concentration_prior_type='dirichlet_process',
            max_iter=500,
            random_state=42
        )
        dpgmm.fit(kl_values)
        Q = dpgmm.predict(kl_values)

        # ----------------- 可视化部分 -----------------
        if show_plot:
            xs = np.linspace(np.min(kl_values), np.max(kl_values), 500).reshape(-1, 1)
            logprob = dpgmm.score_samples(xs)
            responsibilities = dpgmm.predict_proba(xs)
            pdf = np.exp(logprob)

            plt.figure(figsize=(8, 4))
            plt.hist(kl_values, bins=40, density=True, alpha=0.5, label='KL values')
            plt.plot(xs, pdf, '-k', label='DPGMM total PDF')

            for i, comp in enumerate(responsibilities.T):
                plt.plot(xs, comp * pdf, '--', label=f'Component {i}')

            plt.title("DPGMM Fit on KL Divergences")
            plt.xlabel("KL Divergence")
            plt.ylabel("Density")
            plt.legend()
            plt.show()

            # 输出各高斯分布的均值与方差，方便分析
            print("DPGMM component means:", dpgmm.means_.flatten())
            print("DPGMM component variances:", np.array([np.diag(cov)[0] for cov in dpgmm.covariances_]))

        # ----------------- 簇判断逻辑 -----------------
        # 假设 KL 较小的一簇为已知类
        means = dpgmm.means_.flatten()
        known_cluster = np.argmin(means)
        Q_binary = np.array([1 if q == known_cluster else 0 for q in Q])

        return Q_binary