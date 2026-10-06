"""
游戏内决策树与随机森林系统 - 验收测试
运行方式：python -m pytest tests/test_decision_tree.py -q
共 14 个测试用例
"""
import math
import random

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.decision_tree import DecisionTree


def make_game_data(n=300, seed=42):
    """NPC 行为数据：health/distance/ammo -> attack/flee/defend。"""
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        health = rng.uniform(0, 100)
        distance = rng.uniform(0, 50)
        ammo = rng.uniform(0, 20)
        if health < 30:
            label = "flee"
        elif distance < 15 and ammo > 5:
            label = "attack"
        else:
            label = "defend"
        X.append([health, distance, ammo])
        y.append(label)
    return X, y


def make_xor_data(repeat=5):
    X, y = [], []
    for _ in range(repeat):
        for a in (0.0, 1.0):
            for b in (0.0, 1.0):
                X.append([a, b])
                y.append("attack" if a != b else "defend")
    return X, y


class TestDecisionTree:
    def setup_method(self):
        self.system = DecisionTree()

    def test_case_01(self):
        # 熵计算：纯集合为 0，均衡二分类为 1，已知分布取理论值
        assert self.system.entropy([]) == 0.0
        assert self.system.entropy(["attack"] * 4) == 0.0
        assert self.system.entropy(["attack", "attack", "flee", "flee"]) == pytest.approx(1.0)
        expected = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
        assert self.system.entropy(["a", "a", "a", "b"]) == pytest.approx(expected)

    def test_case_02(self):
        # 信息增益：完美分裂增益等于父熵，无信息分裂增益为 0
        parent = ["attack", "attack", "flee", "flee"]
        gain = self.system.information_gain(parent, ["attack", "attack"], ["flee", "flee"])
        assert gain == pytest.approx(1.0)
        no_gain = self.system.information_gain(parent, ["attack", "flee"], ["attack", "flee"])
        assert no_gain == pytest.approx(0.0)
        partial = self.system.information_gain(parent, ["attack", "attack", "flee"], ["flee"])
        assert partial == pytest.approx(1.0 - 0.75 * self.system.entropy(["attack", "attack", "flee"]))

    def test_case_03(self):
        # 信息增益比 = 信息增益 / 分裂信息
        parent = ["attack", "attack", "flee", "flee"]
        ratio = self.system.information_gain_ratio(parent, ["attack", "attack"], ["flee", "flee"])
        assert ratio == pytest.approx(1.0)
        parent8 = ["a"] * 6 + ["b"] * 2
        left, right = ["a"] * 2, ["a"] * 4 + ["b"] * 2
        ig = self.system.information_gain(parent8, left, right)
        split_info = -(0.25 * math.log2(0.25) + 0.75 * math.log2(0.75))
        assert self.system.information_gain_ratio(parent8, left, right) == pytest.approx(ig / split_info)
        assert self.system.information_gain_ratio(parent8, [], parent8) == 0.0

    def test_case_04(self):
        # 基尼不纯度：纯集合为 0，均衡二分类为 0.5
        assert self.system.gini_impurity([]) == 0.0
        assert self.system.gini_impurity(["defend"] * 5) == 0.0
        assert self.system.gini_impurity(["attack", "attack", "flee", "flee"]) == pytest.approx(0.5)
        assert self.system.gini_impurity(["a", "a", "a", "b"]) == pytest.approx(1 - 0.75 ** 2 - 0.25 ** 2)

    def test_case_05(self):
        # 基尼指数（分裂后加权不纯度）与基尼增益
        parent = ["attack", "attack", "flee", "flee"]
        assert self.system.gini_index(parent, ["attack", "attack"], ["flee", "flee"]) == pytest.approx(0.0)
        assert self.system.gini_gain(parent, ["attack", "attack"], ["flee", "flee"]) == pytest.approx(0.5)
        mixed = self.system.gini_index(parent, ["attack", "flee"], ["attack", "flee"])
        assert mixed == pytest.approx(0.5)
        assert self.system.gini_gain(parent, ["attack", "flee"], ["attack", "flee"]) == pytest.approx(0.0)

    def test_case_06(self):
        # 特征选择：根节点必须选中真正有信息量的特征（distance，索引 1）
        rng = random.Random(7)
        X, y = [], []
        for _ in range(200):
            health, distance, ammo = rng.uniform(0, 100), rng.uniform(0, 50), rng.uniform(0, 20)
            X.append([health, distance, ammo])
            y.append("attack" if distance < 25 else "flee")
        root = self.system.build_tree(X, y)
        assert root.feature == 1
        assert self.system.score(X, y) == pytest.approx(1.0)

    def test_case_07(self):
        # 递归分割：XOR 数据单层分裂增益为 0，必须递归到第 2 层才能分开
        X, y = make_xor_data()
        tree = DecisionTree({"min_gain": 0.0})
        tree.build_tree(X, y)
        assert tree.get_depth() == 2
        assert tree.score(X, y) == pytest.approx(1.0)

    def test_case_08(self):
        # 分类准确率：远大于随机猜测的 50%
        X, y = make_game_data(300, seed=1)
        split = 210
        tree = DecisionTree({"max_depth": 8})
        tree.build_tree(X[:split], y[:split])
        accuracy = tree.score(X[split:], y[split:])
        assert accuracy >= 0.8

    def test_case_09(self):
        # 预剪枝：max_depth 与 min_samples_split 在构建时限制树的生长
        X, y = make_game_data(200, seed=2)
        shallow = DecisionTree({"max_depth": 2})
        shallow.build_tree(X, y)
        assert shallow.get_depth() <= 2
        strict = DecisionTree({"min_samples_split": 1000})
        strict.build_tree(X, y)
        assert strict.root.is_leaf
        assert strict.count_nodes() == 1

    def test_case_10(self):
        # 预剪枝：min_gain 增益阈值阻止低增益分裂
        X, y = make_game_data(200, seed=3)
        tree = DecisionTree({"min_gain": 2.0})  # 熵最大为 log2(3) < 2，任何分裂都被预剪枝
        tree.build_tree(X, y)
        assert tree.root.is_leaf
        normal = DecisionTree({"min_gain": 0.01})
        normal.build_tree(X, y)
        assert not normal.root.is_leaf

    def test_case_11(self):
        # 后剪枝：对过拟合的深树剪枝，节点数减少且验证集准确率不下降
        rng = random.Random(11)
        X_train, y_train = [], []
        for _ in range(300):
            f0, f1 = rng.random(), rng.random()
            X_train.append([f0, f1])
            label = "attack" if f0 + f1 > 1 else "defend"
            if rng.random() < 0.15:  # 训练集带噪声，诱导过拟合
                label = "defend" if label == "attack" else "attack"
            y_train.append(label)
        rng_val = random.Random(12)
        X_val = [[rng_val.random(), rng_val.random()] for _ in range(200)]
        y_val = ["attack" if r[0] + r[1] > 1 else "defend" for r in X_val]
        tree = DecisionTree({"max_depth": 20, "min_samples_split": 2})
        tree.build_tree(X_train, y_train)
        before_nodes = tree.count_nodes()
        before_acc = tree.score(X_val, y_val)
        tree.pruning(X_val, y_val)
        after_nodes = tree.count_nodes()
        after_acc = tree.score(X_val, y_val)
        assert after_nodes < before_nodes
        assert after_acc >= before_acc

    def test_case_12(self):
        # 随机森林：多棵树投票分类，准确率达标
        X, y = make_game_data(300, seed=5)
        forest = DecisionTree({"max_depth": 8})
        forest.random_forest(X, y, n_trees=15, seed=99)
        assert len(forest.forest) == 15
        assert forest.forest_predict(X[0]) in ("attack", "flee", "defend")
        assert forest.forest_score(X, y) >= 0.8

    def test_case_13(self):
        # 袋外误差：已计算、取值在 [0, 1]，且在规则数据上较低
        X, y = make_game_data(300, seed=6)
        forest = DecisionTree({"max_depth": 8})
        forest.random_forest(X, y, n_trees=25, seed=7)
        assert forest.oob_error is not None
        assert 0.0 <= forest.oob_error <= 1.0
        assert forest.oob_error < 0.3

    def test_case_14(self):
        # 特征随机化：max_features=1 时各树被迫使用不同特征，森林结构多样
        rng = random.Random(13)
        X, y = [], []
        for _ in range(300):
            f0, f1, f2 = rng.random(), rng.random(), rng.random()
            X.append([f0, f1, f2])
            y.append("attack" if (f0 > 0.5 or f1 > 0.5) else "defend")
        forest = DecisionTree({"max_depth": 6})
        forest.random_forest(X, y, n_trees=20, max_features=1, seed=21)
        root_features = {t.root.feature for t in forest.forest}
        assert len(root_features) >= 2
        assert root_features.issubset({0, 1, 2})
        assert forest.forest_score(X, y) >= 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
