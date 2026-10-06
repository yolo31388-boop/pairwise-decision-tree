"""
游戏内决策树与随机森林系统 - 验收测试
运行方式：python -m pytest tests/test_decision_tree.py -q
共 14 个测试用例
"""
import math
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.decision_tree import DecisionTree


def make_linear_data():
    """特征0可完全分开的线性数据集，特征1为噪声。"""
    X, y = [], []
    for i in range(40):
        X.append([i, i % 3])
        y.append("A" if i < 20 else "B")
    return X, y


def make_xor_data():
    """嵌套数据集：特征0先分组，组内还需特征1递归分割。"""
    X, y = [], []
    for a in (0, 1):
        for b in (0, 1):
            for _ in range(10):
                X.append([a, b])
                y.append(b if a == 0 else 1)
    return X, y


class TestDecisionTree:
    def setup_method(self):
        self.system = DecisionTree()

    # 01 熵计算
    def test_case_01(self):
        assert self.system.entropy([]) == 0.0
        assert self.system.entropy(["A", "A", "A"]) == 0.0
        assert self.system.entropy(["A", "A", "B", "B"]) == pytest.approx(1.0)
        assert self.system.entropy(["A", "A", "A", "B"]) == pytest.approx(0.8112781244591328)

    # 02 信息增益
    def test_case_02(self):
        parent = ["A", "A", "B", "B"]
        # 完美分割：增益等于父节点熵
        gain = self.system.information_gain(parent, [["A", "A"], ["B", "B"]])
        assert gain == pytest.approx(1.0)
        # 无信息分割：增益为 0
        gain0 = self.system.information_gain(parent, [["A", "B"], ["A", "B"]])
        assert gain0 == pytest.approx(0.0)

    # 03 信息增益比
    def test_case_03(self):
        parent = ["A", "A", "B", "B"]
        # 完美二分：增益=1，分裂信息=1，增益比=1
        ratio = self.system.gain_ratio(parent, [["A", "A"], ["B", "B"]])
        assert ratio == pytest.approx(1.0)
        # 极端偏斜分割的增益比应低于完美分割
        skewed = self.system.gain_ratio(parent, [["A"], ["A", "B", "B"]])
        assert 0.0 < skewed < 1.0
        # 不分割时增益比为 0
        assert self.system.gain_ratio(parent, [parent]) == 0.0

    # 04 基尼不纯度
    def test_case_04(self):
        assert self.system.gini_impurity([]) == 0.0
        assert self.system.gini_impurity(["A", "A"]) == 0.0
        assert self.system.gini_impurity(["A", "A", "B", "B"]) == pytest.approx(0.5)
        assert self.system.gini_impurity(["A", "A", "A", "B"]) == pytest.approx(0.375)

    # 05 基尼系数（分割后加权不纯度）
    def test_case_05(self):
        # 完美分割的基尼系数为 0
        assert self.system.gini_index([["A", "A"], ["B", "B"]]) == pytest.approx(0.0)
        # 两个各含 0.5 不纯度的子集，加权后仍为 0.5
        assert self.system.gini_index([["A", "B"], ["A", "B"]]) == pytest.approx(0.5)

    # 06 基尼增益
    def test_case_06(self):
        parent = ["A", "A", "B", "B"]
        gain = self.system.gini_gain(parent, [["A", "A"], ["B", "B"]])
        assert gain == pytest.approx(0.5)
        gain0 = self.system.gini_gain(parent, [["A", "B"], ["A", "B"]])
        assert gain0 == pytest.approx(0.0)

    # 07 树构建后能正确分类
    def test_case_07(self):
        X, y = make_linear_data()
        tree = self.system.build_tree(X, y)
        assert tree["type"] == "node"
        assert self.system.score(X, y) == pytest.approx(1.0)
        assert self.system.predict([5, 0]) == "A"
        assert self.system.predict([35, 2]) == "B"

    # 08 构建时选择最优特征
    def test_case_08(self):
        X, y = make_linear_data()
        tree = self.system.build_tree(X, y)
        # 根节点必须选择有信息的特征0，而不是噪声特征1
        assert tree["feature"] == 0
        assert tree["gain"] > 0.9

    # 09 递归分割节点（XOR 需要深度 >= 2）
    def test_case_09(self):
        X, y = make_xor_data()
        tree = self.system.build_tree(X, y)
        assert self.system.score(X, y) == pytest.approx(1.0)

        def max_depth(node):
            if node["type"] == "leaf":
                return node["depth"]
            return max(max_depth(node["left"]), max_depth(node["right"]))

        assert max_depth(tree) >= 2
        assert tree["left"]["type"] == "node" or tree["right"]["type"] == "node"

    # 10 预剪枝：max_depth 在构建时限制树深
    def test_case_10(self):
        system = DecisionTree({"max_depth": 1})
        X, y = make_xor_data()
        tree = system.build_tree(X, y)

        def max_depth(node):
            if node["type"] == "leaf":
                return node["depth"]
            return max(max_depth(node["left"]), max_depth(node["right"]))

        assert max_depth(tree) <= 1
        # 深度受限后 XOR 无法被完全拟合
        assert system.score(X, y) < 1.0

    # 11 预剪枝：min_samples_split 阻止小样本分割
    def test_case_11(self):
        X, y = make_linear_data()
        system = DecisionTree({"min_samples_split": 100})
        tree = system.build_tree(X, y)
        # 样本数不足，根节点直接退化为叶子
        assert tree["type"] == "leaf"
        # 合理的 min_samples_split 下仍能正常分割
        system2 = DecisionTree({"min_samples_split": 2})
        tree2 = system2.build_tree(X, y)
        assert tree2["type"] == "node"

    # 12 后剪枝：构建后剪枝不降低验证集准确率且不增加节点
    def test_case_12(self):
        # 训练集含噪声样本，使树过拟合
        X, y = make_linear_data()
        X_train = X + [[3, 1], [36, 0]]
        y_train = y + ["B", "A"]  # 噪声标签
        X_val, y_val = make_linear_data()
        self.system.build_tree(X_train, y_train)
        before_nodes = DecisionTree._count_nodes(self.system.tree_)
        before_acc = self.system.score(X_val, y_val)
        self.system.pruning(X_val, y_val)
        after_nodes = DecisionTree._count_nodes(self.system.tree_)
        after_acc = self.system.score(X_val, y_val)
        assert after_nodes < before_nodes
        assert after_acc >= before_acc
        assert after_acc == pytest.approx(1.0)

    # 13 随机森林：多棵树投票分类
    def test_case_13(self):
        X, y = make_xor_data()
        forest = self.system.random_forest(X, y, n_trees=15)
        assert len(forest["trees"]) == 15
        # 森林投票能正确分类训练数据
        assert self.system.forest_score(X, y) >= 0.9
        assert self.system.forest_predict([0, 1]) == 1
        assert self.system.forest_predict([0, 0]) == 0
        assert self.system.forest_predict([1, 1]) == 1

    # 14 随机森林：袋外误差 + 特征随机化
    def test_case_14(self):
        X, y = [], []
        for i in range(60):
            X.append([i, i % 2, i % 5, i % 7])
            y.append("L" if i < 30 else "R")
        forest = self.system.random_forest(X, y, n_trees=20, max_features=2)
        # 袋外误差是 [0, 1] 区间内的合法值
        assert 0.0 <= forest["oob_error"] <= 1.0
        assert self.system.oob_error_ == forest["oob_error"]
        # 特征随机化：每棵树使用的特征子集不同，森林结构存在差异
        signatures = set()
        for tree in forest["trees"]:
            def feats(node):
                if node["type"] == "leaf":
                    return frozenset()
                return frozenset({node["feature"]}) | feats(node["left"]) | feats(node["right"])
            signatures.add(feats(tree))
        assert len(signatures) > 1
        assert forest["max_features"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
