"""
游戏内决策树与随机森林系统 - 核心模块

提供：
- 决策树构建（特征选择 + 递归分割）
- 信息增益 / 熵 / 信息增益比
- 基尼系数 / 基尼不纯度 / 基尼增益
- 预剪枝（构建时）与后剪枝（构建后）
- 随机森林（多树投票、袋外误差、特征随机化）
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import random
import time


@dataclass
class Config:
    pass


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class DecisionTree:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state = {}
        self._history = []
        self.tree_ = None
        self.forest_ = None
        self.oob_error_ = None
        self._rng = random.Random(self.config.get("random_state", 42))

    def update(self, dt: float):
        pass

    def reset(self):
        self._state = {}
        self._history = []
        self.tree_ = None
        self.forest_ = None
        self.oob_error_ = None

    # ------------------------------------------------------------------
    # 不纯度度量：熵 / 基尼不纯度
    # ------------------------------------------------------------------
    @staticmethod
    def entropy(labels: List) -> float:
        """计算一组标签的信息熵。"""
        n = len(labels)
        if n == 0:
            return 0.0
        counts: Dict = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        ent = 0.0
        for c in counts.values():
            p = c / n
            if p > 0:
                ent -= p * math.log2(p)
        return ent

    @staticmethod
    def gini_impurity(labels: List) -> float:
        """计算一组标签的基尼不纯度。"""
        n = len(labels)
        if n == 0:
            return 0.0
        counts: Dict = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        gini = 1.0
        for c in counts.values():
            p = c / n
            gini -= p * p
        return gini

    # ------------------------------------------------------------------
    # 信息增益 / 信息增益比 / 基尼系数 / 基尼增益
    # ------------------------------------------------------------------
    def information_gain(self, parent: List, splits: List[List]) -> float:
        """信息增益 = 父节点熵 - 各子节点加权熵。"""
        n = len(parent)
        if n == 0:
            return 0.0
        gain = self.entropy(parent)
        for subset in splits:
            gain -= (len(subset) / n) * self.entropy(subset)
        return gain

    def gain_ratio(self, parent: List, splits: List[List]) -> float:
        """信息增益比 = 信息增益 / 分裂信息(split information)。"""
        n = len(parent)
        if n == 0:
            return 0.0
        gain = self.information_gain(parent, splits)
        split_info = 0.0
        for subset in splits:
            p = len(subset) / n
            if p > 0:
                split_info -= p * math.log2(p)
        if split_info == 0.0:
            return 0.0
        return gain / split_info

    def gini_index(self, splits: List[List]) -> float:
        """基尼系数：一次分割后各子节点的加权基尼不纯度。"""
        n = sum(len(s) for s in splits)
        if n == 0:
            return 0.0
        index = 0.0
        for subset in splits:
            index += (len(subset) / n) * self.gini_impurity(subset)
        return index

    def gini_gain(self, parent: List, splits: List[List]) -> float:
        """基尼增益 = 父节点基尼不纯度 - 分割后的基尼系数。"""
        return self.gini_impurity(parent) - self.gini_index(splits)

    # ------------------------------------------------------------------
    # 树构建：特征选择 + 递归分割
    # ------------------------------------------------------------------
    def _criterion_score(self, parent: List, splits: List[List]) -> float:
        criterion = self.config.get("criterion", "entropy")
        if criterion == "gini":
            return self.gini_gain(parent, splits)
        if criterion == "gain_ratio":
            return self.gain_ratio(parent, splits)
        return self.information_gain(parent, splits)

    def _candidate_splits(self, X: List[List], y: List, feature: int):
        """为单个特征生成候选二分方案，产出 (split描述, 左标签, 右标签, 左索引, 右索引)。"""
        values = [row[feature] for row in X]
        if all(_is_number(v) for v in values):
            uniq = sorted(set(values))
            if len(uniq) < 2:
                return
            thresholds = [(uniq[i] + uniq[i + 1]) / 2 for i in range(len(uniq) - 1)]
            if len(thresholds) > 16:
                step = len(thresholds) / 16
                thresholds = [thresholds[int(i * step)] for i in range(16)]
            for t in thresholds:
                left_idx = [i for i, v in enumerate(values) if v <= t]
                right_idx = [i for i, v in enumerate(values) if v > t]
                if not left_idx or not right_idx:
                    continue
                split = ("num", feature, t)
                yield split, left_idx, right_idx
        else:
            for v in set(values):
                left_idx = [i for i, val in enumerate(values) if val == v]
                right_idx = [i for i, val in enumerate(values) if val != v]
                if not left_idx or not right_idx:
                    continue
                split = ("cat", feature, v)
                yield split, left_idx, right_idx

    def _best_split(self, X: List[List], y: List, features: List[int]):
        """特征选择：在候选特征中找出增益最大的二分方案。"""
        best = None
        best_score = 0.0
        for feature in features:
            for split, left_idx, right_idx in self._candidate_splits(X, y, feature):
                left_y = [y[i] for i in left_idx]
                right_y = [y[i] for i in right_idx]
                score = self._criterion_score(y, [left_y, right_y])
                if score > best_score:
                    best_score = score
                    best = (split, left_idx, right_idx, score)
        return best

    @staticmethod
    def _majority(y: List):
        counts: Dict = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        return max(counts.items(), key=lambda kv: (kv[1], -list(counts).index(kv[0])))[0]

    def build_tree(self, X: List[List], y: List) -> Dict:
        """构建决策树（含预剪枝），返回根节点。"""
        if not X or not y or len(X) != len(y):
            raise ValueError("X 与 y 必须非空且长度一致")
        n_features = len(X[0])
        self.tree_ = self._build(X, y, list(range(n_features)), depth=0)
        return self.tree_

    def _build(self, X: List[List], y: List, features: List[int], depth: int,
               max_features: Optional[int] = None) -> Dict:
        """递归分割节点；预剪枝在构建时生效。"""
        node = {
            "type": "leaf",
            "label": self._majority(y),
            "count": len(y),
            "depth": depth,
        }

        # 预剪枝 / 停止条件
        max_depth = self.config.get("max_depth")
        min_samples_split = self.config.get("min_samples_split", 2)
        min_gain = self.config.get("min_gain", 1e-12)
        if len(set(y)) == 1:
            return node
        if max_depth is not None and depth >= max_depth:
            return node
        if len(y) < min_samples_split:
            return node
        if not features:
            return node

        # 特征随机化（随机森林使用）
        candidate_features = list(features)
        if max_features is not None and max_features < len(candidate_features):
            candidate_features = self._rng.sample(candidate_features, max_features)

        best = self._best_split(X, y, candidate_features)
        if best is None or best[3] < min_gain:
            return node

        split, left_idx, right_idx, score = best
        left_X = [X[i] for i in left_idx]
        left_y = [y[i] for i in left_idx]
        right_X = [X[i] for i in right_idx]
        right_y = [y[i] for i in right_idx]

        node.update({
            "type": "node",
            "split": split,
            "feature": split[1],
            "gain": score,
            "left": self._build(left_X, left_y, features, depth + 1, max_features),
            "right": self._build(right_X, right_y, features, depth + 1, max_features),
        })
        return node

    # ------------------------------------------------------------------
    # 预测
    # ------------------------------------------------------------------
    def _predict_one(self, node: Dict, x: List):
        while node["type"] == "node":
            kind, feature, value = node["split"]
            if kind == "num":
                node = node["left"] if x[feature] <= value else node["right"]
            else:
                node = node["left"] if x[feature] == value else node["right"]
        return node["label"]

    def predict(self, x: List):
        if self.tree_ is None:
            raise RuntimeError("请先调用 build_tree() 构建决策树")
        return self._predict_one(self.tree_, x)

    def predict_many(self, X: List[List]) -> List:
        return [self.predict(x) for x in X]

    def score(self, X: List[List], y: List) -> float:
        """分类准确率。"""
        if not y:
            return 0.0
        preds = self.predict_many(X)
        return sum(1 for p, t in zip(preds, y) if p == t) / len(y)

    # ------------------------------------------------------------------
    # 剪枝
    # ------------------------------------------------------------------
    @staticmethod
    def _count_nodes(node: Dict) -> int:
        if node["type"] == "leaf":
            return 1
        return 1 + DecisionTree._count_nodes(node["left"]) + DecisionTree._count_nodes(node["right"])

    def pruning(self, X_val: List[List], y_val: List) -> Dict:
        """后剪枝：基于验证集的自底向上降低错误率剪枝。"""
        if self.tree_ is None:
            raise RuntimeError("请先调用 build_tree() 构建决策树")
        best_acc = self.score(X_val, y_val)

        def try_prune(node: Dict) -> float:
            nonlocal best_acc
            if node["type"] == "leaf":
                return best_acc
            try_prune(node["left"])
            try_prune(node["right"])
            saved = {k: node[k] for k in ("type", "split", "feature", "gain", "left", "right")}
            node["type"] = "leaf"
            for k in ("split", "feature", "gain", "left", "right"):
                node.pop(k, None)
            acc = self.score(X_val, y_val)
            if acc >= best_acc:
                best_acc = acc
            else:
                node.update(saved)
            return best_acc

        try_prune(self.tree_)
        return self.tree_

    # ------------------------------------------------------------------
    # 随机森林：多树投票 + 袋外误差 + 特征随机化
    # ------------------------------------------------------------------
    def random_forest(self, X: List[List], y: List, n_trees: int = 10,
                      max_features: Optional[int] = None,
                      sample_ratio: float = 1.0) -> Dict:
        """训练随机森林，返回含树列表与袋外误差的字典。"""
        if not X or not y or len(X) != len(y):
            raise ValueError("X 与 y 必须非空且长度一致")
        n = len(X)
        n_features = len(X[0])
        if max_features is None:
            max_features = max(1, int(math.sqrt(n_features)))
        sample_size = max(1, int(n * sample_ratio))

        trees = []
        oob_votes: List[List] = [[] for _ in range(n)]
        for _ in range(n_trees):
            idx = [self._rng.randrange(n) for _ in range(sample_size)]
            in_bag = set(idx)
            boot_X = [X[i] for i in idx]
            boot_y = [y[i] for i in idx]
            tree = self._build(boot_X, boot_y, list(range(n_features)),
                               depth=0, max_features=max_features)
            trees.append(tree)
            for i in range(n):
                if i not in in_bag:
                    oob_votes[i].append(self._predict_one(tree, X[i]))

        # 袋外误差
        errors = 0
        counted = 0
        for i, votes in enumerate(oob_votes):
            if not votes:
                continue
            counted += 1
            pred = self._majority(votes)
            if pred != y[i]:
                errors += 1
        self.oob_error_ = (errors / counted) if counted else 0.0
        self.forest_ = {"trees": trees, "oob_error": self.oob_error_,
                        "max_features": max_features, "n_trees": n_trees}
        return self.forest_

    def forest_predict(self, x: List):
        """多棵树投票分类。"""
        if not self.forest_:
            raise RuntimeError("请先调用 random_forest() 训练随机森林")
        votes = [self._predict_one(tree, x) for tree in self.forest_["trees"]]
        return self._majority(votes)

    def forest_score(self, X: List[List], y: List) -> float:
        if not y:
            return 0.0
        preds = [self.forest_predict(x) for x in X]
        return sum(1 for p, t in zip(preds, y) if p == t) / len(y)
