"""
游戏内决策树与随机森林系统 - 核心模块

提供：
- build_tree()          树构建（特征选择 + 递归分割，含预剪枝）
- information_gain()    信息增益（熵计算 + 信息增益比）
- gini_index()          基尼系数（基尼不纯度 + 基尼增益）
- pruning()             剪枝（后剪枝：减枝法；预剪枝在构建时生效）
- random_forest()       随机森林（袋外误差 + 特征随机化 + 多树投票）
"""
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import math
import random
import time


@dataclass
class Config:
    max_depth: int = 10
    min_samples_split: int = 2
    min_gain: float = 1e-7
    criterion: str = "entropy"  # "entropy" 或 "gini"


@dataclass
class TreeNode:
    """决策树节点。feature 为 None 表示叶子节点，label 恒为该节点多数类。"""
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    label: Optional[Any] = None
    samples: int = 0
    depth: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


class DecisionTree:
    def __init__(self, config: Optional[Dict] = None):
        if isinstance(config, Config):
            config = {
                "max_depth": config.max_depth,
                "min_samples_split": config.min_samples_split,
                "min_gain": config.min_gain,
                "criterion": config.criterion,
            }
        self.config = config or {}
        self.max_depth = int(self.config.get("max_depth", 10))
        self.min_samples_split = int(self.config.get("min_samples_split", 2))
        self.min_gain = float(self.config.get("min_gain", 1e-7))
        self.criterion = self.config.get("criterion", "entropy")
        self.use_gain_ratio = bool(self.config.get("gain_ratio", False))
        self._state = {}
        self._history = []
        self.root: Optional[TreeNode] = None
        self.n_features: int = 0
        self.forest: List["DecisionTree"] = []
        self.oob_error: Optional[float] = None

    # ------------------------------------------------------------------
    # 游戏主循环钩子
    # ------------------------------------------------------------------
    def update(self, dt: float):
        self._history.append(dt)
        if len(self._history) > 1000:
            self._history = self._history[-1000:]

    def reset(self):
        self._state = {}
        self._history = []
        self.root = None
        self.forest = []
        self.oob_error = None

    # ------------------------------------------------------------------
    # 熵 / 信息增益 / 信息增益比
    # ------------------------------------------------------------------
    @staticmethod
    def entropy(labels: Sequence[Any]) -> float:
        """香农熵 H = -sum(p * log2(p))。"""
        n = len(labels)
        if n == 0:
            return 0.0
        ent = 0.0
        for count in Counter(labels).values():
            p = count / n
            if p > 0:
                ent -= p * math.log2(p)
        return ent

    def information_gain(self, parent: Sequence[Any],
                         left: Sequence[Any], right: Sequence[Any]) -> float:
        """信息增益 IG = H(parent) - 加权子节点熵。"""
        n = len(parent)
        if n == 0:
            return 0.0
        return (self.entropy(parent)
                - (len(left) / n) * self.entropy(left)
                - (len(right) / n) * self.entropy(right))

    def information_gain_ratio(self, parent: Sequence[Any],
                               left: Sequence[Any], right: Sequence[Any]) -> float:
        """信息增益比 = 信息增益 / 分裂信息（SplitInfo）。"""
        n = len(parent)
        if n == 0 or not left or not right:
            return 0.0
        gain = self.information_gain(parent, left, right)
        p_left = len(left) / n
        p_right = len(right) / n
        split_info = -(p_left * math.log2(p_left) + p_right * math.log2(p_right))
        if split_info <= 0:
            return 0.0
        return gain / split_info

    # ------------------------------------------------------------------
    # 基尼不纯度 / 基尼指数 / 基尼增益
    # ------------------------------------------------------------------
    @staticmethod
    def gini_impurity(labels: Sequence[Any]) -> float:
        """基尼不纯度 G = 1 - sum(p^2)。"""
        n = len(labels)
        if n == 0:
            return 0.0
        return 1.0 - sum((count / n) ** 2 for count in Counter(labels).values())

    def gini_index(self, parent: Sequence[Any],
                   left: Sequence[Any], right: Sequence[Any]) -> float:
        """基尼指数：分裂后的加权基尼不纯度（越小越好）。"""
        n = len(parent)
        if n == 0:
            return 0.0
        return ((len(left) / n) * self.gini_impurity(left)
                + (len(right) / n) * self.gini_impurity(right))

    def gini_gain(self, parent: Sequence[Any],
                  left: Sequence[Any], right: Sequence[Any]) -> float:
        """基尼增益 = 父节点基尼不纯度 - 基尼指数。"""
        return self.gini_impurity(parent) - self.gini_index(parent, left, right)

    # ------------------------------------------------------------------
    # 树构建：特征选择 + 递归分割（含预剪枝）
    # ------------------------------------------------------------------
    def _split_score(self, parent, left, right) -> float:
        if self.criterion == "gini":
            return self.gini_gain(parent, left, right)
        if self.use_gain_ratio:
            return self.information_gain_ratio(parent, left, right)
        return self.information_gain(parent, left, right)

    def _best_split(self, X, y, candidate_features):
        """特征选择：在候选特征中找出增益最大的 (特征, 阈值) 二分分裂。"""
        best_feature = None
        best_threshold = None
        best_gain = float("-inf")
        for feature in candidate_features:
            values = sorted({row[feature] for row in X})
            if len(values) <= 1:
                continue
            thresholds = [(values[i] + values[i + 1]) / 2.0
                          for i in range(len(values) - 1)]
            for threshold in thresholds:
                left_y, right_y = [], []
                for row, label in zip(X, y):
                    (left_y if row[feature] <= threshold else right_y).append(label)
                if not left_y or not right_y:
                    continue
                gain = self._split_score(y, left_y, right_y)
                if gain > best_gain:
                    best_feature, best_threshold, best_gain = feature, threshold, gain
        return best_feature, best_threshold, best_gain

    @staticmethod
    def _majority(labels) -> Any:
        return Counter(labels).most_common(1)[0][0]

    def build_tree(self, X, y, max_features: Optional[int] = None,
                   rng: Optional[random.Random] = None) -> TreeNode:
        """构建决策树并返回根节点。max_features 用于随机森林的特征随机化。"""
        X = [list(row) for row in X]
        y = list(y)
        if not X:
            raise ValueError("训练数据为空，无法构建决策树")
        self.n_features = len(X[0])
        self.root = self._build(X, y, depth=0, max_features=max_features,
                                rng=rng or random.Random(0))
        return self.root

    def _build(self, X, y, depth, max_features, rng) -> TreeNode:
        node = TreeNode(label=self._majority(y), samples=len(y), depth=depth)
        # 叶子：样本纯净
        if len(set(y)) == 1:
            return node
        # 预剪枝：深度 / 样本数限制
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return node
        # 特征随机化（随机森林）：只从特征子集中选择
        features = list(range(self.n_features))
        if max_features is not None and 0 < max_features < len(features):
            features = rng.sample(features, max_features)
        feature, threshold, gain = self._best_split(X, y, features)
        # 预剪枝：增益不足则停止分裂
        if feature is None or gain < self.min_gain:
            return node
        left_X, left_y, right_X, right_y = [], [], [], []
        for row, label in zip(X, y):
            if row[feature] <= threshold:
                left_X.append(row)
                left_y.append(label)
            else:
                right_X.append(row)
                right_y.append(label)
        # 递归分割
        node.feature = feature
        node.threshold = threshold
        node.left = self._build(left_X, left_y, depth + 1, max_features, rng)
        node.right = self._build(right_X, right_y, depth + 1, max_features, rng)
        return node

    # ------------------------------------------------------------------
    # 预测
    # ------------------------------------------------------------------
    def _predict_one(self, node: TreeNode, sample) -> Any:
        while not node.is_leaf:
            node = node.left if sample[node.feature] <= node.threshold else node.right
        return node.label

    def predict(self, sample) -> Any:
        if self.root is None:
            raise RuntimeError("决策树尚未构建，请先调用 build_tree()")
        return self._predict_one(self.root, list(sample))

    def classify(self, X) -> List[Any]:
        return [self.predict(row) for row in X]

    def score(self, X, y) -> float:
        y = list(y)
        if not y:
            return 0.0
        predictions = self.classify(X)
        return sum(1 for p, t in zip(predictions, y) if p == t) / len(y)

    # ------------------------------------------------------------------
    # 树结构度量
    # ------------------------------------------------------------------
    def count_nodes(self, node: Optional[TreeNode] = None) -> int:
        if node is None:
            node = self.root
        if node is None:
            return 0
        left = self.count_nodes(node.left) if node.left else 0
        right = self.count_nodes(node.right) if node.right else 0
        return 1 + left + right

    def get_depth(self, node: Optional[TreeNode] = None) -> int:
        if node is None:
            node = self.root
        if node is None or node.is_leaf:
            return 0
        return 1 + max(self.get_depth(node.left), self.get_depth(node.right))

    # ------------------------------------------------------------------
    # 剪枝：后剪枝（减枝法，基于验证集）
    # ------------------------------------------------------------------
    def pruning(self, X_val, y_val) -> TreeNode:
        """后剪枝：自底向上，若子树替换为叶子后验证集准确率不下降则剪枝。"""
        if self.root is None:
            raise RuntimeError("决策树尚未构建，请先调用 build_tree()")
        X_val = [list(row) for row in X_val]
        y_val = list(y_val)
        self._prune_node(self.root, X_val, y_val)
        return self.root

    def _prune_node(self, node: TreeNode, X, y):
        if node.is_leaf:
            return
        left_X, left_y, right_X, right_y = [], [], [], []
        for row, label in zip(X, y):
            if row[node.feature] <= node.threshold:
                left_X.append(row)
                left_y.append(label)
            else:
                right_X.append(row)
                right_y.append(label)
        self._prune_node(node.left, left_X, left_y)
        self._prune_node(node.right, right_X, right_y)
        if not y:
            return
        subtree_correct = sum(1 for row, t in zip(X, y)
                              if self._predict_one(node, row) == t)
        leaf_correct = sum(1 for t in y if t == node.label)
        if leaf_correct >= subtree_correct:
            node.feature = None
            node.threshold = None
            node.left = None
            node.right = None

    # ------------------------------------------------------------------
    # 随机森林：袋装采样 + 特征随机化 + 袋外误差 + 多树投票
    # ------------------------------------------------------------------
    def random_forest(self, X, y, n_trees: int = 10,
                      max_features: Optional[int] = None,
                      sample_ratio: float = 1.0,
                      seed: Optional[int] = None) -> List["DecisionTree"]:
        X = [list(row) for row in X]
        y = list(y)
        if not X:
            raise ValueError("训练数据为空，无法构建随机森林")
        n = len(y)
        n_features = len(X[0])
        if max_features is None:
            max_features = max(1, int(math.sqrt(n_features)))
        rng = random.Random(seed)
        self.forest = []
        oob_votes: List[List[Any]] = [[] for _ in range(n)]
        for _ in range(n_trees):
            # 袋装采样（bootstrap）
            sample_size = max(1, int(round(n * sample_ratio)))
            indices = [rng.randrange(n) for _ in range(sample_size)]
            in_bag = set(indices)
            tree = DecisionTree({
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_gain": self.min_gain,
                "criterion": self.criterion,
                "gain_ratio": self.use_gain_ratio,
            })
            tree.build_tree([X[i] for i in indices], [y[i] for i in indices],
                            max_features=max_features, rng=rng)
            self.forest.append(tree)
            # 袋外样本：用于袋外误差估计
            for i in range(n):
                if i not in in_bag:
                    oob_votes[i].append(tree.predict(X[i]))
        # 袋外误差：对每个样本用其袋外树投票
        errors = 0
        total = 0
        for i, votes in enumerate(oob_votes):
            if not votes:
                continue
            total += 1
            if Counter(votes).most_common(1)[0][0] != y[i]:
                errors += 1
        self.oob_error = errors / total if total else 0.0
        return self.forest

    def forest_predict(self, sample) -> Any:
        """多棵树投票，返回多数类。"""
        if not self.forest:
            raise RuntimeError("随机森林尚未构建，请先调用 random_forest()")
        votes = [tree.predict(sample) for tree in self.forest]
        return Counter(votes).most_common(1)[0][0]

    def forest_classify(self, X) -> List[Any]:
        return [self.forest_predict(row) for row in X]

    def forest_score(self, X, y) -> float:
        y = list(y)
        if not y:
            return 0.0
        predictions = self.forest_classify(X)
        return sum(1 for p, t in zip(predictions, y) if p == t) / len(y)
