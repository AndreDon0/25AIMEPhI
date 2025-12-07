import numpy as np


class Tree:
    class Node:
        def __init__(self, index, t, true_branch, false_branch):
            self.index = index  # индекс признака, по которому ведётся сравнение с порогом в этом узле
            self.t = t  # значение порога
            self.true_branch = true_branch  # поддерево, удовлетворяющее условию в узле
            self.false_branch = false_branch  # поддерево, не удовлетворяющее условию в узле
        
        def forward(self, obj):
            from numbers import Number
            if isinstance(obj[self.index], Number):
                return self.true_branch.forward(obj) if obj[self.index] <= self.t else self.false_branch.forward(obj)
            else:
                return self.true_branch.forward(obj) if obj[self.index] == self.t else self.false_branch.forward(obj)
    
    class Leaf:
        def __init__(self):
            self.value = None
        
        def forward(self, obj):
            return self.value

    def __init__(self, tree_depth=np.inf, min_leaf_count=1, criteria="entropy"):
        if tree_depth < 0 or min_leaf_count < 1:
            raise ValueError("tree_depth and min_leaf_count must be greater than 0")
        if criteria not in ["entropy", "gini", "missclassification"]:
            raise ValueError("criteria must be one of: entropy, gini, missclassification")

        self.tree_depth = tree_depth
        self.min_leaf_count = min_leaf_count
        self.criteria = criteria
        self.classes = None
        self.tree = None

    def fit(self, train_data, train_labels):
        self.data = train_data
        self.labels = train_labels
        self.classes = np.unique(train_labels)
        indices = np.arange(len(train_labels))
        self.tree = self.build_node(indices, self.tree_depth)

    def build_node(self, idx, depth):
        labels = self.labels[idx]

        if depth == 0 or len(idx) < self.min_leaf_count or len(np.unique(labels)) == 1:
            leaf = self.Leaf()
            leaf.value = self.calc_leaf(labels)
            return leaf

        left_idx, right_idx, index, t = self.find_best_split(idx)

        if left_idx is None or right_idx is None:
            leaf = self.Leaf()
            leaf.value = self.calc_leaf(labels)
            return leaf

        true_branch = self.build_node(left_idx, depth - 1)
        false_branch = self.build_node(right_idx, depth - 1)
        return self.Node(index, t, true_branch, false_branch)
    
    def find_best_split(self, idx):
        from numbers import Number
        data, labels = self.data, self.labels
        criteria = getattr(self, self.criteria)

        t_best = None
        index_best = None
        left_best = None
        right_best = None
        q_best = None

        n_total = len(idx)

        for feat in range(len(data[0])):
            feature_vals = [data[i][feat] for i in idx]
            unique_vals = set(feature_vals)
            if len(unique_vals) == 1:
                continue

            for t in unique_vals:
                left, right = [], []
                for i in idx:
                    v = data[i][feat]
                    if isinstance(v, Number):
                        (left if v <= t else right).append(i)
                    else:
                        (left if v == t else right).append(i)

                if len(left) == 0 or len(right) == 0:
                    continue

                left_labels = labels[left]
                right_labels = labels[right]

                p_left = np.array([np.mean(left_labels == c) for c in self.classes])
                p_right = np.array([np.mean(right_labels == c) for c in self.classes])

                imp_left = criteria(p_left)
                imp_right = criteria(p_right)

                q = (len(left) * imp_left + len(right) * imp_right) / n_total

                if q_best is None or q < q_best:
                    q_best = q
                    t_best = t
                    index_best = feat
                    left_best = left
                    right_best = right

        if left_best is None or right_best is None:
            return None, None, None, None

        return np.array(left_best), np.array(right_best), index_best, t_best
    
    def calc_leaf(self, labels):
        vals, counts = np.unique(labels, return_counts=True)
        return vals[np.argmax(counts)]
    
    def predict(self, test_data):
        predictions = []
        for obj in test_data:
            predictions.append(self.tree.forward(obj))
        return np.array(predictions)

    @staticmethod
    def missclassification(p):
        return 1 - np.max(p)

    @staticmethod
    def entropy(p):
        eps = 1e-15
        p += eps
        return -sum(p * np.log2(p))
    
    @staticmethod
    def gini(p):
        return 1 - sum(p ** 2)