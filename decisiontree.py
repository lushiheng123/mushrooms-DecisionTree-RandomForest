"""Classes and methods implementing DecisionTree and RandomForest classifiers
    as described in Chapters 19 and 22 of Zaki & Meira's 2nd Ed of 'Data Mining and Machine Learning'

Top Level Functions in this Module
    split_data(data, attr_split, dtype)
    compute_entropy(data, attr_split, dtype)
    compute_info_gain(data, attr_split, dtype)
    evaluate_numeric_attribute(data, x, dtype)
    evaluate_categorical_attribute(data, x, dtype)

Classes in this Module
    RootNode
        Attributes:
            child
        Methods:
            add_child
            evaluate
    DecisionNode
        Attributes:
            left_child
            right_child
            attribute
            split_point
            split_condition
        Methods:
            add_child
            evaluate
    LeafNode
        Attributes:
            label
            purity
            size
        Methods:
            evaluate
    DecisionTree
        Attributes:
            min_leafsize
            min_leafpurity
            data
            root_node
        Methods:
            fit
            predict
    ForestTree
        Attributes:
            min_leafsize
            min_leafpurity
            num_features
            data
            root_node
        Methods:
            fit
            predict
    RandomForest
        Attributes:
            trees
            random_state
            oob_error_
        Methods:
            fit
            predict
"""

import math
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split


def split_data(data, attr_split, dtype="numeric"):
    """splits a dataset into two datasets based on a given attribute and split point

    :param data: A pandas.DataFrame
    :param attr_split: tuple(attribute_label, split_point)
        attribute_label is the column label for the attribute in data 
            based on which the split will occur.
        split_point is the value on which the attribute will be split.  
            This has only been tested with split_point as a number or a str.
    :param dtype: either "numeric" or "categorical"
        "numeric" if data[attr_split[0]] is a numeric type of data
        "categorical" if data[attr_split[0]] is a categorical type of data
    :returns data_y, data_n: pandas.DataFrame, pandas.DataFrame
        if dtype=="numeric"
            data_y contains all points from data where the attribute <= split_point
            data_n contains all points from data where the attribute > split_point
        else:
            data_y contains all points where the attribute == split_point
            data_n contains all points where the attribute != split_point
    """
    try:
        if dtype=="numeric":
            splitter = data[attr_split[0]] <= float(attr_split[1])
        else:
            splitter = data[attr_split[0]] == attr_split[1]
    except TypeError as terr:
        print(
            f"The argument {attr_split} is an invalid type.",
            "It must be a tuple of the form",
            "\t(col_index_val, float) if dtype='numeric'",
            "\t(col_index_val, str) if dtype='categorical'",
            f"\n{terr}",
            sep="\n",
        )
    except IndexError as ierr:
        print(
            f"The argument {attr_split} is invalid.",
            "It must be a tuple of the form",
            "\t(col_index_val, float) if dtype=='numeric'",
            "\t(col_index_val, str) if dtype=='categorical'",
            f"\n{ierr}",
            sep="\n",
        )
    else:
        return data[splitter], data[~splitter]


def compute_entropy(data, attr_split=tuple(), dtype="numeric"):
    """Calculates the entropy of data, a measure indicating the amount of disorder or uncertainty
        following equations 19.3 and 19.4 in Zaki & Meira's 2nd Ed of 'Data Mining and Machine Learning'

    :param data: A pandas.DataFrame
    :param attr_split: tuple(attribute_label, split_point)
        attribute_label is the column label for the attribute in data 
            based on which the split will occur.
        split_point is the value on which the attribute will be split.  
            This has only been tested with split_point as a number or a str.
    :param dtype: either "numeric" or "categorical"
        "numeric" if data[attr_split[0]] is a numeric type of data
        "categorical" if data[attr_split[0]] is a categorical type of data
    :returns overal_entropy: of data if attr_split is not provided
    :returns split_entropy: of data_y and data_n given split rule described by attr_split
    """
    if not attr_split:
        # compute overall entropy
        label_counts = data.iloc[:, -1].value_counts()
        label_probs = label_counts / data.shape[0]
        overall_entropy = -pd.Series.sum(label_probs * np.log2(label_probs))
        return overall_entropy
    else:
        # compute split entropy
        data_y, data_n = split_data(data, attr_split, dtype)
        return (
            data_y.shape[0] * compute_entropy(data_y, dtype=dtype)
            + data_n.shape[0] * compute_entropy(data_n, dtype=dtype)
        ) / data.shape[0]


def compute_info_gain(data, attr_split=tuple(), dtype="numeric"):
    """Calculates the information gain of data given an split rule
        following equation 19.5 in Zaki & Meira's 2nd Ed of 'Data Mining and Machine Learning'

    :param data: pandas.DataFrame
    :param attr_split: tuple(attribute_label, split_point)
        attribute_label is the column label for the attribute in data 
            based on which the split will occur
        split_point is the ordered value on which the attribute will be split.  It must
            have __leq__() implemented.
    :returns gain: entropy of data minus split entropy of data given the split rule
    """
    return compute_entropy(data, dtype=dtype) - compute_entropy(data, attr_split, dtype=dtype)


def evaluate_numeric_attribute(data, attr_label):
    """Calculates the entropy of data, a measure indicating the amount of disorder or uncertainty
        following Algorithm 19.1 in 2nd Ed. of Zaki and Meria's 'Data Mining and Machine Learning'

    :param data: pandas.DataFrame
    :param attr_label: a column label of data
    :returns max_varg, max_score: the point and information gain reflecting the optimal
        split point for the attribute indicated by attr_label
    """
    d = data.sort_values(by=attr_label).set_index(np.arange(data.shape[0]))
    midpoints = []
    labels = d.iloc[:, -1]
    for j in range(d.shape[0] - 1):
        xj, xj1 = d.loc[j, attr_label], d.loc[j + 1, attr_label]
        if xj1 != xj:
            midpoints.append((xj1 + xj) / 2)
    max_varg, max_score = 0, 0
    for v in midpoints:
        v_score = compute_info_gain(d, attr_split=(attr_label, v))
        if v_score > max_score:
            max_score = v_score
            max_varg = v
    return max_varg, max_score


def evaluate_categorical_attribute(data, attr_label):
    """Calculates the entropy of data, a measure indicating the amount of disorder or uncertainty
        following Algorithm 19.2 in 2nd Ed. of Zaki and Meria's 'Data Mining and Machine Learning'

    :param data: pandas.DataFrame
    :param attr_label: a column label of data
    :returns max_varg, max_score: the point and information gain reflecting the optimal
        split point for the attribute indicated by attr_label
    """
    d = data.copy()
    labels = data.iloc[:, -1]
    max_varg, max_score = str(), 0
    for v in d[attr_label].unique():
        v_score = compute_info_gain(d, attr_split=(attr_label, v), dtype="categorical")
        if v_score > max_score:
            max_score = v_score
            max_varg = v
    return max_varg, max_score

    
class RootNode:
    """Represents the root node of a DecisionTree

    :param child: This root's only child node, either a DecisionNode or a LeafNode
    """
    def __init__(self):
        """Constructs a RootNode for use in a DecisionTree"""
        self.child = None

    def add_child(self, other, side):
        """Adds other as its child node.  Ignores the side parameter.
        """
        self.child = other
        return self.child

    def evaluate(self, data_point):
        """Root node asks its child to start the recursive evaluation of data_point
            in order to find its predicted label.

        :returns: predicted label of data_point
        """
        return self.child.evaluate(data_point)


class DecisionNode:
    """Represents a decision node in a DecisionTree with an attribute and split_point

    :param attribute: a label for an attribute of the dataset to which the DecisionTree is fitted
    :param split_point: the value at which the attribute splits the dataset
    :param split_condition: the condition evaluated to make a decision
        x <= split_point if split_point is a numeric dtype
        x == split_point if split_point is a categorical dtype
    :param left_child: the child DecisionNode or LeafNode for when a data point's split_condition evaluates to True
    :param right_child: the child DecisionNode or LeafNode for when a data point's split_condtion evaluates to False
    """
    def __init__(self, attribute, split_point):
        """Constructs a DecisionNode for use in a DecisionTree
        
        :param attribute: a label for an attribute of the dataset to which the DecisionTree is fitted
        :param split_point: the value at which the attribute splits the dataset
        """
        self.attribute = attribute
        self.split_point = split_point
        if is_numeric_dtype(split_point):
            self.split_condition = lambda x: x <= self.split_point
        else:
            self.split_condition = lambda x: x == self.split_point
        self.left_child = None
        self.right_child = None

    def add_child(self, other, side):
        """Adds a child node on the appropriate side of this node (left or right)

        :param other: A DecisionNode or LeafNode
        :param side: either 'left' or 'right', indicating whether the param other 
            is assigned to self.left_child or self.right_child
        """
        if side == "left":
            self.left_child = other
            return self.left_child
        elif side == "right":
            self.right_child = other
            return self.right_child
        else:
            raise ValueError(
                "The value of side must be either 'right' or 'left'"
            )

    def evaluate(self, data_point):
        """Recursively predicts label for data_point by passing the decision down to the
            correct child node.

        :param data_point: single observation matching the dimensions and type of the data
            to which the DecisionTree was fitted.
        """
        if self.split_condition(data_point.loc[self.attribute]):
            return self.left_child.evaluate(data_point)
        else:
            return self.right_child.evaluate(data_point)


class LeafNode:
    """Represents a leaf node in a DecisionTree, ultimately responsible for predicting a data point's label

    :param label: the predicted label for a data_point that recursively makes its way down to this node in the tree
        when evaluating it.
    :param purity: the purity of the subspace of the training set's data space represented by this leaf node
    :param size: the number of points in the subspace of the training set's data space represented by this leaf node
    """
    def __init__(self, label, purity, size):
        self.label = label
        self.purity = purity
        self.size = size

    def evaluate(self, data_point):
        """As a leaf node, this is the end of the recursive evaluate call.

        :returns: predicted label for data_point
        """
        return self.label


class DecisionTree:
    """Represents a DecisionTree for supervised classification of data points.  The tree is built
        following Algorithm 19.1 of Zaki & Meira's 2nd Ed of 'Data Mining and Machine Learning'

    :param min_leafsize: If a subsection of the data has <= points than this, it is a leaf node
        If 0 <= min_leafsize < 1, it is treated as a percentage of total points in data
        If min_leafsize > 1, it is cast to an int and treated ad the number of leaves.
    :param min_leafpurity: If a subsection of the data has purity >= this, it is a leaf node
    :param data: a pandas.DataFrame with floats in all columns, excepting the last column which is the class label
    :param root_node: a RootNode, used to ask the DecisionTree for a predicted label for a given point
    """
    def __init__(self, min_leafsize, min_leafpurity):
        """Constructs a DecisionTree object
        
        :param min_leafsize: If a subsection of the data has <= points than this, it is a leaf node
            If 0 <= min_leafsize < 1, it is treated as a percentage of total points in data
            If min_leafsize > 1, it is cast to an int and treated ad the number of leaves.
        :param min_leafpurity: If a subsection of the data has purity >= this, it is a leaf node
        """
        if 0 <= min_leafsize < 1:
            self.min_leafsize = min_leafsize
        elif min_leafsize >= 1:
            self.min_leafsize = int(min_leafsize)
        else:
            raise ValueError("min_leafsize must be a float in range [0,1] or a positive integer")
        self.min_leafpurity = min_leafpurity

    def fit(self, data, print_summary=False):
        """Builds a decision tree for classifying points from the population represented
            by the rows of data

        :param data: a pandas.DataFrame with floats in all columns, excepting the last column which is the class label
        :param print_summary: a boolean indicating whether the tree should be printed to the console
        :returns: self
        """
        self.data = data
        self.root_node = RootNode()
        if self.min_leafsize < 1:
            self.min_leafsize = int(self.min_leafsize * self.data.shape[0])
        self.build_decision_tree(
            data, self.root_node, "left", print_summary, level=0
        )
        return self

    def build_decision_tree(
        self, data, parent_node, side, print_summary=False, level=0
    ):
        """Recursively adds nodes to self.root_node until the DecisionTree is fully built
            following Algorithm 19.1 of Zaki & Meira's 2nd Ed of 'Data Mining and Machine Learning'

        :param data: a pandas.DataFrame with floats in all columns, excepting the last column which is the class label
        :param parent_node: the node to which this method will add either a DecisionNode or LeafNode as a child
        :param side: the side of the parent_node to which a DecisionNode or LeafNode will be added as a child
        :param print_summary: a boolean indicating whether the tree should be printed to the console
        :param level: the number of levels down in the tree from the RootNode.  Used to print the print_summary nicely
        """
        size = data.shape[0]
        label_counts = data.iloc[:, -1].value_counts()
        purity = label_counts.max() / size
        indent = "    " * level

        if size <= self.min_leafsize or purity >= self.min_leafpurity:
            if size > 0:
                label = label_counts.index[label_counts.argmax()]
            else:
                label = "None"
            # create leaf node and label it with majority_label
            if print_summary:
                print(
                    f"{indent}If {side=='left'}, Leaf: label= {label}, purity= {round(purity, 3)}, size= {size}"
                )
            parent_node.add_child(LeafNode(label, purity, size), side=side)
            return

        best_split, max_score = 0, 0
        for name, col in data.iloc[:, :-1].items():
            if is_numeric_dtype(col.dtype):
                v, score = evaluate_numeric_attribute(data, name)
                dtype="numeric"
            else:
                v, score = evaluate_categorical_attribute(data, name)
                dtype="categorical"
            if score > max_score:
                best_split, max_score, best_attr = v, score, name

        if print_summary:
            if is_numeric_dtype(data[best_attr].dtype):
                print(
                    f"{indent}{'If ' + str(side=='left') + ', ' if level !=0 else ''}"
                    f"Decision: {best_attr} <= {round(best_split, 3)}, Gain= {round(max_score, 3)}"
                )
            else:
                print(
                    f"{indent}{'If ' + str(side=='left') + ', ' if level !=0 else ''}"
                    f"Decision: {best_attr} = {best_split}, Gain= {round(max_score, 3)}"
                )
        decision_node = parent_node.add_child(
            DecisionNode(best_attr, best_split), side=side
        )
        data_y, data_n = split_data(data, attr_split=(best_attr, best_split), dtype=dtype)
        self.build_decision_tree(
            data_y, decision_node, "left", print_summary, level=level + 1
        )
        self.build_decision_tree(
            data_n, decision_node, "right", print_summary, level=level + 1
        )

    def predict(self, data):
        """Predicts the label for a set of points related to those to which this DecisionTree is fitted

        :param data: a pandas.DataFrame, where each row represents an unlabeled data point
        :returns: a pandas.Series of predicted labels for each point (row) in data
        """
        return pd.Series(
            (self.root_node.evaluate(point) for _, point in data.iterrows()),
            index=data.index,
        )
    
    
class ForestTree(DecisionTree):
    """Represents a DecisionTree for use in a RandomForest classifier.  The tree is built
        following the description on page 577 of Zaki & Meira's 2nd Ed of 'Data Mining and Machine Learning'

    :param min_leafsize: If a subsection of the data has <= points than this, it is a leaf node
        If 0 <= min_leafsize < 1, it is treated as a percentage of total points in data
        If min_leafsize > 1, it is cast to an int and treated ad the number of leaves.
    :param min_leafpurity: If a subsection of the data has purity >= this, it is a leaf node
    :param num_features: The number of features randomly selected for evaluating each split point of the tree
    :param data: a pandas.DataFrame with floats in all columns, excepting the last column which is the class label
    :param root_node: a RootNode, used to ask the DecisionTree for a predicted label for a given point
    """
    def __init__(self, min_leafsize, min_leafpurity, num_features):
        """Constructs a ForestTree object
        
        :param min_leafsize: If a subsection of the data has <= points than this, it is a leaf node
        :param min_leafpurity: If a subsection of the data has purity >= this, it is a leaf node
        :param num_features: The number of features randomly selected for evaluating each split point of the tree
        """
        super().__init__(min_leafsize, min_leafpurity)
        self.num_features = num_features # value further determined in the fit method
    
    def fit(self, data):
        """Builds a decision tree for classifying points from the population represented
            following the description on page 577 of Zaki & Meira's 2nd Ed of 'Data Mining and Machine Learning'

        :param data: a pandas.DataFrame with floats in all columns, excepting the last column which is the class label
        :param print_summary: a boolean indicating whether the tree should be printed to the console
        :returns: self
        """
        self.data = data
        if not self.num_features:
            self.num_features = int(math.sqrt(self.data.shape[1]))
        self.root_node = RootNode()
        if self.min_leafsize < 1:
            self.min_leafsize = int(self.min_leafsize * self.data.shape[0])
        self.build_decision_tree(
            data, self.root_node, "left", level=0
        )
        return self
    
    def build_decision_tree(
        self, data, parent_node, side, print_summary=False, level=0
    ):
        """Recursively adds nodes until the DecisionTree is fully built
            following the description on page 577 of Zaki & Meira's 2nd Ed of 'Data Mining and Machine Learning'

        :param data: a pandas.DataFrame with floats in all columns, excepting the last column which is the class label
        :param parent_node: the node to which this method will add either a DecisionNode or LeafNode as a child
        :param side: the side of the parent_node to which a DecisionNode or LeafNode will be added as a child
        :param print_summary: a boolean indicating whether the tree should be printed to the console
        :param level: the number of levels down in the tree from the RootNode.  Used to print the print_summary nicely
        """
        size = data.shape[0]
        label_counts = data.iloc[:, -1].value_counts()
        purity = label_counts.max() / size
        indent = "    " * level

        if size <= self.min_leafsize or purity >= self.min_leafpurity:
            label = label_counts.index[label_counts.argmax()]
            parent_node.add_child(LeafNode(label, purity, size), side=side)
            return

        selected_features = np.random.choice(data.iloc[:, :-1].columns, size=self.num_features, replace=False)
        best_attr = selected_features[0]
        best_split, max_score = data.loc[:,best_attr].iloc[0], 0
        for name in selected_features:
            col = data[name]
            if is_numeric_dtype(col.dtype):
                v, score = evaluate_numeric_attribute(data, name)
                dtype="numeric"
            else:
                v, score = evaluate_categorical_attribute(data, name)
                dtype="categorical"
            if score > max_score:
                best_split, max_score, best_attr = v, score, name
        
        data_y, data_n = split_data(data, attr_split=(best_attr, best_split), dtype=dtype)
        if data_y.shape[0] == 0 or data_n.shape[0] == 0:
            # if none of the randomly selected features have a split point
            # that results in information gain, then add a leaf node
            label = label_counts.index[label_counts.argmax()]
            parent_node.add_child(LeafNode(label, purity, size), side=side)
            return
        else:
            decision_node = parent_node.add_child(
                DecisionNode(best_attr, best_split), side=side
            )
            self.build_decision_tree(
                data_y, decision_node, "left", print_summary, level=level + 1
            )
            self.build_decision_tree(
                data_n, decision_node, "right", print_summary, level=level + 1
            )
    
    
class RandomForest:
    """Represents a random forest classifier of decision trees
        following Algorithm 22.5 in Zaki & Meira's 2nd Ed of 'Data Mining and Machine Learning'
        
    :param trees: a list of ForestTree objects
    :param random_state: an integer passed to numpy.random.seed for reproducible results
    :param oob_error_: The out of bag error for this model, i.e., the accuracy rate of the model across
        the input data when using only trees for each point in the data that did not learn for each point.
    """
    def __init__(self, num_trees=101, min_leafsize=5, min_leafpurity=0.95, num_features=None, random_state=None):
        self.random_state = random_state
        self.trees = [
            ForestTree(min_leafsize, min_leafpurity, num_features)
            for _ in range(num_trees)
        ]
    
    def fit(self, data):
        """Fits an ensemble of trees to bootstrap samples of data
            following Algorithm 22.5 in Zaki & Meira's 2nd Ed of 'Data Mining and Machine Learning'
        
        :param data: a pandas.DataFrame with floats in all columns, excepting the last column which is the class label
        :returns self:
        """
        np.random.seed(self.random_state)
        true_labels = data.iloc[:, -1]
        oob_matrix = pd.DataFrame(columns=data.index, index=np.arange(len(self.trees))).astype(str)
        oob_matrix.loc[:,:] = pd.NA
        for i, tree in enumerate(self.trees):
            tree.fit(data.sample(frac=1, replace=True))
            oob_index = data.index.difference(tree.data.index)
            oob_data = data.loc[oob_index, :].iloc[:, :-1]
            oob_matrix.loc[i, oob_index] = tree.predict(oob_data)        
        oob_predictions = pd.Series(pd.NA, index=oob_matrix.columns)
        for idx, col in oob_matrix.items():
            counts = col.value_counts()
            if len(counts) > 0:
                oob_predictions[idx] = counts.index[counts.argmax()]
        self.oob_error_ = 1 - (oob_predictions == true_labels).mean()
        return self
    
    def predict(self, data):
        """Classifies data by label.  Forest must be fit before it can predict.
            The label is determined by majority vote of the forest's trees.
        
        :param data: a pandas.DataFrame with each row representing an unlabeled observation
        :returns predicted_labels: a pandas.Series of labels for each observation in data
        """
        predictions = pd.DataFrame(
            tree.predict(data)
            for tree in self.trees
        )
        valcounts = [
            col.value_counts()
            for _, col in predictions.items()
        ]
        possible_labels = [counts.index for counts in valcounts]
        predicted_labels = pd.Series(
            (
                labels[counts.argmax()]
                for labels, counts in zip(possible_labels, valcounts)
            ),
            index = data.index
        )
        return predicted_labels