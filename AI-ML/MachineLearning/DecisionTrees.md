# 🌳 Decision Trees

> **Master decision trees: from mathematical foundations to production implementation**

## 🎯 **Learning Objectives**

- Understand decision tree theory and splitting criteria
- Implement decision trees from scratch in Python and Go
- Master pruning techniques to prevent overfitting
- Handle categorical and numerical features
- Build production-ready tree-based systems

## 📚 **Table of Contents**

1. [Mathematical Foundations](#mathematical-foundations)
2. [Implementation from Scratch](#implementation-from-scratch)
3. [Pruning Techniques](#pruning-techniques)
4. [Feature Handling](#feature-handling)
5. [Production Implementation](#production-implementation)
6. [Interview Questions](#interview-questions)

---

## 🧮 **Mathematical Foundations**

### **Decision Tree Theory**

#### **Concept**
Decision trees make decisions by recursively splitting data based on feature values to maximize information gain.

#### **Math Behind**
- **Entropy**: `H(S) = -∑ᵢ pᵢ log₂(pᵢ)`
- **Information Gain**: `IG(S,A) = H(S) - ∑ᵥ |Sᵥ|/|S| H(Sᵥ)`
- **Gini Impurity**: `Gini(S) = 1 - ∑ᵢ pᵢ²`
- **Variance Reduction**: `VR(S,A) = Var(S) - ∑ᵥ |Sᵥ|/|S| Var(Sᵥ)`

#### **Code Example**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                 criterion='gini', random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.random_state = random_state
        self.tree = None
        self.feature_names = None
    
    def _entropy(self, y):
        """Calculate entropy of target variable"""
        if len(y) == 0:
            return 0
        
        # Count class frequencies
        unique, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def _gini_impurity(self, y):
        """Calculate Gini impurity of target variable"""
        if len(y) == 0:
            return 0
        
        # Count class frequencies
        unique, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        
        # Calculate Gini impurity
        gini = 1 - np.sum(probabilities ** 2)
        return gini
    
    def _variance(self, y):
        """Calculate variance of target variable (for regression)"""
        if len(y) == 0:
            return 0
        return np.var(y)
    
    def _information_gain(self, X, y, feature_idx, threshold):
        """Calculate information gain for a split"""
        # Split data
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        y_left = y[left_mask]
        y_right = y[right_mask]
        
        if len(y_left) == 0 or len(y_right) == 0:
            return 0
        
        # Calculate parent impurity
        if self.criterion == 'gini':
            parent_impurity = self._gini_impurity(y)
            left_impurity = self._gini_impurity(y_left)
            right_impurity = self._gini_impurity(y_right)
        elif self.criterion == 'entropy':
            parent_impurity = self._entropy(y)
            left_impurity = self._entropy(y_left)
            right_impurity = self._entropy(y_right)
        elif self.criterion == 'mse':
            parent_impurity = self._variance(y)
            left_impurity = self._variance(y_left)
            right_impurity = self._variance(y_right)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
        
        # Calculate weighted average of child impurities
        n_left = len(y_left)
        n_right = len(y_right)
        n_total = len(y)
        
        weighted_impurity = (n_left / n_total) * left_impurity + (n_right / n_total) * right_impurity
        
        # Information gain
        information_gain = parent_impurity - weighted_impurity
        return information_gain
    
    def _find_best_split(self, X, y):
        """Find the best split for the current node"""
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature_idx in range(n_features):
            # Get unique values for this feature
            feature_values = np.unique(X[:, feature_idx])
            
            # Try different thresholds
            for i in range(len(feature_values) - 1):
                threshold = (feature_values[i] + feature_values[i + 1]) / 2
                gain = self._information_gain(X, y, feature_idx, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _create_leaf(self, y):
        """Create a leaf node"""
        if self.criterion in ['gini', 'entropy']:
            # Classification: return most common class
            unique, counts = np.unique(y, return_counts=True)
            return {'type': 'leaf', 'prediction': unique[np.argmax(counts)]}
        else:
            # Regression: return mean value
            return {'type': 'leaf', 'prediction': np.mean(y)}
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree"""
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           len(y) < self.min_samples_split or \
           len(np.unique(y)) == 1:
            return self._create_leaf(y)
        
        # Find best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)
        
        # If no good split found, create leaf
        if best_gain == 0:
            return self._create_leaf(y)
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        
        # Check minimum samples per leaf
        if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
            return self._create_leaf(y)
        
        # Recursively build left and right subtrees
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)
        
        # Create internal node
        return {
            'type': 'internal',
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def fit(self, X, y):
        """Fit the decision tree"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        self.tree = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x, node):
        """Predict a single sample"""
        if node['type'] == 'leaf':
            return node['prediction']
        
        if x[node['feature']] <= node['threshold']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])
    
    def predict(self, X):
        """Make predictions"""
        if self.tree is None:
            raise ValueError("Tree must be fitted before making predictions")
        
        predictions = []
        for x in X:
            pred = self._predict_sample(x, self.tree)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def score(self, X, y):
        """Calculate score"""
        predictions = self.predict(X)
        if self.criterion in ['gini', 'entropy']:
            return accuracy_score(y, predictions)
        else:
            return -mean_squared_error(y, predictions)  # Negative MSE for consistency

# Example usage
# Classification example
X_class, y_class = make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Train decision tree
tree_classifier = DecisionTree(max_depth=5, min_samples_split=10, criterion='gini')
tree_classifier.fit(X_train, y_train)

# Make predictions
y_pred = tree_classifier.predict(X_test)
accuracy = tree_classifier.score(X_test, y_test)
print(f"Classification Accuracy: {accuracy:.4f}")

# Regression example
X_reg, y_reg = make_regression(n_samples=1000, n_features=5, noise=0.1, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Train decision tree for regression
tree_regressor = DecisionTree(max_depth=5, min_samples_split=10, criterion='mse')
tree_regressor.fit(X_train_reg, y_train_reg)

# Make predictions
y_pred_reg = tree_regressor.predict(X_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
print(f"Regression MSE: {mse:.4f}")
```

---

## ✂️ **Pruning Techniques**

### **Pre-pruning and Post-pruning**

#### **Concept**
Pruning prevents overfitting by removing unnecessary branches from the tree.

#### **Code Example**

```python
class PrunedDecisionTree(DecisionTree):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                 criterion='gini', random_state=None, ccp_alpha=0.0):
        super().__init__(max_depth, min_samples_split, min_samples_leaf, criterion, random_state)
        self.ccp_alpha = ccp_alpha  # Cost complexity pruning parameter
    
    def _calculate_leaf_error(self, y):
        """Calculate error at leaf node"""
        if self.criterion in ['gini', 'entropy']:
            # Classification error
            unique, counts = np.unique(y, return_counts=True)
            total = len(y)
            error = (total - np.max(counts)) / total
        else:
            # Regression error (MSE)
            error = np.var(y)
        return error
    
    def _calculate_subtree_error(self, node, X, y):
        """Calculate error of subtree"""
        if node['type'] == 'leaf':
            return self._calculate_leaf_error(y)
        
        # Split data
        left_mask = X[:, node['feature']] <= node['threshold']
        right_mask = ~left_mask
        
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        
        # Calculate weighted error
        left_error = self._calculate_subtree_error(node['left'], X_left, y_left)
        right_error = self._calculate_subtree_error(node['right'], X_right, y_right)
        
        total_samples = len(y)
        weighted_error = (len(y_left) / total_samples) * left_error + (len(y_right) / total_samples) * right_error
        
        return weighted_error
    
    def _calculate_alpha(self, node, X, y):
        """Calculate alpha value for cost complexity pruning"""
        if node['type'] == 'leaf':
            return float('inf')
        
        # Calculate errors
        leaf_error = self._calculate_leaf_error(y)
        subtree_error = self._calculate_subtree_error(node, X, y)
        
        # Calculate alpha
        n_leaves = self._count_leaves(node)
        alpha = (leaf_error - subtree_error) / (n_leaves - 1)
        
        return alpha
    
    def _count_leaves(self, node):
        """Count number of leaves in subtree"""
        if node['type'] == 'leaf':
            return 1
        return self._count_leaves(node['left']) + self._count_leaves(node['right'])
    
    def _prune_node(self, node, X, y):
        """Prune a node if it improves the tree"""
        if node['type'] == 'leaf':
            return node
        
        # Recursively prune children
        left_mask = X[:, node['feature']] <= node['threshold']
        right_mask = ~left_mask
        
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        
        node['left'] = self._prune_node(node['left'], X_left, y_left)
        node['right'] = self._prune_node(node['right'], X_right, y_right)
        
        # Check if pruning this node improves the tree
        alpha = self._calculate_alpha(node, X, y)
        
        if alpha <= self.ccp_alpha:
            # Prune this node
            return self._create_leaf(y)
        
        return node
    
    def fit(self, X, y):
        """Fit the decision tree with pruning"""
        # First build the full tree
        super().fit(X, y)
        
        # Then apply pruning if ccp_alpha > 0
        if self.ccp_alpha > 0:
            self.tree = self._prune_node(self.tree, X, y)
        
        return self
    
    def get_pruning_path(self, X, y):
        """Get the pruning path for different alpha values"""
        # Build full tree
        self.ccp_alpha = 0
        super().fit(X, y)
        
        pruning_path = []
        current_alpha = 0
        
        while True:
            # Calculate alpha for current tree
            alpha = self._calculate_alpha(self.tree, X, y)
            
            if alpha == float('inf'):
                break
            
            # Add to pruning path
            pruning_path.append({
                'alpha': current_alpha,
                'n_leaves': self._count_leaves(self.tree),
                'error': self._calculate_subtree_error(self.tree, X, y)
            })
            
            # Prune the tree
            current_alpha = alpha
            self.ccp_alpha = alpha
            self.tree = self._prune_node(self.tree, X, y)
        
        return pruning_path

# Example usage
# Test pruning
X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train tree with pruning
pruned_tree = PrunedDecisionTree(max_depth=10, ccp_alpha=0.01)
pruned_tree.fit(X_train, y_train)

# Get pruning path
pruning_path = pruned_tree.get_pruning_path(X_train, y_train)
print("Pruning Path:")
for step in pruning_path:
    print(f"Alpha: {step['alpha']:.4f}, Leaves: {step['n_leaves']}, Error: {step['error']:.4f}")

# Compare with unpruned tree
unpruned_tree = DecisionTree(max_depth=10)
unpruned_tree.fit(X_train, y_train)

pruned_score = pruned_tree.score(X_test, y_test)
unpruned_score = unpruned_tree.score(X_test, y_test)

print(f"\nPruned Tree Accuracy: {pruned_score:.4f}")
print(f"Unpruned Tree Accuracy: {unpruned_score:.4f}")
```

---

## 🎯 **Feature Handling**

### **Categorical and Numerical Features**

#### **Concept**
Handle different types of features in decision trees.

#### **Code Example**

```python
class AdvancedDecisionTree(DecisionTree):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                 criterion='gini', random_state=None, handle_categorical=True):
        super().__init__(max_depth, min_samples_split, min_samples_leaf, criterion, random_state)
        self.handle_categorical = handle_categorical
        self.feature_types = None
        self.categorical_mappings = None
    
    def _detect_feature_types(self, X):
        """Detect feature types (numerical vs categorical)"""
        feature_types = []
        categorical_mappings = {}
        
        for i in range(X.shape[1]):
            unique_values = np.unique(X[:, i])
            
            # Check if feature is categorical
            if len(unique_values) <= 20 and all(isinstance(val, (str, int)) for val in unique_values):
                feature_types.append('categorical')
                categorical_mappings[i] = {val: idx for idx, val in enumerate(unique_values)}
            else:
                feature_types.append('numerical')
        
        return feature_types, categorical_mappings
    
    def _find_best_split_categorical(self, X, y, feature_idx):
        """Find best split for categorical feature"""
        unique_values = np.unique(X[:, feature_idx])
        best_gain = 0
        best_split = None
        
        # Try all possible binary splits
        for i in range(1, len(unique_values)):
            for split in self._generate_binary_splits(unique_values, i):
                left_mask = np.isin(X[:, feature_idx], split)
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                gain = self._information_gain_categorical(X, y, left_mask, right_mask)
                
                if gain > best_gain:
                    best_gain = gain
                    best_split = split
        
        return best_split, best_gain
    
    def _generate_binary_splits(self, values, n_left):
        """Generate all possible binary splits"""
        from itertools import combinations
        
        for split in combinations(values, n_left):
            yield split
    
    def _information_gain_categorical(self, X, y, left_mask, right_mask):
        """Calculate information gain for categorical split"""
        y_left = y[left_mask]
        y_right = y[right_mask]
        
        if len(y_left) == 0 or len(y_right) == 0:
            return 0
        
        # Calculate parent impurity
        if self.criterion == 'gini':
            parent_impurity = self._gini_impurity(y)
            left_impurity = self._gini_impurity(y_left)
            right_impurity = self._gini_impurity(y_right)
        elif self.criterion == 'entropy':
            parent_impurity = self._entropy(y)
            left_impurity = self._entropy(y_left)
            right_impurity = self._entropy(y_right)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
        
        # Calculate weighted average
        n_left = len(y_left)
        n_right = len(y_right)
        n_total = len(y)
        
        weighted_impurity = (n_left / n_total) * left_impurity + (n_right / n_total) * right_impurity
        
        return parent_impurity - weighted_impurity
    
    def _find_best_split(self, X, y):
        """Find the best split considering feature types"""
        best_gain = 0
        best_feature = None
        best_threshold = None
        best_split_type = None
        
        n_features = X.shape[1]
        
        for feature_idx in range(n_features):
            if self.feature_types[feature_idx] == 'categorical':
                # Handle categorical feature
                split, gain = self._find_best_split_categorical(X, y, feature_idx)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = split
                    best_split_type = 'categorical'
            else:
                # Handle numerical feature
                feature_values = np.unique(X[:, feature_idx])
                
                for i in range(len(feature_values) - 1):
                    threshold = (feature_values[i] + feature_values[i + 1]) / 2
                    gain = self._information_gain(X, y, feature_idx, threshold)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature_idx
                        best_threshold = threshold
                        best_split_type = 'numerical'
        
        return best_feature, best_threshold, best_gain, best_split_type
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree with feature type handling"""
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           len(y) < self.min_samples_split or \
           len(np.unique(y)) == 1:
            return self._create_leaf(y)
        
        # Find best split
        best_feature, best_threshold, best_gain, best_split_type = self._find_best_split(X, y)
        
        # If no good split found, create leaf
        if best_gain == 0:
            return self._create_leaf(y)
        
        # Split data based on feature type
        if best_split_type == 'categorical':
            left_mask = np.isin(X[:, best_feature], best_threshold)
            right_mask = ~left_mask
        else:
            left_mask = X[:, best_feature] <= best_threshold
            right_mask = ~left_mask
        
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        
        # Check minimum samples per leaf
        if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
            return self._create_leaf(y)
        
        # Recursively build left and right subtrees
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)
        
        # Create internal node
        return {
            'type': 'internal',
            'feature': best_feature,
            'threshold': best_threshold,
            'split_type': best_split_type,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def fit(self, X, y):
        """Fit the decision tree with feature type detection"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Detect feature types
        if self.handle_categorical:
            self.feature_types, self.categorical_mappings = self._detect_feature_types(X)
        else:
            self.feature_types = ['numerical'] * X.shape[1]
            self.categorical_mappings = {}
        
        self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        self.tree = self._build_tree(X, y)
        return self

# Example usage
# Create mixed data (numerical + categorical)
np.random.seed(42)
n_samples = 1000

# Numerical features
X_numerical = np.random.randn(n_samples, 3)

# Categorical features
X_categorical = np.random.choice(['A', 'B', 'C'], n_samples).reshape(-1, 1)
X_categorical2 = np.random.choice(['X', 'Y'], n_samples).reshape(-1, 1)

# Combine features
X_mixed = np.hstack([X_numerical, X_categorical, X_categorical2])

# Create target based on features
y = (X_numerical[:, 0] > 0).astype(int)
y[X_categorical.flatten() == 'A'] = 1
y[X_categorical2.flatten() == 'X'] = 1

# Train advanced decision tree
advanced_tree = AdvancedDecisionTree(max_depth=5, handle_categorical=True)
advanced_tree.fit(X_mixed, y)

# Make predictions
predictions = advanced_tree.predict(X_mixed)
accuracy = advanced_tree.score(X_mixed, y)
print(f"Advanced Decision Tree Accuracy: {accuracy:.4f}")
```

---

## 🎯 **Interview Questions**

### **Decision Tree Theory**

#### **Q1: What are the advantages and disadvantages of decision trees?**
**Answer**: 
**Advantages**: Easy to interpret, handle both numerical and categorical data, require little data preparation, can model non-linear relationships
**Disadvantages**: Prone to overfitting, unstable (small changes in data can lead to different trees), biased towards features with many levels

#### **Q2: What is the difference between Gini impurity and entropy?**
**Answer**: 
- **Gini Impurity**: `Gini(S) = 1 - ∑ᵢ pᵢ²`, measures probability of misclassification
- **Entropy**: `H(S) = -∑ᵢ pᵢ log₂(pᵢ)`, measures information content
- **Gini** is computationally faster, **Entropy** is more sensitive to small changes
- Both reach maximum when classes are equally distributed

#### **Q3: How do you prevent overfitting in decision trees?**
**Answer**: 
- **Pre-pruning**: Set max_depth, min_samples_split, min_samples_leaf
- **Post-pruning**: Cost complexity pruning, reduced error pruning
- **Regularization**: Penalize tree complexity
- **Cross-validation**: Use validation set to tune hyperparameters

#### **Q4: What is the time complexity of training a decision tree?**
**Answer**: 
- **Best case**: O(n log n) for balanced tree
- **Worst case**: O(n²) for skewed tree
- **Factors**: Number of samples (n), number of features (m), tree depth (d)
- **Overall**: O(n × m × d × log n) in practice

#### **Q5: How do decision trees handle missing values?**
**Answer**: 
- **Surrogate splits**: Use alternative features when primary feature is missing
- **Imputation**: Fill missing values with mean, median, or mode
- **Separate branch**: Create a separate branch for missing values
- **Weighted splits**: Distribute samples proportionally across branches

### **Implementation Questions**

#### **Q6: Implement decision tree from scratch**
**Answer**: See the implementation above with entropy, Gini impurity, and information gain calculations.

#### **Q7: How would you handle categorical features in decision trees?**
**Answer**: 
- **Binary splits**: Try all possible binary combinations of categories
- **Multi-way splits**: Create separate branch for each category
- **Ordinal encoding**: Convert categories to numbers if ordinal relationship exists
- **One-hot encoding**: Convert to binary features

#### **Q8: What is cost complexity pruning and how does it work?**
**Answer**: 
- **Cost Complexity**: `R_α(T) = R(T) + α|T|` where R(T) is error rate and |T| is number of leaves
- **Process**: Start with full tree, prune nodes that minimize cost complexity
- **Alpha parameter**: Controls trade-off between tree size and accuracy
- **Cross-validation**: Use to find optimal alpha value

---

## 🚀 **Next Steps**

1. **Practice**: Implement all variants and test with different datasets
2. **Optimize**: Focus on performance and memory efficiency
3. **Deploy**: Build production systems with monitoring
4. **Extend**: Learn about ensemble methods like Random Forest
5. **Interview**: Practice decision tree interview questions

---

**Ready to learn about ensemble methods? Let's move to Random Forest!** 🎯
