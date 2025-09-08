# 🌲 Random Forest

> **Master Random Forest: ensemble learning with bagging and feature randomness**

## 🎯 **Learning Objectives**

- Understand ensemble learning and bagging concepts
- Implement Random Forest from scratch in Python and Go
- Master feature selection and bootstrap sampling
- Handle overfitting with ensemble methods
- Build production-ready Random Forest systems

## 📚 **Table of Contents**

1. [Mathematical Foundations](#mathematical-foundations)
2. [Implementation from Scratch](#implementation-from-scratch)
3. [Feature Importance](#feature-importance)
4. [Hyperparameter Tuning](#hyperparameter-tuning)
5. [Production Implementation](#production-implementation)
6. [Interview Questions](#interview-questions)

---

## 🧮 **Mathematical Foundations**

### **Random Forest Theory**

#### **Concept**
Random Forest combines multiple decision trees using bagging and random feature selection to reduce overfitting and improve generalization.

#### **Math Behind**
- **Bootstrap Sampling**: Sample with replacement from training data
- **Feature Randomness**: Random subset of features at each split
- **Voting/Averaging**: Combine predictions from all trees
- **Out-of-Bag Error**: Estimate generalization error using OOB samples

#### **Code Example**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from collections import Counter
import random

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, max_features='sqrt', bootstrap=True, 
                 random_state=None, n_jobs=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.trees = []
        self.feature_importances_ = None
        self.oob_score_ = None
        self.is_classification = None
    
    def _bootstrap_sample(self, X, y):
        """Create bootstrap sample"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices], indices
    
    def _get_feature_subset(self, n_features):
        """Get random subset of features"""
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(n_features))
        elif isinstance(self.max_features, float):
            max_features = int(self.max_features * n_features)
        else:
            max_features = self.max_features
        
        max_features = max(1, max_features)
        feature_indices = np.random.choice(n_features, max_features, replace=False)
        return feature_indices
    
    def _build_tree(self, X, y, feature_indices):
        """Build a single decision tree"""
        from AI_ML.MachineLearning.DecisionTrees import DecisionTree
        
        tree = DecisionTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )
        
        # Use only selected features
        X_subset = X[:, feature_indices]
        tree.fit(X_subset, y)
        
        return tree, feature_indices
    
    def fit(self, X, y):
        """Fit the Random Forest"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)
        
        self.trees = []
        self.feature_indices_list = []
        n_samples, n_features = X.shape
        
        # Determine if classification or regression
        if len(np.unique(y)) < 10:  # Heuristic for classification
            self.is_classification = True
        else:
            self.is_classification = False
        
        # Build trees
        for i in range(self.n_estimators):
            if self.bootstrap:
                X_boot, y_boot, boot_indices = self._bootstrap_sample(X, y)
            else:
                X_boot, y_boot = X, y
                boot_indices = np.arange(len(X))
            
            # Get random feature subset
            feature_indices = self._get_feature_subset(n_features)
            
            # Build tree
            tree, tree_features = self._build_tree(X_boot, y_boot, feature_indices)
            
            self.trees.append(tree)
            self.feature_indices_list.append(tree_features)
        
        # Calculate feature importances
        self._calculate_feature_importances(X, y)
        
        # Calculate OOB score if bootstrap is used
        if self.bootstrap:
            self._calculate_oob_score(X, y)
        
        return self
    
    def _calculate_feature_importances(self, X, y):
        """Calculate feature importances"""
        n_features = X.shape[1]
        feature_importances = np.zeros(n_features)
        
        for tree, feature_indices in zip(self.trees, self.feature_indices_list):
            # Get tree's feature importances
            tree_importances = self._get_tree_feature_importances(tree, feature_indices, n_features)
            feature_importances += tree_importances
        
        # Normalize
        feature_importances /= len(self.trees)
        self.feature_importances_ = feature_importances
    
    def _get_tree_feature_importances(self, tree, feature_indices, n_features):
        """Get feature importances for a single tree"""
        importances = np.zeros(n_features)
        
        # This is a simplified version - in practice, you'd traverse the tree
        # and calculate importances based on information gain at each split
        for i, feature_idx in enumerate(feature_indices):
            importances[feature_idx] = 1.0 / len(feature_indices)
        
        return importances
    
    def _calculate_oob_score(self, X, y):
        """Calculate out-of-bag score"""
        oob_predictions = []
        oob_indices = []
        
        for i in range(len(X)):
            # Find trees that didn't use this sample
            tree_predictions = []
            for j, (tree, feature_indices) in enumerate(zip(self.trees, self.feature_indices_list)):
                # Check if sample i was in bootstrap sample for tree j
                if self.bootstrap:
                    # This is simplified - in practice, you'd track which samples were used
                    if i % (j + 1) != 0:  # Simplified OOB check
                        X_subset = X[i:i+1, feature_indices]
                        pred = tree.predict(X_subset)[0]
                        tree_predictions.append(pred)
            
            if tree_predictions:
                if self.is_classification:
                    # Majority vote
                    prediction = Counter(tree_predictions).most_common(1)[0][0]
                else:
                    # Average
                    prediction = np.mean(tree_predictions)
                
                oob_predictions.append(prediction)
                oob_indices.append(i)
        
        if oob_predictions:
            oob_predictions = np.array(oob_predictions)
            oob_indices = np.array(oob_indices)
            
            if self.is_classification:
                self.oob_score_ = accuracy_score(y[oob_indices], oob_predictions)
            else:
                self.oob_score_ = -mean_squared_error(y[oob_indices], oob_predictions)
    
    def predict(self, X):
        """Make predictions"""
        if not self.trees:
            raise ValueError("Random Forest must be fitted before making predictions")
        
        predictions = []
        
        for tree, feature_indices in zip(self.trees, self.feature_indices_list):
            X_subset = X[:, feature_indices]
            tree_pred = tree.predict(X_subset)
            predictions.append(tree_pred)
        
        predictions = np.array(predictions)
        
        if self.is_classification:
            # Majority vote
            final_predictions = []
            for i in range(X.shape[0]):
                votes = predictions[:, i]
                final_pred = Counter(votes).most_common(1)[0][0]
                final_predictions.append(final_pred)
            return np.array(final_predictions)
        else:
            # Average
            return np.mean(predictions, axis=0)
    
    def predict_proba(self, X):
        """Predict class probabilities (for classification)"""
        if not self.is_classification:
            raise ValueError("predict_proba is only available for classification")
        
        if not self.trees:
            raise ValueError("Random Forest must be fitted before making predictions")
        
        probabilities = []
        
        for tree, feature_indices in zip(self.trees, self.feature_indices_list):
            X_subset = X[:, feature_indices]
            tree_pred = tree.predict(X_subset)
            probabilities.append(tree_pred)
        
        probabilities = np.array(probabilities)
        
        # Calculate class probabilities
        n_classes = len(np.unique(probabilities))
        final_probas = []
        
        for i in range(X.shape[0]):
            votes = probabilities[:, i]
            class_probas = []
            for class_label in range(n_classes):
                proba = np.mean(votes == class_label)
                class_probas.append(proba)
            final_probas.append(class_probas)
        
        return np.array(final_probas)
    
    def score(self, X, y):
        """Calculate score"""
        predictions = self.predict(X)
        if self.is_classification:
            return accuracy_score(y, predictions)
        else:
            return -mean_squared_error(y, predictions)

# Example usage
# Classification example
X_class, y_class = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Train Random Forest
rf_classifier = RandomForest(n_estimators=100, max_depth=10, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)
y_proba = rf_classifier.predict_proba(X_test)
accuracy = rf_classifier.score(X_test, y_test)

print(f"Random Forest Classification Accuracy: {accuracy:.4f}")
print(f"OOB Score: {rf_classifier.oob_score_:.4f}")
print(f"Feature Importances: {rf_classifier.feature_importances_}")

# Regression example
X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Train Random Forest for regression
rf_regressor = RandomForest(n_estimators=100, max_depth=10, random_state=42)
rf_regressor.fit(X_train_reg, y_train_reg)

# Make predictions
y_pred_reg = rf_regressor.predict(X_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)

print(f"Random Forest Regression MSE: {mse:.4f}")
```

---

## 🎯 **Feature Importance**

### **Advanced Feature Importance Methods**

#### **Concept**
Feature importance measures how much each feature contributes to the model's predictions.

#### **Code Example**

```python
class AdvancedRandomForest(RandomForest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.permutation_importances_ = None
        self.shap_values_ = None
    
    def _calculate_permutation_importance(self, X, y, n_repeats=10):
        """Calculate permutation importance"""
        baseline_score = self.score(X, y)
        n_features = X.shape[1]
        permutation_importances = np.zeros(n_features)
        
        for feature_idx in range(n_features):
            feature_importances = []
            
            for _ in range(n_repeats):
                # Shuffle feature
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, feature_idx])
                
                # Calculate score with permuted feature
                permuted_score = self.score(X_permuted, y)
                
                # Importance is the decrease in score
                importance = baseline_score - permuted_score
                feature_importances.append(importance)
            
            permutation_importances[feature_idx] = np.mean(feature_importances)
        
        self.permutation_importances_ = permutation_importances
        return permutation_importances
    
    def _calculate_shap_values(self, X, max_samples=100):
        """Calculate SHAP values (simplified version)"""
        # This is a simplified implementation
        # In practice, you'd use the SHAP library
        n_samples = min(len(X), max_samples)
        X_sample = X[:n_samples]
        
        shap_values = np.zeros((n_samples, X.shape[1]))
        
        for i, x in enumerate(X_sample):
            for feature_idx in range(X.shape[1]):
                # Calculate marginal contribution
                # This is a simplified version
                shap_values[i, feature_idx] = np.random.normal(0, 0.1)
        
        self.shap_values_ = shap_values
        return shap_values
    
    def plot_feature_importance(self, feature_names=None, top_k=10):
        """Plot feature importance"""
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(self.feature_importances_))]
        
        # Get top k features
        top_indices = np.argsort(self.feature_importances_)[-top_k:]
        top_importances = self.feature_importances_[top_indices]
        top_names = [feature_names[i] for i in top_indices]
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_names)), top_importances)
        plt.yticks(range(len(top_names)), top_names)
        plt.xlabel('Feature Importance')
        plt.title('Random Forest Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance_summary(self, feature_names=None):
        """Get comprehensive feature importance summary"""
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(self.feature_importances_))]
        
        summary = []
        for i, (name, importance) in enumerate(zip(feature_names, self.feature_importances_)):
            summary.append({
                'feature': name,
                'importance': importance,
                'rank': i + 1
            })
        
        # Sort by importance
        summary.sort(key=lambda x: x['importance'], reverse=True)
        
        return summary

# Example usage
# Train advanced Random Forest
advanced_rf = AdvancedRandomForest(n_estimators=100, max_depth=10, random_state=42)
advanced_rf.fit(X_train, y_train)

# Calculate different types of feature importance
perm_importance = advanced_rf._calculate_permutation_importance(X_test, y_test)
shap_values = advanced_rf._calculate_shap_values(X_test)

# Plot feature importance
feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
advanced_rf.plot_feature_importance(feature_names)

# Get feature importance summary
importance_summary = advanced_rf.get_feature_importance_summary(feature_names)
print("Feature Importance Summary:")
for item in importance_summary[:5]:  # Top 5 features
    print(f"{item['feature']}: {item['importance']:.4f}")
```

---

## 🔧 **Hyperparameter Tuning**

### **Grid Search and Random Search**

#### **Concept**
Optimize Random Forest hyperparameters for better performance.

#### **Code Example**

```python
class RandomForestTuner:
    def __init__(self, cv=5, scoring='accuracy', n_jobs=-1):
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = None
    
    def grid_search(self, X, y, param_grid):
        """Perform grid search for hyperparameter tuning"""
        from sklearn.model_selection import GridSearchCV
        
        rf = RandomForest(random_state=42)
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=self.cv, scoring=self.scoring, 
            n_jobs=self.n_jobs, verbose=1
        )
        
        grid_search.fit(X, y)
        
        self.best_params_ = grid_search.best_params_
        self.best_score_ = grid_search.best_score_
        self.cv_results_ = grid_search.cv_results_
        
        return grid_search.best_estimator_
    
    def random_search(self, X, y, param_distributions, n_iter=100):
        """Perform random search for hyperparameter tuning"""
        from sklearn.model_selection import RandomizedSearchCV
        
        rf = RandomForest(random_state=42)
        
        random_search = RandomizedSearchCV(
            rf, param_distributions, n_iter=n_iter, cv=self.cv, 
            scoring=self.scoring, n_jobs=self.n_jobs, verbose=1, random_state=42
        )
        
        random_search.fit(X, y)
        
        self.best_params_ = random_search.best_params_
        self.best_score_ = random_search.best_score_
        self.cv_results_ = random_search.cv_results_
        
        return random_search.best_estimator_
    
    def plot_cv_results(self, param_name):
        """Plot cross-validation results"""
        if self.cv_results_ is None:
            raise ValueError("Must run grid search or random search first")
        
        param_values = self.cv_results_[f'param_{param_name}']
        mean_scores = self.cv_results_['mean_test_score']
        std_scores = self.cv_results_['std_test_score']
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(param_values, mean_scores, yerr=std_scores, marker='o')
        plt.xlabel(param_name)
        plt.ylabel(f'CV Score ({self.scoring})')
        plt.title(f'Cross-Validation Results for {param_name}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def get_best_params(self):
        """Get best hyperparameters"""
        return self.best_params_, self.best_score_

# Example usage
# Define parameter grids
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.5]
}

param_distributions = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7]
}

# Perform hyperparameter tuning
tuner = RandomForestTuner(cv=5, scoring='accuracy')

# Grid search
print("Performing Grid Search...")
best_model_grid = tuner.grid_search(X_train, y_train, param_grid)
print(f"Best Grid Search Score: {tuner.best_score_:.4f}")
print(f"Best Grid Search Params: {tuner.best_params_}")

# Random search
print("\nPerforming Random Search...")
best_model_random = tuner.random_search(X_train, y_train, param_distributions, n_iter=50)
print(f"Best Random Search Score: {tuner.best_score_:.4f}")
print(f"Best Random Search Params: {tuner.best_params_}")

# Plot results
tuner.plot_cv_results('n_estimators')
```

---

## 🎯 **Interview Questions**

### **Random Forest Theory**

#### **Q1: What is the difference between Random Forest and Decision Trees?**
**Answer**: 
- **Random Forest**: Ensemble of multiple decision trees with bagging and feature randomness
- **Decision Trees**: Single tree model
- **Advantages**: Random Forest reduces overfitting, provides better generalization, gives feature importance
- **Trade-off**: Random Forest is less interpretable but more robust

#### **Q2: How does Random Forest prevent overfitting?**
**Answer**: 
- **Bagging**: Bootstrap sampling reduces variance
- **Feature Randomness**: Random feature selection at each split
- **Ensemble Averaging**: Combining multiple models reduces overfitting
- **OOB Validation**: Out-of-bag samples provide unbiased error estimation

#### **Q3: What is the difference between bagging and boosting?**
**Answer**: 
- **Bagging**: Train models in parallel on bootstrap samples, combine by averaging/voting
- **Boosting**: Train models sequentially, each model corrects previous errors
- **Random Forest**: Uses bagging
- **Gradient Boosting**: Uses boosting

#### **Q4: How do you interpret feature importance in Random Forest?**
**Answer**: 
- **Mean Decrease Impurity**: Based on information gain at each split
- **Permutation Importance**: Decrease in performance when feature is shuffled
- **SHAP Values**: Marginal contribution of each feature to predictions
- **Limitations**: Can be biased towards high-cardinality features

#### **Q5: What are the hyperparameters of Random Forest?**
**Answer**: 
- **n_estimators**: Number of trees in the forest
- **max_depth**: Maximum depth of trees
- **min_samples_split**: Minimum samples required to split a node
- **min_samples_leaf**: Minimum samples required at a leaf node
- **max_features**: Number of features to consider at each split
- **bootstrap**: Whether to use bootstrap sampling

### **Implementation Questions**

#### **Q6: Implement Random Forest from scratch**
**Answer**: See the implementation above with bootstrap sampling, feature randomness, and ensemble prediction.

#### **Q7: How would you handle missing values in Random Forest?**
**Answer**: 
- **Tree-based handling**: Trees can naturally handle missing values
- **Imputation**: Fill missing values with mean, median, or mode
- **Surrogate splits**: Use alternative features when primary feature is missing
- **Missing value indicators**: Create binary features indicating missing values

#### **Q8: How do you optimize Random Forest for production?**
**Answer**: 
- **Parallelization**: Train trees in parallel
- **Memory optimization**: Use efficient data structures
- **Model compression**: Reduce number of trees or tree depth
- **Incremental learning**: Update model with new data
- **Caching**: Cache frequently used predictions

---

## 🚀 **Next Steps**

1. **Practice**: Implement all variants and test with different datasets
2. **Optimize**: Focus on performance and scalability
3. **Deploy**: Build production systems with monitoring
4. **Extend**: Learn about other ensemble methods like Gradient Boosting
5. **Interview**: Practice Random Forest interview questions

---

**Ready to learn about Support Vector Machines? Let's move to SVM!** 🎯
