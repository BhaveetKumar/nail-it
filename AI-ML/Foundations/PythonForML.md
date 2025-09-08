# 🐍 Python for Machine Learning

> **Master Python programming for AI/ML: NumPy, Pandas, Scikit-learn, and production-ready code**

## 🎯 **Learning Objectives**

- Master NumPy for numerical computing and array operations
- Learn Pandas for data manipulation and analysis
- Understand Scikit-learn for machine learning workflows
- Build production-ready Python code for ML systems
- Implement efficient data processing pipelines

## 📚 **Table of Contents**

1. [NumPy Fundamentals](#numpy-fundamentals)
2. [Pandas Data Manipulation](#pandas-data-manipulation)
3. [Scikit-learn Workflows](#scikit-learn-workflows)
4. [Data Visualization](#data-visualization)
5. [Performance Optimization](#performance-optimization)
6. [Production Code Patterns](#production-code-patterns)
7. [Interview Questions](#interview-questions)

---

## 🔢 **NumPy Fundamentals**

### **Array Operations and Broadcasting**

#### **Concept**
NumPy provides efficient array operations and broadcasting for numerical computing.

#### **Code Example**

```python
import numpy as np
import time

class NumPyOperations:
    def __init__(self):
        self.arrays = {}
    
    def create_arrays(self):
        """Create various types of arrays"""
        # Basic arrays
        arr1d = np.array([1, 2, 3, 4, 5])
        arr2d = np.array([[1, 2, 3], [4, 5, 6]])
        arr3d = np.random.randn(2, 3, 4)
        
        # Special arrays
        zeros = np.zeros((3, 4))
        ones = np.ones((2, 3))
        identity = np.eye(3)
        range_arr = np.arange(0, 10, 2)
        
        return {
            '1d': arr1d, '2d': arr2d, '3d': arr3d,
            'zeros': zeros, 'ones': ones, 'identity': identity,
            'range': range_arr
        }
    
    def array_operations(self, arr1, arr2):
        """Perform array operations"""
        # Element-wise operations
        addition = arr1 + arr2
        multiplication = arr1 * arr2
        division = arr1 / (arr2 + 1e-8)  # Avoid division by zero
        
        # Matrix operations
        dot_product = np.dot(arr1, arr2)
        matrix_mult = np.matmul(arr1, arr2)
        
        # Statistical operations
        mean = np.mean(arr1)
        std = np.std(arr1)
        max_val = np.max(arr1)
        min_val = np.min(arr1)
        
        return {
            'addition': addition, 'multiplication': multiplication,
            'division': division, 'dot_product': dot_product,
            'matrix_mult': matrix_mult, 'mean': mean, 'std': std,
            'max': max_val, 'min': min_val
        }
    
    def broadcasting_example(self):
        """Demonstrate NumPy broadcasting"""
        # Broadcasting with different shapes
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        scalar = 10
        vector = np.array([1, 2, 3])
        
        # Broadcasting operations
        result1 = arr + scalar  # (2,3) + scalar
        result2 = arr + vector  # (2,3) + (3,)
        result3 = arr * vector  # (2,3) * (3,)
        
        return result1, result2, result3

# Example usage
numpy_ops = NumPyOperations()
arrays = numpy_ops.create_arrays()
print("Arrays created successfully")

# Test array operations
arr1 = np.random.randn(3, 4)
arr2 = np.random.randn(4, 5)
operations = numpy_ops.array_operations(arr1, arr2)
print("Array operations completed")

# Test broadcasting
broadcast_results = numpy_ops.broadcasting_example()
print("Broadcasting examples completed")
```

### **Advanced NumPy Techniques**

```python
class AdvancedNumPy:
    def __init__(self):
        self.techniques = {}
    
    def vectorized_operations(self, data):
        """Demonstrate vectorized operations vs loops"""
        # Vectorized approach (fast)
        start_time = time.time()
        vectorized_result = np.sum(data ** 2, axis=1)
        vectorized_time = time.time() - start_time
        
        # Loop approach (slow)
        start_time = time.time()
        loop_result = []
        for row in data:
            loop_result.append(sum(x ** 2 for x in row))
        loop_result = np.array(loop_result)
        loop_time = time.time() - start_time
        
        return vectorized_result, loop_result, vectorized_time, loop_time
    
    def memory_efficient_operations(self, large_array):
        """Memory-efficient operations"""
        # In-place operations
        large_array += 1  # In-place addition
        large_array *= 2  # In-place multiplication
        
        # Memory mapping for large files
        # memmap_array = np.memmap('large_data.dat', dtype='float32', mode='w+', shape=(1000, 1000))
        
        return large_array
    
    def advanced_indexing(self, arr):
        """Advanced indexing techniques"""
        # Boolean indexing
        mask = arr > 0.5
        positive_values = arr[mask]
        
        # Fancy indexing
        indices = [0, 2, 4]
        selected_values = arr[indices]
        
        # Multi-dimensional indexing
        if arr.ndim > 1:
            row_indices = [0, 1]
            col_indices = [1, 2]
            selected_subset = arr[np.ix_(row_indices, col_indices)]
        
        return positive_values, selected_values, selected_subset if arr.ndim > 1 else None

# Example usage
advanced_numpy = AdvancedNumPy()
data = np.random.randn(1000, 100)
vectorized, looped, v_time, l_time = advanced_numpy.vectorized_operations(data)
print(f"Vectorized time: {v_time:.4f}s, Loop time: {l_time:.4f}s")
print(f"Speedup: {l_time/v_time:.2f}x")
```

---

## 📊 **Pandas Data Manipulation**

### **DataFrame Operations**

#### **Concept**
Pandas provides powerful data structures and operations for data analysis.

#### **Code Example**

```python
import pandas as pd
import numpy as np

class PandasOperations:
    def __init__(self):
        self.dataframes = {}
    
    def create_sample_data(self):
        """Create sample datasets"""
        # Sample DataFrame
        data = {
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 28, 32],
            'salary': [50000, 60000, 70000, 55000, 65000],
            'department': ['IT', 'HR', 'IT', 'Finance', 'IT'],
            'experience': [2, 5, 8, 3, 6]
        }
        df = pd.DataFrame(data)
        
        # Time series data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        ts_data = pd.DataFrame({
            'date': dates,
            'value': np.random.randn(100).cumsum(),
            'category': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        return df, ts_data
    
    def data_manipulation(self, df):
        """Perform data manipulation operations"""
        # Basic operations
        df_filtered = df[df['age'] > 30]
        df_sorted = df.sort_values('salary', ascending=False)
        df_grouped = df.groupby('department').agg({
            'salary': ['mean', 'std', 'count'],
            'age': 'mean'
        })
        
        # Data transformation
        df['salary_category'] = pd.cut(df['salary'], 
                                     bins=[0, 55000, 65000, float('inf')], 
                                     labels=['Low', 'Medium', 'High'])
        
        # Pivot table
        pivot_table = df.pivot_table(values='salary', 
                                   index='department', 
                                   columns='salary_category', 
                                   aggfunc='mean')
        
        return df_filtered, df_sorted, df_grouped, pivot_table
    
    def data_cleaning(self, df):
        """Data cleaning operations"""
        # Handle missing values
        df_cleaned = df.copy()
        df_cleaned = df_cleaned.dropna()  # Remove rows with NaN
        df_cleaned = df_cleaned.fillna(df_cleaned.mean())  # Fill with mean
        
        # Remove duplicates
        df_cleaned = df_cleaned.drop_duplicates()
        
        # Data type conversion
        df_cleaned['age'] = df_cleaned['age'].astype(int)
        df_cleaned['salary'] = df_cleaned['salary'].astype(float)
        
        return df_cleaned
    
    def time_series_operations(self, ts_df):
        """Time series specific operations"""
        # Resampling
        daily_avg = ts_df.resample('D', on='date')['value'].mean()
        weekly_sum = ts_df.resample('W', on='date')['value'].sum()
        
        # Rolling operations
        rolling_mean = ts_df['value'].rolling(window=7).mean()
        rolling_std = ts_df['value'].rolling(window=7).std()
        
        # Time-based filtering
        recent_data = ts_df[ts_df['date'] >= '2023-03-01']
        
        return daily_avg, weekly_sum, rolling_mean, rolling_std, recent_data

# Example usage
pandas_ops = PandasOperations()
df, ts_data = pandas_ops.create_sample_data()
print("Sample data created")

# Data manipulation
filtered, sorted_df, grouped, pivot = pandas_ops.data_manipulation(df)
print("Data manipulation completed")

# Data cleaning
cleaned_df = pandas_ops.data_cleaning(df)
print("Data cleaning completed")

# Time series operations
daily, weekly, rolling_mean, rolling_std, recent = pandas_ops.time_series_operations(ts_data)
print("Time series operations completed")
```

---

## 🤖 **Scikit-learn Workflows**

### **Machine Learning Pipeline**

#### **Concept**
Scikit-learn provides a consistent API for machine learning workflows.

#### **Code Example**

```python
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.pipeline import Pipeline

class MLWorkflow:
    def __init__(self):
        self.models = {}
        self.pipelines = {}
    
    def create_sample_data(self):
        """Create sample datasets for classification and regression"""
        # Classification data
        X_class, y_class = make_classification(
            n_samples=1000, n_features=20, n_classes=2, 
            n_redundant=0, random_state=42
        )
        
        # Regression data
        X_reg, y_reg = make_regression(
            n_samples=1000, n_features=20, noise=0.1, random_state=42
        )
        
        return (X_class, y_class), (X_reg, y_reg)
    
    def classification_pipeline(self, X, y):
        """Complete classification pipeline"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        return pipeline, accuracy, report
    
    def hyperparameter_tuning(self, X, y):
        """Hyperparameter tuning with GridSearchCV"""
        # Define parameter grid
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [10, 20, None],
            'classifier__min_samples_split': [2, 5, 10]
        }
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        # Grid search
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    
    def cross_validation(self, X, y, model):
        """Cross-validation evaluation"""
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        return cv_scores.mean(), cv_scores.std()
    
    def feature_importance(self, model, feature_names=None):
        """Extract feature importance"""
        if hasattr(model.named_steps['classifier'], 'feature_importances_'):
            importance = model.named_steps['classifier'].feature_importances_
            if feature_names:
                return dict(zip(feature_names, importance))
            return importance
        return None

# Example usage
ml_workflow = MLWorkflow()
(X_class, y_class), (X_reg, y_reg) = ml_workflow.create_sample_data()

# Classification pipeline
pipeline, accuracy, report = ml_workflow.classification_pipeline(X_class, y_class)
print(f"Classification accuracy: {accuracy:.3f}")

# Hyperparameter tuning
best_model, best_params, best_score = ml_workflow.hyperparameter_tuning(X_class, y_class)
print(f"Best score: {best_score:.3f}")
print(f"Best parameters: {best_params}")

# Cross-validation
cv_mean, cv_std = ml_workflow.cross_validation(X_class, y_class, best_model)
print(f"CV Score: {cv_mean:.3f} (+/- {cv_std*2:.3f})")
```

---

## 📈 **Data Visualization**

### **Matplotlib and Seaborn**

```python
import matplotlib.pyplot as plt
import seaborn as sns

class DataVisualization:
    def __init__(self):
        self.plots = {}
    
    def create_plots(self, df):
        """Create various types of plots"""
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Histogram
        axes[0, 0].hist(df['age'], bins=20, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Age Distribution')
        axes[0, 0].set_xlabel('Age')
        axes[0, 0].set_ylabel('Frequency')
        
        # Scatter plot
        axes[0, 1].scatter(df['age'], df['salary'], alpha=0.6, color='red')
        axes[0, 1].set_title('Age vs Salary')
        axes[0, 1].set_xlabel('Age')
        axes[0, 1].set_ylabel('Salary')
        
        # Box plot
        df.boxplot(column='salary', by='department', ax=axes[1, 0])
        axes[1, 0].set_title('Salary by Department')
        
        # Correlation heatmap
        correlation_matrix = df[['age', 'salary', 'experience']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=axes[1, 1])
        axes[1, 1].set_title('Correlation Matrix')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def time_series_plot(self, ts_df):
        """Create time series plots"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Time series plot
        axes[0].plot(ts_df['date'], ts_df['value'], linewidth=2)
        axes[0].set_title('Time Series Data')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Value')
        axes[0].grid(True, alpha=0.3)
        
        # Rolling average
        rolling_mean = ts_df['value'].rolling(window=7).mean()
        axes[1].plot(ts_df['date'], ts_df['value'], alpha=0.3, label='Original')
        axes[1].plot(ts_df['date'], rolling_mean, linewidth=2, label='7-day Rolling Average')
        axes[1].set_title('Time Series with Rolling Average')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Value')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig

# Example usage
viz = DataVisualization()
df, ts_data = pandas_ops.create_sample_data()

# Create plots
fig1 = viz.create_plots(df)
fig2 = viz.time_series_plot(ts_data)
print("Visualizations created successfully")
```

---

## ⚡ **Performance Optimization**

### **Code Optimization Techniques**

```python
import numba
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

class PerformanceOptimization:
    def __init__(self):
        self.optimized_functions = {}
    
    def numba_optimization(self, data):
        """Use Numba for JIT compilation"""
        @numba.jit(nopython=True)
        def fast_sum_squares(arr):
            result = 0.0
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    result += arr[i, j] ** 2
            return result
        
        return fast_sum_squares(data)
    
    def parallel_processing(self, data, func):
        """Parallel processing with multiprocessing"""
        num_processes = mp.cpu_count()
        
        # Split data
        chunk_size = len(data) // num_processes
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        # Process in parallel
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(executor.map(func, chunks))
        
        return results
    
    def memory_efficient_processing(self, large_data):
        """Memory-efficient processing techniques"""
        # Generator for large datasets
        def data_generator(data, chunk_size=1000):
            for i in range(0, len(data), chunk_size):
                yield data[i:i + chunk_size]
        
        # Process in chunks
        results = []
        for chunk in data_generator(large_data):
            chunk_result = np.sum(chunk, axis=1)
            results.append(chunk_result)
        
        return np.concatenate(results)
    
    def vectorized_operations(self, data):
        """Vectorized operations for better performance"""
        # Vectorized operations are much faster than loops
        start_time = time.time()
        
        # Vectorized approach
        result = np.sum(data ** 2, axis=1)
        
        end_time = time.time()
        return result, end_time - start_time

# Example usage
perf_opt = PerformanceOptimization()
large_data = np.random.randn(10000, 100)

# Test Numba optimization
numba_result = perf_opt.numba_optimization(large_data)
print(f"Numba result: {numba_result}")

# Test vectorized operations
vectorized_result, time_taken = perf_opt.vectorized_operations(large_data)
print(f"Vectorized operation time: {time_taken:.4f}s")
```

---

## 🎯 **Interview Questions**

### **Python and ML Libraries**

#### **Q1: Explain the difference between NumPy arrays and Python lists**
**Answer**: NumPy arrays are homogeneous (same data type), stored in contiguous memory, and optimized for numerical operations. Python lists are heterogeneous, stored as pointers to objects, and slower for numerical computations. NumPy arrays support vectorized operations and broadcasting.

#### **Q2: How do you handle missing data in Pandas?**
**Answer**: Use `dropna()` to remove missing values, `fillna()` to fill with specific values (mean, median, mode), or `interpolate()` for time series data. Choose based on data characteristics and business requirements.

#### **Q3: What is the difference between fit() and transform() in Scikit-learn?**
**Answer**: `fit()` learns parameters from training data (e.g., mean and std for StandardScaler), while `transform()` applies the learned parameters to data. `fit_transform()` does both in one step.

#### **Q4: How do you optimize Python code for ML workloads?**
**Answer**: Use vectorized operations with NumPy, JIT compilation with Numba, parallel processing with multiprocessing, memory-efficient data structures, and profiling tools to identify bottlenecks.

#### **Q5: Explain the concept of broadcasting in NumPy**
**Answer**: Broadcasting allows NumPy to perform operations on arrays of different shapes by automatically expanding smaller arrays to match larger ones, following specific rules for dimension compatibility.

---

## 🚀 **Next Steps**

1. **Practice**: Implement all examples and experiment with different datasets
2. **Optimize**: Focus on performance and memory efficiency
3. **Build**: Create end-to-end ML pipelines
4. **Deploy**: Learn production deployment patterns
5. **Interview**: Practice Python ML interview questions

---

**Ready to dive into machine learning algorithms? Let's move to the Machine Learning section!** 🎯
