# 📊 Probability and Statistics for ML

> **Essential probability theory and statistics for machine learning and data science**

## 🎯 **Learning Objectives**

- Master probability theory fundamentals
- Understand statistical inference and hypothesis testing
- Learn Bayesian statistics and applications
- Implement statistical methods in Python
- Apply statistics to ML model evaluation and selection

## 📚 **Table of Contents**

1. [Probability Fundamentals](#probability-fundamentals)
2. [Statistical Distributions](#statistical-distributions)
3. [Hypothesis Testing](#hypothesis-testing)
4. [Bayesian Statistics](#bayesian-statistics)
5. [Statistical Learning Theory](#statistical-learning-theory)
6. [Implementation Examples](#implementation-examples)
7. [Interview Questions](#interview-questions)

---

## 🎲 **Probability Fundamentals**

### **Basic Probability Concepts**

#### **Concept**
Probability measures the likelihood of events occurring, fundamental for understanding uncertainty in ML.

#### **Math Behind**
- **Probability**: `P(A) = |A|/|S|` where S is sample space
- **Conditional Probability**: `P(A|B) = P(A∩B)/P(B)`
- **Bayes' Theorem**: `P(A|B) = P(B|A)P(A)/P(B)`
- **Independence**: `P(A∩B) = P(A)P(B)`

#### **Code Example**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, binom, poisson

class ProbabilityBasics:
    def __init__(self):
        self.probabilities = {}
    
    def basic_probability(self, outcomes, favorable_outcomes):
        """Calculate basic probability"""
        return len(favorable_outcomes) / len(outcomes)
    
    def conditional_probability(self, p_a_and_b, p_b):
        """Calculate conditional probability P(A|B)"""
        return p_a_and_b / p_b if p_b != 0 else 0
    
    def bayes_theorem(self, p_b_given_a, p_a, p_b):
        """Apply Bayes' theorem"""
        return (p_b_given_a * p_a) / p_b
    
    def independence_test(self, p_a, p_b, p_a_and_b):
        """Test if events A and B are independent"""
        expected = p_a * p_b
        return abs(p_a_and_b - expected) < 1e-6
    
    def law_of_total_probability(self, conditional_probs, marginal_probs):
        """Apply law of total probability"""
        return sum(p * q for p, q in zip(conditional_probs, marginal_probs))

# Example usage
prob_basics = ProbabilityBasics()

# Basic probability example
outcomes = [1, 2, 3, 4, 5, 6]
favorable = [2, 4, 6]  # Even numbers
prob_even = prob_basics.basic_probability(outcomes, favorable)
print(f"Probability of even number: {prob_even}")

# Bayes' theorem example
p_b_given_a = 0.9  # P(positive test | disease)
p_a = 0.01  # P(disease)
p_b = 0.1  # P(positive test)
p_a_given_b = prob_basics.bayes_theorem(p_b_given_a, p_a, p_b)
print(f"P(disease | positive test): {p_a_given_b:.3f}")
```

---

## 📈 **Statistical Distributions**

### **Common Distributions**

#### **Concept**
Statistical distributions model the probability of different outcomes in data.

#### **Code Example**

```python
class StatisticalDistributions:
    def __init__(self):
        self.distributions = {}
    
    def normal_distribution(self, mean=0, std=1, size=1000):
        """Generate and analyze normal distribution"""
        data = np.random.normal(mean, std, size)
        
        # Calculate statistics
        sample_mean = np.mean(data)
        sample_std = np.std(data)
        
        # Probability density function
        x = np.linspace(mean - 4*std, mean + 4*std, 100)
        pdf = norm.pdf(x, mean, std)
        
        return data, sample_mean, sample_std, x, pdf
    
    def binomial_distribution(self, n=10, p=0.5, size=1000):
        """Generate and analyze binomial distribution"""
        data = np.random.binomial(n, p, size)
        
        # Calculate statistics
        mean = n * p
        variance = n * p * (1 - p)
        
        # Probability mass function
        x = np.arange(0, n+1)
        pmf = binom.pmf(x, n, p)
        
        return data, mean, variance, x, pmf
    
    def poisson_distribution(self, lambda_param=3, size=1000):
        """Generate and analyze Poisson distribution"""
        data = np.random.poisson(lambda_param, size)
        
        # Calculate statistics
        mean = lambda_param
        variance = lambda_param
        
        # Probability mass function
        x = np.arange(0, 20)
        pmf = poisson.pmf(x, lambda_param)
        
        return data, mean, variance, x, pmf
    
    def distribution_comparison(self, data1, data2):
        """Compare two distributions"""
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(data1, data2)
        
        # Mann-Whitney U test
        mw_stat, mw_pvalue = stats.mannwhitneyu(data1, data2)
        
        return {
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'mw_statistic': mw_stat,
            'mw_pvalue': mw_pvalue
        }

# Example usage
dist_stats = StatisticalDistributions()

# Normal distribution
normal_data, mean, std, x, pdf = dist_stats.normal_distribution()
print(f"Normal distribution - Mean: {mean:.3f}, Std: {std:.3f}")

# Binomial distribution
binom_data, mean, variance, x, pmf = dist_stats.binomial_distribution()
print(f"Binomial distribution - Mean: {mean}, Variance: {variance}")

# Poisson distribution
poisson_data, mean, variance, x, pmf = dist_stats.poisson_distribution()
print(f"Poisson distribution - Mean: {mean}, Variance: {variance}")
```

---

## 🧪 **Hypothesis Testing**

### **Statistical Tests**

#### **Concept**
Hypothesis testing helps make decisions about population parameters based on sample data.

#### **Code Example**

```python
class HypothesisTesting:
    def __init__(self):
        self.tests = {}
    
    def t_test(self, sample1, sample2=None, alpha=0.05):
        """Perform t-test"""
        if sample2 is None:
            # One-sample t-test
            t_stat, p_value = stats.ttest_1samp(sample1, 0)
            test_type = "One-sample t-test"
        else:
            # Two-sample t-test
            t_stat, p_value = stats.ttest_ind(sample1, sample2)
            test_type = "Two-sample t-test"
        
        # Decision
        decision = "Reject H0" if p_value < alpha else "Fail to reject H0"
        
        return {
            'test_type': test_type,
            't_statistic': t_stat,
            'p_value': p_value,
            'alpha': alpha,
            'decision': decision
        }
    
    def chi_square_test(self, observed, expected=None):
        """Perform chi-square test"""
        if expected is None:
            # Goodness of fit test
            chi2_stat, p_value = stats.chisquare(observed)
            test_type = "Chi-square goodness of fit"
        else:
            # Test of independence
            chi2_stat, p_value = stats.chisquare(observed, expected)
            test_type = "Chi-square test of independence"
        
        return {
            'test_type': test_type,
            'chi2_statistic': chi2_stat,
            'p_value': p_value
        }
    
    def anova_test(self, groups):
        """Perform ANOVA test"""
        f_stat, p_value = stats.f_oneway(*groups)
        
        return {
            'test_type': "One-way ANOVA",
            'f_statistic': f_stat,
            'p_value': p_value
        }
    
    def power_analysis(self, effect_size, sample_size, alpha=0.05):
        """Power analysis for sample size determination"""
        from statsmodels.stats.power import ttest_power
        
        power = ttest_power(effect_size, sample_size, alpha)
        
        return {
            'effect_size': effect_size,
            'sample_size': sample_size,
            'alpha': alpha,
            'power': power
        }

# Example usage
hyp_test = HypothesisTesting()

# Generate sample data
sample1 = np.random.normal(100, 15, 30)
sample2 = np.random.normal(105, 15, 30)

# t-test
t_test_result = hyp_test.t_test(sample1, sample2)
print(f"t-test result: {t_test_result}")

# ANOVA test
group1 = np.random.normal(100, 15, 20)
group2 = np.random.normal(105, 15, 20)
group3 = np.random.normal(110, 15, 20)
anova_result = hyp_test.anova_test([group1, group2, group3])
print(f"ANOVA result: {anova_result}")
```

---

## 🔮 **Bayesian Statistics**

### **Bayesian Inference**

#### **Concept**
Bayesian statistics updates beliefs about parameters using prior knowledge and observed data.

#### **Code Example**

```python
class BayesianStatistics:
    def __init__(self):
        self.priors = {}
        self.posteriors = {}
    
    def bayesian_update(self, prior, likelihood, evidence):
        """Update prior with likelihood to get posterior"""
        posterior = (likelihood * prior) / evidence
        return posterior
    
    def beta_binomial_conjugate(self, alpha, beta, successes, trials):
        """Beta-binomial conjugate prior update"""
        # Prior: Beta(alpha, beta)
        # Likelihood: Binomial(successes, trials)
        # Posterior: Beta(alpha + successes, beta + trials - successes)
        
        posterior_alpha = alpha + successes
        posterior_beta = beta + trials - successes
        
        return posterior_alpha, posterior_beta
    
    def normal_normal_conjugate(self, prior_mean, prior_var, data_mean, data_var, n):
        """Normal-normal conjugate prior update"""
        # Prior: N(prior_mean, prior_var)
        # Likelihood: N(data_mean, data_var/n)
        # Posterior: N(posterior_mean, posterior_var)
        
        posterior_var = 1 / (1/prior_var + n/data_var)
        posterior_mean = posterior_var * (prior_mean/prior_var + n*data_mean/data_var)
        
        return posterior_mean, posterior_var
    
    def markov_chain_monte_carlo(self, target_distribution, n_samples=10000):
        """Simple MCMC implementation"""
        samples = []
        current = 0.0
        
        for i in range(n_samples):
            # Propose new sample
            proposal = current + np.random.normal(0, 1)
            
            # Calculate acceptance probability
            acceptance_prob = min(1, target_distribution(proposal) / target_distribution(current))
            
            # Accept or reject
            if np.random.random() < acceptance_prob:
                current = proposal
            
            samples.append(current)
        
        return samples

# Example usage
bayesian = BayesianStatistics()

# Beta-binomial example
alpha, beta = 1, 1  # Uniform prior
successes, trials = 8, 10
post_alpha, post_beta = bayesian.beta_binomial_conjugate(alpha, beta, successes, trials)
print(f"Posterior: Beta({post_alpha}, {post_beta})")

# Normal-normal example
prior_mean, prior_var = 0, 1
data_mean, data_var, n = 2, 1, 10
post_mean, post_var = bayesian.normal_normal_conjugate(prior_mean, prior_var, data_mean, data_var, n)
print(f"Posterior: N({post_mean:.3f}, {post_var:.3f})")
```

---

## 📊 **Statistical Learning Theory**

### **Model Evaluation and Selection**

#### **Concept**
Statistical learning theory provides theoretical foundations for ML model performance.

#### **Code Example**

```python
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

class StatisticalLearning:
    def __init__(self):
        self.models = {}
        self.evaluations = {}
    
    def bias_variance_decomposition(self, X, y, model, n_bootstrap=100):
        """Estimate bias-variance decomposition"""
        n_samples = len(X)
        predictions = []
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Train model
            model.fit(X_boot, y_boot)
            
            # Predict on original data
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate bias and variance
        mean_pred = np.mean(predictions, axis=0)
        bias = np.mean((mean_pred - y) ** 2)
        variance = np.mean(np.var(predictions, axis=0))
        
        return bias, variance, bias + variance
    
    def learning_curve_analysis(self, X, y, model, train_sizes):
        """Analyze learning curves"""
        train_scores = []
        val_scores = []
        
        for size in train_sizes:
            # Train on subset
            X_subset = X[:size]
            y_subset = y[:size]
            
            # Cross-validation
            train_score = cross_val_score(model, X_subset, y_subset, cv=5).mean()
            val_score = cross_val_score(model, X, y, cv=5).mean()
            
            train_scores.append(train_score)
            val_scores.append(val_score)
        
        return train_scores, val_scores
    
    def model_comparison(self, X, y, models):
        """Compare multiple models statistically"""
        results = {}
        
        for name, model in models.items():
            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=5)
            
            # Statistical test
            mean_score = cv_scores.mean()
            std_score = cv_scores.std()
            
            results[name] = {
                'mean_score': mean_score,
                'std_score': std_score,
                'cv_scores': cv_scores
            }
        
        return results
    
    def confidence_intervals(self, scores, confidence=0.95):
        """Calculate confidence intervals"""
        alpha = 1 - confidence
        n = len(scores)
        
        # t-distribution critical value
        t_critical = stats.t.ppf(1 - alpha/2, n-1)
        
        # Standard error
        se = np.std(scores) / np.sqrt(n)
        
        # Confidence interval
        margin_error = t_critical * se
        ci_lower = np.mean(scores) - margin_error
        ci_upper = np.mean(scores) + margin_error
        
        return ci_lower, ci_upper

# Example usage
stat_learning = StatisticalLearning()

# Generate sample data
X = np.random.randn(100, 5)
y = X @ np.random.randn(5) + np.random.randn(100) * 0.1

# Bias-variance decomposition
model = LinearRegression()
bias, variance, total_error = stat_learning.bias_variance_decomposition(X, y, model)
print(f"Bias: {bias:.3f}, Variance: {variance:.3f}, Total: {total_error:.3f}")

# Model comparison
models = {
    'Linear': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100)
}
results = stat_learning.model_comparison(X, y, models)
print("Model comparison results:", results)
```

---

## 🎯 **Interview Questions**

### **Probability and Statistics**

#### **Q1: Explain the difference between frequentist and Bayesian statistics**
**Answer**: Frequentist statistics treats parameters as fixed but unknown, using sample data to estimate them. Bayesian statistics treats parameters as random variables with prior distributions, updating beliefs with observed data using Bayes' theorem.

#### **Q2: What is the Central Limit Theorem and why is it important?**
**Answer**: The CLT states that the sampling distribution of the mean approaches a normal distribution as sample size increases, regardless of the population distribution. It's crucial for hypothesis testing and confidence intervals.

#### **Q3: Explain Type I and Type II errors in hypothesis testing**
**Answer**: Type I error (α) is rejecting a true null hypothesis (false positive). Type II error (β) is failing to reject a false null hypothesis (false negative). Power = 1 - β is the probability of correctly rejecting a false null hypothesis.

#### **Q4: What is the difference between correlation and causation?**
**Answer**: Correlation measures the strength and direction of linear relationship between variables. Causation implies that one variable directly influences another. Correlation doesn't imply causation due to confounding variables and reverse causality.

#### **Q5: How do you handle multiple comparisons in statistical testing?**
**Answer**: Multiple comparisons increase the chance of Type I errors. Solutions include Bonferroni correction (α/m), False Discovery Rate (FDR), and family-wise error rate (FWER) control methods.

---

## 🚀 **Next Steps**

1. **Practice**: Implement all statistical methods from scratch
2. **Apply**: Use statistics in ML model evaluation and selection
3. **Visualize**: Create plots to understand distributions and relationships
4. **Experiment**: Test statistical concepts with real datasets
5. **Interview**: Practice statistical ML interview questions

---

**Ready to dive into machine learning algorithms? Let's move to the Machine Learning section!** 🎯
