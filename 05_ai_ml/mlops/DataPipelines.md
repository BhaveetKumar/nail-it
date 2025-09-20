# 🔄 Data Pipelines: Building Robust ML Data Infrastructure

> **Complete guide to designing and implementing data pipelines for machine learning**

## 🎯 **Learning Objectives**

- Master data pipeline design patterns and architectures
- Understand ETL/ELT processes for ML workflows
- Implement data validation and quality checks
- Build scalable data processing systems
- Handle real-time and batch data processing

## 📚 **Table of Contents**

1. [Data Pipeline Fundamentals](#data-pipeline-fundamentals)
2. [ETL vs ELT Patterns](#etl-vs-elt-patterns)
3. [Data Validation & Quality](#data-validation--quality)
4. [Real-time Data Processing](#real-time-data-processing)
5. [Batch Processing](#batch-processing)
6. [Pipeline Orchestration](#pipeline-orchestration)
7. [Interview Questions](#interview-questions)

---

## 🔄 **Data Pipeline Fundamentals**

### **Concept**

Data pipelines are automated processes that move and transform data from source systems to destination systems, enabling reliable and scalable data processing for machine learning workflows.

### **Key Components**

1. **Data Sources**: Databases, APIs, files, streams
2. **Data Ingestion**: Collecting data from various sources
3. **Data Transformation**: Cleaning, validating, and enriching data
4. **Data Storage**: Data lakes, warehouses, feature stores
5. **Data Serving**: APIs, databases for model consumption
6. **Monitoring**: Pipeline health, data quality, performance

### **Pipeline Types**

- **Batch Processing**: Process data in large chunks
- **Stream Processing**: Process data in real-time
- **Lambda Architecture**: Combine batch and stream processing
- **Kappa Architecture**: Stream-only processing

---

## 🔄 **ETL vs ELT Patterns**

### **1. ETL (Extract, Transform, Load)**

**Concept**: Extract data from sources, transform it, then load into destination.

**Code Example**:
```python
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging
from datetime import datetime
import json

class ETLPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def extract(self, source_config: Dict[str, Any]) -> pd.DataFrame:
        """Extract data from source"""
        self.logger.info(f"Extracting data from {source_config['type']}")
        
        if source_config['type'] == 'csv':
            return pd.read_csv(source_config['path'])
        elif source_config['type'] == 'json':
            return pd.read_json(source_config['path'])
        elif source_config['type'] == 'database':
            return self._extract_from_database(source_config)
        else:
            raise ValueError(f"Unsupported source type: {source_config['type']}")
    
    def _extract_from_database(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Extract data from database"""
        # Mock database extraction
        return pd.DataFrame({
            'id': range(1000),
            'name': [f'user_{i}' for i in range(1000)],
            'age': np.random.randint(18, 80, 1000),
            'email': [f'user_{i}@example.com' for i in range(1000)]
        })
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data"""
        self.logger.info("Transforming data")
        
        # Data cleaning
        data = self._clean_data(data)
        
        # Data validation
        data = self._validate_data(data)
        
        # Feature engineering
        data = self._engineer_features(data)
        
        return data
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean data"""
        # Remove duplicates
        data = data.drop_duplicates()
        
        # Handle missing values
        data = data.fillna(data.mean(numeric_only=True))
        
        # Remove outliers
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        
        return data
    
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate data"""
        # Check data types
        if 'age' in data.columns:
            data['age'] = pd.to_numeric(data['age'], errors='coerce')
        
        # Check email format
        if 'email' in data.columns:
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            data = data[data['email'].str.match(email_pattern, na=False)]
        
        return data
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features"""
        # Age groups
        if 'age' in data.columns:
            data['age_group'] = pd.cut(data['age'], 
                                     bins=[0, 25, 35, 50, 100], 
                                     labels=['young', 'adult', 'middle', 'senior'])
        
        # Email domain
        if 'email' in data.columns:
            data['email_domain'] = data['email'].str.extract(r'@(.+)')
        
        return data
    
    def load(self, data: pd.DataFrame, destination_config: Dict[str, Any]):
        """Load data to destination"""
        self.logger.info(f"Loading data to {destination_config['type']}")
        
        if destination_config['type'] == 'csv':
            data.to_csv(destination_config['path'], index=False)
        elif destination_config['type'] == 'parquet':
            data.to_parquet(destination_config['path'], index=False)
        elif destination_config['type'] == 'database':
            self._load_to_database(data, destination_config)
        else:
            raise ValueError(f"Unsupported destination type: {destination_config['type']}")
    
    def _load_to_database(self, data: pd.DataFrame, config: Dict[str, Any]):
        """Load data to database"""
        # Mock database loading
        self.logger.info(f"Loading {len(data)} rows to database")
    
    def run(self, source_config: Dict[str, Any], destination_config: Dict[str, Any]):
        """Run ETL pipeline"""
        try:
            # Extract
            data = self.extract(source_config)
            self.logger.info(f"Extracted {len(data)} rows")
            
            # Transform
            transformed_data = self.transform(data)
            self.logger.info(f"Transformed {len(transformed_data)} rows")
            
            # Load
            self.load(transformed_data, destination_config)
            self.logger.info("ETL pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"ETL pipeline failed: {str(e)}")
            raise

# Example usage
def etl_example():
    """Example of ETL pipeline"""
    config = {
        'batch_size': 1000,
        'parallel_workers': 4
    }
    
    pipeline = ETLPipeline(config)
    
    source_config = {
        'type': 'database',
        'connection_string': 'postgresql://user:pass@localhost/db'
    }
    
    destination_config = {
        'type': 'parquet',
        'path': 'output/processed_data.parquet'
    }
    
    pipeline.run(source_config, destination_config)

if __name__ == "__main__":
    etl_example()
```

### **2. ELT (Extract, Load, Transform)**

**Concept**: Extract data, load into destination, then transform.

**Code Example**:
```python
class ELTPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def extract(self, source_config: Dict[str, Any]) -> pd.DataFrame:
        """Extract data from source"""
        self.logger.info(f"Extracting data from {source_config['type']}")
        
        if source_config['type'] == 'api':
            return self._extract_from_api(source_config)
        elif source_config['type'] == 'stream':
            return self._extract_from_stream(source_config)
        else:
            return pd.read_csv(source_config['path'])
    
    def _extract_from_api(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Extract data from API"""
        import requests
        
        response = requests.get(config['url'])
        data = response.json()
        
        return pd.DataFrame(data)
    
    def _extract_from_stream(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Extract data from stream"""
        # Mock stream data
        return pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'),
            'value': np.random.randn(1000),
            'category': np.random.choice(['A', 'B', 'C'], 1000)
        })
    
    def load(self, data: pd.DataFrame, destination_config: Dict[str, Any]):
        """Load raw data to destination"""
        self.logger.info(f"Loading raw data to {destination_config['type']}")
        
        if destination_config['type'] == 'data_lake':
            self._load_to_data_lake(data, destination_config)
        elif destination_config['type'] == 'warehouse':
            self._load_to_warehouse(data, destination_config)
        else:
            data.to_parquet(destination_config['path'], index=False)
    
    def _load_to_data_lake(self, data: pd.DataFrame, config: Dict[str, Any]):
        """Load to data lake"""
        # Mock data lake loading
        self.logger.info(f"Loading {len(data)} rows to data lake")
    
    def _load_to_warehouse(self, data: pd.DataFrame, config: Dict[str, Any]):
        """Load to data warehouse"""
        # Mock warehouse loading
        self.logger.info(f"Loading {len(data)} rows to warehouse")
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data in place"""
        self.logger.info("Transforming data")
        
        # Apply transformations
        data = self._apply_transformations(data)
        
        return data
    
    def _apply_transformations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply data transformations"""
        # Add derived columns
        if 'timestamp' in data.columns:
            data['hour'] = data['timestamp'].dt.hour
            data['day_of_week'] = data['timestamp'].dt.dayofweek
        
        # Aggregate data
        if 'category' in data.columns and 'value' in data.columns:
            aggregated = data.groupby('category')['value'].agg(['mean', 'std', 'count']).reset_index()
            data = data.merge(aggregated, on='category', suffixes=('', '_agg'))
        
        return data
    
    def run(self, source_config: Dict[str, Any], destination_config: Dict[str, Any]):
        """Run ELT pipeline"""
        try:
            # Extract
            data = self.extract(source_config)
            self.logger.info(f"Extracted {len(data)} rows")
            
            # Load raw data
            self.load(data, destination_config)
            self.logger.info("Raw data loaded")
            
            # Transform
            transformed_data = self.transform(data)
            self.logger.info(f"Transformed {len(transformed_data)} rows")
            
            # Load transformed data
            transformed_destination = destination_config.copy()
            transformed_destination['path'] = destination_config['path'].replace('.parquet', '_transformed.parquet')
            self.load(transformed_data, transformed_destination)
            
            self.logger.info("ELT pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"ELT pipeline failed: {str(e)}")
            raise

# Example usage
def elt_example():
    """Example of ELT pipeline"""
    config = {
        'batch_size': 1000,
        'parallel_workers': 4
    }
    
    pipeline = ELTPipeline(config)
    
    source_config = {
        'type': 'stream',
        'url': 'kafka://localhost:9092/topic'
    }
    
    destination_config = {
        'type': 'data_lake',
        'path': 's3://data-lake/raw/'
    }
    
    pipeline.run(source_config, destination_config)

if __name__ == "__main__":
    elt_example()
```

---

## ✅ **Data Validation & Quality**

### **1. Data Quality Framework**

**Code Example**:
```python
from typing import List, Dict, Any, Callable
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

class ValidationResult(Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"

@dataclass
class ValidationCheck:
    name: str
    description: str
    check_function: Callable
    severity: ValidationResult = ValidationResult.FAIL

class DataValidator:
    def __init__(self):
        self.checks = []
        self.results = []
    
    def add_check(self, check: ValidationCheck):
        """Add validation check"""
        self.checks.append(check)
    
    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run all validation checks"""
        results = {
            'total_checks': len(self.checks),
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'details': []
        }
        
        for check in self.checks:
            try:
                result = check.check_function(data)
                if result:
                    results['passed'] += 1
                    status = ValidationResult.PASS
                else:
                    if check.severity == ValidationResult.WARNING:
                        results['warnings'] += 1
                        status = ValidationResult.WARNING
                    else:
                        results['failed'] += 1
                        status = ValidationResult.FAIL
                
                results['details'].append({
                    'name': check.name,
                    'description': check.description,
                    'status': status.value,
                    'result': result
                })
                
            except Exception as e:
                results['failed'] += 1
                results['details'].append({
                    'name': check.name,
                    'description': check.description,
                    'status': ValidationResult.FAIL.value,
                    'error': str(e)
                })
        
        return results
    
    def create_common_checks(self):
        """Create common data quality checks"""
        # Check for missing values
        self.add_check(ValidationCheck(
            name="no_missing_values",
            description="Check for missing values",
            check_function=lambda df: df.isnull().sum().sum() == 0,
            severity=ValidationResult.WARNING
        ))
        
        # Check for duplicates
        self.add_check(ValidationCheck(
            name="no_duplicates",
            description="Check for duplicate rows",
            check_function=lambda df: df.duplicated().sum() == 0
        ))
        
        # Check data types
        self.add_check(ValidationCheck(
            name="correct_data_types",
            description="Check data types",
            check_function=self._check_data_types
        ))
        
        # Check value ranges
        self.add_check(ValidationCheck(
            name="value_ranges",
            description="Check value ranges",
            check_function=self._check_value_ranges
        ))
        
        # Check data freshness
        self.add_check(ValidationCheck(
            name="data_freshness",
            description="Check data freshness",
            check_function=self._check_data_freshness
        ))
    
    def _check_data_types(self, data: pd.DataFrame) -> bool:
        """Check data types"""
        # Mock implementation
        return True
    
    def _check_value_ranges(self, data: pd.DataFrame) -> bool:
        """Check value ranges"""
        # Mock implementation
        return True
    
    def _check_data_freshness(self, data: pd.DataFrame) -> bool:
        """Check data freshness"""
        # Mock implementation
        return True

# Example usage
def data_validation_example():
    """Example of data validation"""
    # Create sample data
    data = pd.DataFrame({
        'id': range(100),
        'name': [f'user_{i}' for i in range(100)],
        'age': np.random.randint(18, 80, 100),
        'email': [f'user_{i}@example.com' for i in range(100)]
    })
    
    # Create validator
    validator = DataValidator()
    validator.create_common_checks()
    
    # Run validation
    results = validator.validate(data)
    
    print("Data Validation Results:")
    print(f"Total checks: {results['total_checks']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Warnings: {results['warnings']}")
    
    for detail in results['details']:
        print(f"- {detail['name']}: {detail['status']}")

if __name__ == "__main__":
    data_validation_example()
```

### **2. Data Quality Monitoring**

**Code Example**:
```python
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

class DataQualityMonitor:
    def __init__(self):
        self.metrics = {}
        self.alerts = []
        self.thresholds = {
            'missing_value_rate': 0.05,  # 5%
            'duplicate_rate': 0.01,      # 1%
            'outlier_rate': 0.10,        # 10%
            'freshness_hours': 24        # 24 hours
        }
    
    def calculate_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data quality metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'row_count': len(data),
            'column_count': len(data.columns),
            'missing_value_rate': data.isnull().sum().sum() / (len(data) * len(data.columns)),
            'duplicate_rate': data.duplicated().sum() / len(data),
            'outlier_rate': self._calculate_outlier_rate(data),
            'freshness': self._calculate_freshness(data)
        }
        
        return metrics
    
    def _calculate_outlier_rate(self, data: pd.DataFrame) -> float:
        """Calculate outlier rate"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            return 0.0
        
        outlier_count = 0
        total_count = 0
        
        for col in numeric_columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            outlier_count += len(outliers)
            total_count += len(data)
        
        return outlier_count / total_count if total_count > 0 else 0.0
    
    def _calculate_freshness(self, data: pd.DataFrame) -> float:
        """Calculate data freshness in hours"""
        if 'timestamp' in data.columns:
            latest_timestamp = data['timestamp'].max()
            if pd.isna(latest_timestamp):
                return float('inf')
            
            current_time = datetime.now()
            if isinstance(latest_timestamp, pd.Timestamp):
                latest_time = latest_timestamp.to_pydatetime()
            else:
                latest_time = latest_timestamp
            
            freshness_hours = (current_time - latest_time).total_seconds() / 3600
            return freshness_hours
        
        return 0.0
    
    def check_thresholds(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check metrics against thresholds"""
        alerts = []
        
        for metric, threshold in self.thresholds.items():
            if metric in metrics:
                if metric == 'freshness_hours':
                    if metrics['freshness'] > threshold:
                        alerts.append({
                            'metric': metric,
                            'value': metrics['freshness'],
                            'threshold': threshold,
                            'severity': 'high',
                            'message': f"Data is {metrics['freshness']:.1f} hours old, exceeds {threshold} hour threshold"
                        })
                else:
                    if metrics[metric] > threshold:
                        alerts.append({
                            'metric': metric,
                            'value': metrics[metric],
                            'threshold': threshold,
                            'severity': 'high',
                            'message': f"{metric} is {metrics[metric]:.3f}, exceeds {threshold} threshold"
                        })
        
        return alerts
    
    def monitor(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Monitor data quality"""
        metrics = self.calculate_metrics(data)
        alerts = self.check_thresholds(metrics)
        
        result = {
            'metrics': metrics,
            'alerts': alerts,
            'status': 'healthy' if len(alerts) == 0 else 'unhealthy'
        }
        
        # Store metrics
        self.metrics[metrics['timestamp']] = metrics
        self.alerts.extend(alerts)
        
        return result
    
    def get_historical_metrics(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical metrics"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        historical_metrics = []
        for timestamp, metrics in self.metrics.items():
            if datetime.fromisoformat(timestamp) >= cutoff_time:
                historical_metrics.append(metrics)
        
        return sorted(historical_metrics, key=lambda x: x['timestamp'])

# Example usage
def data_quality_monitoring_example():
    """Example of data quality monitoring"""
    # Create sample data
    data = pd.DataFrame({
        'id': range(100),
        'name': [f'user_{i}' for i in range(100)],
        'age': np.random.randint(18, 80, 100),
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H')
    })
    
    # Create monitor
    monitor = DataQualityMonitor()
    
    # Monitor data quality
    result = monitor.monitor(data)
    
    print("Data Quality Monitoring Results:")
    print(f"Status: {result['status']}")
    print(f"Metrics: {result['metrics']}")
    print(f"Alerts: {result['alerts']}")

if __name__ == "__main__":
    data_quality_monitoring_example()
```

---

## 🎯 **Interview Questions**

### **1. What's the difference between ETL and ELT?**

**Answer:**
- **ETL**: Extract → Transform → Load
  - Transform data before loading
  - Good for structured data
  - Requires more processing power upfront
  - Better for data warehouses
  
- **ELT**: Extract → Load → Transform
  - Load raw data first, then transform
  - Good for unstructured data
  - Leverages destination system's processing power
  - Better for data lakes

### **2. How do you handle data quality issues in pipelines?**

**Answer:**
- **Validation**: Check data types, ranges, and formats
- **Monitoring**: Track data quality metrics over time
- **Alerting**: Notify when quality thresholds are breached
- **Cleaning**: Remove duplicates, handle missing values
- **Documentation**: Document data quality rules and expectations
- **Testing**: Automated tests for data quality

### **3. What are the challenges of real-time data processing?**

**Answer:**
- **Latency**: Minimizing processing delay
- **Throughput**: Handling high-volume data streams
- **Fault Tolerance**: Handling failures gracefully
- **State Management**: Managing state across distributed systems
- **Backpressure**: Handling data faster than processing capacity
- **Consistency**: Ensuring data consistency across systems

### **4. How do you design a scalable data pipeline?**

**Answer:**
- **Partitioning**: Split data into manageable chunks
- **Parallel Processing**: Process data in parallel
- **Load Balancing**: Distribute workload evenly
- **Auto-scaling**: Automatically adjust resources
- **Monitoring**: Track performance and health
- **Error Handling**: Graceful failure recovery

### **5. What are the key considerations for data pipeline security?**

**Answer:**
- **Encryption**: Encrypt data in transit and at rest
- **Access Control**: Implement proper authentication and authorization
- **Data Masking**: Mask sensitive data in non-production environments
- **Audit Logging**: Log all data access and modifications
- **Compliance**: Meet regulatory requirements (GDPR, HIPAA)
- **Network Security**: Secure network connections and firewalls

---

**🎉 Data pipelines are the backbone of reliable ML systems and require careful design and monitoring!**


## Batch Processing

<!-- AUTO-GENERATED ANCHOR: originally referenced as #batch-processing -->

Placeholder content. Please replace with proper section.


## Pipeline Orchestration

<!-- AUTO-GENERATED ANCHOR: originally referenced as #pipeline-orchestration -->

Placeholder content. Please replace with proper section.
