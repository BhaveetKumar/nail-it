# 🚀 GitHub Actions: CI/CD Automation and Workflow Orchestration

> **Master GitHub Actions for automated CI/CD pipelines and workflow orchestration**

## 📚 Concept

GitHub Actions is a CI/CD platform that allows you to automate workflows directly in your GitHub repository. It provides powerful automation capabilities for building, testing, and deploying applications.

### Key Features
- **Workflow Automation**: Define custom workflows with YAML
- **Event-Driven**: Trigger workflows on various GitHub events
- **Matrix Builds**: Run jobs across multiple configurations
- **Secrets Management**: Secure storage of sensitive data
- **Artifact Management**: Store and share build artifacts
- **Self-Hosted Runners**: Run workflows on your own infrastructure

## 🏗️ GitHub Actions Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    GitHub Repository                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Events    │  │  Workflows  │  │   Actions   │     │
│  │   Push      │  │   Build     │  │   Custom    │     │
│  │   PR        │  │   Test      │  │   Marketplace│     │
│  │   Release   │  │   Deploy    │  │   Composite │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              GitHub Actions Runner                │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Ubuntu    │  │   Windows   │  │    macOS    │ │ │
│  │  │   Runner    │  │   Runner    │  │   Runner    │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Docker    │  │   AWS/GCP   │  │   Kubernetes│     │
│  │   Registry  │  │   Deploy    │  │   Deploy    │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Hands-on Example

### Complete CI/CD Workflow

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  release:
    types: [ published ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # Code Quality and Testing
  test:
    name: Test and Code Quality
    runs-on: ubuntu-latest
    strategy:
      matrix:
        go-version: [1.19, 1.20, 1.21]
        os: [ubuntu-latest, windows-latest, macos-latest]
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      with:
        fetch-depth: 0

    - name: Set up Go
      uses: actions/setup-go@v4
      with:
        go-version: ${{ matrix.go-version }}

    - name: Cache Go modules
      uses: actions/cache@v3
      with:
        path: |
          ~/.cache/go-build
          ~/go/pkg/mod
        key: ${{ runner.os }}-go-${{ hashFiles('**/go.sum') }}
        restore-keys: |
          ${{ runner.os }}-go-

    - name: Install dependencies
      run: go mod download

    - name: Run tests
      run: |
        go test -v -race -coverprofile=coverage.out ./...
        go test -v -race -coverprofile=coverage.out -covermode=atomic ./...

    - name: Run benchmarks
      run: go test -bench=. -benchmem ./...

    - name: Run linter
      uses: golangci/golangci-lint-action@v3
      with:
        version: latest
        args: --timeout=5m

    - name: Run security scan
      uses: securecodewarrior/github-action-add-sarif@v1
      with:
        sarif-file: 'gosec-results.sarif'

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.out
        flags: unittests
        name: codecov-umbrella
        fail_ci_if_error: false

    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results-${{ matrix.go-version }}-${{ matrix.os }}
        path: |
          coverage.out
          test-results.xml

  # Security Scanning
  security:
    name: Security Scan
    runs-on: ubuntu-latest
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'

    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'

    - name: Run Snyk security scan
      uses: snyk/actions/go@master
      env:
        SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
      with:
        args: --severity-threshold=high

  # Build and Package
  build:
    name: Build Application
    runs-on: ubuntu-latest
    needs: [test, security]
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Go
      uses: actions/setup-go@v4
      with:
        go-version: '1.21'

    - name: Build application
      run: |
        go build -ldflags="-s -w -X main.version=${{ github.sha }}" -o app ./cmd/server

    - name: Build Docker image
      run: |
        docker build -t ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} .
        docker build -t ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest .

    - name: Upload build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: build-artifacts
        path: |
          app
          Dockerfile

  # Docker Registry
  push:
    name: Push to Registry
    runs-on: ubuntu-latest
    needs: build
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=sha,prefix={{branch}}-

    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  # Deploy to Staging
  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: [build, push]
    if: github.ref == 'refs/heads/develop'
    environment: staging
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-west-2

    - name: Deploy to EKS
      run: |
        aws eks update-kubeconfig --region us-west-2 --name staging-cluster
        kubectl set image deployment/app-deployment app=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} -n staging
        kubectl rollout status deployment/app-deployment -n staging

    - name: Run smoke tests
      run: |
        kubectl get pods -n staging
        kubectl get services -n staging
        # Add smoke test commands here

  # Deploy to Production
  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: [build, push]
    if: github.event_name == 'release'
    environment: production
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-west-2

    - name: Deploy to EKS
      run: |
        aws eks update-kubeconfig --region us-west-2 --name production-cluster
        kubectl set image deployment/app-deployment app=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.event.release.tag_name }} -n production
        kubectl rollout status deployment/app-deployment -n production

    - name: Run health checks
      run: |
        kubectl get pods -n production
        kubectl get services -n production
        # Add health check commands here

    - name: Notify deployment
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        channel: '#deployments'
        webhook_url: ${{ secrets.SLACK_WEBHOOK }}
      if: always()

  # Cleanup
  cleanup:
    name: Cleanup
    runs-on: ubuntu-latest
    needs: [deploy-staging, deploy-production]
    if: always()
    
    steps:
    - name: Clean up old artifacts
      uses: actions/github-script@v6
      with:
        script: |
          const artifacts = await github.rest.actions.listWorkflowRunArtifacts({
            owner: context.repo.owner,
            repo: context.repo.repo,
            run_id: context.runId,
          });
          
          for (const artifact of artifacts.data.artifacts) {
            if (artifact.name.includes('test-results') || artifact.name.includes('build-artifacts')) {
              await github.rest.actions.deleteArtifact({
                owner: context.repo.owner,
                repo: context.repo.repo,
                artifact_id: artifact.id,
              });
            }
          }
```

### Go Application Workflow

```yaml
# .github/workflows/go.yml
name: Go Application CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  GO_VERSION: '1.21'
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    name: Test
    runs-on: ubuntu-latest
    strategy:
      matrix:
        go-version: [1.19, 1.20, 1.21]
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Go
      uses: actions/setup-go@v4
      with:
        go-version: ${{ matrix.go-version }}

    - name: Cache Go modules
      uses: actions/cache@v3
      with:
        path: |
          ~/.cache/go-build
          ~/go/pkg/mod
        key: ${{ runner.os }}-go-${{ hashFiles('**/go.sum') }}
        restore-keys: |
          ${{ runner.os }}-go-

    - name: Install dependencies
      run: go mod download

    - name: Run tests
      run: |
        go test -v -race -coverprofile=coverage.out ./...
        go test -v -race -coverprofile=coverage.out -covermode=atomic ./...

    - name: Run benchmarks
      run: go test -bench=. -benchmem ./...

    - name: Run linter
      uses: golangci/golangci-lint-action@v3
      with:
        version: latest
        args: --timeout=5m

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.out
        flags: unittests
        name: codecov-umbrella
        fail_ci_if_error: false

  build:
    name: Build
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Go
      uses: actions/setup-go@v4
      with:
        go-version: ${{ env.GO_VERSION }}

    - name: Build application
      run: |
        go build -ldflags="-s -w -X main.version=${{ github.sha }}" -o app ./cmd/server

    - name: Upload build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: build-artifacts
        path: app

  docker:
    name: Docker Build
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=sha,prefix={{branch}}-

    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy:
    name: Deploy
    runs-on: ubuntu-latest
    needs: [build, docker]
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-west-2

    - name: Deploy to EKS
      run: |
        aws eks update-kubeconfig --region us-west-2 --name production-cluster
        kubectl set image deployment/app-deployment app=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} -n production
        kubectl rollout status deployment/app-deployment -n production

    - name: Run health checks
      run: |
        kubectl get pods -n production
        kubectl get services -n production
        # Add health check commands here
```

### Node.js Application Workflow

```yaml
# .github/workflows/nodejs.yml
name: Node.js Application CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  NODE_VERSION: '18'
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    name: Test
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node-version: [16, 18, 20]
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Node.js
      uses: actions/setup-node@v3
      with:
        node-version: ${{ matrix.node-version }}
        cache: 'npm'

    - name: Install dependencies
      run: npm ci

    - name: Run tests
      run: |
        npm test
        npm run test:coverage

    - name: Run linter
      run: npm run lint

    - name: Run security audit
      run: npm audit --audit-level=high

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage/lcov.info
        flags: unittests
        name: codecov-umbrella
        fail_ci_if_error: false

  build:
    name: Build
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Node.js
      uses: actions/setup-node@v3
      with:
        node-version: ${{ env.NODE_VERSION }}
        cache: 'npm'

    - name: Install dependencies
      run: npm ci

    - name: Build application
      run: npm run build

    - name: Upload build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: build-artifacts
        path: |
          dist/
          package.json
          package-lock.json

  docker:
    name: Docker Build
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=sha,prefix={{branch}}-

    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy:
    name: Deploy
    runs-on: ubuntu-latest
    needs: [build, docker]
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-west-2

    - name: Deploy to EKS
      run: |
        aws eks update-kubeconfig --region us-west-2 --name production-cluster
        kubectl set image deployment/app-deployment app=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} -n production
        kubectl rollout status deployment/app-deployment -n production

    - name: Run health checks
      run: |
        kubectl get pods -n production
        kubectl get services -n production
        # Add health check commands here
```

### Python Application Workflow

```yaml
# .github/workflows/python.yml
name: Python Application CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  PYTHON_VERSION: '3.11'
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    name: Test
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml --cov-report=html

    - name: Run linter
      run: |
        flake8 src/
        black --check src/
        isort --check-only src/

    - name: Run type checker
      run: mypy src/

    - name: Run security scan
      run: bandit -r src/

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
        fail_ci_if_error: false

  build:
    name: Build
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Build package
      run: |
        python setup.py sdist bdist_wheel

    - name: Upload build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: build-artifacts
        path: |
          dist/
          *.whl
          *.tar.gz

  docker:
    name: Docker Build
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=sha,prefix={{branch}}-

    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy:
    name: Deploy
    runs-on: ubuntu-latest
    needs: [build, docker]
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-west-2

    - name: Deploy to EKS
      run: |
        aws eks update-kubeconfig --region us-west-2 --name production-cluster
        kubectl set image deployment/app-deployment app=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} -n production
        kubectl rollout status deployment/app-deployment -n production

    - name: Run health checks
      run: |
        kubectl get pods -n production
        kubectl get services -n production
        # Add health check commands here
```

## 🚀 Best Practices

### 1. Workflow Organization
```yaml
# Organize workflows by purpose
name: Feature Branch Workflow
on:
  push:
    branches: [ develop, feature/* ]
  pull_request:
    branches: [ develop ]

name: Release Workflow
on:
  push:
    branches: [ main ]
  release:
    types: [ published ]

name: Hotfix Workflow
on:
  push:
    branches: [ hotfix/* ]
```

### 2. Security Best Practices
```yaml
# Use secrets for sensitive data
- name: Deploy to production
  run: |
    echo "Deploying to production..."
  env:
    API_KEY: ${{ secrets.API_KEY }}
    DATABASE_URL: ${{ secrets.DATABASE_URL }}
    ENCRYPTION_KEY: ${{ secrets.ENCRYPTION_KEY }}

# Use environment protection rules
environment: production
```

### 3. Performance Optimization
```yaml
# Use caching for dependencies
- name: Cache Go modules
  uses: actions/cache@v3
  with:
    path: |
      ~/.cache/go-build
      ~/go/pkg/mod
    key: ${{ runner.os }}-go-${{ hashFiles('**/go.sum') }}
    restore-keys: |
      ${{ runner.os }}-go-

# Use matrix builds for parallel execution
strategy:
  matrix:
    go-version: [1.19, 1.20, 1.21]
    os: [ubuntu-latest, windows-latest, macos-latest]
```

## 🏢 Industry Insights

### GitHub's Actions Usage
- **Open Source**: Public repositories
- **Enterprise**: Private repositories
- **Marketplace**: Custom actions
- **Self-Hosted**: Custom runners

### Netflix's GitHub Actions Strategy
- **Microservices**: Individual service pipelines
- **Security**: Automated security scanning
- **Deployment**: Multi-environment deployment
- **Monitoring**: Automated monitoring setup

### Spotify's GitHub Actions Approach
- **Music Processing**: Audio file processing
- **Data Pipeline**: ETL operations
- **Real-time Analytics**: User listening data
- **Cost Efficiency**: Serverless computing

## 🎯 Interview Questions

### Basic Level
1. **What are GitHub Actions?**
   - CI/CD platform
   - Workflow automation
   - Event-driven execution
   - Integrated with GitHub

2. **What are GitHub Actions triggers?**
   - Push events
   - Pull request events
   - Release events
   - Scheduled events

3. **What are GitHub Actions limitations?**
   - 6-hour job timeout
   - 2GB memory limit
   - 10GB disk space
   - Rate limits

### Intermediate Level
4. **How do you optimize GitHub Actions performance?**
   ```yaml
   # Use caching and matrix builds
   - name: Cache dependencies
     uses: actions/cache@v3
     with:
       path: ~/.cache
       key: ${{ runner.os }}-cache-${{ hashFiles('**/lockfile') }}
   
   strategy:
     matrix:
       version: [1.19, 1.20, 1.21]
   ```

5. **How do you handle GitHub Actions secrets?**
   - Use repository secrets
   - Use environment secrets
   - Use organization secrets
   - Use encrypted secrets

6. **How do you secure GitHub Actions?**
   - Use least privilege
   - Use environment protection
   - Use secret scanning
   - Use dependency scanning

### Advanced Level
7. **How do you implement GitHub Actions patterns?**
   - Reusable workflows
   - Composite actions
   - Custom actions
   - Workflow templates

8. **How do you handle GitHub Actions scaling?**
   - Self-hosted runners
   - Matrix builds
   - Parallel jobs
   - Resource optimization

9. **How do you implement GitHub Actions testing?**
   - Unit testing
   - Integration testing
   - End-to-end testing
   - Performance testing

---

**Next**: [Jenkins](./Jenkins.md) - Self-hosted CI/CD, pipeline automation
