# 🦊 GitLab CI: Integrated CI/CD and Container Registry

> **Master GitLab CI for integrated CI/CD pipelines and container management**

## 📚 Concept

GitLab CI is an integrated CI/CD platform that provides automated testing, building, and deployment directly within GitLab. It offers a comprehensive DevOps platform with built-in container registry, security scanning, and deployment tools.

### Key Features
- **Integrated CI/CD**: Built into GitLab platform
- **Container Registry**: Built-in Docker registry
- **Security Scanning**: Automated vulnerability scanning
- **Deployment**: Multi-environment deployment
- **Pipeline as Code**: YAML-based configuration
- **Auto DevOps**: Automated pipeline generation

## 🏗️ GitLab CI Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    GitLab Platform                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Git       │  │   CI/CD     │  │   Registry  │     │
│  │   Repository│  │   Pipelines │  │   Container │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              GitLab Runners                       │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Shared    │  │   Group     │  │   Project   │ │ │
│  │  │   Runners   │  │   Runners   │  │   Runners   │ │ │
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

### Complete GitLab CI Pipeline

```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - security
  - deploy

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"
  IMAGE_TAG: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  LATEST_TAG: $CI_REGISTRY_IMAGE:latest

# Test stage
test:go:
  stage: test
  image: golang:1.21
  script:
    - go mod download
    - go test -v -race -coverprofile=coverage.out ./...
    - go test -bench=. -benchmem ./...
  coverage: '/coverage: \d+\.\d+%/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
    paths:
      - coverage.out
    expire_in: 1 week

test:lint:
  stage: test
  image: golang:1.21
  script:
    - go mod download
    - golangci-lint run --timeout=5m
  allow_failure: true

# Build stage
build:docker:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - docker build -t $IMAGE_TAG .
    - docker tag $IMAGE_TAG $LATEST_TAG
    - docker push $IMAGE_TAG
    - docker push $LATEST_TAG
  only:
    - main
    - develop

# Security stage
security:scan:
  stage: security
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker run --rm -v /var/run/docker.sock:/var/run/docker.sock
      -v $PWD:/app aquasec/trivy image $IMAGE_TAG
  allow_failure: true

# Deploy stages
deploy:staging:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl config use-context staging
    - kubectl set image deployment/app-deployment app=$IMAGE_TAG -n staging
    - kubectl rollout status deployment/app-deployment -n staging
  environment:
    name: staging
    url: https://staging.example.com
  only:
    - develop

deploy:production:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl config use-context production
    - kubectl set image deployment/app-deployment app=$IMAGE_TAG -n production
    - kubectl rollout status deployment/app-deployment -n production
  environment:
    name: production
    url: https://example.com
  when: manual
  only:
    - main
```

### Go Application Pipeline

```yaml
# .gitlab-ci.yml for Go
stages:
  - test
  - build
  - deploy

variables:
  GO_VERSION: "1.21"
  CGO_ENABLED: "0"

test:
  stage: test
  image: golang:${GO_VERSION}
  script:
    - go mod download
    - go test -v -race -coverprofile=coverage.out ./...
    - go test -bench=. -benchmem ./...
  coverage: '/coverage: \d+\.\d+%/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

build:
  stage: build
  image: golang:${GO_VERSION}
  script:
    - go build -ldflags="-s -w -X main.version=$CI_COMMIT_SHA" -o app ./cmd/server
  artifacts:
    paths:
      - app
    expire_in: 1 week

deploy:
  stage: deploy
  image: alpine:latest
  script:
    - apk add --no-cache curl
    - curl -X POST $DEPLOY_WEBHOOK_URL
  only:
    - main
```

### Node.js Application Pipeline

```yaml
# .gitlab-ci.yml for Node.js
stages:
  - test
  - build
  - deploy

variables:
  NODE_VERSION: "18"

test:
  stage: test
  image: node:${NODE_VERSION}
  cache:
    paths:
      - node_modules/
  script:
    - npm ci
    - npm test
    - npm run lint
  coverage: '/Lines\s*:\s*(\d+\.\d+)%/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage/cobertura-coverage.xml

build:
  stage: build
  image: node:${NODE_VERSION}
  cache:
    paths:
      - node_modules/
  script:
    - npm ci
    - npm run build
  artifacts:
    paths:
      - dist/
    expire_in: 1 week

deploy:
  stage: deploy
  image: alpine:latest
  script:
    - apk add --no-cache curl
    - curl -X POST $DEPLOY_WEBHOOK_URL
  only:
    - main
```

### Python Application Pipeline

```yaml
# .gitlab-ci.yml for Python
stages:
  - test
  - build
  - deploy

variables:
  PYTHON_VERSION: "3.11"

test:
  stage: test
  image: python:${PYTHON_VERSION}
  cache:
    paths:
      - .cache/pip/
  script:
    - pip install -r requirements.txt
    - pip install -r requirements-dev.txt
    - pytest --cov=src --cov-report=xml
    - flake8 src/
    - black --check src/
  coverage: '/TOTAL.*\s+(\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

build:
  stage: build
  image: python:${PYTHON_VERSION}
  script:
    - pip install -r requirements.txt
    - python setup.py sdist bdist_wheel
  artifacts:
    paths:
      - dist/
    expire_in: 1 week

deploy:
  stage: deploy
  image: alpine:latest
  script:
    - apk add --no-cache curl
    - curl -X POST $DEPLOY_WEBHOOK_URL
  only:
    - main
```

## 🚀 Best Practices

### 1. Pipeline Organization
```yaml
# Use stages and dependencies
stages:
  - test
  - build
  - security
  - deploy

test:unit:
  stage: test
  script:
    - go test ./...

test:integration:
  stage: test
  script:
    - go test -tags=integration ./...

build:docker:
  stage: build
  dependencies:
    - test:unit
    - test:integration
  script:
    - docker build -t $IMAGE_TAG .
```

### 2. Security Best Practices
```yaml
# Use variables for secrets
variables:
  DOCKER_REGISTRY: $CI_REGISTRY
  IMAGE_TAG: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA

deploy:
  stage: deploy
  script:
    - kubectl set image deployment/app-deployment app=$IMAGE_TAG -n production
  only:
    - main
  when: manual
```

### 3. Performance Optimization
```yaml
# Use caching and parallel jobs
test:unit:
  stage: test
  image: golang:1.21
  cache:
    paths:
      - .go/pkg/mod/
  script:
    - go test ./...

test:integration:
  stage: test
  image: golang:1.21
  cache:
    paths:
      - .go/pkg/mod/
  script:
    - go test -tags=integration ./...
```

## 🏢 Industry Insights

### GitLab's CI Usage
- **Integrated Platform**: All-in-one DevOps
- **Container Registry**: Built-in Docker registry
- **Security Scanning**: Automated vulnerability scanning
- **Auto DevOps**: Automated pipeline generation

### Enterprise GitLab Strategy
- **Self-Hosted**: Full control over infrastructure
- **Security**: Built-in security scanning
- **Compliance**: Audit trails and compliance
- **Scalability**: Enterprise-grade performance

## 🎯 Interview Questions

### Basic Level
1. **What is GitLab CI?**
   - Integrated CI/CD platform
   - Built into GitLab
   - YAML-based configuration
   - Container registry

2. **What are GitLab CI stages?**
   - Test stage
   - Build stage
   - Deploy stage
   - Security stage

3. **What are GitLab CI variables?**
   - Predefined variables
   - Custom variables
   - Secret variables
   - Environment variables

### Intermediate Level
4. **How do you optimize GitLab CI performance?**
   ```yaml
   # Use caching and parallel jobs
   test:unit:
     stage: test
     cache:
       paths:
         - .go/pkg/mod/
     script:
       - go test ./...
   ```

5. **How do you handle GitLab CI security?**
   - Use secret variables
   - Enable security scanning
   - Use protected branches
   - Implement access controls

6. **How do you implement GitLab CI testing?**
   - Unit testing
   - Integration testing
   - End-to-end testing
   - Performance testing

### Advanced Level
7. **How do you implement GitLab CI patterns?**
   - Pipeline templates
   - Include files
   - Reusable components
   - Multi-project pipelines

8. **How do you handle GitLab CI scaling?**
   - Runner configuration
   - Resource optimization
   - Load balancing
   - Performance monitoring

9. **How do you implement GitLab CI monitoring?**
   - Pipeline monitoring
   - Performance metrics
   - Alerting
   - Logging

---

**Next**: [ArgoCD](./ArgoCD.md) - GitOps deployment, continuous delivery
