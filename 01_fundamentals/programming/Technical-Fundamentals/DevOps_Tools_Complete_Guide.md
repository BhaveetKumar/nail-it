---
# Auto-generated front matter
Title: Devops Tools Complete Guide
LastUpdated: 2025-11-06T20:45:59.104752
Tags: []
Status: draft
---

# 🚀 DevOps Tools Complete Guide - Node.js Perspective

> **Comprehensive guide to DevOps tools and practices with Node.js implementations**

## 🎯 **Overview**

This guide covers essential DevOps tools, CI/CD pipelines, containerization, orchestration, monitoring, and infrastructure as code. Each tool is explained with practical Node.js examples and production-ready configurations.

## 📚 **Table of Contents**

1. [Version Control with Git](#version-control-with-git)
2. [CI/CD Pipelines](#cicd-pipelines)
3. [Containerization with Docker](#containerization-with-docker)
4. [Orchestration with Kubernetes](#orchestration-with-kubernetes)
5. [Infrastructure as Code](#infrastructure-as-code)
6. [Monitoring and Observability](#monitoring-and-observability)
7. [Security and Compliance](#security-and-compliance)

---

## 📝 **Version Control with Git**

### **Git Workflow Management**

```javascript
// Git Operations with Node.js
const { execSync } = require('child_process');
const fs = require('fs').promises;
const path = require('path');

class GitManager {
    constructor(repoPath = process.cwd()) {
        this.repoPath = repoPath;
    }
    
    async executeGitCommand(command) {
        try {
            const result = execSync(command, { 
                cwd: this.repoPath, 
                encoding: 'utf8',
                stdio: 'pipe'
            });
            return { success: true, output: result.trim() };
        } catch (error) {
            return { 
                success: false, 
                error: error.message,
                stderr: error.stderr?.toString() || ''
            };
        }
    }
    
    async getStatus() {
        return await this.executeGitCommand('git status --porcelain');
    }
    
    async addFiles(files = []) {
        if (files.length === 0) {
            return await this.executeGitCommand('git add .');
        }
        return await this.executeGitCommand(`git add ${files.join(' ')}`);
    }
    
    async commit(message, options = {}) {
        const flags = [];
        if (options.amend) flags.push('--amend');
        if (options.noVerify) flags.push('--no-verify');
        
        const command = `git commit ${flags.join(' ')} -m "${message}"`;
        return await this.executeGitCommand(command);
    }
    
    async push(branch = 'main', options = {}) {
        const flags = [];
        if (options.force) flags.push('--force');
        if (options.setUpstream) flags.push('--set-upstream');
        
        const command = `git push ${flags.join(' ')} origin ${branch}`;
        return await this.executeGitCommand(command);
    }
    
    async pull(branch = 'main') {
        return await this.executeGitCommand(`git pull origin ${branch}`);
    }
    
    async createBranch(branchName, fromBranch = 'main') {
        const result = await this.executeGitCommand(`git checkout -b ${branchName} ${fromBranch}`);
        if (result.success) {
            return await this.executeGitCommand(`git push --set-upstream origin ${branchName}`);
        }
        return result;
    }
    
    async mergeBranch(sourceBranch, targetBranch = 'main') {
        // Switch to target branch
        await this.executeGitCommand(`git checkout ${targetBranch}`);
        await this.pull(targetBranch);
        
        // Merge source branch
        return await this.executeGitCommand(`git merge ${sourceBranch}`);
    }
    
    async getBranches() {
        const result = await this.executeGitCommand('git branch -a');
        if (result.success) {
            return result.output.split('\n')
                .map(branch => branch.trim())
                .filter(branch => branch.length > 0);
        }
        return [];
    }
    
    async getLog(limit = 10) {
        const result = await this.executeGitCommand(`git log --oneline -${limit}`);
        if (result.success) {
            return result.output.split('\n')
                .map(commit => {
                    const [hash, ...messageParts] = commit.split(' ');
                    return {
                        hash,
                        message: messageParts.join(' ')
                    };
                });
        }
        return [];
    }
}

// Git Hooks Management
class GitHooksManager {
    constructor(repoPath = process.cwd()) {
        this.hooksPath = path.join(repoPath, '.git', 'hooks');
    }
    
    async createPreCommitHook() {
        const hookContent = `#!/bin/sh
# Pre-commit hook for Node.js project

echo "Running pre-commit checks..."

# Run linting
npm run lint
if [ $? -ne 0 ]; then
    echo "Linting failed. Please fix errors before committing."
    exit 1
fi

# Run tests
npm test
if [ $? -ne 0 ]; then
    echo "Tests failed. Please fix tests before committing."
    exit 1
fi

# Check for console.log statements
if git diff --cached --name-only | xargs grep -l "console\\.log"; then
    echo "Warning: console.log statements found in staged files."
    echo "Consider removing them before committing."
fi

echo "Pre-commit checks passed!"
exit 0
`;

        const hookPath = path.join(this.hooksPath, 'pre-commit');
        await fs.writeFile(hookPath, hookContent, { mode: 0o755 });
        
        return { success: true, path: hookPath };
    }
    
    async createCommitMsgHook() {
        const hookContent = `#!/bin/sh
# Commit message hook

commit_regex='^(feat|fix|docs|style|refactor|test|chore)(\\(.+\\))?: .{1,50}'

if ! grep -qE "$commit_regex" "$1"; then
    echo "Invalid commit message format!"
    echo "Format: type(scope): description"
    echo "Types: feat, fix, docs, style, refactor, test, chore"
    exit 1
fi
`;

        const hookPath = path.join(this.hooksPath, 'commit-msg');
        await fs.writeFile(hookPath, hookContent, { mode: 0o755 });
        
        return { success: true, path: hookPath };
    }
}
```

---

## 🔄 **CI/CD Pipelines**

### **GitHub Actions Workflow**

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        node-version: [16.x, 18.x, 20.x]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Use Node.js ${{ matrix.node-version }}
      uses: actions/setup-node@v3
      with:
        node-version: ${{ matrix.node-version }}
        cache: 'npm'
    
    - name: Install dependencies
      run: npm ci
    
    - name: Run linting
      run: npm run lint
    
    - name: Run tests
      run: npm test
    
    - name: Run security audit
      run: npm audit --audit-level moderate
    
    - name: Generate coverage report
      run: npm run test:coverage
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Use Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18.x'
        cache: 'npm'
    
    - name: Install dependencies
      run: npm ci
    
    - name: Build application
      run: npm run build
    
    - name: Build Docker image
      run: docker build -t ${{ github.repository }}:${{ github.sha }} .
    
    - name: Login to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
    
    - name: Push Docker image
      run: docker push ${{ github.repository }}:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Deploy to production
      run: |
        echo "Deploying to production..."
        # Add deployment commands here
```

### **Jenkins Pipeline**

```groovy
// Jenkinsfile
pipeline {
    agent any
    
    environment {
        NODE_VERSION = '18'
        DOCKER_IMAGE = "${env.JOB_NAME}:${env.BUILD_NUMBER}"
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Install Dependencies') {
            steps {
                sh 'npm ci'
            }
        }
        
        stage('Lint') {
            steps {
                sh 'npm run lint'
            }
        }
        
        stage('Test') {
            steps {
                sh 'npm test'
            }
            post {
                always {
                    publishTestResults testResultsPattern: 'test-results.xml'
                    publishCoverage adapters: [
                        coberturaAdapter('coverage/cobertura-coverage.xml')
                    ]
                }
            }
        }
        
        stage('Build') {
            steps {
                sh 'npm run build'
            }
        }
        
        stage('Docker Build') {
            steps {
                script {
                    def image = docker.build("${DOCKER_IMAGE}")
                    docker.withRegistry('https://registry.hub.docker.com', 'docker-hub-credentials') {
                        image.push()
                    }
                }
            }
        }
        
        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                sh 'kubectl set image deployment/app app=${DOCKER_IMAGE}'
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
        failure {
            emailext (
                subject: "Build Failed: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                body: "Build failed. Check console output for details.",
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
    }
}
```

---

## 🐳 **Containerization with Docker**

### **Docker Configuration**

```dockerfile
# Dockerfile
FROM node:18-alpine AS base

# Install dependencies only when needed
FROM base AS deps
RUN apk add --no-cache libc6-compat
WORKDIR /app

# Install dependencies based on the preferred package manager
COPY package.json package-lock.json* ./
RUN npm ci --only=production

# Rebuild the source code only when needed
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Build the application
RUN npm run build

# Production image, copy all the files and run next
FROM base AS runner
WORKDIR /app

ENV NODE_ENV production

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public

# Set the correct permission for prerender cache
RUN mkdir .next
RUN chown nextjs:nodejs .next

# Automatically leverage output traces to reduce image size
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

ENV PORT 3000
ENV HOSTNAME "0.0.0.0"

CMD ["node", "server.js"]
```

### **Docker Compose Configuration**

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - DATABASE_URL=postgresql://user:password@db:5432/mydb
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=mydb
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    command: redis-server --appendonly yes

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### **Docker Management with Node.js**

```javascript
// Docker Operations with Node.js
const { execSync } = require('child_process');
const fs = require('fs').promises;

class DockerManager {
    constructor() {
        this.registry = process.env.DOCKER_REGISTRY || 'docker.io';
        this.namespace = process.env.DOCKER_NAMESPACE || 'myorg';
    }
    
    async executeDockerCommand(command) {
        try {
            const result = execSync(command, { 
                encoding: 'utf8',
                stdio: 'pipe'
            });
            return { success: true, output: result.trim() };
        } catch (error) {
            return { 
                success: false, 
                error: error.message,
                stderr: error.stderr?.toString() || ''
            };
        }
    }
    
    async buildImage(imageName, tag = 'latest', context = '.') {
        const fullImageName = `${this.namespace}/${imageName}:${tag}`;
        const command = `docker build -t ${fullImageName} ${context}`;
        
        const result = await this.executeDockerCommand(command);
        if (result.success) {
            return { success: true, imageName: fullImageName };
        }
        return result;
    }
    
    async pushImage(imageName, tag = 'latest') {
        const fullImageName = `${this.namespace}/${imageName}:${tag}`;
        const command = `docker push ${fullImageName}`;
        
        return await this.executeDockerCommand(command);
    }
    
    async pullImage(imageName, tag = 'latest') {
        const fullImageName = `${this.namespace}/${imageName}:${tag}`;
        const command = `docker pull ${fullImageName}`;
        
        return await this.executeDockerCommand(command);
    }
    
    async runContainer(imageName, options = {}) {
        const {
            tag = 'latest',
            name,
            ports = {},
            volumes = {},
            environment = {},
            detach = true
        } = options;
        
        const fullImageName = `${this.namespace}/${imageName}:${tag}`;
        let command = 'docker run';
        
        if (detach) command += ' -d';
        if (name) command += ` --name ${name}`;
        
        // Add port mappings
        Object.entries(ports).forEach(([host, container]) => {
            command += ` -p ${host}:${container}`;
        });
        
        // Add volume mappings
        Object.entries(volumes).forEach(([host, container]) => {
            command += ` -v ${host}:${container}`;
        });
        
        // Add environment variables
        Object.entries(environment).forEach(([key, value]) => {
            command += ` -e ${key}=${value}`;
        });
        
        command += ` ${fullImageName}`;
        
        return await this.executeDockerCommand(command);
    }
    
    async stopContainer(containerName) {
        return await this.executeDockerCommand(`docker stop ${containerName}`);
    }
    
    async removeContainer(containerName) {
        return await this.executeDockerCommand(`docker rm ${containerName}`);
    }
    
    async listContainers(all = false) {
        const flag = all ? '-a' : '';
        const result = await this.executeDockerCommand(`docker ps ${flag}`);
        
        if (result.success) {
            const lines = result.output.split('\n').slice(1);
            return lines.map(line => {
                const parts = line.split(/\s+/);
                return {
                    containerId: parts[0],
                    image: parts[1],
                    command: parts[2],
                    created: parts[3],
                    status: parts[4],
                    ports: parts[5],
                    names: parts[6]
                };
            });
        }
        return [];
    }
    
    async getContainerLogs(containerName, tail = 100) {
        return await this.executeDockerCommand(`docker logs --tail ${tail} ${containerName}`);
    }
    
    async executeInContainer(containerName, command) {
        return await this.executeDockerCommand(`docker exec ${containerName} ${command}`);
    }
    
    async createDockerfile(projectPath, options = {}) {
        const {
            nodeVersion = '18',
            baseImage = 'alpine',
            port = 3000,
            workingDir = '/app'
        } = options;
        
        const dockerfileContent = `FROM node:${nodeVersion}-${baseImage}

WORKDIR ${workingDir}

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY . .

# Build application
RUN npm run build

# Expose port
EXPOSE ${port}

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
  CMD curl -f http://localhost:${port}/health || exit 1

# Start application
CMD ["npm", "start"]
`;

        const dockerfilePath = `${projectPath}/Dockerfile`;
        await fs.writeFile(dockerfilePath, dockerfileContent);
        
        return { success: true, path: dockerfilePath };
    }
}
```

---

## ☸️ **Orchestration with Kubernetes**

### **Kubernetes Manifests**

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nodejs-app
  labels:
    app: nodejs-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nodejs-app
  template:
    metadata:
      labels:
        app: nodejs-app
    spec:
      containers:
      - name: nodejs-app
        image: myorg/nodejs-app:latest
        ports:
        - containerPort: 3000
        env:
        - name: NODE_ENV
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: database-url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: nodejs-app-service
spec:
  selector:
    app: nodejs-app
  ports:
  - protocol: TCP
    port: 80
    targetPort: 3000
  type: LoadBalancer
```

### **Kubernetes Management with Node.js**

```javascript
// Kubernetes Operations with Node.js
const k8s = require('@kubernetes/client-node');

class KubernetesManager {
    constructor() {
        this.kc = new k8s.KubeConfig();
        this.kc.loadFromDefault();
        
        this.k8sApi = this.kc.makeApiClient(k8s.CoreV1Api);
        this.appsV1Api = this.kc.makeApiClient(k8s.AppsV1Api);
    }
    
    async createDeployment(deploymentConfig) {
        try {
            const deployment = new k8s.V1Deployment();
            deployment.apiVersion = 'apps/v1';
            deployment.kind = 'Deployment';
            deployment.metadata = deploymentConfig.metadata;
            deployment.spec = deploymentConfig.spec;
            
            const result = await this.appsV1Api.createNamespacedDeployment(
                deploymentConfig.metadata.namespace || 'default',
                deployment
            );
            
            return { success: true, deployment: result.body };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async createService(serviceConfig) {
        try {
            const service = new k8s.V1Service();
            service.apiVersion = 'v1';
            service.kind = 'Service';
            service.metadata = serviceConfig.metadata;
            service.spec = serviceConfig.spec;
            
            const result = await this.k8sApi.createNamespacedService(
                serviceConfig.metadata.namespace || 'default',
                service
            );
            
            return { success: true, service: result.body };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async getPods(namespace = 'default') {
        try {
            const result = await this.k8sApi.listNamespacedPod(namespace);
            return { success: true, pods: result.body.items };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async getDeployments(namespace = 'default') {
        try {
            const result = await this.appsV1Api.listNamespacedDeployment(namespace);
            return { success: true, deployments: result.body.items };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async scaleDeployment(name, replicas, namespace = 'default') {
        try {
            const result = await this.appsV1Api.patchNamespacedDeploymentScale(
                name,
                namespace,
                { spec: { replicas } },
                undefined,
                undefined,
                undefined,
                undefined,
                { headers: { 'Content-Type': 'application/merge-patch+json' } }
            );
            
            return { success: true, deployment: result.body };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async updateDeploymentImage(name, image, namespace = 'default') {
        try {
            const deployment = await this.appsV1Api.readNamespacedDeployment(name, namespace);
            deployment.body.spec.template.spec.containers[0].image = image;
            
            const result = await this.appsV1Api.replaceNamespacedDeployment(
                name,
                namespace,
                deployment.body
            );
            
            return { success: true, deployment: result.body };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async createConfigMap(name, data, namespace = 'default') {
        try {
            const configMap = new k8s.V1ConfigMap();
            configMap.apiVersion = 'v1';
            configMap.kind = 'ConfigMap';
            configMap.metadata = { name, namespace };
            configMap.data = data;
            
            const result = await this.k8sApi.createNamespacedConfigMap(namespace, configMap);
            return { success: true, configMap: result.body };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async createSecret(name, data, namespace = 'default') {
        try {
            const secret = new k8s.V1Secret();
            secret.apiVersion = 'v1';
            secret.kind = 'Secret';
            secret.metadata = { name, namespace };
            secret.type = 'Opaque';
            secret.data = {};
            
            // Encode data to base64
            Object.entries(data).forEach(([key, value]) => {
                secret.data[key] = Buffer.from(value).toString('base64');
            });
            
            const result = await this.k8sApi.createNamespacedSecret(namespace, secret);
            return { success: true, secret: result.body };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
}
```

---

## 🏗️ **Infrastructure as Code**

### **Terraform Configuration**

```hcl
# terraform/main.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# VPC
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${var.project_name}-vpc"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${var.project_name}-igw"
  }
}

# Public Subnets
resource "aws_subnet" "public" {
  count = length(var.availability_zones)

  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.${count.index + 1}.0/24"
  availability_zone       = var.availability_zones[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.project_name}-public-subnet-${count.index + 1}"
  }
}

# EKS Cluster
resource "aws_eks_cluster" "main" {
  name     = "${var.project_name}-cluster"
  role_arn = aws_iam_role.eks_cluster.arn
  version  = var.kubernetes_version

  vpc_config {
    subnet_ids = aws_subnet.public[*].id
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
  ]

  tags = {
    Name = "${var.project_name}-cluster"
  }
}

# EKS Node Group
resource "aws_eks_node_group" "main" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "${var.project_name}-nodes"
  node_role_arn   = aws_iam_role.eks_node_group.arn
  subnet_ids      = aws_subnet.public[*].id

  scaling_config {
    desired_size = var.node_group_desired_size
    max_size     = var.node_group_max_size
    min_size     = var.node_group_min_size
  }

  instance_types = var.node_group_instance_types

  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.eks_container_registry_policy,
  ]

  tags = {
    Name = "${var.project_name}-node-group"
  }
}
```

### **Terraform Management with Node.js**

```javascript
// Terraform Operations with Node.js
const { execSync } = require('child_process');
const fs = require('fs').promises;
const path = require('path');

class TerraformManager {
    constructor(workingDir = './terraform') {
        this.workingDir = workingDir;
    }
    
    async executeTerraformCommand(command) {
        try {
            const result = execSync(command, { 
                cwd: this.workingDir,
                encoding: 'utf8',
                stdio: 'pipe'
            });
            return { success: true, output: result.trim() };
        } catch (error) {
            return { 
                success: false, 
                error: error.message,
                stderr: error.stderr?.toString() || ''
            };
        }
    }
    
    async init() {
        return await this.executeTerraformCommand('terraform init');
    }
    
    async plan(variables = {}) {
        let command = 'terraform plan';
        
        Object.entries(variables).forEach(([key, value]) => {
            command += ` -var="${key}=${value}"`;
        });
        
        return await this.executeTerraformCommand(command);
    }
    
    async apply(variables = {}, autoApprove = false) {
        let command = 'terraform apply';
        
        if (autoApprove) {
            command += ' -auto-approve';
        }
        
        Object.entries(variables).forEach(([key, value]) => {
            command += ` -var="${key}=${value}"`;
        });
        
        return await this.executeTerraformCommand(command);
    }
    
    async destroy(variables = {}, autoApprove = false) {
        let command = 'terraform destroy';
        
        if (autoApprove) {
            command += ' -auto-approve';
        }
        
        Object.entries(variables).forEach(([key, value]) => {
            command += ` -var="${key}=${value}"`;
        });
        
        return await this.executeTerraformCommand(command);
    }
    
    async output() {
        return await this.executeTerraformCommand('terraform output -json');
    }
    
    async show() {
        return await this.executeTerraformCommand('terraform show -json');
    }
    
    async validate() {
        return await this.executeTerraformCommand('terraform validate');
    }
    
    async format() {
        return await this.executeTerraformCommand('terraform fmt -recursive');
    }
    
    async createVariablesFile(variables) {
        const variablesContent = Object.entries(variables)
            .map(([key, value]) => `variable "${key}" {\n  default = "${value}"\n}`)
            .join('\n\n');
        
        const variablesPath = path.join(this.workingDir, 'variables.tf');
        await fs.writeFile(variablesPath, variablesContent);
        
        return { success: true, path: variablesPath };
    }
    
    async createOutputsFile(outputs) {
        const outputsContent = Object.entries(outputs)
            .map(([key, value]) => `output "${key}" {\n  value = ${value}\n}`)
            .join('\n\n');
        
        const outputsPath = path.join(this.workingDir, 'outputs.tf');
        await fs.writeFile(outputsPath, outputsContent);
        
        return { success: true, path: outputsPath };
    }
}
```

---

## 📊 **Monitoring and Observability**

### **Prometheus and Grafana Setup**

```javascript
// Prometheus Metrics Collection
const client = require('prom-client');

class MetricsCollector {
    constructor() {
        this.register = new client.Registry();
        
        // Add default metrics
        client.collectDefaultMetrics({ register: this.register });
        
        // Custom metrics
        this.httpRequestDuration = new client.Histogram({
            name: 'http_request_duration_seconds',
            help: 'Duration of HTTP requests in seconds',
            labelNames: ['method', 'route', 'status_code'],
            buckets: [0.1, 0.3, 0.5, 0.7, 1, 3, 5, 7, 10]
        });
        
        this.httpRequestTotal = new client.Counter({
            name: 'http_requests_total',
            help: 'Total number of HTTP requests',
            labelNames: ['method', 'route', 'status_code']
        });
        
        this.activeConnections = new client.Gauge({
            name: 'active_connections',
            help: 'Number of active connections'
        });
        
        this.databaseQueryDuration = new client.Histogram({
            name: 'database_query_duration_seconds',
            help: 'Duration of database queries in seconds',
            labelNames: ['operation', 'table']
        });
        
        this.register.registerMetric(this.httpRequestDuration);
        this.register.registerMetric(this.httpRequestTotal);
        this.register.registerMetric(this.activeConnections);
        this.register.registerMetric(this.databaseQueryDuration);
    }
    
    recordHttpRequest(method, route, statusCode, duration) {
        this.httpRequestDuration
            .labels(method, route, statusCode)
            .observe(duration);
        
        this.httpRequestTotal
            .labels(method, route, statusCode)
            .inc();
    }
    
    setActiveConnections(count) {
        this.activeConnections.set(count);
    }
    
    recordDatabaseQuery(operation, table, duration) {
        this.databaseQueryDuration
            .labels(operation, table)
            .observe(duration);
    }
    
    async getMetrics() {
        return await this.register.metrics();
    }
}

// Express.js Middleware for Metrics
function metricsMiddleware(metricsCollector) {
    return (req, res, next) => {
        const start = Date.now();
        
        res.on('finish', () => {
            const duration = (Date.now() - start) / 1000;
            metricsCollector.recordHttpRequest(
                req.method,
                req.route?.path || req.path,
                res.statusCode,
                duration
            );
        });
        
        next();
    };
}

// Health Check Endpoint
function createHealthCheckEndpoint(app, metricsCollector) {
    app.get('/health', (req, res) => {
        res.json({
            status: 'healthy',
            timestamp: new Date().toISOString(),
            uptime: process.uptime(),
            memory: process.memoryUsage(),
            version: process.env.npm_package_version || '1.0.0'
        });
    });
    
    app.get('/metrics', async (req, res) => {
        res.set('Content-Type', client.register.contentType);
        const metrics = await metricsCollector.getMetrics();
        res.end(metrics);
    });
}
```

### **Logging with Winston**

```javascript
// Advanced Logging Configuration
const winston = require('winston');
const { combine, timestamp, errors, json, printf, colorize } = winston.format;

class LoggerManager {
    constructor(options = {}) {
        this.options = {
            level: options.level || 'info',
            service: options.service || 'nodejs-app',
            environment: options.environment || 'development',
            ...options
        };
        
        this.logger = this.createLogger();
    }
    
    createLogger() {
        const logFormat = printf(({ level, message, timestamp, service, ...meta }) => {
            return JSON.stringify({
                timestamp,
                level,
                service,
                message,
                ...meta
            });
        });
        
        const transports = [
            new winston.transports.Console({
                format: combine(
                    colorize(),
                    timestamp(),
                    errors({ stack: true }),
                    logFormat
                )
            })
        ];
        
        // Add file transport in production
        if (this.options.environment === 'production') {
            transports.push(
                new winston.transports.File({
                    filename: 'logs/error.log',
                    level: 'error',
                    format: combine(timestamp(), errors({ stack: true }), json())
                }),
                new winston.transports.File({
                    filename: 'logs/combined.log',
                    format: combine(timestamp(), errors({ stack: true }), json())
                })
            );
        }
        
        return winston.createLogger({
            level: this.options.level,
            format: combine(
                timestamp(),
                errors({ stack: true }),
                json()
            ),
            defaultMeta: {
                service: this.options.service,
                environment: this.options.environment
            },
            transports
        });
    }
    
    info(message, meta = {}) {
        this.logger.info(message, meta);
    }
    
    error(message, error = null, meta = {}) {
        if (error instanceof Error) {
            this.logger.error(message, { error: error.message, stack: error.stack, ...meta });
        } else {
            this.logger.error(message, { error, ...meta });
        }
    }
    
    warn(message, meta = {}) {
        this.logger.warn(message, meta);
    }
    
    debug(message, meta = {}) {
        this.logger.debug(message, meta);
    }
    
    // Structured logging for specific use cases
    logHttpRequest(req, res, responseTime) {
        this.info('HTTP Request', {
            method: req.method,
            url: req.url,
            statusCode: res.statusCode,
            responseTime: `${responseTime}ms`,
            userAgent: req.get('User-Agent'),
            ip: req.ip
        });
    }
    
    logDatabaseQuery(query, duration, error = null) {
        if (error) {
            this.error('Database Query Failed', error, { query, duration });
        } else {
            this.debug('Database Query', { query, duration });
        }
    }
    
    logBusinessEvent(event, data = {}) {
        this.info('Business Event', { event, ...data });
    }
}
```

---

## 🔒 **Security and Compliance**

### **Security Scanning and Compliance**

```javascript
// Security Scanner
const { execSync } = require('child_process');
const fs = require('fs').promises;

class SecurityScanner {
    constructor() {
        this.scanResults = [];
    }
    
    async runNpmAudit() {
        try {
            const result = execSync('npm audit --json', { 
                encoding: 'utf8',
                stdio: 'pipe'
            });
            
            const auditData = JSON.parse(result);
            return {
                success: true,
                vulnerabilities: auditData.vulnerabilities,
                summary: auditData.metadata
            };
        } catch (error) {
            return {
                success: false,
                error: error.message
            };
        }
    }
    
    async runSnykScan() {
        try {
            const result = execSync('snyk test --json', { 
                encoding: 'utf8',
                stdio: 'pipe'
            });
            
            const snykData = JSON.parse(result);
            return {
                success: true,
                vulnerabilities: snykData.vulnerabilities
            };
        } catch (error) {
            return {
                success: false,
                error: error.message
            };
        }
    }
    
    async scanForSecrets() {
        const secretPatterns = [
            /password\s*=\s*['"][^'"]+['"]/gi,
            /api[_-]?key\s*=\s*['"][^'"]+['"]/gi,
            /secret\s*=\s*['"][^'"]+['"]/gi,
            /token\s*=\s*['"][^'"]+['"]/gi,
            /private[_-]?key\s*=\s*['"][^'"]+['"]/gi
        ];
        
        const files = await this.getAllFiles('./src');
        const findings = [];
        
        for (const file of files) {
            try {
                const content = await fs.readFile(file, 'utf8');
                
                secretPatterns.forEach((pattern, index) => {
                    const matches = content.match(pattern);
                    if (matches) {
                        findings.push({
                            file,
                            pattern: index,
                            matches: matches.length
                        });
                    }
                });
            } catch (error) {
                // Skip files that can't be read
            }
        }
        
        return { success: true, findings };
    }
    
    async getAllFiles(dir) {
        const files = [];
        const entries = await fs.readdir(dir, { withFileTypes: true });
        
        for (const entry of entries) {
            const fullPath = `${dir}/${entry.name}`;
            
            if (entry.isDirectory()) {
                const subFiles = await this.getAllFiles(fullPath);
                files.push(...subFiles);
            } else if (entry.isFile() && this.isCodeFile(entry.name)) {
                files.push(fullPath);
            }
        }
        
        return files;
    }
    
    isCodeFile(filename) {
        const codeExtensions = ['.js', '.ts', '.jsx', '.tsx', '.json', '.env'];
        return codeExtensions.some(ext => filename.endsWith(ext));
    }
    
    async generateSecurityReport() {
        const report = {
            timestamp: new Date().toISOString(),
            npmAudit: await this.runNpmAudit(),
            snykScan: await this.runSnykScan(),
            secretScan: await this.scanForSecrets()
        };
        
        await fs.writeFile('security-report.json', JSON.stringify(report, null, 2));
        return report;
    }
}
```

---

## 🎯 **Key Takeaways**

### **Version Control**
- Use Git hooks for automated quality checks
- Implement proper branching strategies
- Automate common Git operations

### **CI/CD Pipelines**
- Automate testing, building, and deployment
- Use matrix builds for multiple environments
- Implement proper security scanning

### **Containerization**
- Use multi-stage builds for optimization
- Implement health checks and proper logging
- Use Docker Compose for local development

### **Orchestration**
- Use Kubernetes for production deployments
- Implement proper resource limits and health checks
- Use ConfigMaps and Secrets for configuration

### **Infrastructure as Code**
- Use Terraform for infrastructure management
- Implement proper state management
- Use modules for reusability

### **Monitoring**
- Implement comprehensive metrics collection
- Use structured logging
- Set up proper alerting

### **Security**
- Regular security scanning
- Secret management
- Compliance monitoring

---

**🎉 This comprehensive guide covers all essential DevOps tools and practices with Node.js implementations!**
