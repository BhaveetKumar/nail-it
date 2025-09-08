# 🔧 Jenkins: Self-Hosted CI/CD and Pipeline Automation

> **Master Jenkins for self-hosted CI/CD pipelines and advanced automation**

## 📚 Concept

Jenkins is an open-source automation server that provides continuous integration and continuous delivery (CI/CD) capabilities. It's highly extensible and supports building, testing, and deploying applications.

### Key Features
- **Pipeline as Code**: Define pipelines using Groovy DSL
- **Plugin Ecosystem**: Extensive plugin library
- **Distributed Builds**: Master-slave architecture
- **Blue Ocean**: Modern UI for pipeline visualization
- **Multibranch Pipelines**: Automatic branch detection
- **Self-Hosted**: Full control over infrastructure

## 🏗️ Jenkins Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Jenkins Master                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Web UI    │  │  Pipeline   │  │   Plugin    │     │
│  │   Dashboard │  │   Engine    │  │   Manager   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Jenkins Agents                        │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Linux     │  │   Windows   │  │    macOS    │ │ │
│  │  │   Agent     │  │   Agent     │  │   Agent     │ │ │
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

### Jenkinsfile (Declarative Pipeline)

```groovy
// Jenkinsfile
pipeline {
    agent any
    
    environment {
        DOCKER_REGISTRY = 'your-registry.com'
        IMAGE_NAME = 'your-app'
        KUBECONFIG = credentials('kubeconfig')
        AWS_ACCESS_KEY_ID = credentials('aws-access-key')
        AWS_SECRET_ACCESS_KEY = credentials('aws-secret-key')
    }
    
    options {
        buildDiscarder(logRotator(numToKeepStr: '10'))
        timeout(time: 30, unit: 'MINUTES')
        timestamps()
        ansiColor('xterm')
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
                script {
                    env.GIT_COMMIT_SHORT = sh(
                        script: 'git rev-parse --short HEAD',
                        returnStdout: true
                    ).trim()
                }
            }
        }
        
        stage('Build') {
            parallel {
                stage('Build Go App') {
                    when {
                        anyOf {
                            branch 'main'
                            branch 'develop'
                            changeRequest()
                        }
                    }
                    steps {
                        script {
                            def goVersion = '1.21'
                            sh """
                                docker run --rm -v \${PWD}:/app -w /app golang:${goVersion} \
                                go build -ldflags="-s -w -X main.version=\${GIT_COMMIT_SHORT}" \
                                -o app ./cmd/server
                            """
                        }
                    }
                }
                
                stage('Build Docker Image') {
                    steps {
                        script {
                            def imageTag = "${env.DOCKER_REGISTRY}/${env.IMAGE_NAME}:${env.GIT_COMMIT_SHORT}"
                            def latestTag = "${env.DOCKER_REGISTRY}/${env.IMAGE_NAME}:latest"
                            
                            sh """
                                docker build -t ${imageTag} .
                                docker tag ${imageTag} ${latestTag}
                            """
                            
                            env.DOCKER_IMAGE = imageTag
                        }
                    }
                }
            }
        }
        
        stage('Test') {
            parallel {
                stage('Unit Tests') {
                    steps {
                        sh """
                            docker run --rm -v \${PWD}:/app -w /app golang:1.21 \
                            go test -v -race -coverprofile=coverage.out ./...
                        """
                    }
                    post {
                        always {
                            publishTestResults testResultsPattern: 'test-results.xml'
                            publishCoverage adapters: [
                                coberturaAdapter('coverage.xml')
                            ], sourceFileResolver: sourceFiles('STORE_LAST_BUILD')
                        }
                    }
                }
                
                stage('Integration Tests') {
                    steps {
                        sh """
                            docker run --rm -v \${PWD}:/app -w /app golang:1.21 \
                            go test -v -tags=integration ./tests/integration/...
                        """
                    }
                }
                
                stage('Security Scan') {
                    steps {
                        script {
                            sh """
                                docker run --rm -v \${PWD}:/app -w /app \
                                securecodewarrior/gosec ./...
                            """
                        }
                    }
                }
            }
        }
        
        stage('Quality Gate') {
            steps {
                script {
                    def qualityGate = sh(
                        script: 'go test -v -coverprofile=coverage.out ./... | grep -o "coverage: [0-9.]*%" | cut -d" " -f2 | cut -d"%" -f1',
                        returnStdout: true
                    ).trim()
                    
                    def coverage = qualityGate as Float
                    if (coverage < 80.0) {
                        error "Coverage ${coverage}% is below threshold of 80%"
                    }
                    
                    echo "Quality gate passed with ${coverage}% coverage"
                }
            }
        }
        
        stage('Push to Registry') {
            when {
                anyOf {
                    branch 'main'
                    branch 'develop'
                }
            }
            steps {
                script {
                    sh """
                        docker login ${env.DOCKER_REGISTRY} -u \${DOCKER_USERNAME} -p \${DOCKER_PASSWORD}
                        docker push ${env.DOCKER_IMAGE}
                        docker push ${env.DOCKER_REGISTRY}/${env.IMAGE_NAME}:latest
                    """
                }
            }
        }
        
        stage('Deploy to Staging') {
            when {
                branch 'develop'
            }
            steps {
                script {
                    sh """
                        kubectl config use-context staging
                        kubectl set image deployment/app-deployment \
                        app=${env.DOCKER_IMAGE} -n staging
                        kubectl rollout status deployment/app-deployment -n staging
                    """
                }
            }
            post {
                success {
                    sh """
                        kubectl get pods -n staging
                        kubectl get services -n staging
                    """
                }
            }
        }
        
        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            steps {
                script {
                    sh """
                        kubectl config use-context production
                        kubectl set image deployment/app-deployment \
                        app=${env.DOCKER_IMAGE} -n production
                        kubectl rollout status deployment/app-deployment -n production
                    """
                }
            }
            post {
                success {
                    sh """
                        kubectl get pods -n production
                        kubectl get services -n production
                    """
                }
            }
        }
        
        stage('Smoke Tests') {
            when {
                anyOf {
                    branch 'main'
                    branch 'develop'
                }
            }
            steps {
                script {
                    def namespace = env.BRANCH_NAME == 'main' ? 'production' : 'staging'
                    sh """
                        kubectl config use-context ${namespace}
                        kubectl get pods -n ${namespace}
                        kubectl get services -n ${namespace}
                        # Add smoke test commands here
                    """
                }
            }
        }
    }
    
    post {
        always {
            cleanWs()
            script {
                if (env.DOCKER_IMAGE) {
                    sh "docker rmi ${env.DOCKER_IMAGE} || true"
                }
            }
        }
        success {
            script {
                if (env.BRANCH_NAME == 'main') {
                    slackSend(
                        channel: '#deployments',
                        color: 'good',
                        message: "✅ Production deployment successful: ${env.BUILD_URL}"
                    )
                }
            }
        }
        failure {
            script {
                slackSend(
                    channel: '#deployments',
                    color: 'danger',
                    message: "❌ Build failed: ${env.BUILD_URL}"
                )
            }
        }
        unstable {
            script {
                slackSend(
                    channel: '#deployments',
                    color: 'warning',
                    message: "⚠️ Build unstable: ${env.BUILD_URL}"
                )
            }
        }
    }
}
```

### Jenkinsfile (Scripted Pipeline)

```groovy
// Jenkinsfile (Scripted)
node {
    def dockerImage
    def gitCommit
    
    stage('Checkout') {
        checkout scm
        gitCommit = sh(
            script: 'git rev-parse --short HEAD',
            returnStdout: true
        ).trim()
    }
    
    stage('Build') {
        dockerImage = docker.build("your-registry.com/your-app:${gitCommit}")
    }
    
    stage('Test') {
        dockerImage.inside {
            sh 'go test -v -race ./...'
        }
    }
    
    stage('Push') {
        docker.withRegistry('https://your-registry.com', 'docker-registry-credentials') {
            dockerImage.push()
            dockerImage.push('latest')
        }
    }
    
    stage('Deploy') {
        if (env.BRANCH_NAME == 'main') {
            sh 'kubectl set image deployment/app-deployment app=your-registry.com/your-app:${gitCommit} -n production'
        } else if (env.BRANCH_NAME == 'develop') {
            sh 'kubectl set image deployment/app-deployment app=your-registry.com/your-app:${gitCommit} -n staging'
        }
    }
}
```

### Multibranch Pipeline Configuration

```groovy
// Jenkinsfile for Multibranch Pipeline
pipeline {
    agent any
    
    options {
        buildDiscarder(logRotator(numToKeepStr: '10'))
        timeout(time: 30, unit: 'MINUTES')
        timestamps()
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Build') {
            steps {
                sh 'go build -o app ./cmd/server'
            }
        }
        
        stage('Test') {
            steps {
                sh 'go test -v ./...'
            }
        }
        
        stage('Deploy') {
            when {
                anyOf {
                    branch 'main'
                    branch 'develop'
                }
            }
            steps {
                script {
                    def environment = env.BRANCH_NAME == 'main' ? 'production' : 'staging'
                    sh "kubectl set image deployment/app-deployment app=your-registry.com/your-app:${env.BRANCH_NAME} -n ${environment}"
                }
            }
        }
    }
}
```

### Jenkins Configuration as Code (JCasC)

```yaml
# jenkins.yaml
jenkins:
  systemMessage: "Jenkins configured automatically by Jenkins Configuration as Code plugin\n\n"
  numExecutors: 2
  scmCheckoutRetryCount: 3
  mode: NORMAL
  
  securityRealm:
    local:
      allowsSignup: false
      users:
        - id: "admin"
          password: "${JENKINS_ADMIN_PASSWORD}"
        - id: "jenkins"
          password: "${JENKINS_PASSWORD}"
  
  authorizationStrategy:
    loggedInUsersCanDoAnything:
      allowAnonymousRead: false
  
  remotingSecurity:
    enabled: true
  
  clouds:
    - kubernetes:
        name: "kubernetes"
        serverUrl: "https://kubernetes.default"
        namespace: "jenkins"
        jenkinsUrl: "http://jenkins:8080"
        jenkinsTunnel: "jenkins-agent:50000"
        containerCapStr: "100"
        maxRequestsPerHostStr: "32"
        templates:
          - name: "jnlp"
            label: "jnlp"
            containers:
              - name: "jnlp"
                image: "jenkins/inbound-agent:latest"
                resourceRequestCpu: "100m"
                resourceRequestMemory: "128Mi"
                resourceLimitCpu: "500m"
                resourceLimitMemory: "512Mi"
            yaml: |
              spec:
                securityContext:
                  runAsUser: 1000
                  runAsGroup: 1000
                  fsGroup: 1000
                containers:
                - name: jnlp
                  securityContext:
                    runAsUser: 1000
                    runAsGroup: 1000
                  resources:
                    requests:
                      memory: "128Mi"
                      cpu: "100m"
                    limits:
                      memory: "512Mi"
                      cpu: "500m"
  
  tools:
    git:
      installations:
        - name: "Default"
          home: "git"
    go:
      installations:
        - name: "go1.21"
          properties:
            - installSource:
                installers:
                  - goInstaller:
                      id: "1.21.0"
    docker:
      installations:
        - name: "docker"
          home: "/usr/bin/docker"
  
  globalLibraries:
    libraries:
      - name: "shared-library"
        defaultVersion: "main"
        retriever:
          modernSCM:
            scm:
              git:
                remote: "https://github.com/your-org/jenkins-shared-library.git"
                credentialsId: "github-credentials"
  
  unclassified:
    location:
      url: "http://jenkins:8080/"
      adminAddress: "admin@your-org.com"
    
    globalLibraries:
      libraries:
        - name: "shared-library"
          defaultVersion: "main"
          retriever:
            modernSCM:
              scm:
                git:
                  remote: "https://github.com/your-org/jenkins-shared-library.git"
                  credentialsId: "github-credentials"
    
    slack:
      teamDomain: "your-team"
      token: "${SLACK_TOKEN}"
      room: "#jenkins"
      startNotification: true
      notifySuccess: true
      notifyAborted: true
      notifyNotBuilt: true
      notifyUnstable: true
      notifyRegression: true
      notifyFailure: true
      notifyBackToNormal: true
      notifyRepeatedFailure: true
      includeTestResults: true
      includeCustomMessage: true
      customMessage: "Build ${env.BUILD_STATUS}: ${env.JOB_NAME} - ${env.BUILD_NUMBER}"
      sendAs: "Jenkins"
      commitInfoChoice: "AUTHORS_AND_TITLES"
      includeCustomMessage: true
      customMessage: "Build ${env.BUILD_STATUS}: ${env.JOB_NAME} - ${env.BUILD_NUMBER}"
      sendAs: "Jenkins"
      commitInfoChoice: "AUTHORS_AND_TITLES"
    
    sonarGlobalConfiguration:
      installations:
        - name: "SonarQube"
          serverUrl: "http://sonarqube:9000"
          credentialsId: "sonar-credentials"
    
    dockerTool:
      installations:
        - name: "docker"
          home: "/usr/bin/docker"
    
    gitTool:
      installations:
        - name: "Default"
          home: "git"
    
    goTool:
      installations:
        - name: "go1.21"
          properties:
            - installSource:
                installers:
                  - goInstaller:
                      id: "1.21.0"
  
  security:
    queueItemAuthenticator:
      authenticators:
        - global:
            strategy: triggeringUsersAuthorizationStrategy
```

### Shared Library

```groovy
// vars/buildGoApp.groovy
def call(Map config) {
    def defaultConfig = [
        goVersion: '1.21',
        buildArgs: '',
        testArgs: '-v -race',
        coverage: true
    ]
    
    config = defaultConfig + config
    
    pipeline {
        agent any
        
        stages {
            stage('Build') {
                steps {
                    script {
                        sh """
                            docker run --rm -v \${PWD}:/app -w /app golang:${config.goVersion} \
                            go build ${config.buildArgs} -o app ./cmd/server
                        """
                    }
                }
            }
            
            stage('Test') {
                steps {
                    script {
                        def testCmd = "go test ${config.testArgs}"
                        if (config.coverage) {
                            testCmd += " -coverprofile=coverage.out"
                        }
                        testCmd += " ./..."
                        
                        sh """
                            docker run --rm -v \${PWD}:/app -w /app golang:${config.goVersion} \
                            ${testCmd}
                        """
                    }
                }
            }
        }
    }
}
```

### Docker Compose for Jenkins

```yaml
# docker-compose.yml
version: '3.8'

services:
  jenkins:
    image: jenkins/jenkins:lts
    container_name: jenkins
    ports:
      - "8080:8080"
      - "50000:50000"
    volumes:
      - jenkins_home:/var/jenkins_home
      - /var/run/docker.sock:/var/run/docker.sock
      - /usr/bin/docker:/usr/bin/docker
    environment:
      - JENKINS_OPTS=--httpPort=8080
      - JAVA_OPTS=-Djenkins.install.runSetupWizard=false
    networks:
      - jenkins

  jenkins-agent:
    image: jenkins/inbound-agent:latest
    container_name: jenkins-agent
    environment:
      - JENKINS_URL=http://jenkins:8080
      - JENKINS_SECRET=your-agent-secret
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - /usr/bin/docker:/usr/bin/docker
    networks:
      - jenkins

volumes:
  jenkins_home:

networks:
  jenkins:
    driver: bridge
```

### Kubernetes Jenkins Deployment

```yaml
# jenkins-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jenkins
  namespace: jenkins
spec:
  replicas: 1
  selector:
    matchLabels:
      app: jenkins
  template:
    metadata:
      labels:
        app: jenkins
    spec:
      containers:
      - name: jenkins
        image: jenkins/jenkins:lts
        ports:
        - containerPort: 8080
        - containerPort: 50000
        env:
        - name: JAVA_OPTS
          value: "-Djenkins.install.runSetupWizard=false"
        - name: JENKINS_OPTS
          value: "--httpPort=8080"
        volumeMounts:
        - name: jenkins-home
          mountPath: /var/jenkins_home
        - name: docker-sock
          mountPath: /var/run/docker.sock
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      volumes:
      - name: jenkins-home
        persistentVolumeClaim:
          claimName: jenkins-pvc
      - name: docker-sock
        hostPath:
          path: /var/run/docker.sock
---
apiVersion: v1
kind: Service
metadata:
  name: jenkins
  namespace: jenkins
spec:
  selector:
    app: jenkins
  ports:
  - name: http
    port: 8080
    targetPort: 8080
  - name: agent
    port: 50000
    targetPort: 50000
  type: LoadBalancer
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: jenkins-pvc
  namespace: jenkins
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

## 🚀 Best Practices

### 1. Pipeline Organization
```groovy
// Use shared libraries for common functionality
@Library('shared-library@main') _

pipeline {
    agent any
    
    stages {
        stage('Build') {
            steps {
                buildGoApp(
                    goVersion: '1.21',
                    buildArgs: '-ldflags="-s -w"',
                    testArgs: '-v -race -coverprofile=coverage.out'
                )
            }
        }
    }
}
```

### 2. Security Best Practices
```groovy
// Use credentials for sensitive data
pipeline {
    agent any
    
    environment {
        DOCKER_REGISTRY = credentials('docker-registry')
        KUBECONFIG = credentials('kubeconfig')
        AWS_ACCESS_KEY_ID = credentials('aws-access-key')
        AWS_SECRET_ACCESS_KEY = credentials('aws-secret-key')
    }
    
    stages {
        stage('Deploy') {
            steps {
                sh 'kubectl set image deployment/app-deployment app=${DOCKER_REGISTRY}/app:${GIT_COMMIT} -n production'
            }
        }
    }
}
```

### 3. Performance Optimization
```groovy
// Use parallel stages and caching
pipeline {
    agent any
    
    stages {
        stage('Build') {
            parallel {
                stage('Build App') {
                    steps {
                        sh 'go build -o app ./cmd/server'
                    }
                }
                stage('Build Docker') {
                    steps {
                        sh 'docker build -t app:${GIT_COMMIT} .'
                    }
                }
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
    }
}
```

## 🏢 Industry Insights

### Jenkins Usage Patterns
- **Enterprise**: Large-scale deployments
- **Self-Hosted**: Full control over infrastructure
- **Plugin Ecosystem**: Extensive customization
- **Pipeline as Code**: Version-controlled pipelines

### Netflix's Jenkins Strategy
- **Microservices**: Individual service pipelines
- **Security**: Automated security scanning
- **Deployment**: Multi-environment deployment
- **Monitoring**: Automated monitoring setup

### Spotify's Jenkins Approach
- **Music Processing**: Audio file processing
- **Data Pipeline**: ETL operations
- **Real-time Analytics**: User listening data
- **Cost Efficiency**: Self-hosted infrastructure

## 🎯 Interview Questions

### Basic Level
1. **What is Jenkins?**
   - Open-source automation server
   - CI/CD platform
   - Pipeline as code
   - Plugin ecosystem

2. **What are Jenkins pipelines?**
   - Declarative pipelines
   - Scripted pipelines
   - Multibranch pipelines
   - Pipeline as code

3. **What are Jenkins agents?**
   - Master-slave architecture
   - Distributed builds
   - Remote execution
   - Resource management

### Intermediate Level
4. **How do you optimize Jenkins performance?**
   ```groovy
   // Use parallel stages and caching
   pipeline {
     agent any
     
     stages {
       stage('Build') {
         parallel {
           stage('Build App') {
             steps {
               sh 'go build -o app ./cmd/server'
             }
           }
           stage('Build Docker') {
             steps {
               sh 'docker build -t app:${GIT_COMMIT} .'
             }
           }
         }
       }
     }
   }
   ```

5. **How do you handle Jenkins security?**
   - Use credentials
   - Role-based access control
   - Plugin security
   - Network security

6. **How do you implement Jenkins testing?**
   - Unit testing
   - Integration testing
   - End-to-end testing
   - Performance testing

### Advanced Level
7. **How do you implement Jenkins patterns?**
   - Shared libraries
   - Pipeline templates
   - Custom steps
   - Reusable workflows

8. **How do you handle Jenkins scaling?**
   - Master-slave architecture
   - Kubernetes agents
   - Resource optimization
   - Load balancing

9. **How do you implement Jenkins monitoring?**
   - Build monitoring
   - Performance metrics
   - Alerting
   - Logging

---

**Next**: [GitLab CI](./GitLabCI.md) - Integrated CI/CD, container registry, security scanning
