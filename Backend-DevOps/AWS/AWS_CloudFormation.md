# ☁️ AWS CloudFormation: Infrastructure as Code

> **Master AWS CloudFormation for declarative infrastructure management**

## 📚 Concept

**Detailed Explanation:**
AWS CloudFormation is a powerful Infrastructure as Code (IaC) service that enables developers and DevOps engineers to model, provision, and manage AWS resources using declarative templates. Instead of manually creating and configuring resources through the AWS console, CloudFormation allows you to define your entire infrastructure in code, making it version-controlled, repeatable, and automated. This approach transforms infrastructure management from a manual, error-prone process into a systematic, reliable operation.

**Core Philosophy:**

- **Infrastructure as Code**: Treat infrastructure the same way as application code
- **Declarative Approach**: Define what you want, not how to get there
- **Idempotent Operations**: Same template produces the same result every time
- **Dependency Management**: Automatic handling of resource dependencies
- **State Management**: Track and manage infrastructure state
- **Change Management**: Controlled and predictable infrastructure changes

**Why CloudFormation Matters:**

- **Consistency**: Ensures identical infrastructure across environments
- **Reproducibility**: Recreate entire environments from templates
- **Version Control**: Track infrastructure changes over time
- **Automation**: Integrate infrastructure provisioning into CI/CD pipelines
- **Cost Management**: Track and optimize infrastructure costs
- **Compliance**: Maintain consistent security and compliance standards

### Key Features

**Detailed Feature Breakdown:**

**1. Infrastructure as Code:**

- **Declarative Templates**: Define desired state rather than procedural steps
- **Version Control**: Store templates in Git repositories for change tracking
- **Code Review**: Apply software development practices to infrastructure
- **Testing**: Validate templates before deployment
- **Documentation**: Self-documenting infrastructure through templates
- **Reusability**: Share and reuse template components across projects

**2. Template-based Management:**

- **JSON/YAML Support**: Use familiar markup languages for template definition
- **Intrinsic Functions**: Built-in functions for dynamic value generation
- **Parameters**: Make templates configurable and reusable
- **Conditions**: Environment-specific resource creation
- **Mappings**: Static value lookups for different environments
- **Outputs**: Expose important values for other stacks or applications

**3. Stack Management:**

- **Resource Grouping**: Organize related resources into logical units
- **Stack Lifecycle**: Create, update, and delete entire stacks atomically
- **Stack Dependencies**: Reference resources from other stacks
- **Stack Policies**: Control which resources can be modified
- **Stack Drift Detection**: Identify manual changes to stack resources
- **Stack Sets**: Deploy stacks across multiple accounts and regions

**4. Dependency Management:**

- **Automatic Resolution**: CloudFormation determines resource creation order
- **Reference-based Dependencies**: Resources depend on other resources through references
- **Explicit Dependencies**: Use DependsOn for custom dependency ordering
- **Circular Dependency Detection**: Prevent infinite dependency loops
- **Parallel Execution**: Create independent resources simultaneously
- **Rollback on Failure**: Automatic cleanup when dependencies fail

**5. Change Management:**

- **Change Sets**: Preview changes before applying them
- **Stack Updates**: Modify existing stacks with controlled changes
- **Rollback Capability**: Revert to previous stack state on failure
- **Stack Policies**: Protect critical resources from accidental changes
- **Drift Detection**: Identify and reconcile manual changes
- **Stack Events**: Track all stack operations and changes

**6. Integration and Automation:**

- **AWS CLI Integration**: Command-line interface for stack operations
- **SDK Support**: Programmatic access through AWS SDKs
- **CI/CD Integration**: Automated deployment through pipelines
- **CloudFormation Registry**: Extend CloudFormation with custom resources
- **Custom Resources**: Integrate with external systems and services
- **Stack Notifications**: SNS integration for stack event notifications

**Discussion Questions & Answers:**

**Q1: How do you design a scalable CloudFormation architecture for a large enterprise?**

**Answer:** Enterprise CloudFormation architecture design:

**Template Organization:**

- Use nested stacks to break down large templates into manageable modules
- Implement cross-stack references for sharing resources between stacks
- Create reusable template components for common infrastructure patterns
- Use AWS CloudFormation Registry for custom resource types
- Implement template validation and testing in CI/CD pipelines

**Environment Management:**

- Use parameter files for environment-specific configurations
- Implement stack sets for multi-account and multi-region deployments
- Use AWS Organizations for centralized stack management
- Implement stack policies to protect production resources
- Use AWS Config for compliance monitoring and drift detection

**Security and Governance:**

- Implement least privilege IAM policies for CloudFormation operations
- Use AWS KMS for encrypting sensitive parameters and outputs
- Implement stack policies to prevent accidental resource deletion
- Use AWS CloudTrail for auditing all CloudFormation operations
- Implement automated security scanning of templates

**Monitoring and Operations:**

- Use CloudWatch for monitoring stack operations and resource health
- Implement automated rollback policies for failed deployments
- Use AWS Systems Manager for parameter management
- Implement stack drift detection and automated remediation
- Use AWS Service Catalog for standardized infrastructure offerings

**Q2: What are the best practices for managing CloudFormation templates at scale?**

**Answer:** CloudFormation template management best practices:

**Template Design:**

- Keep templates focused and single-purpose to improve maintainability
- Use parameters for configurable values and conditions for environment-specific logic
- Implement proper naming conventions and tagging strategies
- Use intrinsic functions for dynamic value generation
- Avoid hardcoded values and use mappings for static lookups

**Version Control and Collaboration:**

- Store templates in version control systems with proper branching strategies
- Implement code review processes for template changes
- Use semantic versioning for template releases
- Document template parameters and outputs clearly
- Implement automated testing for template validation

**Deployment Strategies:**

- Use change sets to preview changes before execution
- Implement blue-green deployments for zero-downtime updates
- Use stack policies to protect critical resources
- Implement automated rollback policies for failed deployments
- Use stack sets for consistent multi-environment deployments

**Performance Optimization:**

- Minimize template size by using nested stacks and cross-stack references
- Use parallel resource creation where possible
- Implement proper dependency management to avoid unnecessary delays
- Use AWS CloudFormation Registry for custom resource optimization
- Monitor and optimize stack creation and update times

**Q3: How do you implement comprehensive security and compliance with CloudFormation?**

**Answer:** CloudFormation security and compliance implementation:

**Template Security:**

- Implement least privilege IAM policies for all resources
- Use AWS KMS for encrypting sensitive data and parameters
- Implement proper security group rules and network ACLs
- Use AWS Secrets Manager for sensitive configuration data
- Implement resource-level permissions and access controls

**Compliance and Governance:**

- Use AWS Config rules for compliance monitoring
- Implement stack policies to enforce organizational policies
- Use AWS Organizations for centralized governance
- Implement automated compliance scanning of templates
- Use AWS CloudTrail for comprehensive audit logging

**Data Protection:**

- Enable encryption at rest for all storage resources
- Use SSL/TLS for all data in transit
- Implement proper backup and disaster recovery procedures
- Use AWS Systems Manager Parameter Store for secure parameter management
- Implement data classification and handling procedures

**Monitoring and Incident Response:**

- Use CloudWatch for security event monitoring
- Implement automated alerting for security violations
- Use AWS Security Hub for centralized security findings
- Implement incident response procedures for security events
- Use AWS GuardDuty for threat detection and response

## 🏗️ CloudFormation Architecture

```
┌─────────────────────────────────────────────────────────┐
│                CloudFormation Architecture             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Template  │  │   Stack     │  │   Resources │     │
│  │   (JSON/    │  │   Manager   │  │   (AWS      │     │
│  │   YAML)     │  │             │  │   Services) │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              CloudFormation Engine                 │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Parser    │  │   Planner   │  │   Executor  │ │ │
│  │  │             │  │             │  │             │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Change    │  │   Events    │  │   Outputs   │     │
│  │   Sets      │  │   & Logs    │  │   & Exports │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Hands-on Example

### CloudFormation Template

```yaml
# infrastructure.yaml
AWSTemplateFormatVersion: "2010-09-09"
Description: "Complete web application infrastructure with RDS, EC2, and ALB"

Parameters:
  Environment:
    Type: String
    Default: production
    AllowedValues: [development, staging, production]
    Description: Environment name

  InstanceType:
    Type: String
    Default: t3.micro
    AllowedValues: [t3.micro, t3.small, t3.medium, t3.large]
    Description: EC2 instance type

  DBInstanceClass:
    Type: String
    Default: db.t3.micro
    AllowedValues: [db.t3.micro, db.t3.small, db.t3.medium]
    Description: RDS instance class

  DBPassword:
    Type: String
    NoEcho: true
    MinLength: 8
    MaxLength: 128
    Description: Database password

  KeyPairName:
    Type: AWS::EC2::KeyPair::KeyName
    Description: EC2 Key Pair name

Conditions:
  IsProduction: !Equals [!Ref Environment, production]
  IsDevelopment: !Equals [!Ref Environment, development]

Resources:
  # VPC and Networking
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsHostnames: true
      EnableDnsSupport: true
      Tags:
        - Key: Name
          Value: !Sub "${Environment}-vpc"
        - Key: Environment
          Value: !Ref Environment

  InternetGateway:
    Type: AWS::EC2::InternetGateway
    Properties:
      Tags:
        - Key: Name
          Value: !Sub "${Environment}-igw"
        - Key: Environment
          Value: !Ref Environment

  InternetGatewayAttachment:
    Type: AWS::EC2::VPCGatewayAttachment
    Properties:
      InternetGatewayId: !Ref InternetGateway
      VpcId: !Ref VPC

  # Public Subnets
  PublicSubnet1:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [0, !GetAZs ""]
      CidrBlock: 10.0.1.0/24
      MapPublicIpOnLaunch: true
      Tags:
        - Key: Name
          Value: !Sub "${Environment}-public-subnet-1"
        - Key: Environment
          Value: !Ref Environment

  PublicSubnet2:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [1, !GetAZs ""]
      CidrBlock: 10.0.2.0/24
      MapPublicIpOnLaunch: true
      Tags:
        - Key: Name
          Value: !Sub "${Environment}-public-subnet-2"
        - Key: Environment
          Value: !Ref Environment

  # Private Subnets
  PrivateSubnet1:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [0, !GetAZs ""]
      CidrBlock: 10.0.3.0/24
      Tags:
        - Key: Name
          Value: !Sub "${Environment}-private-subnet-1"
        - Key: Environment
          Value: !Ref Environment

  PrivateSubnet2:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [1, !GetAZs ""]
      CidrBlock: 10.0.4.0/24
      Tags:
        - Key: Name
          Value: !Sub "${Environment}-private-subnet-2"
        - Key: Environment
          Value: !Ref Environment

  # Route Tables
  PublicRouteTable:
    Type: AWS::EC2::RouteTable
    Properties:
      VpcId: !Ref VPC
      Tags:
        - Key: Name
          Value: !Sub "${Environment}-public-rt"
        - Key: Environment
          Value: !Ref Environment

  DefaultPublicRoute:
    Type: AWS::EC2::Route
    DependsOn: InternetGatewayAttachment
    Properties:
      RouteTableId: !Ref PublicRouteTable
      DestinationCidrBlock: 0.0.0.0/0
      GatewayId: !Ref InternetGateway

  PublicSubnet1RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      RouteTableId: !Ref PublicRouteTable
      SubnetId: !Ref PublicSubnet1

  PublicSubnet2RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      RouteTableId: !Ref PublicRouteTable
      SubnetId: !Ref PublicSubnet2

  PrivateRouteTable:
    Type: AWS::EC2::RouteTable
    Properties:
      VpcId: !Ref VPC
      Tags:
        - Key: Name
          Value: !Sub "${Environment}-private-rt"
        - Key: Environment
          Value: !Ref Environment

  PrivateSubnet1RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      RouteTableId: !Ref PrivateRouteTable
      SubnetId: !Ref PrivateSubnet1

  PrivateSubnet2RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      RouteTableId: !Ref PrivateRouteTable
      SubnetId: !Ref PrivateSubnet2

  # Security Groups
  WebServerSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupName: !Sub "${Environment}-web-sg"
      GroupDescription: Security group for web servers
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 80
          ToPort: 80
          SourceSecurityGroupId: !Ref LoadBalancerSecurityGroup
        - IpProtocol: tcp
          FromPort: 443
          ToPort: 443
          SourceSecurityGroupId: !Ref LoadBalancerSecurityGroup
        - IpProtocol: tcp
          FromPort: 22
          ToPort: 22
          CidrIp: 0.0.0.0/0
      Tags:
        - Key: Name
          Value: !Sub "${Environment}-web-sg"
        - Key: Environment
          Value: !Ref Environment

  LoadBalancerSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupName: !Sub "${Environment}-alb-sg"
      GroupDescription: Security group for Application Load Balancer
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 80
          ToPort: 80
          CidrIp: 0.0.0.0/0
        - IpProtocol: tcp
          FromPort: 443
          ToPort: 443
          CidrIp: 0.0.0.0/0
      Tags:
        - Key: Name
          Value: !Sub "${Environment}-alb-sg"
        - Key: Environment
          Value: !Ref Environment

  DatabaseSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupName: !Sub "${Environment}-db-sg"
      GroupDescription: Security group for database
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 5432
          ToPort: 5432
          SourceSecurityGroupId: !Ref WebServerSecurityGroup
      Tags:
        - Key: Name
          Value: !Sub "${Environment}-db-sg"
        - Key: Environment
          Value: !Ref Environment

  # Application Load Balancer
  ApplicationLoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Name: !Sub "${Environment}-alb"
      Scheme: internet-facing
      Type: application
      Subnets:
        - !Ref PublicSubnet1
        - !Ref PublicSubnet2
      SecurityGroups:
        - !Ref LoadBalancerSecurityGroup
      Tags:
        - Key: Name
          Value: !Sub "${Environment}-alb"
        - Key: Environment
          Value: !Ref Environment

  # Target Group
  TargetGroup:
    Type: AWS::ElasticLoadBalancingV2::TargetGroup
    Properties:
      Name: !Sub "${Environment}-tg"
      Port: 80
      Protocol: HTTP
      VpcId: !Ref VPC
      HealthCheckPath: /health
      HealthCheckProtocol: HTTP
      HealthCheckIntervalSeconds: 30
      HealthCheckTimeoutSeconds: 5
      HealthyThresholdCount: 2
      UnhealthyThresholdCount: 3
      Tags:
        - Key: Name
          Value: !Sub "${Environment}-tg"
        - Key: Environment
          Value: !Ref Environment

  # Load Balancer Listener
  LoadBalancerListener:
    Type: AWS::ElasticLoadBalancingV2::Listener
    Properties:
      DefaultActions:
        - Type: forward
          TargetGroupArn: !Ref TargetGroup
      LoadBalancerArn: !Ref ApplicationLoadBalancer
      Port: 80
      Protocol: HTTP

  # Launch Template
  LaunchTemplate:
    Type: AWS::EC2::LaunchTemplate
    Properties:
      LaunchTemplateName: !Sub "${Environment}-lt"
      LaunchTemplateData:
        ImageId: ami-0c02fb55956c7d316 # Amazon Linux 2 AMI
        InstanceType: !Ref InstanceType
        KeyName: !Ref KeyPairName
        SecurityGroupIds:
          - !Ref WebServerSecurityGroup
        UserData:
          Fn::Base64: !Sub |
            #!/bin/bash
            yum update -y
            yum install -y docker
            systemctl start docker
            systemctl enable docker
            usermod -a -G docker ec2-user

            # Install Docker Compose
            curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
            chmod +x /usr/local/bin/docker-compose

            # Create application directory
            mkdir -p /opt/app
            cd /opt/app

            # Create docker-compose.yml
            cat > docker-compose.yml << 'EOF'
            version: '3.8'
            services:
              app:
                image: nginx:alpine
                ports:
                  - "80:80"
                volumes:
                  - ./nginx.conf:/etc/nginx/nginx.conf
                restart: unless-stopped
            EOF

            # Create nginx configuration
            cat > nginx.conf << 'EOF'
            events {
                worker_connections 1024;
            }
            http {
                server {
                    listen 80;
                    location / {
                        return 200 'Hello from ${Environment} environment!';
                        add_header Content-Type text/plain;
                    }
                    location /health {
                        return 200 'OK';
                        add_header Content-Type text/plain;
                    }
                }
            }
            EOF

            # Start the application
            docker-compose up -d
        TagSpecifications:
          - ResourceType: instance
            Tags:
              - Key: Name
                Value: !Sub "${Environment}-instance"
              - Key: Environment
                Value: !Ref Environment

  # Auto Scaling Group
  AutoScalingGroup:
    Type: AWS::AutoScaling::AutoScalingGroup
    Properties:
      AutoScalingGroupName: !Sub "${Environment}-asg"
      VPCZoneIdentifier:
        - !Ref PrivateSubnet1
        - !Ref PrivateSubnet2
      LaunchTemplate:
        LaunchTemplateId: !Ref LaunchTemplate
        Version: !GetAtt LaunchTemplate.LatestVersionNumber
      MinSize: !If [IsProduction, 2, 1]
      MaxSize: !If [IsProduction, 10, 3]
      DesiredCapacity: !If [IsProduction, 3, 1]
      TargetGroupARNs:
        - !Ref TargetGroup
      HealthCheckType: ELB
      HealthCheckGracePeriod: 300
      Tags:
        - Key: Name
          Value: !Sub "${Environment}-asg"
          PropagateAtLaunch: true
        - Key: Environment
          Value: !Ref Environment
          PropagateAtLaunch: true

  # Scaling Policies
  ScaleUpPolicy:
    Type: AWS::AutoScaling::ScalingPolicy
    Properties:
      AdjustmentType: ChangeInCapacity
      AutoScalingGroupName: !Ref AutoScalingGroup
      Cooldown: 300
      ScalingAdjustment: 1

  ScaleDownPolicy:
    Type: AWS::AutoScaling::ScalingPolicy
    Properties:
      AdjustmentType: ChangeInCapacity
      AutoScalingGroupName: !Ref AutoScalingGroup
      Cooldown: 300
      ScalingAdjustment: -1

  # CloudWatch Alarms
  CPUAlarmHigh:
    Type: AWS::CloudWatch::Alarm
    Properties:
      AlarmName: !Sub "${Environment}-cpu-high"
      AlarmDescription: Alarm when CPU exceeds 70%
      MetricName: CPUUtilization
      Namespace: AWS/EC2
      Statistic: Average
      Period: 300
      EvaluationPeriods: 2
      Threshold: 70
      ComparisonOperator: GreaterThanThreshold
      Dimensions:
        - Name: AutoScalingGroupName
          Value: !Ref AutoScalingGroup
      AlarmActions:
        - !Ref ScaleUpPolicy

  CPUAlarmLow:
    Type: AWS::CloudWatch::Alarm
    Properties:
      AlarmName: !Sub "${Environment}-cpu-low"
      AlarmDescription: Alarm when CPU falls below 20%
      MetricName: CPUUtilization
      Namespace: AWS/EC2
      Statistic: Average
      Period: 300
      EvaluationPeriods: 2
      Threshold: 20
      ComparisonOperator: LessThanThreshold
      Dimensions:
        - Name: AutoScalingGroupName
          Value: !Ref AutoScalingGroup
      AlarmActions:
        - !Ref ScaleDownPolicy

  # RDS Subnet Group
  DBSubnetGroup:
    Type: AWS::RDS::DBSubnetGroup
    Properties:
      DBSubnetGroupName: !Sub "${Environment}-db-subnet-group"
      DBSubnetGroupDescription: Subnet group for RDS database
      SubnetIds:
        - !Ref PrivateSubnet1
        - !Ref PrivateSubnet2
      Tags:
        - Key: Name
          Value: !Sub "${Environment}-db-subnet-group"
        - Key: Environment
          Value: !Ref Environment

  # RDS Instance
  Database:
    Type: AWS::RDS::DBInstance
    Properties:
      DBInstanceIdentifier: !Sub "${Environment}-database"
      DBName: myapp
      DBInstanceClass: !Ref DBInstanceClass
      Engine: postgres
      EngineVersion: "14.7"
      MasterUsername: admin
      MasterUserPassword: !Ref DBPassword
      AllocatedStorage: 20
      MaxAllocatedStorage: 100
      StorageType: gp2
      StorageEncrypted: true
      VPCSecurityGroups:
        - !Ref DatabaseSecurityGroup
      DBSubnetGroupName: !Ref DBSubnetGroup
      BackupRetentionPeriod: 7
      BackupWindow: "03:00-04:00"
      MaintenanceWindow: "sun:04:00-sun:05:00"
      MultiAZ: !If [IsProduction, true, false]
      DeletionProtection: !If [IsProduction, true, false]
      Tags:
        - Key: Name
          Value: !Sub "${Environment}-database"
        - Key: Environment
          Value: !Ref Environment

  # S3 Bucket for application logs
  LogsBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub "${Environment}-app-logs-${AWS::AccountId}"
      VersioningConfiguration:
        Status: Enabled
      LifecycleConfiguration:
        Rules:
          - Id: DeleteOldLogs
            Status: Enabled
            ExpirationInDays: 30
      PublicAccessBlockConfiguration:
        BlockPublicAcls: true
        BlockPublicPolicy: true
        IgnorePublicAcls: true
        RestrictPublicBuckets: true
      Tags:
        - Key: Name
          Value: !Sub "${Environment}-app-logs"
        - Key: Environment
          Value: !Ref Environment

  # IAM Role for EC2 instances
  EC2Role:
    Type: AWS::IAM::Role
    Properties:
      RoleName: !Sub "${Environment}-ec2-role"
      AssumeRolePolicyDocument:
        Version: "2012-10-17"
        Statement:
          - Effect: Allow
            Principal:
              Service: ec2.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
        - arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy
      Tags:
        - Key: Name
          Value: !Sub "${Environment}-ec2-role"
        - Key: Environment
          Value: !Ref Environment

  # Instance Profile
  EC2InstanceProfile:
    Type: AWS::IAM::InstanceProfile
    Properties:
      Roles:
        - !Ref EC2Role

Outputs:
  VPCId:
    Description: VPC ID
    Value: !Ref VPC
    Export:
      Name: !Sub "${Environment}-VPC-ID"

  LoadBalancerDNS:
    Description: Application Load Balancer DNS name
    Value: !GetAtt ApplicationLoadBalancer.DNSName
    Export:
      Name: !Sub "${Environment}-ALB-DNS"

  DatabaseEndpoint:
    Description: RDS instance endpoint
    Value: !GetAtt Database.Endpoint.Address
    Export:
      Name: !Sub "${Environment}-DB-Endpoint"

  LogsBucketName:
    Description: S3 bucket for application logs
    Value: !Ref LogsBucket
    Export:
      Name: !Sub "${Environment}-Logs-Bucket"

  AutoScalingGroupName:
    Description: Auto Scaling Group name
    Value: !Ref AutoScalingGroup
    Export:
      Name: !Sub "${Environment}-ASG-Name"
```

### CloudFormation Deployment Script

```bash
#!/bin/bash
# deploy.sh

set -e

# Configuration
STACK_NAME="web-app-infrastructure"
TEMPLATE_FILE="infrastructure.yaml"
PARAMETERS_FILE="parameters.json"
REGION="us-east-1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    print_error "AWS CLI is not installed. Please install it first."
    exit 1
fi

# Check if template file exists
if [ ! -f "$TEMPLATE_FILE" ]; then
    print_error "Template file $TEMPLATE_FILE not found."
    exit 1
fi

# Validate template
print_status "Validating CloudFormation template..."
aws cloudformation validate-template \
    --template-body file://$TEMPLATE_FILE \
    --region $REGION

if [ $? -eq 0 ]; then
    print_status "Template validation successful."
else
    print_error "Template validation failed."
    exit 1
fi

# Check if stack exists
print_status "Checking if stack exists..."
if aws cloudformation describe-stacks --stack-name $STACK_NAME --region $REGION &> /dev/null; then
    print_warning "Stack $STACK_NAME already exists. Updating..."

    # Create change set
    CHANGE_SET_NAME="update-$(date +%Y%m%d-%H%M%S)"
    print_status "Creating change set: $CHANGE_SET_NAME"

    aws cloudformation create-change-set \
        --stack-name $STACK_NAME \
        --change-set-name $CHANGE_SET_NAME \
        --template-body file://$TEMPLATE_FILE \
        --parameters file://$PARAMETERS_FILE \
        --capabilities CAPABILITY_IAM \
        --region $REGION

    # Wait for change set creation
    print_status "Waiting for change set creation..."
    aws cloudformation wait change-set-create-complete \
        --stack-name $STACK_NAME \
        --change-set-name $CHANGE_SET_NAME \
        --region $REGION

    # Describe change set
    print_status "Change set created. Reviewing changes..."
    aws cloudformation describe-change-set \
        --stack-name $STACK_NAME \
        --change-set-name $CHANGE_SET_NAME \
        --region $REGION

    # Ask for confirmation
    read -p "Do you want to execute the change set? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Executing change set..."
        aws cloudformation execute-change-set \
            --stack-name $STACK_NAME \
            --change-set-name $CHANGE_SET_NAME \
            --region $REGION

        # Wait for stack update
        print_status "Waiting for stack update to complete..."
        aws cloudformation wait stack-update-complete \
            --stack-name $STACK_NAME \
            --region $REGION

        print_status "Stack update completed successfully!"
    else
        print_warning "Change set execution cancelled."
        aws cloudformation delete-change-set \
            --stack-name $STACK_NAME \
            --change-set-name $CHANGE_SET_NAME \
            --region $REGION
        exit 0
    fi
else
    print_status "Stack $STACK_NAME does not exist. Creating new stack..."

    # Create stack
    aws cloudformation create-stack \
        --stack-name $STACK_NAME \
        --template-body file://$TEMPLATE_FILE \
        --parameters file://$PARAMETERS_FILE \
        --capabilities CAPABILITY_IAM \
        --region $REGION

    # Wait for stack creation
    print_status "Waiting for stack creation to complete..."
    aws cloudformation wait stack-create-complete \
        --stack-name $STACK_NAME \
        --region $REGION

    print_status "Stack creation completed successfully!"
fi

# Get stack outputs
print_status "Retrieving stack outputs..."
aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --region $REGION \
    --query 'Stacks[0].Outputs' \
    --output table

print_status "Deployment completed successfully!"
```

### Parameters File

```json
{
  "Parameters": {
    "Environment": "production",
    "InstanceType": "t3.small",
    "DBInstanceClass": "db.t3.small",
    "DBPassword": "SecurePassword123!",
    "KeyPairName": "my-key-pair"
  }
}
```

## 🚀 Best Practices

### 1. Template Organization

```yaml
# Use conditions for environment-specific resources
Conditions:
  IsProduction: !Equals [!Ref Environment, production]
  IsDevelopment: !Equals [!Ref Environment, development]

# Use conditions in resource properties
MultiAZ: !If [IsProduction, true, false]
```

### 2. Parameter Validation

```yaml
Parameters:
  InstanceType:
    Type: String
    Default: t3.micro
    AllowedValues: [t3.micro, t3.small, t3.medium]
    Description: EC2 instance type
```

### 3. Outputs and Exports

```yaml
Outputs:
  VPCId:
    Description: VPC ID
    Value: !Ref VPC
    Export:
      Name: !Sub "${Environment}-VPC-ID"
```

## 🏢 Industry Insights

### CloudFormation Usage Patterns

- **Infrastructure as Code**: Version control for infrastructure
- **Stack Management**: Group related resources
- **Change Sets**: Preview changes before execution
- **Nested Stacks**: Modular template organization

### Enterprise CloudFormation Strategy

- **Template Validation**: Automated template validation
- **Stack Policies**: Resource protection policies
- **Cross-Stack References**: Share resources between stacks
- **Automated Deployment**: CI/CD integration

## 🎯 Interview Questions

### Basic Level

1. **What is AWS CloudFormation?**

   - Infrastructure as Code service
   - Declarative resource management
   - Template-based provisioning

2. **What are CloudFormation templates?**

   - JSON or YAML files
   - Resource definitions
   - Parameter and output specifications

3. **What are the benefits of CloudFormation?**
   - Infrastructure as Code
   - Automated provisioning
   - Rollback support
   - Cost tracking

### Intermediate Level

4. **How do you handle dependencies in CloudFormation?**

   - Automatic dependency resolution
   - DependsOn attribute
   - Reference-based dependencies

5. **How do you implement change management with CloudFormation?**

   - Change sets
   - Stack policies
   - Rollback on failure

6. **How do you organize large CloudFormation templates?**
   - Nested stacks
   - Cross-stack references
   - Template modules

### Advanced Level

7. **How do you implement CI/CD with CloudFormation?**

   - Automated deployment pipelines
   - Template validation
   - Change set automation

8. **How do you handle secrets in CloudFormation?**

   - AWS Systems Manager Parameter Store
   - AWS Secrets Manager
   - NoEcho parameters

9. **How do you implement disaster recovery with CloudFormation?**
   - Cross-region stacks
   - Backup and restore
   - Multi-region deployment
