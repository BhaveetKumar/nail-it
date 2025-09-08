# ☁️ Cloud Computing Basics: IaaS, PaaS, SaaS, and Cloud Models

> **Master cloud computing fundamentals for modern backend development**

## 📚 Concept

Cloud computing delivers computing services over the internet, providing on-demand access to resources without direct management.

### Cloud Service Models

- **IaaS (Infrastructure as a Service)**: Virtual machines, storage, networking
- **PaaS (Platform as a Service)**: Runtime environment, databases, development tools
- **SaaS (Software as a Service)**: Complete applications delivered over the internet

### Cloud Deployment Models

- **Public Cloud**: Shared infrastructure (AWS, GCP, Azure)
- **Private Cloud**: Dedicated infrastructure
- **Hybrid Cloud**: Combination of public and private
- **Multi-Cloud**: Multiple cloud providers

## 🏗️ Cloud Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Cloud Provider                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Compute   │  │   Storage   │  │  Networking │     │
│  │   Services  │  │   Services  │  │   Services  │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  Database   │  │  Security   │  │ Monitoring  │     │
│  │  Services   │  │  Services   │  │  Services   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Hands-on Example

### AWS CloudFormation Template

```yaml
# cloudformation-template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Basic cloud infrastructure template'

Parameters:
  Environment:
    Type: String
    Default: dev
    AllowedValues: [dev, staging, prod]

  InstanceType:
    Type: String
    Default: t3.micro
    AllowedValues: [t3.micro, t3.small, t3.medium]

Resources:
  # VPC
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsHostnames: true
      EnableDnsSupport: true
      Tags:
        - Key: Name
          Value: !Sub '${Environment}-vpc'

  # Internet Gateway
  InternetGateway:
    Type: AWS::EC2::InternetGateway
    Properties:
      Tags:
        - Key: Name
          Value: !Sub '${Environment}-igw'

  # Attach Internet Gateway to VPC
  InternetGatewayAttachment:
    Type: AWS::EC2::VPCGatewayAttachment
    Properties:
      InternetGatewayId: !Ref InternetGateway
      VpcId: !Ref VPC

  # Public Subnet
  PublicSubnet:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [0, !GetAZs '']
      CidrBlock: 10.0.1.0/24
      MapPublicIpOnLaunch: true
      Tags:
        - Key: Name
          Value: !Sub '${Environment}-public-subnet'

  # Private Subnet
  PrivateSubnet:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [1, !GetAZs '']
      CidrBlock: 10.0.2.0/24
      Tags:
        - Key: Name
          Value: !Sub '${Environment}-private-subnet'

  # Route Table for Public Subnet
  PublicRouteTable:
    Type: AWS::EC2::RouteTable
    Properties:
      VpcId: !Ref VPC
      Tags:
        - Key: Name
          Value: !Sub '${Environment}-public-rt'

  # Route Table for Private Subnet
  PrivateRouteTable:
    Type: AWS::EC2::RouteTable
    Properties:
      VpcId: !Ref VPC
      Tags:
        - Key: Name
          Value: !Sub '${Environment}-private-rt'

  # Default Route for Public Subnet
  DefaultPublicRoute:
    Type: AWS::EC2::Route
    DependsOn: InternetGatewayAttachment
    Properties:
      RouteTableId: !Ref PublicRouteTable
      DestinationCidrBlock: 0.0.0.0/0
      GatewayId: !Ref InternetGateway

  # Associate Public Subnet with Route Table
  PublicSubnetRouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      RouteTableId: !Ref PublicRouteTable
      SubnetId: !Ref PublicSubnet

  # Associate Private Subnet with Route Table
  PrivateSubnetRouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      RouteTableId: !Ref PrivateRouteTable
      SubnetId: !Ref PrivateSubnet

  # Security Group for Web Servers
  WebServerSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupName: !Sub '${Environment}-web-sg'
      GroupDescription: Security group for web servers
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
        - IpProtocol: tcp
          FromPort: 22
          ToPort: 22
          CidrIp: 0.0.0.0/0
      Tags:
        - Key: Name
          Value: !Sub '${Environment}-web-sg'

  # Security Group for Database
  DatabaseSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupName: !Sub '${Environment}-db-sg'
      GroupDescription: Security group for database
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 3306
          ToPort: 3306
          SourceSecurityGroupId: !Ref WebServerSecurityGroup
      Tags:
        - Key: Name
          Value: !Sub '${Environment}-db-sg'

  # EC2 Instance
  WebServer:
    Type: AWS::EC2::Instance
    Properties:
      ImageId: ami-0c02fb55956c7d316  # Amazon Linux 2
      InstanceType: !Ref InstanceType
      SubnetId: !Ref PublicSubnet
      SecurityGroupIds:
        - !Ref WebServerSecurityGroup
      UserData:
        Fn::Base64: !Sub |
          #!/bin/bash
          yum update -y
          yum install -y httpd
          systemctl start httpd
          systemctl enable httpd
          echo "<h1>Hello from ${Environment} environment!</h1>" > /var/www/html/index.html
      Tags:
        - Key: Name
          Value: !Sub '${Environment}-web-server'

  # RDS Database
  Database:
    Type: AWS::RDS::DBInstance
    Properties:
      DBInstanceIdentifier: !Sub '${Environment}-database'
      DBInstanceClass: db.t3.micro
      Engine: MySQL
      EngineVersion: '8.0'
      MasterUsername: admin
      MasterUserPassword: !Ref DatabasePassword
      AllocatedStorage: 20
      StorageType: gp2
      VPCSecurityGroups:
        - !Ref DatabaseSecurityGroup
      DBSubnetGroupName: !Ref DatabaseSubnetGroup
      BackupRetentionPeriod: 7
      MultiAZ: false
      Tags:
        - Key: Name
          Value: !Sub '${Environment}-database'

  # Database Subnet Group
  DatabaseSubnetGroup:
    Type: AWS::RDS::DBSubnetGroup
    Properties:
      DBSubnetGroupDescription: Subnet group for database
      SubnetIds:
        - !Ref PrivateSubnet
        - !Ref PublicSubnet
      Tags:
        - Key: Name
          Value: !Sub '${Environment}-db-subnet-group'

  # S3 Bucket
  StorageBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub '${Environment}-storage-bucket-${AWS::AccountId}'
      VersioningConfiguration:
        Status: Enabled
      PublicAccessBlockConfiguration:
        BlockPublicAcls: true
        BlockPublicPolicy: true
        IgnorePublicAcls: true
        RestrictPublicBuckets: true
      Tags:
        - Key: Name
          Value: !Sub '${Environment}-storage-bucket'

Parameters:
  DatabasePassword:
    Type: String
    NoEcho: true
    MinLength: 8
    MaxLength: 128
    Description: Password for the database

Outputs:
  VPCId:
    Description: VPC ID
    Value: !Ref VPC
    Export:
      Name: !Sub '${Environment}-VPC-ID'

  PublicSubnetId:
    Description: Public Subnet ID
    Value: !Ref PublicSubnet
    Export:
      Name: !Sub '${Environment}-Public-Subnet-ID'

  PrivateSubnetId:
    Description: Private Subnet ID
    Value: !Ref PrivateSubnet
    Export:
      Name: !Sub '${Environment}-Private-Subnet-ID'

  WebServerPublicIP:
    Description: Public IP address of the web server
    Value: !GetAtt WebServer.PublicIp

  DatabaseEndpoint:
    Description: Database endpoint
    Value: !GetAtt Database.Endpoint.Address

  StorageBucketName:
    Description: S3 bucket name
    Value: !Ref StorageBucket
```

### GCP Deployment Manager Template

```yaml
# gcp-deployment.yaml
imports:
  - path: compute-instance.jinja
  - path: network.jinja

resources:
  - name: my-network
    type: network.jinja
    properties:
      name: my-network
      autoCreateSubnetworks: false

  - name: my-subnet
    type: compute.v1.subnetwork
    properties:
      name: my-subnet
      region: us-central1
      network: $(ref.my-network.selfLink)
      ipCidrRange: 10.0.0.0/24

  - name: my-instance
    type: compute-instance.jinja
    properties:
      name: my-instance
      zone: us-central1-a
      machineType: f1-micro
      image: projects/debian-cloud/global/images/family/debian-11
      network: $(ref.my-network.selfLink)
      subnetwork: $(ref.my-subnet.selfLink)
      tags:
        - web-server
      metadata:
        startup-script: |
          #!/bin/bash
          apt-get update
          apt-get install -y apache2
          systemctl start apache2
          systemctl enable apache2
          echo "<h1>Hello from GCP!</h1>" > /var/www/html/index.html

  - name: my-firewall-rule
    type: compute.v1.firewall
    properties:
      name: my-firewall-rule
      network: $(ref.my-network.selfLink)
      sourceRanges: ["0.0.0.0/0"]
      allowed:
        - IPProtocol: TCP
          ports: ["80", "443", "22"]
      targetTags: ["web-server"]

  - name: my-database
    type: sqladmin.v1beta4.instance
    properties:
      name: my-database
      region: us-central1
      settings:
        tier: db-f1-micro
        dataDiskSizeGb: 10
        dataDiskType: PD_SSD
        backupConfiguration:
          enabled: true
          startTime: "03:00"
        ipConfiguration:
          ipv4Enabled: true
          authorizedNetworks:
            - value: "0.0.0.0/0"
      databaseVersion: MYSQL_8_0
      rootPassword: "my-secure-password"

  - name: my-storage-bucket
    type: storage.v1.bucket
    properties:
      name: my-storage-bucket-${PROJECT_ID}
      location: US
      storageClass: STANDARD
      versioning:
        enabled: true
      lifecycle:
        rule:
          - action:
              type: Delete
            condition:
              age: 365
```

## 🚀 Best Practices

### 1. Cost Optimization

```yaml
# Auto Scaling Group for cost optimization
AutoScalingGroup:
  Type: AWS::AutoScaling::AutoScalingGroup
  Properties:
    MinSize: 1
    MaxSize: 10
    DesiredCapacity: 2
    VPCZoneIdentifier:
      - !Ref PublicSubnet
    LaunchTemplate:
      LaunchTemplateId: !Ref LaunchTemplate
      Version: !GetAtt LaunchTemplate.LatestVersionNumber
    TargetGroupARNs:
      - !Ref TargetGroup
    HealthCheckType: ELB
    HealthCheckGracePeriod: 300
    Tags:
      - Key: Name
        Value: !Sub "${Environment}-asg"
        PropagateAtLaunch: true
```

### 2. Security Best Practices

```yaml
# Security Group with least privilege
WebServerSecurityGroup:
  Type: AWS::EC2::SecurityGroup
  Properties:
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
    SecurityGroupEgress:
      - IpProtocol: tcp
        FromPort: 443
        ToPort: 443
        CidrIp: 0.0.0.0/0
      - IpProtocol: tcp
        FromPort: 3306
        ToPort: 3306
        SourceSecurityGroupId: !Ref DatabaseSecurityGroup
```

### 3. High Availability

```yaml
# Multi-AZ deployment
Database:
  Type: AWS::RDS::DBInstance
  Properties:
    DBInstanceClass: db.t3.micro
    Engine: MySQL
    EngineVersion: "8.0"
    MultiAZ: true
    BackupRetentionPeriod: 7
    PreferredBackupWindow: "03:00-04:00"
    PreferredMaintenanceWindow: "sun:04:00-sun:05:00"
    StorageEncrypted: true
    DeletionProtection: true
```

## 🏢 Industry Insights

### AWS Cloud Services

- **Compute**: EC2, Lambda, ECS, EKS
- **Storage**: S3, EBS, EFS, Glacier
- **Database**: RDS, DynamoDB, ElastiCache
- **Networking**: VPC, CloudFront, Route 53

### Google Cloud Services

- **Compute**: Compute Engine, Cloud Functions, GKE
- **Storage**: Cloud Storage, Persistent Disk
- **Database**: Cloud SQL, Firestore, Bigtable
- **Networking**: VPC, Cloud CDN, Cloud DNS

### Microsoft Azure Services

- **Compute**: Virtual Machines, Functions, AKS
- **Storage**: Blob Storage, Managed Disks
- **Database**: SQL Database, Cosmos DB
- **Networking**: Virtual Network, CDN, DNS

## 🎯 Interview Questions

### Basic Level

1. **What are the main cloud service models?**

   - IaaS: Infrastructure as a Service
   - PaaS: Platform as a Service
   - SaaS: Software as a Service

2. **What are the benefits of cloud computing?**

   - Cost reduction
   - Scalability
   - Flexibility
   - Reliability
   - Security

3. **What is the difference between public and private cloud?**
   - Public: Shared infrastructure, pay-per-use
   - Private: Dedicated infrastructure, higher control

### Intermediate Level

4. **How do you design for high availability?**

   - Multi-AZ deployment
   - Load balancing
   - Auto scaling
   - Health checks
   - Backup and recovery

5. **How do you optimize cloud costs?**

   - Right-sizing instances
   - Reserved instances
   - Spot instances
   - Auto scaling
   - Resource tagging

6. **What is Infrastructure as Code?**
   - Managing infrastructure through code
   - Version control for infrastructure
   - Automated provisioning
   - Examples: CloudFormation, Terraform

### Advanced Level

7. **How do you implement disaster recovery?**

   - Multi-region deployment
   - Backup strategies
   - RTO and RPO planning
   - Failover mechanisms
   - Testing procedures

8. **How do you handle cloud security?**

   - Identity and access management
   - Network security
   - Data encryption
   - Compliance frameworks
   - Security monitoring

9. **How do you implement cloud governance?**
   - Resource policies
   - Cost management
   - Compliance monitoring
   - Security policies
   - Operational procedures

---

**Next**: [Virtualization vs Containers](./VirtualizationVsContainers.md) - VMs vs containers, Docker basics
