# 🌐 Networking in Cloud: VPC, Subnets, Security Groups, Load Balancers

> **Master cloud networking fundamentals for secure and scalable backend systems**

## 📚 Concept

Cloud networking provides the foundation for secure, scalable, and reliable communication between cloud resources and external networks.

### Key Components

- **VPC (Virtual Private Cloud)**: Isolated network environment
- **Subnets**: Network segments within VPC
- **Security Groups**: Firewall rules for instances
- **Load Balancers**: Traffic distribution across instances

## 🏗️ Cloud Networking Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Internet Gateway                     │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                    VPC (10.0.0.0/16)                   │
│  ┌─────────────┐              ┌─────────────┐          │
│  │ Public      │              │ Private     │          │
│  │ Subnet      │              │ Subnet      │          │
│  │ 10.0.1.0/24 │              │ 10.0.2.0/24 │          │
│  │             │              │             │          │
│  │ ┌─────────┐ │              │ ┌─────────┐ │          │
│  │ │ Web     │ │              │ │ DB      │ │          │
│  │ │ Server  │ │              │ │ Server  │ │          │
│  │ └─────────┘ │              │ └─────────┘ │          │
│  └─────────────┘              └─────────────┘          │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Hands-on Example

### AWS VPC Configuration

```yaml
# cloudformation-vpc.yaml
AWSTemplateFormatVersion: "2010-09-09"
Description: "VPC with public and private subnets"

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
          Value: Production-VPC

  # Internet Gateway
  InternetGateway:
    Type: AWS::EC2::InternetGateway
    Properties:
      Tags:
        - Key: Name
          Value: Production-IGW

  # Attach Internet Gateway
  InternetGatewayAttachment:
    Type: AWS::EC2::VPCGatewayAttachment
    Properties:
      InternetGatewayId: !Ref InternetGateway
      VpcId: !Ref VPC

  # Public Subnet 1
  PublicSubnet1:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [0, !GetAZs ""]
      CidrBlock: 10.0.1.0/24
      MapPublicIpOnLaunch: true
      Tags:
        - Key: Name
          Value: Public-Subnet-1

  # Public Subnet 2
  PublicSubnet2:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [1, !GetAZs ""]
      CidrBlock: 10.0.2.0/24
      MapPublicIpOnLaunch: true
      Tags:
        - Key: Name
          Value: Public-Subnet-2

  # Private Subnet 1
  PrivateSubnet1:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [0, !GetAZs ""]
      CidrBlock: 10.0.3.0/24
      Tags:
        - Key: Name
          Value: Private-Subnet-1

  # Private Subnet 2
  PrivateSubnet2:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [1, !GetAZs ""]
      CidrBlock: 10.0.4.0/24
      Tags:
        - Key: Name
          Value: Private-Subnet-2

  # NAT Gateway 1
  NatGateway1EIP:
    Type: AWS::EC2::EIP
    DependsOn: InternetGatewayAttachment
    Properties:
      Domain: vpc

  NatGateway1:
    Type: AWS::EC2::NatGateway
    Properties:
      AllocationId: !GetAtt NatGateway1EIP.AllocationId
      SubnetId: !Ref PublicSubnet1

  # NAT Gateway 2
  NatGateway2EIP:
    Type: AWS::EC2::EIP
    DependsOn: InternetGatewayAttachment
    Properties:
      Domain: vpc

  NatGateway2:
    Type: AWS::EC2::NatGateway
    Properties:
      AllocationId: !GetAtt NatGateway2EIP.AllocationId
      SubnetId: !Ref PublicSubnet2

  # Route Table for Public Subnets
  PublicRouteTable:
    Type: AWS::EC2::RouteTable
    Properties:
      VpcId: !Ref VPC
      Tags:
        - Key: Name
          Value: Public-Route-Table

  # Default Route for Public Subnets
  DefaultPublicRoute:
    Type: AWS::EC2::Route
    DependsOn: InternetGatewayAttachment
    Properties:
      RouteTableId: !Ref PublicRouteTable
      DestinationCidrBlock: 0.0.0.0/0
      GatewayId: !Ref InternetGateway

  # Route Table for Private Subnet 1
  PrivateRouteTable1:
    Type: AWS::EC2::RouteTable
    Properties:
      VpcId: !Ref VPC
      Tags:
        - Key: Name
          Value: Private-Route-Table-1

  # Route Table for Private Subnet 2
  PrivateRouteTable2:
    Type: AWS::EC2::RouteTable
    Properties:
      VpcId: !Ref VPC
      Tags:
        - Key: Name
          Value: Private-Route-Table-2

  # Route for Private Subnet 1
  DefaultPrivateRoute1:
    Type: AWS::EC2::Route
    Properties:
      RouteTableId: !Ref PrivateRouteTable1
      DestinationCidrBlock: 0.0.0.0/0
      NatGatewayId: !Ref NatGateway1

  # Route for Private Subnet 2
  DefaultPrivateRoute2:
    Type: AWS::EC2::Route
    Properties:
      RouteTableId: !Ref PrivateRouteTable2
      DestinationCidrBlock: 0.0.0.0/0
      NatGatewayId: !Ref NatGateway2

  # Associate Public Subnets with Route Table
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

  # Associate Private Subnets with Route Tables
  PrivateSubnet1RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      RouteTableId: !Ref PrivateRouteTable1
      SubnetId: !Ref PrivateSubnet1

  PrivateSubnet2RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      RouteTableId: !Ref PrivateRouteTable2
      SubnetId: !Ref PrivateSubnet2

  # Security Group for Web Servers
  WebServerSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
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
          CidrIp: 10.0.0.0/16
      SecurityGroupEgress:
        - IpProtocol: tcp
          FromPort: 443
          ToPort: 443
          CidrIp: 0.0.0.0/0
        - IpProtocol: tcp
          FromPort: 3306
          ToPort: 3306
          SourceSecurityGroupId: !Ref DatabaseSecurityGroup
      Tags:
        - Key: Name
          Value: Web-Server-SG

  # Security Group for Database
  DatabaseSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Security group for database
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 3306
          ToPort: 3306
          SourceSecurityGroupId: !Ref WebServerSecurityGroup
      Tags:
        - Key: Name
          Value: Database-SG

  # Application Load Balancer
  ApplicationLoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Name: Production-ALB
      Scheme: internet-facing
      Type: application
      Subnets:
        - !Ref PublicSubnet1
        - !Ref PublicSubnet2
      SecurityGroups:
        - !Ref LoadBalancerSecurityGroup
      Tags:
        - Key: Name
          Value: Production-ALB

  # Security Group for Load Balancer
  LoadBalancerSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Security group for load balancer
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
          Value: Load-Balancer-SG

  # Target Group
  TargetGroup:
    Type: AWS::ElasticLoadBalancingV2::TargetGroup
    Properties:
      Name: Production-TG
      Port: 8080
      Protocol: HTTP
      VpcId: !Ref VPC
      HealthCheckPath: /health
      HealthCheckProtocol: HTTP
      HealthCheckIntervalSeconds: 30
      HealthCheckTimeoutSeconds: 5
      HealthyThresholdCount: 2
      UnhealthyThresholdCount: 3
      TargetType: instance
      Tags:
        - Key: Name
          Value: Production-TG

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

Outputs:
  VPCId:
    Description: VPC ID
    Value: !Ref VPC
    Export:
      Name: Production-VPC-ID

  PublicSubnet1Id:
    Description: Public Subnet 1 ID
    Value: !Ref PublicSubnet1
    Export:
      Name: Production-Public-Subnet-1-ID

  PublicSubnet2Id:
    Description: Public Subnet 2 ID
    Value: !Ref PublicSubnet2
    Export:
      Name: Production-Public-Subnet-2-ID

  PrivateSubnet1Id:
    Description: Private Subnet 1 ID
    Value: !Ref PrivateSubnet1
    Export:
      Name: Production-Private-Subnet-1-ID

  PrivateSubnet2Id:
    Description: Private Subnet 2 ID
    Value: !Ref PrivateSubnet2
    Export:
      Name: Production-Private-Subnet-2-ID

  LoadBalancerDNS:
    Description: Load Balancer DNS Name
    Value: !GetAtt ApplicationLoadBalancer.DNSName
    Export:
      Name: Production-ALB-DNS
```

### GCP VPC Configuration

```yaml
# gcp-vpc.yaml
resources:
  - name: production-vpc
    type: compute.v1.network
    properties:
      name: production-vpc
      autoCreateSubnetworks: false
      routingConfig:
        routingMode: REGIONAL

  - name: production-subnet-1
    type: compute.v1.subnetwork
    properties:
      name: production-subnet-1
      region: us-central1
      network: $(ref.production-vpc.selfLink)
      ipCidrRange: 10.0.1.0/24
      privateIpGoogleAccess: false

  - name: production-subnet-2
    type: compute.v1.subnetwork
    properties:
      name: production-subnet-2
      region: us-central1
      network: $(ref.production-vpc.selfLink)
      ipCidrRange: 10.0.2.0/24
      privateIpGoogleAccess: true

  - name: production-firewall-rule
    type: compute.v1.firewall
    properties:
      name: production-firewall-rule
      network: $(ref.production-vpc.selfLink)
      sourceRanges: ["0.0.0.0/0"]
      allowed:
        - IPProtocol: TCP
          ports: ["80", "443", "22"]
      targetTags: ["web-server"]

  - name: production-load-balancer
    type: compute.v1.urlMap
    properties:
      name: production-load-balancer
      defaultService: $(ref.production-backend-service.selfLink)

  - name: production-backend-service
    type: compute.v1.backendService
    properties:
      name: production-backend-service
      protocol: HTTP
      port: 8080
      timeoutSec: 30
      healthChecks:
        - $(ref.production-health-check.selfLink)
      backends:
        - group: $(ref.production-instance-group.selfLink)
          balancingMode: UTILIZATION
          maxUtilization: 0.8

  - name: production-health-check
    type: compute.v1.httpHealthCheck
    properties:
      name: production-health-check
      requestPath: /health
      port: 8080
      checkIntervalSec: 30
      timeoutSec: 5
      healthyThreshold: 2
      unhealthyThreshold: 3

  - name: production-instance-group
    type: compute.v1.instanceGroup
    properties:
      name: production-instance-group
      zone: us-central1-a
      network: $(ref.production-vpc.selfLink)
      subnetwork: $(ref.production-subnet-1.selfLink)
```

### Load Balancer Configuration

```yaml
# load-balancer-config.yaml
apiVersion: v1
kind: Service
metadata:
  name: web-service
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: nlb
    service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled: "true"
spec:
  type: LoadBalancer
  ports:
    - port: 80
      targetPort: 8080
      protocol: TCP
    - port: 443
      targetPort: 8443
      protocol: TCP
  selector:
    app: web
---
apiVersion: v1
kind: Service
metadata:
  name: web-service-internal
spec:
  type: ClusterIP
  ports:
    - port: 8080
      targetPort: 8080
      protocol: TCP
  selector:
    app: web
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: web-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
    - hosts:
        - api.example.com
      secretName: api-tls
  rules:
    - host: api.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: web-service
                port:
                  number: 80
```

## 🚀 Best Practices

### 1. Network Security

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

### 2. High Availability

```yaml
# Multi-AZ deployment
ApplicationLoadBalancer:
  Type: AWS::ElasticLoadBalancingV2::LoadBalancer
  Properties:
    Name: Production-ALB
    Scheme: internet-facing
    Type: application
    Subnets:
      - !Ref PublicSubnet1
      - !Ref PublicSubnet2
    HealthCheckSettings:
      HealthyThresholdCount: 2
      UnhealthyThresholdCount: 3
      Timeout: 5
      Interval: 30
      Path: /health
      Port: 8080
      Protocol: HTTP
```

### 3. Network Monitoring

```yaml
# VPC Flow Logs
VPCFlowLogs:
  Type: AWS::EC2::FlowLog
  Properties:
    ResourceType: VPC
    ResourceIds:
      - !Ref VPC
    TrafficType: ALL
    LogDestinationType: cloud-watch-logs
    LogGroupName: !Ref VPCFlowLogsGroup
    DeliverLogsPermissionArn: !GetAtt VPCFlowLogsRole.Arn

VPCFlowLogsGroup:
  Type: AWS::Logs::LogGroup
  Properties:
    LogGroupName: VPCFlowLogs
    RetentionInDays: 30

VPCFlowLogsRole:
  Type: AWS::IAM::Role
  Properties:
    AssumeRolePolicyDocument:
      Version: "2012-10-17"
      Statement:
        - Effect: Allow
          Principal:
            Service: vpc-flow-logs.amazonaws.com
          Action: sts:AssumeRole
    Policies:
      - PolicyName: VPCFlowLogsPolicy
        PolicyDocument:
          Version: "2012-10-17"
          Statement:
            - Effect: Allow
              Action:
                - logs:CreateLogGroup
                - logs:CreateLogStream
                - logs:PutLogEvents
                - logs:DescribeLogGroups
                - logs:DescribeLogStreams
              Resource: "*"
```

## 🏢 Industry Insights

### AWS Networking

- **VPC**: Isolated network environment
- **ALB/NLB**: Application and network load balancers
- **CloudFront**: Global CDN
- **Route 53**: DNS service

### Google Cloud Networking

- **VPC**: Global network
- **Cloud Load Balancing**: Global load balancing
- **Cloud CDN**: Content delivery network
- **Cloud DNS**: Managed DNS service

### Microsoft Azure Networking

- **Virtual Network**: Isolated network
- **Load Balancer**: Traffic distribution
- **CDN**: Content delivery network
- **DNS**: Domain name resolution

## 🎯 Interview Questions

### Basic Level

1. **What is a VPC?**

   - Virtual Private Cloud
   - Isolated network environment
   - Custom IP address range
   - Subnets, route tables, gateways

2. **What are security groups?**

   - Virtual firewall
   - Inbound and outbound rules
   - Stateful filtering
   - Instance-level security

3. **What is a load balancer?**
   - Traffic distribution
   - High availability
   - Health checks
   - SSL termination

### Intermediate Level

4. **How do you design a secure network architecture?**

   - Public and private subnets
   - NAT gateways for outbound traffic
   - Security groups with least privilege
   - Network ACLs for additional security

5. **How do you implement high availability?**

   - Multi-AZ deployment
   - Load balancers across zones
   - Health checks and auto-scaling
   - Redundant components

6. **How do you handle network monitoring?**
   - VPC Flow Logs
   - CloudWatch metrics
   - Network performance monitoring
   - Security monitoring

### Advanced Level

7. **How do you implement microservices networking?**

   - Service mesh
   - Service discovery
   - Load balancing
   - Circuit breakers

8. **How do you handle network security?**

   - Network segmentation
   - Zero-trust architecture
   - Encryption in transit
   - Network policies

9. **How do you optimize network performance?**
   - CDN implementation
   - Connection pooling
   - Network optimization
   - Latency reduction

---

**Next**: [AWS Services](./AWS/) - EC2, S3, Lambda, RDS, CloudFormation, IAM, EKS
