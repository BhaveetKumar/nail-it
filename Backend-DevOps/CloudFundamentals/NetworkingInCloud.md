# 🌐 Networking in Cloud: VPC, Subnets, Security Groups, Load Balancers

> **Master cloud networking fundamentals for secure and scalable backend systems**

## 📚 Concept

**Detailed Explanation:**
Cloud networking is the foundational layer that enables secure, scalable, and reliable communication between cloud resources, external networks, and users. It provides the infrastructure and services necessary to build, deploy, and manage applications in the cloud while maintaining security, performance, and availability. Cloud networking abstracts the complexity of physical network infrastructure while providing powerful tools for network management and optimization.

**Core Philosophy:**

- **Software-Defined Networking**: Network configuration and management through software rather than hardware
- **Scalability**: Automatically scale network resources based on demand
- **Security-First**: Built-in security controls and isolation mechanisms
- **Global Reach**: Connect resources across multiple regions and availability zones
- **Cost Optimization**: Pay only for the network resources you use
- **High Availability**: Built-in redundancy and failover capabilities

**Why Cloud Networking Matters:**

- **Security**: Implement network isolation, access controls, and traffic filtering
- **Scalability**: Handle varying traffic loads and resource requirements
- **Reliability**: Ensure high availability and fault tolerance
- **Performance**: Optimize network performance and reduce latency
- **Cost Efficiency**: Reduce infrastructure costs through shared resources
- **Global Distribution**: Deploy applications across multiple regions
- **Compliance**: Meet regulatory requirements for data residency and security
- **Innovation**: Enable new architectures like microservices and serverless

**Key Components:**

**1. VPC (Virtual Private Cloud):**

- **Definition**: Isolated network environment within the cloud provider's infrastructure
- **Purpose**: Provide secure, isolated network space for cloud resources
- **Benefits**: Network isolation, custom IP addressing, control over network topology
- **Use Cases**: Multi-tenant applications, compliance requirements, network segmentation
- **Best Practices**: Use private IP ranges, implement proper subnetting, enable DNS resolution

**2. Subnets:**

- **Definition**: Network segments within a VPC that group related resources
- **Purpose**: Organize resources and control network traffic flow
- **Benefits**: Network segmentation, traffic isolation, security boundaries
- **Use Cases**: Public/private resource separation, multi-tier applications, compliance
- **Best Practices**: Use appropriate CIDR blocks, separate public and private subnets, plan for growth

**3. Security Groups:**

- **Definition**: Virtual firewalls that control inbound and outbound traffic for instances
- **Purpose**: Implement network-level security controls and access policies
- **Benefits**: Stateful filtering, instance-level security, easy management
- **Use Cases**: Application security, access control, traffic filtering
- **Best Practices**: Use least privilege principle, separate security groups by function, regular review

**4. Load Balancers:**

- **Definition**: Services that distribute incoming traffic across multiple targets
- **Purpose**: Ensure high availability, scalability, and performance
- **Benefits**: Traffic distribution, health checking, SSL termination, auto-scaling
- **Use Cases**: High availability, traffic management, performance optimization
- **Best Practices**: Use appropriate load balancer type, implement health checks, enable SSL/TLS

**Advanced Cloud Networking Concepts:**

- **NAT Gateways**: Provide outbound internet access for private subnets
- **Internet Gateways**: Enable internet access for public subnets
- **Route Tables**: Control traffic routing within VPCs
- **Network ACLs**: Additional layer of security at subnet level
- **VPC Peering**: Connect VPCs for resource sharing
- **Transit Gateways**: Centralized network connectivity hub
- **VPN Connections**: Secure connections between on-premises and cloud
- **Direct Connect**: Dedicated network connections to cloud providers

**Discussion Questions & Answers:**

**Q1: How do you design a comprehensive cloud networking architecture for a large-scale, multi-region application with strict security and compliance requirements?**

**Answer:** Comprehensive cloud networking architecture design:

- **Multi-Region VPC Design**: Implement VPCs in multiple regions with consistent CIDR blocks and naming conventions
- **Network Segmentation**: Use public, private, and database subnets with appropriate security groups
- **Cross-Region Connectivity**: Implement VPC peering or transit gateways for inter-region communication
- **Security Architecture**: Use security groups, NACLs, and network firewalls for defense in depth
- **Load Balancing Strategy**: Implement global load balancers with regional failover capabilities
- **DNS Management**: Use managed DNS services with health checks and failover
- **Monitoring and Logging**: Implement VPC Flow Logs, network monitoring, and security logging
- **Compliance Controls**: Ensure network architecture meets regulatory requirements
- **Disaster Recovery**: Design for regional failures with automated failover
- **Cost Optimization**: Use appropriate instance types and network services
- **Documentation**: Maintain comprehensive network documentation and diagrams
- **Testing**: Implement network testing and validation procedures

**Q2: What are the key considerations when implementing network security for a microservices architecture with service-to-service communication?**

**Answer:** Microservices network security implementation:

- **Service Mesh**: Implement service mesh (Istio, Linkerd) for secure service-to-service communication
- **Network Policies**: Use Kubernetes network policies or cloud-native network policies
- **Zero Trust Architecture**: Implement zero trust principles with mutual TLS and identity verification
- **API Gateway Security**: Centralize authentication and authorization at the API gateway
- **Network Segmentation**: Isolate microservices using separate subnets and security groups
- **Traffic Encryption**: Encrypt all service-to-service communication using TLS
- **Service Discovery**: Implement secure service discovery with proper authentication
- **Monitoring**: Deploy network security monitoring and anomaly detection
- **Compliance**: Ensure microservices communication meets compliance requirements
- **Incident Response**: Have clear procedures for network security incidents
- **Regular Audits**: Conduct regular network security audits and penetration testing
- **Training**: Provide security training for teams working with microservices

**Q3: How do you optimize cloud networking performance and costs for high-traffic applications while maintaining security and reliability?**

**Answer:** Cloud networking performance and cost optimization:

- **CDN Implementation**: Use Content Delivery Networks for static content and global distribution
- **Load Balancer Optimization**: Choose appropriate load balancer types and configurations
- **Connection Pooling**: Implement connection pooling and keep-alive connections
- **Network Monitoring**: Use network performance monitoring to identify bottlenecks
- **Traffic Analysis**: Analyze traffic patterns and optimize routing accordingly
- **Caching Strategies**: Implement caching at multiple levels to reduce network traffic
- **Compression**: Use compression for network traffic to reduce bandwidth costs
- **Regional Optimization**: Deploy resources closer to users to reduce latency
- **Auto-scaling**: Implement auto-scaling to handle traffic spikes efficiently
- **Cost Monitoring**: Monitor network costs and optimize resource usage
- **Performance Testing**: Conduct regular performance testing and optimization
- **Documentation**: Maintain performance baselines and optimization procedures

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
