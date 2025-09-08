# 🚀 Pulumi: Modern Infrastructure as Code

> **Master Pulumi for infrastructure provisioning with real programming languages**

## 📚 Concept

Pulumi is a modern Infrastructure as Code platform that allows you to define infrastructure using real programming languages like TypeScript, Python, Go, and C#. It provides a unified approach to managing cloud resources across multiple providers.

### Key Features

- **Real Programming Languages**: TypeScript, Python, Go, C#
- **Multi-Cloud**: Support for AWS, Azure, GCP, Kubernetes
- **State Management**: Built-in state management
- **Testing**: Unit and integration testing
- **Policy as Code**: CrossGuard policy engine
- **Secrets Management**: Built-in secret handling

## 🏗️ Pulumi Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Pulumi Workflow                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Write     │  │    Plan     │  │   Deploy    │     │
│  │   Code      │  │   Preview   │  │  Resources  │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Pulumi Engine                        │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Language  │  │   Resource  │  │   Provider  │ │ │
│  │  │   Host      │  │   Graph     │  │   Plugins   │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   AWS       │  │   GCP       │  │   Azure     │     │
│  │  Provider   │  │  Provider   │  │  Provider   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Hands-on Example

### TypeScript Configuration

```typescript
// index.ts
import * as pulumi from "@pulumi/pulumi";
import * as aws from "@pulumi/aws";
import * as awsx from "@pulumi/awsx";

// Configuration
const config = new pulumi.Config();
const environment = config.get("environment") || "dev";
const projectName = config.get("projectName") || "my-app";

// VPC
const vpc = new awsx.ec2.Vpc(`${projectName}-vpc`, {
  cidrBlock: "10.0.0.0/16",
  numberOfAvailabilityZones: 3,
  enableDnsHostnames: true,
  enableDnsSupport: true,
  tags: {
    Name: `${projectName}-vpc`,
    Environment: environment,
  },
});

// Security Groups
const webSecurityGroup = new aws.ec2.SecurityGroup(`${projectName}-web-sg`, {
  vpcId: vpc.vpcId,
  description: "Security group for web servers",
  ingress: [
    {
      protocol: "tcp",
      fromPort: 80,
      toPort: 80,
      cidrBlocks: ["0.0.0.0/0"],
    },
    {
      protocol: "tcp",
      fromPort: 443,
      toPort: 443,
      cidrBlocks: ["0.0.0.0/0"],
    },
    {
      protocol: "tcp",
      fromPort: 22,
      toPort: 22,
      cidrBlocks: [vpc.vpc.cidrBlock],
    },
  ],
  egress: [
    {
      protocol: "-1",
      fromPort: 0,
      toPort: 0,
      cidrBlocks: ["0.0.0.0/0"],
    },
  ],
  tags: {
    Name: `${projectName}-web-sg`,
    Environment: environment,
  },
});

const dbSecurityGroup = new aws.ec2.SecurityGroup(`${projectName}-db-sg`, {
  vpcId: vpc.vpcId,
  description: "Security group for database",
  ingress: [
    {
      protocol: "tcp",
      fromPort: 5432,
      toPort: 5432,
      securityGroups: [webSecurityGroup.id],
    },
  ],
  egress: [
    {
      protocol: "-1",
      fromPort: 0,
      toPort: 0,
      cidrBlocks: ["0.0.0.0/0"],
    },
  ],
  tags: {
    Name: `${projectName}-db-sg`,
    Environment: environment,
  },
});

// Application Load Balancer
const alb = new awsx.lb.ApplicationLoadBalancer(`${projectName}-alb`, {
  vpc: vpc,
  securityGroups: [webSecurityGroup.id],
  tags: {
    Name: `${projectName}-alb`,
    Environment: environment,
  },
});

// Target Group
const targetGroup = new aws.lb.TargetGroup(`${projectName}-tg`, {
  port: 80,
  protocol: "HTTP",
  vpcId: vpc.vpcId,
  targetType: "instance",
  healthCheck: {
    enabled: true,
    healthyThreshold: 2,
    unhealthyThreshold: 3,
    timeout: 5,
    interval: 30,
    path: "/health",
    matcher: "200",
    port: "traffic-port",
    protocol: "HTTP",
  },
  tags: {
    Name: `${projectName}-tg`,
    Environment: environment,
  },
});

// Load Balancer Listener
const listener = new aws.lb.Listener(`${projectName}-listener`, {
  loadBalancerArn: alb.loadBalancer.arn,
  port: 80,
  protocol: "HTTP",
  defaultActions: [
    {
      type: "forward",
      targetGroupArn: targetGroup.arn,
    },
  ],
});

// Launch Template
const launchTemplate = new aws.ec2.LaunchTemplate(`${projectName}-lt`, {
  imageId: aws.ec2
    .getAmi({
      mostRecent: true,
      owners: ["amazon"],
      filters: [
        {
          name: "name",
          values: ["amzn2-ami-hvm-*-x86_64-gp2"],
        },
      ],
    })
    .then((ami) => ami.id),
  instanceType: "t3.micro",
  keyName: config.require("keyPairName"),
  vpcSecurityGroupIds: [webSecurityGroup.id],
  userData: pulumi.interpolate`#!/bin/bash
yum update -y
yum install -y docker
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
mkdir -p /opt/app
cd /opt/app
cat > docker-compose.yml << EOF
version: '3.8'
services:
  app:
    image: ${projectName}:latest
    ports:
      - "80:8080"
    environment:
      - ENV=${environment}
      - DATABASE_URL=postgresql://postgres:password@${rds.endpoint}:5432/myapp
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
EOF
docker-compose up -d`,
  tagSpecifications: [
    {
      resourceType: "instance",
      tags: {
        Name: `${projectName}-instance`,
        Environment: environment,
      },
    },
  ],
  tags: {
    Name: `${projectName}-lt`,
    Environment: environment,
  },
});

// Auto Scaling Group
const asg = new aws.autoscaling.Group(`${projectName}-asg`, {
  vpcZoneIdentifiers: vpc.privateSubnetIds,
  targetGroupArns: [targetGroup.arn],
  healthCheckType: "ELB",
  healthCheckGracePeriod: 300,
  minSize: 2,
  maxSize: 10,
  desiredCapacity: 3,
  launchTemplate: {
    id: launchTemplate.id,
    version: "$Latest",
  },
  tags: [
    {
      key: "Name",
      value: `${projectName}-asg`,
      propagateAtLaunch: true,
    },
    {
      key: "Environment",
      value: environment,
      propagateAtLaunch: true,
    },
  ],
});

// Auto Scaling Policies
const scaleUpPolicy = new aws.autoscaling.Policy(`${projectName}-scale-up`, {
  scalingAdjustment: 1,
  adjustmentType: "ChangeInCapacity",
  cooldown: 300,
  autoscalingGroupName: asg.name,
});

const scaleDownPolicy = new aws.autoscaling.Policy(
  `${projectName}-scale-down`,
  {
    scalingAdjustment: -1,
    adjustmentType: "ChangeInCapacity",
    cooldown: 300,
    autoscalingGroupName: asg.name,
  }
);

// CloudWatch Alarms
const cpuHighAlarm = new aws.cloudwatch.MetricAlarm(`${projectName}-cpu-high`, {
  alarmName: `${projectName}-cpu-high`,
  comparisonOperator: "GreaterThanThreshold",
  evaluationPeriods: 2,
  metricName: "CPUUtilization",
  namespace: "AWS/EC2",
  period: 300,
  statistic: "Average",
  threshold: 70,
  alarmDescription: "This metric monitors ec2 cpu utilization",
  alarmActions: [scaleUpPolicy.arn],
  dimensions: {
    AutoScalingGroupName: asg.name,
  },
});

const cpuLowAlarm = new aws.cloudwatch.MetricAlarm(`${projectName}-cpu-low`, {
  alarmName: `${projectName}-cpu-low`,
  comparisonOperator: "LessThanThreshold",
  evaluationPeriods: 2,
  metricName: "CPUUtilization",
  namespace: "AWS/EC2",
  period: 300,
  statistic: "Average",
  threshold: 20,
  alarmDescription: "This metric monitors ec2 cpu utilization",
  alarmActions: [scaleDownPolicy.arn],
  dimensions: {
    AutoScalingGroupName: asg.name,
  },
});

// RDS Database
const dbSubnetGroup = new aws.rds.SubnetGroup(
  `${projectName}-db-subnet-group`,
  {
    subnetIds: vpc.privateSubnetIds,
    tags: {
      Name: `${projectName}-db-subnet-group`,
      Environment: environment,
    },
  }
);

const rds = new aws.rds.Instance(`${projectName}-db`, {
  identifier: `${projectName}-db`,
  engine: "postgres",
  engineVersion: "14.7",
  instanceClass: "db.t3.micro",
  allocatedStorage: 20,
  maxAllocatedStorage: 100,
  storageType: "gp2",
  storageEncrypted: true,
  dbName: "myapp",
  username: "postgres",
  password: config.requireSecret("dbPassword"),
  vpcSecurityGroupIds: [dbSecurityGroup.id],
  dbSubnetGroupName: dbSubnetGroup.name,
  backupRetentionPeriod: 7,
  backupWindow: "03:00-04:00",
  maintenanceWindow: "sun:04:00-sun:05:00",
  skipFinalSnapshot: true,
  deletionProtection: false,
  tags: {
    Name: `${projectName}-db`,
    Environment: environment,
  },
});

// Outputs
export const vpcId = vpc.vpcId;
export const vpcCidrBlock = vpc.vpc.cidrBlock;
export const publicSubnetIds = vpc.publicSubnetIds;
export const privateSubnetIds = vpc.privateSubnetIds;
export const loadBalancerDns = alb.loadBalancer.dnsName;
export const loadBalancerZoneId = alb.loadBalancer.zoneId;
export const autoScalingGroupName = asg.name;
export const databaseEndpoint = rds.endpoint;
export const databasePort = rds.port;
```

### Python Configuration

```python
# __main__.py
import pulumi
import pulumi_aws as aws
import pulumi_awsx as awsx

# Configuration
config = pulumi.Config()
environment = config.get("environment") or "dev"
project_name = config.get("projectName") or "my-app"

# VPC
vpc = awsx.ec2.Vpc(f"{project_name}-vpc",
    cidr_block="10.0.0.0/16",
    number_of_availability_zones=3,
    enable_dns_hostnames=True,
    enable_dns_support=True,
    tags={
        "Name": f"{project_name}-vpc",
        "Environment": environment,
    }
)

# Security Groups
web_security_group = aws.ec2.SecurityGroup(f"{project_name}-web-sg",
    vpc_id=vpc.vpc_id,
    description="Security group for web servers",
    ingress=[
        aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp",
            from_port=80,
            to_port=80,
            cidr_blocks=["0.0.0.0/0"],
        ),
        aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp",
            from_port=443,
            to_port=443,
            cidr_blocks=["0.0.0.0/0"],
        ),
        aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp",
            from_port=22,
            to_port=22,
            cidr_blocks=[vpc.vpc.cidr_block],
        ),
    ],
    egress=[
        aws.ec2.SecurityGroupEgressArgs(
            protocol="-1",
            from_port=0,
            to_port=0,
            cidr_blocks=["0.0.0.0/0"],
        ),
    ],
    tags={
        "Name": f"{project_name}-web-sg",
        "Environment": environment,
    }
)

# Application Load Balancer
alb = awsx.lb.ApplicationLoadBalancer(f"{project_name}-alb",
    vpc=vpc,
    security_groups=[web_security_group.id],
    tags={
        "Name": f"{project_name}-alb",
        "Environment": environment,
    }
)

# Target Group
target_group = aws.lb.TargetGroup(f"{project_name}-tg",
    port=80,
    protocol="HTTP",
    vpc_id=vpc.vpc_id,
    target_type="instance",
    health_check=aws.lb.TargetGroupHealthCheckArgs(
        enabled=True,
        healthy_threshold=2,
        unhealthy_threshold=3,
        timeout=5,
        interval=30,
        path="/health",
        matcher="200",
        port="traffic-port",
        protocol="HTTP",
    ),
    tags={
        "Name": f"{project_name}-tg",
        "Environment": environment,
    }
)

# Load Balancer Listener
listener = aws.lb.Listener(f"{project_name}-listener",
    load_balancer_arn=alb.load_balancer.arn,
    port=80,
    protocol="HTTP",
    default_actions=[
        aws.lb.ListenerDefaultActionArgs(
            type="forward",
            target_group_arn=target_group.arn,
        ),
    ]
)

# Launch Template
launch_template = aws.ec2.LaunchTemplate(f"{project_name}-lt",
    image_id=aws.ec2.get_ami(
        most_recent=True,
        owners=["amazon"],
        filters=[
            aws.ec2.GetAmiFilterArgs(
                name="name",
                values=["amzn2-ami-hvm-*-x86_64-gp2"],
            ),
        ]
    ).id,
    instance_type="t3.micro",
    key_name=config.require("keyPairName"),
    vpc_security_group_ids=[web_security_group.id],
    user_data=pulumi.Output.concat(
        "#!/bin/bash\n",
        "yum update -y\n",
        "yum install -y docker\n",
        "systemctl start docker\n",
        "systemctl enable docker\n",
        "usermod -a -G docker ec2-user\n",
        "curl -L \"https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)\" -o /usr/local/bin/docker-compose\n",
        "chmod +x /usr/local/bin/docker-compose\n",
        "mkdir -p /opt/app\n",
        "cd /opt/app\n",
        "cat > docker-compose.yml << EOF\n",
        "version: '3.8'\n",
        "services:\n",
        "  app:\n",
        "    image: ", project_name, ":latest\n",
        "    ports:\n",
        "      - \"80:8080\"\n",
        "    environment:\n",
        "      - ENV=", environment, "\n",
        "    restart: unless-stopped\n",
        "EOF\n",
        "docker-compose up -d"
    ),
    tag_specifications=[
        aws.ec2.LaunchTemplateTagSpecificationArgs(
            resource_type="instance",
            tags={
                "Name": f"{project_name}-instance",
                "Environment": environment,
            },
        ),
    ],
    tags={
        "Name": f"{project_name}-lt",
        "Environment": environment,
    }
)

# Auto Scaling Group
asg = aws.autoscaling.Group(f"{project_name}-asg",
    vpc_zone_identifiers=vpc.private_subnet_ids,
    target_group_arns=[target_group.arn],
    health_check_type="ELB",
    health_check_grace_period=300,
    min_size=2,
    max_size=10,
    desired_capacity=3,
    launch_template=aws.autoscaling.GroupLaunchTemplateArgs(
        id=launch_template.id,
        version="$Latest",
    ),
    tags=[
        aws.autoscaling.GroupTagArgs(
            key="Name",
            value=f"{project_name}-asg",
            propagate_at_launch=True,
        ),
        aws.autoscaling.GroupTagArgs(
            key="Environment",
            value=environment,
            propagate_at_launch=True,
        ),
    ]
)

# RDS Database
db_subnet_group = aws.rds.SubnetGroup(f"{project_name}-db-subnet-group",
    subnet_ids=vpc.private_subnet_ids,
    tags={
        "Name": f"{project_name}-db-subnet-group",
        "Environment": environment,
    }
)

rds = aws.rds.Instance(f"{project_name}-db",
    identifier=f"{project_name}-db",
    engine="postgres",
    engine_version="14.7",
    instance_class="db.t3.micro",
    allocated_storage=20,
    max_allocated_storage=100,
    storage_type="gp2",
    storage_encrypted=True,
    db_name="myapp",
    username="postgres",
    password=config.require_secret("dbPassword"),
    vpc_security_group_ids=[db_security_group.id],
    db_subnet_group_name=db_subnet_group.name,
    backup_retention_period=7,
    backup_window="03:00-04:00",
    maintenance_window="sun:04:00-sun:05:00",
    skip_final_snapshot=True,
    deletion_protection=False,
    tags={
        "Name": f"{project_name}-db",
        "Environment": environment,
    }
)

# Outputs
pulumi.export("vpc_id", vpc.vpc_id)
pulumi.export("vpc_cidr_block", vpc.vpc.cidr_block)
pulumi.export("public_subnet_ids", vpc.public_subnet_ids)
pulumi.export("private_subnet_ids", vpc.private_subnet_ids)
pulumi.export("load_balancer_dns", alb.load_balancer.dns_name)
pulumi.export("load_balancer_zone_id", alb.load_balancer.zone_id)
pulumi.export("auto_scaling_group_name", asg.name)
pulumi.export("database_endpoint", rds.endpoint)
pulumi.export("database_port", rds.port)
```

### Pulumi Commands

```bash
# Initialize Pulumi
pulumi new typescript
pulumi new python
pulumi new go
pulumi new csharp

# Install dependencies
npm install
pip install -r requirements.txt
go mod tidy
dotnet restore

# Preview changes
pulumi preview

# Deploy infrastructure
pulumi up

# Deploy with auto-approve
pulumi up --yes

# Destroy infrastructure
pulumi destroy

# Show current state
pulumi stack output

# List resources
pulumi stack --show-urns

# Import existing resource
pulumi import aws:ec2/instance:Instance my-instance i-1234567890abcdef0

# Remove resource from state
pulumi state delete urn:pulumi:stack::project::aws:ec2/instance:Instance::my-instance

# Refresh state
pulumi refresh

# Stack management
pulumi stack init dev
pulumi stack select dev
pulumi stack ls
pulumi stack rm dev

# Configuration management
pulumi config set aws:region us-west-2
pulumi config set environment production
pulumi config set --secret dbPassword mypassword

# Policy management
pulumi policy new aws-typescript
pulumi policy publish
pulumi policy enable aws-typescript

# Testing
pulumi test
pulumi test --parallel 4
```

## 🚀 Best Practices

### 1. Project Structure

```
my-project/
├── Pulumi.yaml
├── Pulumi.dev.yaml
├── Pulumi.prod.yaml
├── index.ts
├── components/
│   ├── vpc.ts
│   ├── database.ts
│   └── application.ts
└── tests/
    └── index.test.ts
```

### 2. Configuration Management

```typescript
// Use configuration for environment-specific values
const config = new pulumi.Config();
const environment = config.get("environment") || "dev";
const dbPassword = config.requireSecret("dbPassword");
```

### 3. Resource Organization

```typescript
// Use components for reusable infrastructure
export class Database extends pulumi.ComponentResource {
  public readonly instance: aws.rds.Instance;

  constructor(
    name: string,
    args: DatabaseArgs,
    opts?: pulumi.ComponentResourceOptions
  ) {
    super("custom:Database", name, args, opts);

    this.instance = new aws.rds.Instance(
      `${name}-db`,
      {
        // ... configuration
      },
      { parent: this }
    );
  }
}
```

## 🏢 Industry Insights

### Pulumi Usage Patterns

- **Infrastructure as Code**: Modern IaC with real languages
- **Multi-Cloud**: Cross-cloud deployments
- **Testing**: Infrastructure testing
- **Policy**: Governance and compliance

### Enterprise Pulumi Strategy

- **State Management**: Centralized state
- **Policy as Code**: CrossGuard policies
- **Testing**: Automated testing
- **CI/CD**: Integration with pipelines

## 🎯 Interview Questions

### Basic Level

1. **What is Pulumi?**

   - Modern Infrastructure as Code platform
   - Real programming languages
   - Multi-cloud support
   - State management

2. **What programming languages does Pulumi support?**

   - TypeScript/JavaScript
   - Python
   - Go
   - C#

3. **What is Pulumi state?**
   - Infrastructure state tracking
   - Resource mapping
   - Dependency management
   - Change detection

### Intermediate Level

4. **How do you handle Pulumi configuration?**

   ```typescript
   const config = new pulumi.Config();
   const environment = config.get("environment");
   const secret = config.requireSecret("password");
   ```

5. **How do you implement Pulumi components?**

   - Component resources
   - Reusable infrastructure
   - Input/output types
   - Resource composition

6. **How do you handle Pulumi testing?**
   - Unit testing
   - Integration testing
   - Mocks and stubs
   - Test automation

### Advanced Level

7. **How do you implement Pulumi patterns?**

   - Component composition
   - Resource dependencies
   - Output handling
   - Error management

8. **How do you handle Pulumi security?**

   - Secret management
   - State encryption
   - Access control
   - Policy enforcement

9. **How do you implement Pulumi automation?**
   - CI/CD integration
   - Automated testing
   - Policy enforcement
   - Monitoring

---

**Next**: [Observability](./Observability/) - Logging, monitoring, tracing, alerting
