# 🔧 Ansible: Configuration Management and Automation

> **Master Ansible for configuration management, automation, and orchestration**

## 📚 Concept

Ansible is an open-source automation platform that provides configuration management, application deployment, and orchestration. It uses YAML-based playbooks to define automation tasks and executes them over SSH without requiring agents on target systems.

### Key Features
- **Agentless**: No agents required on target systems
- **Idempotent**: Safe to run multiple times
- **YAML-based**: Human-readable configuration
- **Inventory Management**: Dynamic host management
- **Modules**: Extensive library of automation modules
- **Playbooks**: Orchestration and workflow automation

## 🏗️ Ansible Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Ansible Control Node                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  Playbooks  │  │  Inventory  │  │   Modules   │     │
│  │   (YAML)    │  │   (Hosts)   │  │  (Tasks)    │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Ansible Engine                       │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Parser    │  │   Executor  │  │   Callback  │ │ │
│  │  │   Engine    │  │   Engine    │  │   Plugins   │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   SSH       │  │   WinRM     │  │   API       │     │
│  │ Connection  │  │ Connection  │  │ Connection  │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Hands-on Example

### Inventory Configuration

```ini
# inventory.ini
[webservers]
web1 ansible_host=10.0.1.10 ansible_user=ubuntu
web2 ansible_host=10.0.1.11 ansible_user=ubuntu
web3 ansible_host=10.0.1.12 ansible_user=ubuntu

[databases]
db1 ansible_host=10.0.2.10 ansible_user=ubuntu
db2 ansible_host=10.0.2.11 ansible_user=ubuntu

[load_balancers]
lb1 ansible_host=10.0.3.10 ansible_user=ubuntu

[production:children]
webservers
databases
load_balancers

[production:vars]
ansible_ssh_private_key_file=~/.ssh/production_key.pem
ansible_ssh_common_args='-o StrictHostKeyChecking=no'
```

### Dynamic Inventory Script

```python
#!/usr/bin/env python3
# aws_ec2.py

import json
import boto3
import os

def get_instances():
    ec2 = boto3.client('ec2')
    
    response = ec2.describe_instances(
        Filters=[
            {'Name': 'instance-state-name', 'Values': ['running']},
            {'Name': 'tag:Environment', 'Values': ['production']}
        ]
    )
    
    inventory = {
        '_meta': {
            'hostvars': {}
        },
        'webservers': {
            'hosts': [],
            'vars': {
                'ansible_user': 'ubuntu',
                'ansible_ssh_private_key_file': '~/.ssh/production_key.pem'
            }
        },
        'databases': {
            'hosts': [],
            'vars': {
                'ansible_user': 'ubuntu',
                'ansible_ssh_private_key_file': '~/.ssh/production_key.pem'
            }
        }
    }
    
    for reservation in response['Reservations']:
        for instance in reservation['Instances']:
            hostname = None
            group = None
            
            for tag in instance.get('Tags', []):
                if tag['Key'] == 'Name':
                    hostname = tag['Value']
                elif tag['Key'] == 'Role':
                    group = tag['Value'].lower() + 's'
            
            if hostname and group:
                inventory[group]['hosts'].append(hostname)
                inventory['_meta']['hostvars'][hostname] = {
                    'ansible_host': instance['PrivateIpAddress'],
                    'instance_id': instance['InstanceId'],
                    'availability_zone': instance['Placement']['AvailabilityZone']
                }
    
    return inventory

if __name__ == '__main__':
    print(json.dumps(get_instances(), indent=2))
```

### Basic Playbook

```yaml
# site.yml
---
- name: Configure Web Servers
  hosts: webservers
  become: yes
  vars:
    app_name: "my-app"
    app_port: 8080
    app_user: "appuser"
    app_group: "appgroup"
    app_home: "/opt/{{ app_name }}"
    app_repo: "https://github.com/company/my-app.git"
    app_branch: "main"
    database_host: "{{ hostvars[groups['databases'][0]]['ansible_host'] }}"
    database_port: 5432
    database_name: "myapp"
    database_user: "myapp"
    database_password: "{{ vault_database_password }}"
  
  pre_tasks:
    - name: Update package cache
      apt:
        update_cache: yes
        cache_valid_time: 3600
      when: ansible_os_family == "Debian"
    
    - name: Update package cache
      yum:
        update_cache: yes
      when: ansible_os_family == "RedHat"
  
  tasks:
    - name: Install required packages
      package:
        name:
          - python3
          - python3-pip
          - git
          - nginx
          - postgresql-client
          - curl
          - htop
          - unzip
        state: present
    
    - name: Create application user
      user:
        name: "{{ app_user }}"
        group: "{{ app_group }}"
        home: "{{ app_home }}"
        shell: /bin/bash
        create_home: yes
        state: present
    
    - name: Create application directory
      file:
        path: "{{ app_home }}"
        state: directory
        owner: "{{ app_user }}"
        group: "{{ app_group }}"
        mode: '0755'
    
    - name: Clone application repository
      git:
        repo: "{{ app_repo }}"
        dest: "{{ app_home }}/app"
        version: "{{ app_branch }}"
        force: yes
      become_user: "{{ app_user }}"
    
    - name: Install Python dependencies
      pip:
        requirements: "{{ app_home }}/app/requirements.txt"
        virtualenv: "{{ app_home }}/venv"
        virtualenv_python: python3
      become_user: "{{ app_user }}"
    
    - name: Create application configuration
      template:
        src: app.conf.j2
        dest: "{{ app_home }}/app/config.py"
        owner: "{{ app_user }}"
        group: "{{ app_group }}"
        mode: '0644'
      become_user: "{{ app_user }}"
    
    - name: Create systemd service file
      template:
        src: app.service.j2
        dest: /etc/systemd/system/{{ app_name }}.service
        mode: '0644'
      notify: restart app service
    
    - name: Enable and start application service
      systemd:
        name: "{{ app_name }}"
        enabled: yes
        state: started
        daemon_reload: yes
    
    - name: Configure Nginx
      template:
        src: nginx.conf.j2
        dest: /etc/nginx/sites-available/{{ app_name }}
        mode: '0644'
      notify: restart nginx
    
    - name: Enable Nginx site
      file:
        src: /etc/nginx/sites-available/{{ app_name }}
        dest: /etc/nginx/sites-enabled/{{ app_name }}
        state: link
      notify: restart nginx
    
    - name: Remove default Nginx site
      file:
        path: /etc/nginx/sites-enabled/default
        state: absent
      notify: restart nginx
    
    - name: Test Nginx configuration
      command: nginx -t
      register: nginx_test
      failed_when: nginx_test.rc != 0
    
    - name: Install CloudWatch agent
      get_url:
        url: https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
        dest: /tmp/amazon-cloudwatch-agent.deb
        mode: '0644'
    
    - name: Install CloudWatch agent package
      apt:
        deb: /tmp/amazon-cloudwatch-agent.deb
        state: present
    
    - name: Configure CloudWatch agent
      template:
        src: cloudwatch-config.json.j2
        dest: /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json
        mode: '0644'
      notify: restart cloudwatch agent
    
    - name: Start CloudWatch agent
      systemd:
        name: amazon-cloudwatch-agent
        enabled: yes
        state: started
  
  handlers:
    - name: restart app service
      systemd:
        name: "{{ app_name }}"
        state: restarted
    
    - name: restart nginx
      systemd:
        name: nginx
        state: restarted
    
    - name: restart cloudwatch agent
      systemd:
        name: amazon-cloudwatch-agent
        state: restarted
  
  post_tasks:
    - name: Verify application is running
      uri:
        url: "http://localhost:{{ app_port }}/health"
        method: GET
        status_code: 200
      retries: 5
      delay: 10
    
    - name: Display application status
      debug:
        msg: "Application {{ app_name }} is running on port {{ app_port }}"
```

### Database Configuration Playbook

```yaml
# database.yml
---
- name: Configure Database Servers
  hosts: databases
  become: yes
  vars:
    postgres_version: "14"
    postgres_data_dir: "/var/lib/postgresql/{{ postgres_version }}/main"
    postgres_config_dir: "/etc/postgresql/{{ postgres_version }}/main"
    database_name: "myapp"
    database_user: "myapp"
    database_password: "{{ vault_database_password }}"
    backup_retention_days: 7
    backup_schedule: "0 2 * * *"
  
  tasks:
    - name: Install PostgreSQL
      package:
        name: "postgresql-{{ postgres_version }}"
        state: present
    
    - name: Install PostgreSQL contrib packages
      package:
        name: "postgresql-contrib-{{ postgres_version }}"
        state: present
    
    - name: Start and enable PostgreSQL
      systemd:
        name: "postgresql"
        enabled: yes
        state: started
    
    - name: Configure PostgreSQL
      template:
        src: postgresql.conf.j2
        dest: "{{ postgres_config_dir }}/postgresql.conf"
        mode: '0644'
      notify: restart postgresql
    
    - name: Configure PostgreSQL authentication
      template:
        src: pg_hba.conf.j2
        dest: "{{ postgres_config_dir }}/pg_hba.conf"
        mode: '0644'
      notify: restart postgresql
    
    - name: Create database
      postgresql_db:
        name: "{{ database_name }}"
        state: present
    
    - name: Create database user
      postgresql_user:
        name: "{{ database_user }}"
        password: "{{ database_password }}"
        priv: "{{ database_name }}:ALL"
        state: present
    
    - name: Install backup tools
      package:
        name:
          - postgresql-client-{{ postgres_version }}
          - awscli
        state: present
    
    - name: Create backup directory
      file:
        path: /opt/backups
        state: directory
        mode: '0755'
    
    - name: Create backup script
      template:
        src: backup.sh.j2
        dest: /opt/backups/backup.sh
        mode: '0755'
    
    - name: Setup backup cron job
      cron:
        name: "Database backup"
        job: "/opt/backups/backup.sh"
        minute: "0"
        hour: "2"
        user: postgres
    
    - name: Install monitoring tools
      package:
        name:
          - postgresql-client-{{ postgres_version }}
          - python3-psycopg2
        state: present
    
    - name: Create monitoring script
      template:
        src: monitor.sh.j2
        dest: /opt/monitor.sh
        mode: '0755'
    
    - name: Setup monitoring cron job
      cron:
        name: "Database monitoring"
        job: "/opt/monitor.sh"
        minute: "*/5"
        user: postgres
  
  handlers:
    - name: restart postgresql
      systemd:
        name: "postgresql"
        state: restarted
```

### Load Balancer Configuration Playbook

```yaml
# loadbalancer.yml
---
- name: Configure Load Balancer
  hosts: load_balancers
  become: yes
  vars:
    app_name: "my-app"
    app_port: 8080
    ssl_cert_path: "/etc/ssl/certs/{{ app_name }}.crt"
    ssl_key_path: "/etc/ssl/private/{{ app_name }}.key"
    upstream_servers: "{{ groups['webservers'] }}"
  
  tasks:
    - name: Install Nginx
      package:
        name: nginx
        state: present
    
    - name: Install SSL tools
      package:
        name:
          - certbot
          - python3-certbot-nginx
        state: present
    
    - name: Start and enable Nginx
      systemd:
        name: nginx
        enabled: yes
        state: started
    
    - name: Configure Nginx load balancer
      template:
        src: nginx-lb.conf.j2
        dest: /etc/nginx/sites-available/{{ app_name }}-lb
        mode: '0644'
      notify: restart nginx
    
    - name: Enable load balancer site
      file:
        src: /etc/nginx/sites-available/{{ app_name }}-lb
        dest: /etc/nginx/sites-enabled/{{ app_name }}-lb
        state: link
      notify: restart nginx
    
    - name: Remove default Nginx site
      file:
        path: /etc/nginx/sites-enabled/default
        state: absent
      notify: restart nginx
    
    - name: Test Nginx configuration
      command: nginx -t
      register: nginx_test
      failed_when: nginx_test.rc != 0
    
    - name: Setup SSL certificate
      command: certbot --nginx -d {{ app_name }}.example.com --non-interactive --agree-tos --email admin@example.com
      when: ssl_cert_path is not defined
    
    - name: Configure log rotation
      template:
        src: nginx-logrotate.j2
        dest: /etc/logrotate.d/nginx
        mode: '0644'
    
    - name: Install monitoring tools
      package:
        name:
          - htop
          - iotop
          - nethogs
        state: present
    
    - name: Create monitoring script
      template:
        src: lb-monitor.sh.j2
        dest: /opt/lb-monitor.sh
        mode: '0755'
    
    - name: Setup monitoring cron job
      cron:
        name: "Load balancer monitoring"
        job: "/opt/lb-monitor.sh"
        minute: "*/5"
  
  handlers:
    - name: restart nginx
      systemd:
        name: nginx
        state: restarted
```

### Template Files

```jinja2
# app.conf.j2
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'postgresql://{{ database_user }}:{{ database_password }}@{{ database_host }}:{{ database_port }}/{{ database_name }}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Redis configuration
    REDIS_URL = os.environ.get('REDIS_URL') or 'redis://localhost:6379/0'
    
    # Logging configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL') or 'INFO'
    LOG_FILE = os.environ.get('LOG_FILE') or '/var/log/{{ app_name }}/app.log'
    
    # Application settings
    APP_NAME = '{{ app_name }}'
    APP_VERSION = '1.0.0'
    DEBUG = False
    
    # Security settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Rate limiting
    RATELIMIT_STORAGE_URL = os.environ.get('REDIS_URL') or 'redis://localhost:6379/0'
    RATELIMIT_DEFAULT = "100 per hour"
```

```jinja2
# app.service.j2
[Unit]
Description={{ app_name }} Application
After=network.target

[Service]
Type=simple
User={{ app_user }}
Group={{ app_group }}
WorkingDirectory={{ app_home }}/app
Environment=PATH={{ app_home }}/venv/bin
ExecStart={{ app_home }}/venv/bin/python app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```jinja2
# nginx.conf.j2
server {
    listen 80;
    server_name {{ app_name }}.example.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name {{ app_name }}.example.com;
    
    # SSL configuration
    ssl_certificate {{ ssl_cert_path }};
    ssl_certificate_key {{ ssl_key_path }};
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=login:10m rate=1r/s;
    
    # API endpoints
    location /api/ {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://127.0.0.1:{{ app_port }};
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
    
    # Login endpoint
    location /login {
        limit_req zone=login burst=5 nodelay;
        proxy_pass http://127.0.0.1:{{ app_port }};
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Static files
    location /static/ {
        alias {{ app_home }}/app/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # Health check
    location /health {
        proxy_pass http://127.0.0.1:{{ app_port }};
        proxy_set_header Host $host;
        access_log off;
    }
    
    # Logging
    access_log /var/log/nginx/{{ app_name }}_access.log;
    error_log /var/log/nginx/{{ app_name }}_error.log;
}
```

### Ansible Commands

```bash
# Run playbook
ansible-playbook -i inventory.ini site.yml

# Run playbook with specific hosts
ansible-playbook -i inventory.ini site.yml --limit webservers

# Run playbook with tags
ansible-playbook -i inventory.ini site.yml --tags "nginx,ssl"

# Run playbook with check mode
ansible-playbook -i inventory.ini site.yml --check

# Run playbook with diff
ansible-playbook -i inventory.ini site.yml --diff

# Run playbook with verbose output
ansible-playbook -i inventory.ini site.yml -v

# Run playbook with extra variables
ansible-playbook -i inventory.ini site.yml -e "app_version=2.0.0"

# Run playbook with vault password
ansible-playbook -i inventory.ini site.yml --vault-password-file .vault_pass

# Run ad-hoc commands
ansible webservers -i inventory.ini -m ping
ansible webservers -i inventory.ini -m shell -a "uptime"
ansible webservers -i inventory.ini -m copy -a "src=file.txt dest=/tmp/file.txt"

# Use dynamic inventory
ansible-playbook -i aws_ec2.py site.yml

# Test connectivity
ansible all -i inventory.ini -m ping

# Gather facts
ansible all -i inventory.ini -m setup

# Run with parallel execution
ansible-playbook -i inventory.ini site.yml -f 10

# Run with specific user
ansible-playbook -i inventory.ini site.yml -u ubuntu

# Run with sudo
ansible-playbook -i inventory.ini site.yml --become

# Run with specific become method
ansible-playbook -i inventory.ini site.yml --become-method sudo
```

## 🚀 Best Practices

### 1. Inventory Management
```ini
# Use groups and variables
[webservers:vars]
ansible_user=ubuntu
ansible_ssh_private_key_file=~/.ssh/production_key.pem

[webservers]
web1 ansible_host=10.0.1.10
web2 ansible_host=10.0.1.11
```

### 2. Playbook Structure
```yaml
# Use roles for organization
- name: Configure Web Servers
  hosts: webservers
  roles:
    - common
    - nginx
    - app
  vars:
    app_name: "my-app"
```

### 3. Variable Management
```yaml
# Use group_vars and host_vars
# group_vars/webservers.yml
app_name: "my-app"
app_port: 8080

# host_vars/web1.yml
app_version: "2.0.0"
```

## 🏢 Industry Insights

### Ansible Usage Patterns
- **Configuration Management**: Server configuration
- **Application Deployment**: Automated deployments
- **Infrastructure Provisioning**: Cloud resource management
- **Orchestration**: Multi-step workflows

### Enterprise Ansible Strategy
- **Tower/AWX**: Centralized management
- **Roles**: Reusable components
- **Vault**: Secret management
- **Testing**: Molecule and Testinfra

## 🎯 Interview Questions

### Basic Level
1. **What is Ansible?**
   - Configuration management tool
   - Agentless automation
   - YAML-based playbooks
   - SSH-based execution

2. **What is an Ansible playbook?**
   - YAML configuration file
   - Task definitions
   - Host targeting
   - Variable management

3. **What is Ansible inventory?**
   - Host definitions
   - Group organization
   - Variable assignment
   - Connection parameters

### Intermediate Level
4. **How do you handle Ansible variables?**
   ```yaml
   # Variable precedence
   - command line variables
   - playbook variables
   - inventory variables
   - group_vars
   - host_vars
   ```

5. **How do you implement Ansible roles?**
   - Role structure
   - Task organization
   - Variable management
   - Handler definitions

6. **How do you handle Ansible secrets?**
   - Ansible Vault
   - Encrypted variables
   - Secret management
   - Access control

### Advanced Level
7. **How do you implement Ansible patterns?**
   - Role composition
   - Dynamic inventory
   - Conditional execution
   - Error handling

8. **How do you handle Ansible scaling?**
   - Parallel execution
   - Inventory optimization
   - Performance tuning
   - Resource management

9. **How do you implement Ansible testing?**
   - Molecule testing
   - Testinfra validation
   - CI/CD integration
   - Quality assurance

---

**Next**: [Pulumi](./Pulumi.md) - Modern infrastructure as code, multi-language support
