# 🚨 Alerting: Intelligent Alert Management and Notification Systems

> **Master alerting strategies, notification channels, and escalation policies for production systems**

## 📚 Concept

**Detailed Explanation:**
Alerting is a critical component of modern observability that automatically detects anomalies, failures, and performance issues in systems, applications, and infrastructure. It provides real-time notifications to appropriate teams and stakeholders, enabling rapid response to incidents and maintaining system reliability. Effective alerting systems are essential for meeting service level objectives (SLOs) and ensuring business continuity.

**Core Philosophy:**

- **Proactive Monitoring**: Detect issues before they impact users
- **Intelligent Routing**: Send alerts to the right people at the right time
- **Contextual Information**: Provide actionable information with alerts
- **Reduced Noise**: Minimize false positives and alert fatigue
- **Automated Response**: Integrate with runbooks and automated remediation
- **Continuous Improvement**: Learn from incidents to improve alerting

**Why Alerting Matters:**

- **Incident Response**: Enable rapid detection and response to system issues
- **Service Reliability**: Maintain high availability and performance standards
- **Business Continuity**: Prevent revenue loss and customer impact
- **Team Productivity**: Reduce manual monitoring overhead
- **Compliance**: Meet regulatory requirements for system monitoring
- **Cost Optimization**: Prevent expensive outages and performance degradation
- **User Experience**: Maintain optimal user experience and satisfaction
- **Operational Excellence**: Enable data-driven decision making and continuous improvement

**Key Features:**

**1. Real-time Detection:**

- **Immediate Identification**: Detect issues as they occur or before they impact users
- **Continuous Monitoring**: 24/7 monitoring of system health and performance
- **Threshold-based Alerts**: Set appropriate thresholds for different metrics
- **Anomaly Detection**: Use machine learning to detect unusual patterns
- **Benefits**: Faster incident response, reduced mean time to detection (MTTD)
- **Use Cases**: System failures, performance degradation, security threats, business anomalies

**2. Multi-channel Notifications:**

- **Email**: Detailed notifications with context and runbooks
- **SMS**: Critical alerts for immediate attention
- **Slack/Teams**: Team collaboration and real-time communication
- **PagerDuty**: On-call management and escalation
- **Webhooks**: Custom integrations with internal systems
- **Benefits**: Ensure alerts reach the right people through their preferred channels
- **Use Cases**: Different severity levels, team preferences, escalation scenarios

**3. Escalation Policies:**

- **Automatic Escalation**: Escalate alerts based on time and severity
- **Team-based Routing**: Route alerts to appropriate teams based on service or component
- **On-call Management**: Integrate with on-call schedules and rotations
- **Escalation Chains**: Define clear escalation paths for different scenarios
- **Benefits**: Ensure critical issues receive appropriate attention
- **Use Cases**: Critical incidents, team availability, expertise requirements

**4. Alert Grouping:**

- **Related Alerts**: Group related alerts to reduce noise
- **Time-based Grouping**: Group alerts that occur within a time window
- **Service-based Grouping**: Group alerts by service or component
- **Deduplication**: Prevent duplicate alerts for the same issue
- **Benefits**: Reduce alert fatigue, improve signal-to-noise ratio
- **Use Cases**: Cascading failures, related system issues, maintenance windows

**5. Suppression Rules:**

- **Inhibition Rules**: Suppress lower-priority alerts when higher-priority ones are active
- **Time-based Suppression**: Suppress alerts during maintenance windows
- **Condition-based Suppression**: Suppress alerts based on specific conditions
- **Manual Suppression**: Allow manual suppression of alerts
- **Benefits**: Reduce alert noise, focus on actionable alerts
- **Use Cases**: Maintenance windows, known issues, planned changes

**6. Runbook Integration:**

- **Automated Response**: Trigger automated remediation procedures
- **Contextual Information**: Include relevant runbooks and documentation
- **Step-by-step Guidance**: Provide clear instructions for incident response
- **Integration**: Connect with ticketing systems and incident management tools
- **Benefits**: Faster incident resolution, consistent response procedures
- **Use Cases**: Common issues, standard procedures, automated remediation

**Advanced Alerting Concepts:**

- **Alert Correlation**: Identify relationships between different alerts
- **Machine Learning**: Use ML for anomaly detection and alert optimization
- **Alert Fatigue Management**: Implement strategies to reduce alert noise
- **SLO-based Alerting**: Alert based on service level objectives
- **Error Budget Alerting**: Alert when error budgets are at risk
- **Business Impact Alerting**: Alert based on business metrics and KPIs
- **Predictive Alerting**: Use predictive analytics to alert before issues occur

**Discussion Questions & Answers:**

**Q1: How do you design a comprehensive alerting strategy for a large-scale, multi-service application with complex dependencies and strict SLA requirements?**

**Answer:** Comprehensive alerting strategy design:

- **Service Mapping**: Map all services, dependencies, and critical paths
- **SLO Definition**: Define service level objectives for each service and component
- **Alert Hierarchy**: Create alert hierarchy based on business impact and severity
- **Multi-layer Monitoring**: Implement infrastructure, application, and business layer monitoring
- **Escalation Policies**: Define clear escalation paths for different teams and scenarios
- **Alert Grouping**: Group related alerts to reduce noise and improve signal
- **Runbook Integration**: Create comprehensive runbooks for common scenarios
- **Testing**: Implement alert testing and validation procedures
- **Documentation**: Maintain detailed documentation for alert rules and procedures
- **Continuous Improvement**: Regular review and optimization of alert rules
- **Training**: Provide training for teams on alert response procedures
- **Metrics**: Track alert effectiveness and response times

**Q2: What are the key considerations when implementing alerting for a production environment with strict compliance requirements and multiple teams?**

**Answer:** Production alerting implementation considerations:

- **Compliance Requirements**: Ensure alerting meets regulatory and compliance requirements
- **Team Structure**: Design alerting based on team responsibilities and expertise
- **On-call Management**: Implement proper on-call schedules and rotations
- **Escalation Procedures**: Define clear escalation procedures for different scenarios
- **Documentation**: Maintain comprehensive documentation for all alert rules and procedures
- **Testing**: Implement regular testing of alert rules and notification channels
- **Monitoring**: Monitor the alerting system itself for reliability and performance
- **Security**: Implement proper security controls for alerting systems
- **Audit Trail**: Maintain audit trails for all alerting activities
- **Incident Response**: Integrate alerting with incident response procedures
- **Training**: Provide training for teams on alert response and procedures
- **Regular Reviews**: Conduct regular reviews of alert effectiveness and optimization

**Q3: How do you optimize alerting for effectiveness, reliability, and team productivity in enterprise environments?**

**Answer:** Enterprise alerting optimization strategies:

- **Alert Tuning**: Continuously tune alert thresholds and rules based on historical data
- **Noise Reduction**: Implement strategies to reduce false positives and alert fatigue
- **Context Enhancement**: Provide rich context and actionable information with alerts
- **Automation**: Implement automated response and remediation where possible
- **Performance Monitoring**: Monitor alerting system performance and reliability
- **Team Feedback**: Collect and incorporate team feedback on alert effectiveness
- **Metrics and Analytics**: Use metrics and analytics to optimize alerting performance
- **Documentation**: Maintain comprehensive documentation and runbooks
- **Training**: Provide ongoing training for teams on alert response procedures
- **Regular Reviews**: Conduct regular reviews of alert rules and effectiveness
- **Incident Learning**: Learn from incidents to improve alerting and response procedures
- **Continuous Improvement**: Implement continuous improvement processes for alerting

## 🏗️ Alerting Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Alerting Pipeline                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Metrics   │  │   Logs      │  │   Traces    │     │
│  │  (Prometheus)│  │ (Elasticsearch)│ (Jaeger)   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Alert Rules Engine                   │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Prometheus│  │   Grafana   │  │   Custom    │ │ │
│  │  │   Rules     │  │   Alerts    │  │   Rules     │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Alert Manager                         │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Grouping  │  │   Routing   │  │   Inhibition│ │ │
│  │  │   Engine    │  │   Engine    │  │   Engine    │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Email     │  │   Slack     │  │  PagerDuty  │     │
│  │  Notifier   │  │  Notifier   │  │  Notifier   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Hands-on Example

### Prometheus Alert Rules

```yaml
# alerts.yml
groups:
  - name: application.rules
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
          team: backend
          service: api
        annotations:
          summary: "High error rate detected in {{ $labels.service }}"
          description: "Error rate is {{ $value | humanize }} errors per second for {{ $labels.service }}"
          runbook_url: "https://runbooks.example.com/high-error-rate"
          dashboard_url: "https://grafana.example.com/d/api-overview"

      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
          team: backend
          service: api
        annotations:
          summary: "High response time detected in {{ $labels.service }}"
          description: "95th percentile response time is {{ $value | humanize }} seconds for {{ $labels.service }}"
          runbook_url: "https://runbooks.example.com/high-response-time"

      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
          team: infrastructure
        annotations:
          summary: "Service {{ $labels.instance }} is down"
          description: "Service {{ $labels.instance }} has been down for more than 1 minute"
          runbook_url: "https://runbooks.example.com/service-down"

      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
        for: 5m
        labels:
          severity: warning
          team: infrastructure
        annotations:
          summary: "High memory usage on {{ $labels.instance }}"
          description: "Memory usage is {{ $value | humanizePercentage }} on {{ $labels.instance }}"
          runbook_url: "https://runbooks.example.com/high-memory-usage"

      - alert: HighCPUUsage
        expr: 100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 5m
        labels:
          severity: warning
          team: infrastructure
        annotations:
          summary: "High CPU usage on {{ $labels.instance }}"
          description: "CPU usage is {{ $value | humanize }}% on {{ $labels.instance }}"
          runbook_url: "https://runbooks.example.com/high-cpu-usage"

      - alert: DiskSpaceLow
        expr: (node_filesystem_avail_bytes / node_filesystem_size_bytes) < 0.1
        for: 5m
        labels:
          severity: critical
          team: infrastructure
        annotations:
          summary: "Disk space low on {{ $labels.instance }}"
          description: "Disk space is {{ $value | humanizePercentage }} on {{ $labels.instance }}"
          runbook_url: "https://runbooks.example.com/disk-space-low"

      - alert: DatabaseConnectionsHigh
        expr: database_connections_active > 80
        for: 5m
        labels:
          severity: warning
          team: database
        annotations:
          summary: "High database connections"
          description: "Database connections are {{ $value }} (threshold: 80)"
          runbook_url: "https://runbooks.example.com/high-db-connections"

      - alert: BusinessOperationFailure
        expr: rate(business_operations_total{status="error"}[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
          team: business
        annotations:
          summary: "Business operation failure rate high"
          description: "Failure rate is {{ $value | humanize }} failures per second"
          runbook_url: "https://runbooks.example.com/business-operation-failure"

      - alert: QueueDepthHigh
        expr: queue_depth > 1000
        for: 5m
        labels:
          severity: warning
          team: backend
        annotations:
          summary: "Queue depth is high"
          description: "Queue depth is {{ $value }} (threshold: 1000)"
          runbook_url: "https://runbooks.example.com/queue-depth-high"

      - alert: CertificateExpiring
        expr: (ssl_certificate_expiry_timestamp - time()) / 86400 < 30
        for: 1h
        labels:
          severity: warning
          team: security
        annotations:
          summary: "SSL certificate expiring soon"
          description: "SSL certificate for {{ $labels.instance }} expires in {{ $value | humanize }} days"
          runbook_url: "https://runbooks.example.com/certificate-expiring"

  - name: business.rules
    rules:
      - alert: RevenueDrop
        expr: rate(revenue_total[1h]) < 0.8 * rate(revenue_total[1h] offset 1d)
        for: 30m
        labels:
          severity: critical
          team: business
        annotations:
          summary: "Revenue drop detected"
          description: "Revenue is {{ $value | humanizePercentage }} of yesterday's revenue"
          runbook_url: "https://runbooks.example.com/revenue-drop"

      - alert: UserRegistrationDrop
        expr: rate(user_registrations_total[1h]) < 0.5 * rate(user_registrations_total[1h] offset 1d)
        for: 1h
        labels:
          severity: warning
          team: business
        annotations:
          summary: "User registration drop detected"
          description: "User registrations are {{ $value | humanizePercentage }} of yesterday's registrations"
          runbook_url: "https://runbooks.example.com/user-registration-drop"
```

### Alertmanager Configuration

```yaml
# alertmanager.yml
global:
  smtp_smarthost: "smtp.gmail.com:587"
  smtp_from: "alerts@example.com"
  smtp_auth_username: "alerts@example.com"
  smtp_auth_password: "password"
  smtp_require_tls: true

  slack_api_url: "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
  pagerduty_url: "https://events.pagerduty.com/v2/enqueue"

route:
  group_by: ["alertname", "cluster", "service"]
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: "default"
  routes:
    - match:
        severity: critical
      receiver: "critical-alerts"
      group_wait: 5s
      repeat_interval: 30m
      routes:
        - match:
            team: infrastructure
          receiver: "infrastructure-critical"
        - match:
            team: database
          receiver: "database-critical"
        - match:
            team: business
          receiver: "business-critical"

    - match:
        severity: warning
      receiver: "warning-alerts"
      group_wait: 30s
      repeat_interval: 2h

    - match:
        team: security
      receiver: "security-alerts"
      group_wait: 5s
      repeat_interval: 1h

inhibit_rules:
  - source_match:
      severity: "critical"
    target_match:
      severity: "warning"
    equal: ["alertname", "cluster", "service"]

  - source_match:
      alertname: "ServiceDown"
    target_match:
      alertname: "HighResponseTime"
    equal: ["instance"]

receivers:
  - name: "default"
    webhook_configs:
      - url: "http://webhook:5001/"
        send_resolved: true

  - name: "critical-alerts"
    email_configs:
      - to: "critical@example.com"
        subject: "🚨 CRITICAL: {{ .GroupLabels.alertname }}"
        body: |
          {{ range .Alerts }}
          **Alert:** {{ .Annotations.summary }}
          **Description:** {{ .Annotations.description }}
          **Severity:** {{ .Labels.severity }}
          **Team:** {{ .Labels.team }}
          **Service:** {{ .Labels.service }}
          **Instance:** {{ .Labels.instance }}
          **Runbook:** {{ .Annotations.runbook_url }}
          **Dashboard:** {{ .Annotations.dashboard_url }}
          **Time:** {{ .StartsAt }}
          {{ end }}
        headers:
          X-Priority: "1"
          X-MSMail-Priority: "High"

    slack_configs:
      - api_url: "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
        channel: "#critical-alerts"
        title: "🚨 Critical Alert"
        text: |
          {{ range .Alerts }}
          *Alert:* {{ .Annotations.summary }}
          *Description:* {{ .Annotations.description }}
          *Severity:* {{ .Labels.severity }}
          *Team:* {{ .Labels.team }}
          *Service:* {{ .Labels.service }}
          *Instance:* {{ .Labels.instance }}
          *Runbook:* {{ .Annotations.runbook_url }}
          *Dashboard:* {{ .Annotations.dashboard_url }}
          *Time:* {{ .StartsAt }}
          {{ end }}
        color: "danger"

    pagerduty_configs:
      - routing_key: "YOUR_PAGERDUTY_ROUTING_KEY"
        description: "{{ .GroupLabels.alertname }}"
        details:
          summary: "{{ .Annotations.summary }}"
          description: "{{ .Annotations.description }}"
          severity: "{{ .Labels.severity }}"
          team: "{{ .Labels.team }}"
          service: "{{ .Labels.service }}"
          instance: "{{ .Labels.instance }}"
          runbook_url: "{{ .Annotations.runbook_url }}"
          dashboard_url: "{{ .Annotations.dashboard_url }}"

  - name: "warning-alerts"
    email_configs:
      - to: "warnings@example.com"
        subject: "⚠️ WARNING: {{ .GroupLabels.alertname }}"
        body: |
          {{ range .Alerts }}
          **Alert:** {{ .Annotations.summary }}
          **Description:** {{ .Annotations.description }}
          **Severity:** {{ .Labels.severity }}
          **Team:** {{ .Labels.team }}
          **Service:** {{ .Labels.service }}
          **Instance:** {{ .Labels.instance }}
          **Runbook:** {{ .Annotations.runbook_url }}
          **Dashboard:** {{ .Annotations.dashboard_url }}
          **Time:** {{ .StartsAt }}
          {{ end }}

    slack_configs:
      - api_url: "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
        channel: "#warnings"
        title: "⚠️ Warning Alert"
        text: |
          {{ range .Alerts }}
          *Alert:* {{ .Annotations.summary }}
          *Description:* {{ .Annotations.description }}
          *Severity:* {{ .Labels.severity }}
          *Team:* {{ .Labels.team }}
          *Service:* {{ .Labels.service }}
          *Instance:* {{ .Labels.instance }}
          *Runbook:* {{ .Annotations.runbook_url }}
          *Dashboard:* {{ .Annotations.dashboard_url }}
          *Time:* {{ .StartsAt }}
          {{ end }}
        color: "warning"

  - name: "infrastructure-critical"
    email_configs:
      - to: "infrastructure@example.com"
        subject: "🚨 INFRASTRUCTURE CRITICAL: {{ .GroupLabels.alertname }}"
        body: |
          {{ range .Alerts }}
          **Alert:** {{ .Annotations.summary }}
          **Description:** {{ .Annotations.description }}
          **Instance:** {{ .Labels.instance }}
          **Runbook:** {{ .Annotations.runbook_url }}
          **Dashboard:** {{ .Annotations.dashboard_url }}
          **Time:** {{ .StartsAt }}
          {{ end }}

    pagerduty_configs:
      - routing_key: "YOUR_INFRASTRUCTURE_PAGERDUTY_KEY"
        description: "Infrastructure Critical: {{ .GroupLabels.alertname }}"
        details:
          summary: "{{ .Annotations.summary }}"
          description: "{{ .Annotations.description }}"
          instance: "{{ .Labels.instance }}"
          runbook_url: "{{ .Annotations.runbook_url }}"

  - name: "database-critical"
    email_configs:
      - to: "database@example.com"
        subject: "🚨 DATABASE CRITICAL: {{ .GroupLabels.alertname }}"
        body: |
          {{ range .Alerts }}
          **Alert:** {{ .Annotations.summary }}
          **Description:** {{ .Annotations.description }}
          **Instance:** {{ .Labels.instance }}
          **Runbook:** {{ .Annotations.runbook_url }}
          **Dashboard:** {{ .Annotations.dashboard_url }}
          **Time:** {{ .StartsAt }}
          {{ end }}

    pagerduty_configs:
      - routing_key: "YOUR_DATABASE_PAGERDUTY_KEY"
        description: "Database Critical: {{ .GroupLabels.alertname }}"
        details:
          summary: "{{ .Annotations.summary }}"
          description: "{{ .Annotations.description }}"
          instance: "{{ .Labels.instance }}"
          runbook_url: "{{ .Annotations.runbook_url }}"

  - name: "business-critical"
    email_configs:
      - to: "business@example.com"
        subject: "🚨 BUSINESS CRITICAL: {{ .GroupLabels.alertname }}"
        body: |
          {{ range .Alerts }}
          **Alert:** {{ .Annotations.summary }}
          **Description:** {{ .Annotations.description }}
          **Runbook:** {{ .Annotations.runbook_url }}
          **Dashboard:** {{ .Annotations.dashboard_url }}
          **Time:** {{ .StartsAt }}
          {{ end }}

    pagerduty_configs:
      - routing_key: "YOUR_BUSINESS_PAGERDUTY_KEY"
        description: "Business Critical: {{ .GroupLabels.alertname }}"
        details:
          summary: "{{ .Annotations.summary }}"
          description: "{{ .Annotations.description }}"
          runbook_url: "{{ .Annotations.runbook_url }}"

  - name: "security-alerts"
    email_configs:
      - to: "security@example.com"
        subject: "🔒 SECURITY: {{ .GroupLabels.alertname }}"
        body: |
          {{ range .Alerts }}
          **Alert:** {{ .Annotations.summary }}
          **Description:** {{ .Annotations.description }}
          **Instance:** {{ .Labels.instance }}
          **Runbook:** {{ .Annotations.runbook_url }}
          **Dashboard:** {{ .Annotations.dashboard_url }}
          **Time:** {{ .StartsAt }}
          {{ end }}

    slack_configs:
      - api_url: "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
        channel: "#security-alerts"
        title: "🔒 Security Alert"
        text: |
          {{ range .Alerts }}
          *Alert:* {{ .Annotations.summary }}
          *Description:* {{ .Annotations.description }}
          *Instance:* {{ .Labels.instance }}
          *Runbook:* {{ .Annotations.runbook_url }}
          *Dashboard:* {{ .Annotations.dashboard_url }}
          *Time:* {{ .StartsAt }}
          {{ end }}
        color: "danger"
```

### Grafana Alerting Configuration

```json
{
  "alerting": {
    "enabled": true,
    "execute_alerts": true,
    "error_or_timeout": "alerting",
    "nodata_or_nullvalues": "alerting",
    "concurrent_query_limit": 10,
    "evaluation_timeout_seconds": 30,
    "max_attempts": 3,
    "min_interval_seconds": 10,
    "max_annotation_age": "24h",
    "max_annotations_to_keep": 0
  },
  "unified_alerting": {
    "enabled": true,
    "min_interval": "10s",
    "max_attempts": 3,
    "evaluation_timeout": "30s",
    "rule_group_max_rules": 1000,
    "rule_group_max_evaluation_duration": "30s",
    "admin_config_poll_interval": "10s",
    "alertmanager_config_poll_interval": "10s",
    "notification_logs": {
      "enabled": true,
      "max_size": "1GB",
      "max_age": "24h",
      "max_silence_age": "5d"
    }
  }
}
```

### Custom Alert Webhook

```go
// webhook.go
package main

import (
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "os"
    "time"

    "github.com/gin-gonic/gin"
    "github.com/slack-go/slack"
)

type Alert struct {
    Status      string            `json:"status"`
    Labels      map[string]string `json:"labels"`
    Annotations map[string]string `json:"annotations"`
    StartsAt    time.Time         `json:"startsAt"`
    EndsAt      time.Time         `json:"endsAt"`
    GeneratorURL string           `json:"generatorURL"`
}

type AlertGroup struct {
    GroupKey       string            `json:"groupKey"`
    Status         string            `json:"status"`
    GroupLabels    map[string]string `json:"groupLabels"`
    CommonLabels   map[string]string `json:"commonLabels"`
    CommonAnnotations map[string]string `json:"commonAnnotations"`
    ExternalURL    string            `json:"externalURL"`
    Alerts         []Alert           `json:"alerts"`
}

type AlertManager struct {
    slackClient *slack.Client
    teamsClient *TeamsClient
}

func NewAlertManager() *AlertManager {
    return &AlertManager{
        slackClient: slack.New(os.Getenv("SLACK_TOKEN")),
        teamsClient: NewTeamsClient(os.Getenv("TEAMS_WEBHOOK_URL")),
    }
}

func (am *AlertManager) HandleAlert(c *gin.Context) {
    var alertGroup AlertGroup
    if err := c.ShouldBindJSON(&alertGroup); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
        return
    }

    // Process alerts
    for _, alert := range alertGroup.Alerts {
        am.processAlert(alert, alertGroup)
    }

    c.JSON(http.StatusOK, gin.H{"status": "processed"})
}

func (am *AlertManager) processAlert(alert Alert, group AlertGroup) {
    // Determine severity and routing
    severity := alert.Labels["severity"]
    team := alert.Labels["team"]
    service := alert.Labels["service"]

    // Create alert message
    message := am.createAlertMessage(alert, group)

    // Route to appropriate channels
    switch severity {
    case "critical":
        am.sendCriticalAlert(message, team, service)
    case "warning":
        am.sendWarningAlert(message, team, service)
    default:
        am.sendInfoAlert(message, team, service)
    }

    // Log alert
    am.logAlert(alert, group)
}

func (am *AlertManager) createAlertMessage(alert Alert, group AlertGroup) string {
    status := "🚨"
    if alert.Status == "resolved" {
        status = "✅"
    }

    message := fmt.Sprintf("%s **%s**\n", status, alert.Annotations["summary"])
    message += fmt.Sprintf("**Description:** %s\n", alert.Annotations["description"])
    message += fmt.Sprintf("**Severity:** %s\n", alert.Labels["severity"])
    message += fmt.Sprintf("**Team:** %s\n", alert.Labels["team"])
    message += fmt.Sprintf("**Service:** %s\n", alert.Labels["service"])
    message += fmt.Sprintf("**Instance:** %s\n", alert.Labels["instance"])
    message += fmt.Sprintf("**Time:** %s\n", alert.StartsAt.Format(time.RFC3339))

    if runbookURL := alert.Annotations["runbook_url"]; runbookURL != "" {
        message += fmt.Sprintf("**Runbook:** %s\n", runbookURL)
    }

    if dashboardURL := alert.Annotations["dashboard_url"]; dashboardURL != "" {
        message += fmt.Sprintf("**Dashboard:** %s\n", dashboardURL)
    }

    return message
}

func (am *AlertManager) sendCriticalAlert(message, team, service string) {
    // Send to Slack
    am.sendSlackMessage(message, "#critical-alerts", "danger")

    // Send to Teams
    am.sendTeamsMessage(message, "critical")

    // Send to PagerDuty
    am.sendPagerDutyAlert(message, team, service, "critical")

    // Send to email
    am.sendEmailAlert(message, "critical@example.com", "CRITICAL ALERT")
}

func (am *AlertManager) sendWarningAlert(message, team, service string) {
    // Send to Slack
    am.sendSlackMessage(message, "#warnings", "warning")

    // Send to Teams
    am.sendTeamsMessage(message, "warning")

    // Send to email
    am.sendEmailAlert(message, "warnings@example.com", "WARNING ALERT")
}

func (am *AlertManager) sendInfoAlert(message, team, service string) {
    // Send to Slack
    am.sendSlackMessage(message, "#info-alerts", "good")

    // Send to Teams
    am.sendTeamsMessage(message, "info")
}

func (am *AlertManager) sendSlackMessage(message, channel, color string) {
    attachment := slack.Attachment{
        Color:      color,
        Text:       message,
        Timestamp:  time.Now().Unix(),
        Footer:     "Alert Manager",
        FooterIcon: "https://platform.slack-edge.com/img/default_application_icon.png",
    }

    _, _, err := am.slackClient.PostMessage(
        channel,
        slack.MsgOptionAttachments(attachment),
    )
    if err != nil {
        log.Printf("Failed to send Slack message: %v", err)
    }
}

func (am *AlertManager) sendTeamsMessage(message, severity string) {
    // Teams webhook implementation
    teamsMessage := map[string]interface{}{
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "themeColor": am.getTeamsColor(severity),
        "summary": "Alert Notification",
        "sections": []map[string]interface{}{
            {
                "activityTitle": "Alert Notification",
                "activitySubtitle": message,
                "markdown": true,
            },
        },
    }

    am.teamsClient.SendMessage(teamsMessage)
}

func (am *AlertManager) getTeamsColor(severity string) string {
    switch severity {
    case "critical":
        return "FF0000"
    case "warning":
        return "FFA500"
    case "info":
        return "00FF00"
    default:
        return "808080"
    }
}

func (am *AlertManager) sendPagerDutyAlert(message, team, service, severity string) {
    // PagerDuty integration
    pagerDutyEvent := map[string]interface{}{
        "routing_key": am.getPagerDutyRoutingKey(team),
        "event_action": "trigger",
        "dedup_key": fmt.Sprintf("%s-%s-%s", team, service, severity),
        "payload": map[string]interface{}{
            "summary": message,
            "severity": severity,
            "source": service,
            "component": team,
            "group": team,
            "class": "alert",
        },
    }

    // Send to PagerDuty
    am.sendPagerDutyEvent(pagerDutyEvent)
}

func (am *AlertManager) getPagerDutyRoutingKey(team string) string {
    routingKeys := map[string]string{
        "infrastructure": os.Getenv("PAGERDUTY_INFRASTRUCTURE_KEY"),
        "database":       os.Getenv("PAGERDUTY_DATABASE_KEY"),
        "business":       os.Getenv("PAGERDUTY_BUSINESS_KEY"),
        "security":       os.Getenv("PAGERDUTY_SECURITY_KEY"),
    }
    return routingKeys[team]
}

func (am *AlertManager) sendEmailAlert(message, to, subject string) {
    // Email integration
    emailData := map[string]interface{}{
        "to":      to,
        "subject": subject,
        "body":    message,
        "html":    true,
    }

    // Send email
    am.sendEmail(emailData)
}

func (am *AlertManager) logAlert(alert Alert, group AlertGroup) {
    logData := map[string]interface{}{
        "alert":      alert,
        "group":      group,
        "timestamp":  time.Now(),
        "processed":  true,
    }

    // Log to structured logging system
    log.Printf("Alert processed: %+v", logData)
}

func main() {
    router := gin.Default()

    alertManager := NewAlertManager()

    router.POST("/alerts", alertManager.HandleAlert)

    port := os.Getenv("PORT")
    if port == "" {
        port = "5001"
    }

    router.Run(":" + port)
}
```

### Docker Compose for Alerting Stack

```yaml
# docker-compose.yml
version: "3.8"

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - ./alerts.yml:/etc/prometheus/alerts.yml
      - prometheus_data:/prometheus
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.path=/prometheus"
      - "--web.console.libraries=/etc/prometheus/console_libraries"
      - "--web.console.templates=/etc/prometheus/consoles"
      - "--storage.tsdb.retention.time=200h"
      - "--web.enable-lifecycle"
      - "--web.enable-admin-api"

  alertmanager:
    image: prom/alertmanager:latest
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
      - alertmanager_data:/alertmanager
    command:
      - "--config.file=/etc/alertmanager/alertmanager.yml"
      - "--storage.path=/alertmanager"
      - "--web.external-url=http://localhost:9093"
      - "--web.enable-lifecycle"

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_ALERTING_ENABLED=true
      - GF_UNIFIED_ALERTING_ENABLED=true
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning

  webhook:
    build: .
    ports:
      - "5001:5001"
    environment:
      - SLACK_TOKEN=your-slack-token
      - TEAMS_WEBHOOK_URL=your-teams-webhook-url
      - PAGERDUTY_INFRASTRUCTURE_KEY=your-pagerduty-key
      - PAGERDUTY_DATABASE_KEY=your-pagerduty-key
      - PAGERDUTY_BUSINESS_KEY=your-pagerduty-key
      - PAGERDUTY_SECURITY_KEY=your-pagerduty-key

  my-app:
    build: .
    ports:
      - "8080:8080"
    environment:
      - LOG_LEVEL=info
      - ENVIRONMENT=production
    depends_on:
      - prometheus

volumes:
  prometheus_data:
  alertmanager_data:
  grafana_data:
```

## 🚀 Best Practices

### 1. Alert Rule Design

```yaml
# Use appropriate thresholds and time windows
- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
  for: 5m
  labels:
    severity: critical
```

### 2. Alert Grouping

```yaml
# Group related alerts
route:
  group_by: ["alertname", "cluster", "service"]
  group_wait: 10s
  group_interval: 10s
```

### 3. Escalation Policies

```yaml
# Implement escalation
routes:
  - match:
      severity: critical
    receiver: "critical-alerts"
    repeat_interval: 30m
```

## 🏢 Industry Insights

### Alerting Usage Patterns

- **Infrastructure Monitoring**: System health and performance
- **Application Monitoring**: Business logic and errors
- **Security Monitoring**: Threats and vulnerabilities
- **Business Monitoring**: KPIs and revenue metrics

### Enterprise Alerting Strategy

- **Multi-channel**: Email, SMS, Slack, PagerDuty
- **Escalation**: Automatic escalation policies
- **Suppression**: Reduce alert fatigue
- **Runbooks**: Automated response procedures

## 🎯 Interview Questions

### Basic Level

1. **What is alerting?**

   - Automatic issue detection
   - Notification systems
   - Escalation policies
   - Response procedures

2. **What are alert rules?**

   - Condition definitions
   - Threshold settings
   - Time windows
   - Severity levels

3. **What is Alertmanager?**
   - Alert routing
   - Grouping and inhibition
   - Notification channels
   - Escalation policies

### Intermediate Level

4. **How do you design alert rules?**

   ```yaml
   - alert: HighErrorRate
     expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
     for: 5m
     labels:
       severity: critical
   ```

5. **How do you handle alert grouping?**

   - Group by labels
   - Group wait time
   - Group interval
   - Repeat interval

6. **How do you implement escalation policies?**
   - Severity-based routing
   - Time-based escalation
   - Team-based routing
   - Channel selection

### Advanced Level

7. **How do you implement alert suppression?**

   - Inhibition rules
   - Time-based suppression
   - Condition-based suppression
   - Manual suppression

8. **How do you handle alert fatigue?**

   - Smart grouping
   - Appropriate thresholds
   - Runbook integration
   - Alert correlation

9. **How do you implement alert testing?**
   - Alert simulation
   - End-to-end testing
   - Channel validation
   - Escalation testing

---

**Next**: [Security](./Security/) - Secrets management, zero-trust architecture, secure APIs
