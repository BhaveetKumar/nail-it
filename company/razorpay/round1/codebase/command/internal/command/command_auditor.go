package command

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// CommandAuditorImpl implements CommandAuditor interface
type CommandAuditorImpl struct {
	auditLogs map[string]*AuditLog
	mu        sync.RWMutex
}

// NewCommandAuditor creates a new command auditor
func NewCommandAuditor() *CommandAuditorImpl {
	return &CommandAuditorImpl{
		auditLogs: make(map[string]*AuditLog),
	}
}

// Audit audits command execution
func (ca *CommandAuditorImpl) Audit(ctx context.Context, command Command, result *CommandResult) error {
	ca.mu.Lock()
	defer ca.mu.Unlock()

	auditLog := &AuditLog{
		LogID:       fmt.Sprintf("audit_%s_%d", command.GetCommandID(), time.Now().UnixNano()),
		CommandID:   command.GetCommandID(),
		CommandType: command.GetCommandType(),
		Action:      "execute",
		Status:      "success",
		Timestamp:   time.Now(),
		Duration:    result.Duration,
		Data:        result.Data,
		Metadata:    result.Metadata,
	}

	if !result.Success {
		auditLog.Status = "failure"
		auditLog.Error = result.Error
	}

	ca.auditLogs[auditLog.LogID] = auditLog
	return nil
}

// GetAuditLog returns audit log by ID
func (ca *CommandAuditorImpl) GetAuditLog(logID string) (*AuditLog, error) {
	ca.mu.RLock()
	defer ca.mu.RUnlock()

	auditLog, exists := ca.auditLogs[logID]
	if !exists {
		return nil, fmt.Errorf("audit log not found: %s", logID)
	}

	return auditLog, nil
}

// GetAuditLogs returns audit logs with pagination
func (ca *CommandAuditorImpl) GetAuditLogs(limit, offset int) ([]*AuditLog, error) {
	ca.mu.RLock()
	defer ca.mu.RUnlock()

	var logs []*AuditLog
	count := 0
	skipped := 0

	for _, log := range ca.auditLogs {
		if skipped < offset {
			skipped++
			continue
		}

		if count >= limit {
			break
		}

		logs = append(logs, log)
		count++
	}

	return logs, nil
}

// GetAuditLogsByType returns audit logs by command type
func (ca *CommandAuditorImpl) GetAuditLogsByType(commandType string, limit, offset int) ([]*AuditLog, error) {
	ca.mu.RLock()
	defer ca.mu.RUnlock()

	var logs []*AuditLog
	count := 0
	skipped := 0

	for _, log := range ca.auditLogs {
		if log.CommandType != commandType {
			continue
		}

		if skipped < offset {
			skipped++
			continue
		}

		if count >= limit {
			break
		}

		logs = append(logs, log)
		count++
	}

	return logs, nil
}

// GetAuditLogsByTimeRange returns audit logs by time range
func (ca *CommandAuditorImpl) GetAuditLogsByTimeRange(start, end time.Time) ([]*AuditLog, error) {
	ca.mu.RLock()
	defer ca.mu.RUnlock()

	var logs []*AuditLog

	for _, log := range ca.auditLogs {
		if log.Timestamp.After(start) && log.Timestamp.Before(end) {
			logs = append(logs, log)
		}
	}

	return logs, nil
}
