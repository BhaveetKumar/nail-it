package mediator

import (
	"time"
)

// BaseColleague provides common functionality for all colleagues
type BaseColleague struct {
	ID           string    `json:"id"`
	Name         string    `json:"name"`
	Type         string    `json:"type"`
	Active       bool      `json:"active"`
	LastActivity time.Time `json:"last_activity"`
	Mediator     Mediator  `json:"-"`
}

// GetID returns the colleague ID
func (bc *BaseColleague) GetID() string {
	return bc.ID
}

// GetName returns the colleague name
func (bc *BaseColleague) GetName() string {
	return bc.Name
}

// GetType returns the colleague type
func (bc *BaseColleague) GetType() string {
	return bc.Type
}

// IsActive returns whether the colleague is active
func (bc *BaseColleague) IsActive() bool {
	return bc.Active
}

// SetActive sets the colleague active status
func (bc *BaseColleague) SetActive(active bool) {
	bc.Active = active
}

// GetLastActivity returns the last activity time
func (bc *BaseColleague) GetLastActivity() time.Time {
	return bc.LastActivity
}

// UpdateActivity updates the last activity time
func (bc *BaseColleague) UpdateActivity() {
	bc.LastActivity = time.Now()
}

// SetMediator sets the mediator
func (bc *BaseColleague) SetMediator(mediator Mediator) {
	bc.Mediator = mediator
}

// GetMediator returns the mediator
func (bc *BaseColleague) GetMediator() Mediator {
	return bc.Mediator
}

// SendMessage sends a message through the mediator
func (bc *BaseColleague) SendMessage(recipientID string, message interface{}) error {
	if bc.Mediator == nil {
		return ErrNoMediator
	}
	return bc.Mediator.SendMessage(bc.ID, recipientID, message)
}

// BroadcastMessage broadcasts a message through the mediator
func (bc *BaseColleague) BroadcastMessage(message interface{}) error {
	if bc.Mediator == nil {
		return ErrNoMediator
	}
	return bc.Mediator.BroadcastMessage(bc.ID, message)
}

// BaseMessage provides common functionality for all messages
type BaseMessage struct {
	ID         string      `json:"id"`
	Type       string      `json:"type"`
	Content    interface{} `json:"content"`
	SenderID   string      `json:"sender_id"`
	RecipientID string     `json:"recipient_id"`
	Timestamp  time.Time   `json:"timestamp"`
	Priority   int         `json:"priority"`
	Broadcast  bool        `json:"broadcast"`
}

// GetID returns the message ID
func (bm *BaseMessage) GetID() string {
	return bm.ID
}

// GetType returns the message type
func (bm *BaseMessage) GetType() string {
	return bm.Type
}

// GetContent returns the message content
func (bm *BaseMessage) GetContent() interface{} {
	return bm.Content
}

// GetSenderID returns the sender ID
func (bm *BaseMessage) GetSenderID() string {
	return bm.SenderID
}

// GetRecipientID returns the recipient ID
func (bm *BaseMessage) GetRecipientID() string {
	return bm.RecipientID
}

// GetTimestamp returns the timestamp
func (bm *BaseMessage) GetTimestamp() time.Time {
	return bm.Timestamp
}

// GetPriority returns the priority
func (bm *BaseMessage) GetPriority() int {
	return bm.Priority
}

// SetPriority sets the priority
func (bm *BaseMessage) SetPriority(priority int) {
	bm.Priority = priority
}

// IsBroadcast returns whether the message is a broadcast
func (bm *BaseMessage) IsBroadcast() bool {
	return bm.Broadcast
}

// SetBroadcast sets the broadcast flag
func (bm *BaseMessage) SetBroadcast(broadcast bool) {
	bm.Broadcast = broadcast
}

// BaseEvent provides common functionality for all events
type BaseEvent struct {
	ID        string      `json:"id"`
	Type      string      `json:"type"`
	Data      interface{} `json:"data"`
	Timestamp time.Time   `json:"timestamp"`
	Source    string      `json:"source"`
}

// GetID returns the event ID
func (be *BaseEvent) GetID() string {
	return be.ID
}

// GetType returns the event type
func (be *BaseEvent) GetType() string {
	return be.Type
}

// GetData returns the event data
func (be *BaseEvent) GetData() interface{} {
	return be.Data
}

// GetTimestamp returns the timestamp
func (be *BaseEvent) GetTimestamp() time.Time {
	return be.Timestamp
}

// GetSource returns the source
func (be *BaseEvent) GetSource() string {
	return be.Source
}

// BaseCommand provides common functionality for all commands
type BaseCommand struct {
	ID        string      `json:"id"`
	Type      string      `json:"type"`
	Data      interface{} `json:"data"`
	Timestamp time.Time   `json:"timestamp"`
	Source    string      `json:"source"`
}

// GetID returns the command ID
func (bc *BaseCommand) GetID() string {
	return bc.ID
}

// GetType returns the command type
func (bc *BaseCommand) GetType() string {
	return bc.Type
}

// GetData returns the command data
func (bc *BaseCommand) GetData() interface{} {
	return bc.Data
}

// GetTimestamp returns the timestamp
func (bc *BaseCommand) GetTimestamp() time.Time {
	return bc.Timestamp
}

// GetSource returns the source
func (bc *BaseCommand) GetSource() string {
	return bc.Source
}

// Execute executes the command
func (bc *BaseCommand) Execute() error {
	// Default implementation - override in specific commands
	return nil
}

// Undo undoes the command
func (bc *BaseCommand) Undo() error {
	// Default implementation - override in specific commands
	return nil
}

// CanUndo returns whether the command can be undone
func (bc *BaseCommand) CanUndo() bool {
	// Default implementation - override in specific commands
	return false
}

// BaseQuery provides common functionality for all queries
type BaseQuery struct {
	ID        string      `json:"id"`
	Type      string      `json:"type"`
	Data      interface{} `json:"data"`
	Timestamp time.Time   `json:"timestamp"`
	Source    string      `json:"source"`
}

// GetID returns the query ID
func (bq *BaseQuery) GetID() string {
	return bq.ID
}

// GetType returns the query type
func (bq *BaseQuery) GetType() string {
	return bq.Type
}

// GetData returns the query data
func (bq *BaseQuery) GetData() interface{} {
	return bq.Data
}

// GetTimestamp returns the timestamp
func (bq *BaseQuery) GetTimestamp() time.Time {
	return bq.Timestamp
}

// GetSource returns the source
func (bq *BaseQuery) GetSource() string {
	return bq.Source
}

// Execute executes the query
func (bq *BaseQuery) Execute() (interface{}, error) {
	// Default implementation - override in specific queries
	return nil, nil
}

// BaseNotification provides common functionality for all notifications
type BaseNotification struct {
	ID          string      `json:"id"`
	Type        string      `json:"type"`
	Content     interface{} `json:"content"`
	RecipientID string      `json:"recipient_id"`
	Timestamp   time.Time   `json:"timestamp"`
	Priority    int         `json:"priority"`
	Read        bool        `json:"read"`
	ReadAt      time.Time   `json:"read_at"`
}

// GetID returns the notification ID
func (bn *BaseNotification) GetID() string {
	return bn.ID
}

// GetType returns the notification type
func (bn *BaseNotification) GetType() string {
	return bn.Type
}

// GetContent returns the notification content
func (bn *BaseNotification) GetContent() interface{} {
	return bn.Content
}

// GetRecipientID returns the recipient ID
func (bn *BaseNotification) GetRecipientID() string {
	return bn.RecipientID
}

// GetTimestamp returns the timestamp
func (bn *BaseNotification) GetTimestamp() time.Time {
	return bn.Timestamp
}

// GetPriority returns the priority
func (bn *BaseNotification) GetPriority() int {
	return bn.Priority
}

// IsRead returns whether the notification is read
func (bn *BaseNotification) IsRead() bool {
	return bn.Read
}

// SetRead sets the read status
func (bn *BaseNotification) SetRead(read bool) {
	bn.Read = read
	if read {
		bn.ReadAt = time.Now()
	}
}

// GetReadAt returns the read timestamp
func (bn *BaseNotification) GetReadAt() time.Time {
	return bn.ReadAt
}

// SetReadAt sets the read timestamp
func (bn *BaseNotification) SetReadAt(readAt time.Time) {
	bn.ReadAt = readAt
}

// BaseWorkflow provides common functionality for all workflows
type BaseWorkflow struct {
	ID            string         `json:"id"`
	Name          string         `json:"name"`
	Steps         []WorkflowStep `json:"steps"`
	CurrentStep   int            `json:"current_step"`
	Status        string         `json:"status"`
	Data          interface{}    `json:"data"`
	Progress      float64        `json:"progress"`
	EstimatedTime time.Duration  `json:"estimated_time"`
	ActualTime    time.Duration  `json:"actual_time"`
	StartTime     time.Time      `json:"start_time"`
	EndTime       time.Time      `json:"end_time"`
}

// GetID returns the workflow ID
func (bw *BaseWorkflow) GetID() string {
	return bw.ID
}

// GetName returns the workflow name
func (bw *BaseWorkflow) GetName() string {
	return bw.Name
}

// GetSteps returns the workflow steps
func (bw *BaseWorkflow) GetSteps() []WorkflowStep {
	return bw.Steps
}

// GetCurrentStep returns the current step
func (bw *BaseWorkflow) GetCurrentStep() int {
	return bw.CurrentStep
}

// GetStatus returns the workflow status
func (bw *BaseWorkflow) GetStatus() string {
	return bw.Status
}

// GetData returns the workflow data
func (bw *BaseWorkflow) GetData() interface{} {
	return bw.Data
}

// Execute executes the workflow
func (bw *BaseWorkflow) Execute() error {
	// Default implementation - override in specific workflows
	return nil
}

// Pause pauses the workflow
func (bw *BaseWorkflow) Pause() error {
	// Default implementation - override in specific workflows
	return nil
}

// Resume resumes the workflow
func (bw *BaseWorkflow) Resume() error {
	// Default implementation - override in specific workflows
	return nil
}

// Cancel cancels the workflow
func (bw *BaseWorkflow) Cancel() error {
	// Default implementation - override in specific workflows
	return nil
}

// Complete completes the workflow
func (bw *BaseWorkflow) Complete() error {
	// Default implementation - override in specific workflows
	return nil
}

// GetProgress returns the workflow progress
func (bw *BaseWorkflow) GetProgress() float64 {
	return bw.Progress
}

// GetEstimatedTime returns the estimated time
func (bw *BaseWorkflow) GetEstimatedTime() time.Duration {
	return bw.EstimatedTime
}

// GetActualTime returns the actual time
func (bw *BaseWorkflow) GetActualTime() time.Duration {
	return bw.ActualTime
}

// BaseWorkflowStep provides common functionality for all workflow steps
type BaseWorkflowStep struct {
	ID            string        `json:"id"`
	Name          string        `json:"name"`
	Type          string        `json:"type"`
	Status        string        `json:"status"`
	Data          interface{}   `json:"data"`
	Dependencies  []string      `json:"dependencies"`
	EstimatedTime time.Duration `json:"estimated_time"`
	ActualTime    time.Duration `json:"actual_time"`
	StartTime     time.Time     `json:"start_time"`
	EndTime       time.Time     `json:"end_time"`
}

// GetID returns the step ID
func (bws *BaseWorkflowStep) GetID() string {
	return bws.ID
}

// GetName returns the step name
func (bws *BaseWorkflowStep) GetName() string {
	return bws.Name
}

// GetType returns the step type
func (bws *BaseWorkflowStep) GetType() string {
	return bws.Type
}

// GetStatus returns the step status
func (bws *BaseWorkflowStep) GetStatus() string {
	return bws.Status
}

// GetData returns the step data
func (bws *BaseWorkflowStep) GetData() interface{} {
	return bws.Data
}

// Execute executes the step
func (bws *BaseWorkflowStep) Execute() error {
	// Default implementation - override in specific steps
	return nil
}

// CanExecute returns whether the step can be executed
func (bws *BaseWorkflowStep) CanExecute() bool {
	// Default implementation - override in specific steps
	return true
}

// GetDependencies returns the step dependencies
func (bws *BaseWorkflowStep) GetDependencies() []string {
	return bws.Dependencies
}

// GetEstimatedTime returns the estimated time
func (bws *BaseWorkflowStep) GetEstimatedTime() time.Duration {
	return bws.EstimatedTime
}

// GetActualTime returns the actual time
func (bws *BaseWorkflowStep) GetActualTime() time.Duration {
	return bws.ActualTime
}

// BaseService provides common functionality for all services
type BaseService struct {
	ID           string                 `json:"id"`
	Name         string                 `json:"name"`
	Type         string                 `json:"type"`
	Status       string                 `json:"status"`
	Health       bool                   `json:"health"`
	Metrics      map[string]interface{} `json:"metrics"`
	Dependencies []string               `json:"dependencies"`
	Dependents   []string               `json:"dependents"`
	StartTime    time.Time              `json:"start_time"`
	StopTime     time.Time              `json:"stop_time"`
}

// GetID returns the service ID
func (bs *BaseService) GetID() string {
	return bs.ID
}

// GetName returns the service name
func (bs *BaseService) GetName() string {
	return bs.Name
}

// GetType returns the service type
func (bs *BaseService) GetType() string {
	return bs.Type
}

// GetStatus returns the service status
func (bs *BaseService) GetStatus() string {
	return bs.Status
}

// GetHealth returns the service health
func (bs *BaseService) GetHealth() bool {
	return bs.Health
}

// GetMetrics returns the service metrics
func (bs *BaseService) GetMetrics() map[string]interface{} {
	return bs.Metrics
}

// Start starts the service
func (bs *BaseService) Start() error {
	// Default implementation - override in specific services
	return nil
}

// Stop stops the service
func (bs *BaseService) Stop() error {
	// Default implementation - override in specific services
	return nil
}

// Restart restarts the service
func (bs *BaseService) Restart() error {
	// Default implementation - override in specific services
	return nil
}

// GetDependencies returns the service dependencies
func (bs *BaseService) GetDependencies() []string {
	return bs.Dependencies
}

// GetDependents returns the service dependents
func (bs *BaseService) GetDependents() []string {
	return bs.Dependents
}

// BaseResource provides common functionality for all resources
type BaseResource struct {
	ID           string  `json:"id"`
	Name         string  `json:"name"`
	Type         string  `json:"type"`
	Status       string  `json:"status"`
	Capacity     int     `json:"capacity"`
	Used         int     `json:"used"`
	Available    int     `json:"available"`
	Utilization  float64 `json:"utilization"`
	WaitTime     time.Duration `json:"wait_time"`
	LastAccess   time.Time     `json:"last_access"`
}

// GetID returns the resource ID
func (br *BaseResource) GetID() string {
	return br.ID
}

// GetName returns the resource name
func (br *BaseResource) GetName() string {
	return br.Name
}

// GetType returns the resource type
func (br *BaseResource) GetType() string {
	return br.Type
}

// GetStatus returns the resource status
func (br *BaseResource) GetStatus() string {
	return br.Status
}

// GetCapacity returns the resource capacity
func (br *BaseResource) GetCapacity() int {
	return br.Capacity
}

// GetUsed returns the used amount
func (br *BaseResource) GetUsed() int {
	return br.Used
}

// GetAvailable returns the available amount
func (br *BaseResource) GetAvailable() int {
	return br.Available
}

// GetUtilization returns the utilization percentage
func (br *BaseResource) GetUtilization() float64 {
	return br.Utilization
}

// Acquire acquires the resource
func (br *BaseResource) Acquire() error {
	// Default implementation - override in specific resources
	return nil
}

// Release releases the resource
func (br *BaseResource) Release() error {
	// Default implementation - override in specific resources
	return nil
}

// IsAvailable returns whether the resource is available
func (br *BaseResource) IsAvailable() bool {
	return br.Available > 0
}

// GetWaitTime returns the wait time
func (br *BaseResource) GetWaitTime() time.Duration {
	return br.WaitTime
}

// BaseTask provides common functionality for all tasks
type BaseTask struct {
	ID            string        `json:"id"`
	Name          string        `json:"name"`
	Type          string        `json:"type"`
	Status        string        `json:"status"`
	Priority      int           `json:"priority"`
	Data          interface{}   `json:"data"`
	Dependencies  []string      `json:"dependencies"`
	Dependents    []string      `json:"dependents"`
	EstimatedTime time.Duration `json:"estimated_time"`
	ActualTime    time.Duration `json:"actual_time"`
	Progress      float64       `json:"progress"`
	StartTime     time.Time     `json:"start_time"`
	EndTime       time.Time     `json:"end_time"`
}

// GetID returns the task ID
func (bt *BaseTask) GetID() string {
	return bt.ID
}

// GetName returns the task name
func (bt *BaseTask) GetName() string {
	return bt.Name
}

// GetType returns the task type
func (bt *BaseTask) GetType() string {
	return bt.Type
}

// GetStatus returns the task status
func (bt *BaseTask) GetStatus() string {
	return bt.Status
}

// GetPriority returns the task priority
func (bt *BaseTask) GetPriority() int {
	return bt.Priority
}

// GetData returns the task data
func (bt *BaseTask) GetData() interface{} {
	return bt.Data
}

// GetDependencies returns the task dependencies
func (bt *BaseTask) GetDependencies() []string {
	return bt.Dependencies
}

// GetDependents returns the task dependents
func (bt *BaseTask) GetDependents() []string {
	return bt.Dependents
}

// Execute executes the task
func (bt *BaseTask) Execute() error {
	// Default implementation - override in specific tasks
	return nil
}

// CanExecute returns whether the task can be executed
func (bt *BaseTask) CanExecute() bool {
	// Default implementation - override in specific tasks
	return true
}

// GetEstimatedTime returns the estimated time
func (bt *BaseTask) GetEstimatedTime() time.Duration {
	return bt.EstimatedTime
}

// GetActualTime returns the actual time
func (bt *BaseTask) GetActualTime() time.Duration {
	return bt.ActualTime
}

// GetProgress returns the task progress
func (bt *BaseTask) GetProgress() float64 {
	return bt.Progress
}

// BaseJob provides common functionality for all jobs
type BaseJob struct {
	ID            string        `json:"id"`
	Name          string        `json:"name"`
	Type          string        `json:"type"`
	Status        string        `json:"status"`
	Priority      int           `json:"priority"`
	Data          interface{}   `json:"data"`
	Schedule      string        `json:"schedule"`
	NextRun       time.Time     `json:"next_run"`
	LastRun       time.Time     `json:"last_run"`
	EstimatedTime time.Duration `json:"estimated_time"`
	ActualTime    time.Duration `json:"actual_time"`
	Progress      float64       `json:"progress"`
	StartTime     time.Time     `json:"start_time"`
	EndTime       time.Time     `json:"end_time"`
}

// GetID returns the job ID
func (bj *BaseJob) GetID() string {
	return bj.ID
}

// GetName returns the job name
func (bj *BaseJob) GetName() string {
	return bj.Name
}

// GetType returns the job type
func (bj *BaseJob) GetType() string {
	return bj.Type
}

// GetStatus returns the job status
func (bj *BaseJob) GetStatus() string {
	return bj.Status
}

// GetPriority returns the job priority
func (bj *BaseJob) GetPriority() int {
	return bj.Priority
}

// GetData returns the job data
func (bj *BaseJob) GetData() interface{} {
	return bj.Data
}

// GetSchedule returns the job schedule
func (bj *BaseJob) GetSchedule() string {
	return bj.Schedule
}

// GetNextRun returns the next run time
func (bj *BaseJob) GetNextRun() time.Time {
	return bj.NextRun
}

// GetLastRun returns the last run time
func (bj *BaseJob) GetLastRun() time.Time {
	return bj.LastRun
}

// Execute executes the job
func (bj *BaseJob) Execute() error {
	// Default implementation - override in specific jobs
	return nil
}

// CanExecute returns whether the job can be executed
func (bj *BaseJob) CanExecute() bool {
	// Default implementation - override in specific jobs
	return true
}

// GetEstimatedTime returns the estimated time
func (bj *BaseJob) GetEstimatedTime() time.Duration {
	return bj.EstimatedTime
}

// GetActualTime returns the actual time
func (bj *BaseJob) GetActualTime() time.Duration {
	return bj.ActualTime
}

// GetProgress returns the job progress
func (bj *BaseJob) GetProgress() float64 {
	return bj.Progress
}

// BaseAlert provides common functionality for all alerts
type BaseAlert struct {
	ID        string                 `json:"id"`
	Type      string                 `json:"type"`
	Severity  string                 `json:"severity"`
	Message   string                 `json:"message"`
	Timestamp time.Time              `json:"timestamp"`
	Status    string                 `json:"status"`
	Source    string                 `json:"source"`
	Data      map[string]interface{} `json:"data"`
	ResolvedAt time.Time             `json:"resolved_at"`
}

// GetID returns the alert ID
func (ba *BaseAlert) GetID() string {
	return ba.ID
}

// GetType returns the alert type
func (ba *BaseAlert) GetType() string {
	return ba.Type
}

// GetSeverity returns the alert severity
func (ba *BaseAlert) GetSeverity() string {
	return ba.Severity
}

// GetMessage returns the alert message
func (ba *BaseAlert) GetMessage() string {
	return ba.Message
}

// GetTimestamp returns the alert timestamp
func (ba *BaseAlert) GetTimestamp() time.Time {
	return ba.Timestamp
}

// GetStatus returns the alert status
func (ba *BaseAlert) GetStatus() string {
	return ba.Status
}

// GetSource returns the alert source
func (ba *BaseAlert) GetSource() string {
	return ba.Source
}

// GetData returns the alert data
func (ba *BaseAlert) GetData() map[string]interface{} {
	return ba.Data
}

// SetStatus sets the alert status
func (ba *BaseAlert) SetStatus(status string) {
	ba.Status = status
}

// Resolve resolves the alert
func (ba *BaseAlert) Resolve() error {
	ba.Status = "resolved"
	ba.ResolvedAt = time.Now()
	return nil
}

// IsActive returns whether the alert is active
func (ba *BaseAlert) IsActive() bool {
	return ba.Status == "active"
}

// IsResolved returns whether the alert is resolved
func (ba *BaseAlert) IsResolved() bool {
	return ba.Status == "resolved"
}
