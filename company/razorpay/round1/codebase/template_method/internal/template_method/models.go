package template_method

import (
	"time"
)

// BaseTemplateMethod provides common functionality for all template methods
type BaseTemplateMethod struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Steps       []Step                 `json:"steps"`
	CurrentStep int                    `json:"current_step"`
	Status      string                 `json:"status"`
	Data        interface{}            `json:"data"`
	Result      interface{}            `json:"result"`
	StartTime   time.Time              `json:"start_time"`
	EndTime     time.Time              `json:"end_time"`
	Error       error                  `json:"error"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// GetName returns the template method name
func (btm *BaseTemplateMethod) GetName() string {
	return btm.Name
}

// GetDescription returns the template method description
func (btm *BaseTemplateMethod) GetDescription() string {
	return btm.Description
}

// GetSteps returns the template method steps
func (btm *BaseTemplateMethod) GetSteps() []Step {
	return btm.Steps
}

// GetCurrentStep returns the current step index
func (btm *BaseTemplateMethod) GetCurrentStep() int {
	return btm.CurrentStep
}

// GetStatus returns the template method status
func (btm *BaseTemplateMethod) GetStatus() string {
	return btm.Status
}

// GetData returns the template method data
func (btm *BaseTemplateMethod) GetData() interface{} {
	return btm.Data
}

// SetData sets the template method data
func (btm *BaseTemplateMethod) SetData(data interface{}) error {
	btm.Data = data
	return nil
}

// GetResult returns the template method result
func (btm *BaseTemplateMethod) GetResult() interface{} {
	return btm.Result
}

// SetResult sets the template method result
func (btm *BaseTemplateMethod) SetResult(result interface{}) error {
	btm.Result = result
	return nil
}

// GetStartTime returns the start time
func (btm *BaseTemplateMethod) GetStartTime() time.Time {
	return btm.StartTime
}

// GetEndTime returns the end time
func (btm *BaseTemplateMethod) GetEndTime() time.Time {
	return btm.EndTime
}

// GetDuration returns the duration
func (btm *BaseTemplateMethod) GetDuration() time.Duration {
	if btm.EndTime.IsZero() {
		return time.Since(btm.StartTime)
	}
	return btm.EndTime.Sub(btm.StartTime)
}

// IsCompleted returns whether the template method is completed
func (btm *BaseTemplateMethod) IsCompleted() bool {
	return btm.Status == "completed"
}

// IsFailed returns whether the template method failed
func (btm *BaseTemplateMethod) IsFailed() bool {
	return btm.Status == "failed"
}

// IsRunning returns whether the template method is running
func (btm *BaseTemplateMethod) IsRunning() bool {
	return btm.Status == "running"
}

// GetError returns the error
func (btm *BaseTemplateMethod) GetError() error {
	return btm.Error
}

// SetError sets the error
func (btm *BaseTemplateMethod) SetError(err error) {
	btm.Error = err
}

// GetMetadata returns the metadata
func (btm *BaseTemplateMethod) GetMetadata() map[string]interface{} {
	return btm.Metadata
}

// SetMetadata sets the metadata
func (btm *BaseTemplateMethod) SetMetadata(metadata map[string]interface{}) {
	btm.Metadata = metadata
}

// Execute executes the template method
func (btm *BaseTemplateMethod) Execute() error {
	btm.Status = "running"
	btm.StartTime = time.Now()
	btm.CurrentStep = 0

	for i, step := range btm.Steps {
		btm.CurrentStep = i
		
		if !step.CanExecute() {
			btm.Status = "failed"
			btm.Error = ErrStepCannotExecute
			btm.EndTime = time.Now()
			return btm.Error
		}

		if err := step.Execute(); err != nil {
			btm.Status = "failed"
			btm.Error = err
			btm.EndTime = time.Now()
			return err
		}
	}

	btm.Status = "completed"
	btm.EndTime = time.Now()
	return nil
}

// BaseStep provides common functionality for all steps
type BaseStep struct {
	Name         string                 `json:"name"`
	Description  string                 `json:"description"`
	Type         string                 `json:"type"`
	Status       string                 `json:"status"`
	Data         interface{}            `json:"data"`
	Result       interface{}            `json:"result"`
	StartTime    time.Time              `json:"start_time"`
	EndTime      time.Time              `json:"end_time"`
	Error        error                  `json:"error"`
	Dependencies []string               `json:"dependencies"`
	Metadata     map[string]interface{} `json:"metadata"`
}

// GetName returns the step name
func (bs *BaseStep) GetName() string {
	return bs.Name
}

// GetDescription returns the step description
func (bs *BaseStep) GetDescription() string {
	return bs.Description
}

// GetType returns the step type
func (bs *BaseStep) GetType() string {
	return bs.Type
}

// GetStatus returns the step status
func (bs *BaseStep) GetStatus() string {
	return bs.Status
}

// GetData returns the step data
func (bs *BaseStep) GetData() interface{} {
	return bs.Data
}

// SetData sets the step data
func (bs *BaseStep) SetData(data interface{}) error {
	bs.Data = data
	return nil
}

// GetResult returns the step result
func (bs *BaseStep) GetResult() interface{} {
	return bs.Result
}

// SetResult sets the step result
func (bs *BaseStep) SetResult(result interface{}) error {
	bs.Result = result
	return nil
}

// GetStartTime returns the start time
func (bs *BaseStep) GetStartTime() time.Time {
	return bs.StartTime
}

// GetEndTime returns the end time
func (bs *BaseStep) GetEndTime() time.Time {
	return bs.EndTime
}

// GetDuration returns the duration
func (bs *BaseStep) GetDuration() time.Duration {
	if bs.EndTime.IsZero() {
		return time.Since(bs.StartTime)
	}
	return bs.EndTime.Sub(bs.StartTime)
}

// IsCompleted returns whether the step is completed
func (bs *BaseStep) IsCompleted() bool {
	return bs.Status == "completed"
}

// IsFailed returns whether the step failed
func (bs *BaseStep) IsFailed() bool {
	return bs.Status == "failed"
}

// IsRunning returns whether the step is running
func (bs *BaseStep) IsRunning() bool {
	return bs.Status == "running"
}

// GetError returns the error
func (bs *BaseStep) GetError() error {
	return bs.Error
}

// SetError sets the error
func (bs *BaseStep) SetError(err error) {
	bs.Error = err
}

// GetDependencies returns the step dependencies
func (bs *BaseStep) GetDependencies() []string {
	return bs.Dependencies
}

// SetDependencies sets the step dependencies
func (bs *BaseStep) SetDependencies(dependencies []string) {
	bs.Dependencies = dependencies
}

// GetMetadata returns the metadata
func (bs *BaseStep) GetMetadata() map[string]interface{} {
	return bs.Metadata
}

// SetMetadata sets the metadata
func (bs *BaseStep) SetMetadata(metadata map[string]interface{}) {
	bs.Metadata = metadata
}

// CanExecute returns whether the step can be executed
func (bs *BaseStep) CanExecute() bool {
	return bs.Status == "pending" || bs.Status == "ready"
}

// Execute executes the step
func (bs *BaseStep) Execute() error {
	bs.Status = "running"
	bs.StartTime = time.Now()
	
	// Default implementation - override in specific steps
	bs.Status = "completed"
	bs.EndTime = time.Now()
	return nil
}

// Validate validates the step
func (bs *BaseStep) Validate() error {
	// Default implementation - override in specific steps
	return nil
}

// ConcreteTemplateMethod implements a concrete template method
type ConcreteTemplateMethod struct {
	BaseTemplateMethod
	TemplateType string `json:"template_type"`
}

// NewConcreteTemplateMethod creates a new concrete template method
func NewConcreteTemplateMethod(name, description, templateType string) *ConcreteTemplateMethod {
	return &ConcreteTemplateMethod{
		BaseTemplateMethod: BaseTemplateMethod{
			Name:        name,
			Description: description,
			Steps:       make([]Step, 0),
			CurrentStep: 0,
			Status:      "pending",
			Data:        make(map[string]interface{}),
			Result:      make(map[string]interface{}),
			StartTime:   time.Time{},
			EndTime:     time.Time{},
			Error:       nil,
			Metadata:    make(map[string]interface{}),
		},
		TemplateType: templateType,
	}
}

// AddStep adds a step to the template method
func (ctm *ConcreteTemplateMethod) AddStep(step Step) error {
	ctm.Steps = append(ctm.Steps, step)
	return nil
}

// RemoveStep removes a step from the template method
func (ctm *ConcreteTemplateMethod) RemoveStep(stepName string) error {
	for i, step := range ctm.Steps {
		if step.GetName() == stepName {
			ctm.Steps = append(ctm.Steps[:i], ctm.Steps[i+1:]...)
			return nil
		}
	}
	return ErrStepNotFound
}

// GetStep returns a step by name
func (ctm *ConcreteTemplateMethod) GetStep(stepName string) (Step, error) {
	for _, step := range ctm.Steps {
		if step.GetName() == stepName {
			return step, nil
		}
	}
	return nil, ErrStepNotFound
}

// GetStepByIndex returns a step by index
func (ctm *ConcreteTemplateMethod) GetStepByIndex(index int) (Step, error) {
	if index < 0 || index >= len(ctm.Steps) {
		return nil, ErrInvalidStepIndex
	}
	return ctm.Steps[index], nil
}

// GetStepCount returns the number of steps
func (ctm *ConcreteTemplateMethod) GetStepCount() int {
	return len(ctm.Steps)
}

// GetTemplateType returns the template type
func (ctm *ConcreteTemplateMethod) GetTemplateType() string {
	return ctm.TemplateType
}

// SetTemplateType sets the template type
func (ctm *ConcreteTemplateMethod) SetTemplateType(templateType string) {
	ctm.TemplateType = templateType
}

// ConcreteStep implements a concrete step
type ConcreteStep struct {
	BaseStep
	StepType string `json:"step_type"`
}

// NewConcreteStep creates a new concrete step
func NewConcreteStep(name, description, stepType string) *ConcreteStep {
	return &ConcreteStep{
		BaseStep: BaseStep{
			Name:         name,
			Description:  description,
			Type:         stepType,
			Status:       "pending",
			Data:         make(map[string]interface{}),
			Result:       make(map[string]interface{}),
			StartTime:    time.Time{},
			EndTime:      time.Time{},
			Error:        nil,
			Dependencies: make([]string, 0),
			Metadata:     make(map[string]interface{}),
		},
		StepType: stepType,
	}
}

// GetStepType returns the step type
func (cs *ConcreteStep) GetStepType() string {
	return cs.StepType
}

// SetStepType sets the step type
func (cs *ConcreteStep) SetStepType(stepType string) {
	cs.StepType = stepType
}

// Execute executes the step
func (cs *ConcreteStep) Execute() error {
	cs.Status = "running"
	cs.StartTime = time.Now()
	
	// Default implementation - override in specific steps
	cs.Status = "completed"
	cs.EndTime = time.Now()
	return nil
}

// Validate validates the step
func (cs *ConcreteStep) Validate() error {
	// Default implementation - override in specific steps
	return nil
}

// DocumentProcessingTemplateMethod implements document processing template method
type DocumentProcessingTemplateMethod struct {
	ConcreteTemplateMethod
	DocumentType string `json:"document_type"`
	ProcessingRules []string `json:"processing_rules"`
}

// NewDocumentProcessingTemplateMethod creates a new document processing template method
func NewDocumentProcessingTemplateMethod(name, description, documentType string) *DocumentProcessingTemplateMethod {
	return &DocumentProcessingTemplateMethod{
		ConcreteTemplateMethod: *NewConcreteTemplateMethod(name, description, "document_processing"),
		DocumentType: documentType,
		ProcessingRules: make([]string, 0),
	}
}

// AddProcessingRule adds a processing rule
func (dptm *DocumentProcessingTemplateMethod) AddProcessingRule(rule string) error {
	dptm.ProcessingRules = append(dptm.ProcessingRules, rule)
	return nil
}

// GetProcessingRules returns the processing rules
func (dptm *DocumentProcessingTemplateMethod) GetProcessingRules() []string {
	return dptm.ProcessingRules
}

// GetDocumentType returns the document type
func (dptm *DocumentProcessingTemplateMethod) GetDocumentType() string {
	return dptm.DocumentType
}

// SetDocumentType sets the document type
func (dptm *DocumentProcessingTemplateMethod) SetDocumentType(documentType string) {
	dptm.DocumentType = documentType
}

// DataValidationTemplateMethod implements data validation template method
type DataValidationTemplateMethod struct {
	ConcreteTemplateMethod
	ValidationRules []string `json:"validation_rules"`
	ValidationLevel string   `json:"validation_level"`
}

// NewDataValidationTemplateMethod creates a new data validation template method
func NewDataValidationTemplateMethod(name, description, validationLevel string) *DataValidationTemplateMethod {
	return &DataValidationTemplateMethod{
		ConcreteTemplateMethod: *NewConcreteTemplateMethod(name, description, "data_validation"),
		ValidationRules: make([]string, 0),
		ValidationLevel: validationLevel,
	}
}

// AddValidationRule adds a validation rule
func (dvtm *DataValidationTemplateMethod) AddValidationRule(rule string) error {
	dvtm.ValidationRules = append(dvtm.ValidationRules, rule)
	return nil
}

// GetValidationRules returns the validation rules
func (dvtm *DataValidationTemplateMethod) GetValidationRules() []string {
	return dvtm.ValidationRules
}

// GetValidationLevel returns the validation level
func (dvtm *DataValidationTemplateMethod) GetValidationLevel() string {
	return dvtm.ValidationLevel
}

// SetValidationLevel sets the validation level
func (dvtm *DataValidationTemplateMethod) SetValidationLevel(validationLevel string) {
	dvtm.ValidationLevel = validationLevel
}

// WorkflowTemplateMethod implements workflow template method
type WorkflowTemplateMethod struct {
	ConcreteTemplateMethod
	WorkflowType string                 `json:"workflow_type"`
	WorkflowData map[string]interface{} `json:"workflow_data"`
}

// NewWorkflowTemplateMethod creates a new workflow template method
func NewWorkflowTemplateMethod(name, description, workflowType string) *WorkflowTemplateMethod {
	return &WorkflowTemplateMethod{
		ConcreteTemplateMethod: *NewConcreteTemplateMethod(name, description, "workflow"),
		WorkflowType: workflowType,
		WorkflowData: make(map[string]interface{}),
	}
}

// GetWorkflowType returns the workflow type
func (wtm *WorkflowTemplateMethod) GetWorkflowType() string {
	return wtm.WorkflowType
}

// SetWorkflowType sets the workflow type
func (wtm *WorkflowTemplateMethod) SetWorkflowType(workflowType string) {
	wtm.WorkflowType = workflowType
}

// GetWorkflowData returns the workflow data
func (wtm *WorkflowTemplateMethod) GetWorkflowData() map[string]interface{} {
	return wtm.WorkflowData
}

// SetWorkflowData sets the workflow data
func (wtm *WorkflowTemplateMethod) SetWorkflowData(workflowData map[string]interface{}) {
	wtm.WorkflowData = workflowData
}

// APIRequestTemplateMethod implements API request template method
type APIRequestTemplateMethod struct {
	ConcreteTemplateMethod
	RequestType string                 `json:"request_type"`
	RequestData map[string]interface{} `json:"request_data"`
	ResponseData map[string]interface{} `json:"response_data"`
}

// NewAPIRequestTemplateMethod creates a new API request template method
func NewAPIRequestTemplateMethod(name, description, requestType string) *APIRequestTemplateMethod {
	return &APIRequestTemplateMethod{
		ConcreteTemplateMethod: *NewConcreteTemplateMethod(name, description, "api_request"),
		RequestType: requestType,
		RequestData: make(map[string]interface{}),
		ResponseData: make(map[string]interface{}),
	}
}

// GetRequestType returns the request type
func (artm *APIRequestTemplateMethod) GetRequestType() string {
	return artm.RequestType
}

// SetRequestType sets the request type
func (artm *APIRequestTemplateMethod) SetRequestType(requestType string) {
	artm.RequestType = requestType
}

// GetRequestData returns the request data
func (artm *APIRequestTemplateMethod) GetRequestData() map[string]interface{} {
	return artm.RequestData
}

// SetRequestData sets the request data
func (artm *APIRequestTemplateMethod) SetRequestData(requestData map[string]interface{}) {
	artm.RequestData = requestData
}

// GetResponseData returns the response data
func (artm *APIRequestTemplateMethod) GetResponseData() map[string]interface{} {
	return artm.ResponseData
}

// SetResponseData sets the response data
func (artm *APIRequestTemplateMethod) SetResponseData(responseData map[string]interface{}) {
	artm.ResponseData = responseData
}

// DatabaseOperationTemplateMethod implements database operation template method
type DatabaseOperationTemplateMethod struct {
	ConcreteTemplateMethod
	OperationType string                 `json:"operation_type"`
	TableName     string                 `json:"table_name"`
	QueryData     map[string]interface{} `json:"query_data"`
}

// NewDatabaseOperationTemplateMethod creates a new database operation template method
func NewDatabaseOperationTemplateMethod(name, description, operationType, tableName string) *DatabaseOperationTemplateMethod {
	return &DatabaseOperationTemplateMethod{
		ConcreteTemplateMethod: *NewConcreteTemplateMethod(name, description, "database_operation"),
		OperationType: operationType,
		TableName: tableName,
		QueryData: make(map[string]interface{}),
	}
}

// GetOperationType returns the operation type
func (dotm *DatabaseOperationTemplateMethod) GetOperationType() string {
	return dotm.OperationType
}

// SetOperationType sets the operation type
func (dotm *DatabaseOperationTemplateMethod) SetOperationType(operationType string) {
	dotm.OperationType = operationType
}

// GetTableName returns the table name
func (dotm *DatabaseOperationTemplateMethod) GetTableName() string {
	return dotm.TableName
}

// SetTableName sets the table name
func (dotm *DatabaseOperationTemplateMethod) SetTableName(tableName string) {
	dotm.TableName = tableName
}

// GetQueryData returns the query data
func (dotm *DatabaseOperationTemplateMethod) GetQueryData() map[string]interface{} {
	return dotm.QueryData
}

// SetQueryData sets the query data
func (dotm *DatabaseOperationTemplateMethod) SetQueryData(queryData map[string]interface{}) {
	dotm.QueryData = queryData
}

// Helper functions

// generateID generates a unique ID
func generateID() string {
	return time.Now().Format("20060102150405") + "-" + randomString(8)
}

// randomString generates a random string
func randomString(length int) string {
	const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	b := make([]byte, length)
	for i := range b {
		b[i] = charset[time.Now().UnixNano()%int64(len(charset))]
	}
	return string(b)
}
