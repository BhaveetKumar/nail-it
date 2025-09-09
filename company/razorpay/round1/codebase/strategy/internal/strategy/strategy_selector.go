package strategy

import (
	"context"
	"fmt"
	"sort"
)

// StrategySelectorImpl implements StrategySelector interface
type StrategySelectorImpl struct {
	paymentManager        StrategyManager
	notificationManager   StrategyManager
	pricingManager        StrategyManager
	authManager           StrategyManager
	cachingManager        StrategyManager
	loggingManager        StrategyManager
	dataProcessingManager StrategyManager
}

// NewStrategySelector creates a new strategy selector
func NewStrategySelector(
	paymentManager StrategyManager,
	notificationManager StrategyManager,
	pricingManager StrategyManager,
	authManager StrategyManager,
	cachingManager StrategyManager,
	loggingManager StrategyManager,
	dataProcessingManager StrategyManager,
) *StrategySelectorImpl {
	return &StrategySelectorImpl{
		paymentManager:        paymentManager,
		notificationManager:   notificationManager,
		pricingManager:        pricingManager,
		authManager:           authManager,
		cachingManager:        cachingManager,
		loggingManager:        loggingManager,
		dataProcessingManager: dataProcessingManager,
	}
}

// SelectPaymentStrategy selects the best payment strategy based on request
func (ss *StrategySelectorImpl) SelectPaymentStrategy(ctx context.Context, request PaymentRequest) (PaymentStrategy, error) {
	// Get available strategies
	strategies := ss.paymentManager.GetAvailableStrategies()

	// Filter strategies based on currency support
	var supportedStrategies []PaymentStrategy
	for _, strategyName := range strategies {
		strategy, err := ss.paymentManager.GetStrategy(strategyName)
		if err != nil {
			continue
		}

		paymentStrategy, ok := strategy.(PaymentStrategy)
		if !ok {
			continue
		}

		// Check if strategy supports the currency
		supportedCurrencies := paymentStrategy.GetSupportedCurrencies()
		for _, currency := range supportedCurrencies {
			if currency == request.Currency {
				supportedStrategies = append(supportedStrategies, paymentStrategy)
				break
			}
		}
	}

	if len(supportedStrategies) == 0 {
		return nil, fmt.Errorf("no payment strategy supports currency: %s", request.Currency)
	}

	// Select strategy based on priority (fastest processing time)
	sort.Slice(supportedStrategies, func(i, j int) bool {
		return supportedStrategies[i].GetProcessingTime() < supportedStrategies[j].GetProcessingTime()
	})

	return supportedStrategies[0], nil
}

// SelectNotificationStrategy selects the best notification strategy based on request
func (ss *StrategySelectorImpl) SelectNotificationStrategy(ctx context.Context, request NotificationRequest) (NotificationStrategy, error) {
	// Get available strategies
	strategies := ss.notificationManager.GetAvailableStrategies()

	// Filter strategies based on channel support
	var supportedStrategies []NotificationStrategy
	for _, strategyName := range strategies {
		strategy, err := ss.notificationManager.GetStrategy(strategyName)
		if err != nil {
			continue
		}

		notificationStrategy, ok := strategy.(NotificationStrategy)
		if !ok {
			continue
		}

		// Check if strategy supports the channel
		supportedChannels := notificationStrategy.GetSupportedChannels()
		for _, channel := range supportedChannels {
			if channel == request.Channel {
				supportedStrategies = append(supportedStrategies, notificationStrategy)
				break
			}
		}
	}

	if len(supportedStrategies) == 0 {
		return nil, fmt.Errorf("no notification strategy supports channel: %s", request.Channel)
	}

	// Select strategy based on priority (fastest delivery time)
	sort.Slice(supportedStrategies, func(i, j int) bool {
		return supportedStrategies[i].GetDeliveryTime() < supportedStrategies[j].GetDeliveryTime()
	})

	return supportedStrategies[0], nil
}

// SelectPricingStrategy selects the best pricing strategy based on request
func (ss *StrategySelectorImpl) SelectPricingStrategy(ctx context.Context, request PricingRequest) (PricingStrategy, error) {
	// Get available strategies
	strategies := ss.pricingManager.GetAvailableStrategies()

	// Filter strategies based on product support
	var supportedStrategies []PricingStrategy
	for _, strategyName := range strategies {
		strategy, err := ss.pricingManager.GetStrategy(strategyName)
		if err != nil {
			continue
		}

		pricingStrategy, ok := strategy.(PricingStrategy)
		if !ok {
			continue
		}

		// Check if strategy supports the product
		supportedProducts := pricingStrategy.GetSupportedProducts()
		for _, product := range supportedProducts {
			if product == "all" || product == request.ProductID {
				supportedStrategies = append(supportedStrategies, pricingStrategy)
				break
			}
		}
	}

	if len(supportedStrategies) == 0 {
		return nil, fmt.Errorf("no pricing strategy supports product: %s", request.ProductID)
	}

	// Select strategy based on priority (fastest calculation time)
	sort.Slice(supportedStrategies, func(i, j int) bool {
		return supportedStrategies[i].GetCalculationTime() < supportedStrategies[j].GetCalculationTime()
	})

	return supportedStrategies[0], nil
}

// SelectAuthenticationStrategy selects the best authentication strategy based on request
func (ss *StrategySelectorImpl) SelectAuthenticationStrategy(ctx context.Context, request AuthRequest) (AuthenticationStrategy, error) {
	// Get available strategies
	strategies := ss.authManager.GetAvailableStrategies()

	// Filter strategies based on method support
	var supportedStrategies []AuthenticationStrategy
	for _, strategyName := range strategies {
		strategy, err := ss.authManager.GetStrategy(strategyName)
		if err != nil {
			continue
		}

		authStrategy, ok := strategy.(AuthenticationStrategy)
		if !ok {
			continue
		}

		// Check if strategy supports the method
		supportedMethods := authStrategy.GetSupportedMethods()
		for _, method := range supportedMethods {
			if method == request.Method {
				supportedStrategies = append(supportedStrategies, authStrategy)
				break
			}
		}
	}

	if len(supportedStrategies) == 0 {
		return nil, fmt.Errorf("no authentication strategy supports method: %s", request.Method)
	}

	// Select strategy based on priority (fastest auth time)
	sort.Slice(supportedStrategies, func(i, j int) bool {
		return supportedStrategies[i].GetAuthTime() < supportedStrategies[j].GetAuthTime()
	})

	return supportedStrategies[0], nil
}

// SelectCachingStrategy selects the best caching strategy based on key
func (ss *StrategySelectorImpl) SelectCachingStrategy(ctx context.Context, key string) (CachingStrategy, error) {
	// Get available strategies
	strategies := ss.cachingManager.GetAvailableStrategies()

	// Filter strategies based on type support
	var supportedStrategies []CachingStrategy
	for _, strategyName := range strategies {
		strategy, err := ss.cachingManager.GetStrategy(strategyName)
		if err != nil {
			continue
		}

		cachingStrategy, ok := strategy.(CachingStrategy)
		if !ok {
			continue
		}

		// Check if strategy supports the type
		supportedTypes := cachingStrategy.GetSupportedTypes()
		for _, cacheType := range supportedTypes {
			if cacheType == "all" || cacheType == "general" {
				supportedStrategies = append(supportedStrategies, cachingStrategy)
				break
			}
		}
	}

	if len(supportedStrategies) == 0 {
		return nil, fmt.Errorf("no caching strategy available")
	}

	// Select strategy based on priority (fastest access time)
	sort.Slice(supportedStrategies, func(i, j int) bool {
		return supportedStrategies[i].GetAccessTime() < supportedStrategies[j].GetAccessTime()
	})

	return supportedStrategies[0], nil
}

// SelectLoggingStrategy selects the best logging strategy based on level
func (ss *StrategySelectorImpl) SelectLoggingStrategy(ctx context.Context, level LogLevel) (LoggingStrategy, error) {
	// Get available strategies
	strategies := ss.loggingManager.GetAvailableStrategies()

	// Filter strategies based on level support
	var supportedStrategies []LoggingStrategy
	for _, strategyName := range strategies {
		strategy, err := ss.loggingManager.GetStrategy(strategyName)
		if err != nil {
			continue
		}

		loggingStrategy, ok := strategy.(LoggingStrategy)
		if !ok {
			continue
		}

		// Check if strategy supports the level
		supportedLevels := loggingStrategy.GetSupportedLevels()
		for _, supportedLevel := range supportedLevels {
			if supportedLevel == level {
				supportedStrategies = append(supportedStrategies, loggingStrategy)
				break
			}
		}
	}

	if len(supportedStrategies) == 0 {
		return nil, fmt.Errorf("no logging strategy supports level: %s", level.String())
	}

	// Select strategy based on priority (fastest log time)
	sort.Slice(supportedStrategies, func(i, j int) bool {
		return supportedStrategies[i].GetLogTime() < supportedStrategies[j].GetLogTime()
	})

	return supportedStrategies[0], nil
}

// SelectDataProcessingStrategy selects the best data processing strategy based on data
func (ss *StrategySelectorImpl) SelectDataProcessingStrategy(ctx context.Context, data interface{}) (DataProcessingStrategy, error) {
	// Get available strategies
	strategies := ss.dataProcessingManager.GetAvailableStrategies()

	// Filter strategies based on format support
	var supportedStrategies []DataProcessingStrategy
	for _, strategyName := range strategies {
		strategy, err := ss.dataProcessingManager.GetStrategy(strategyName)
		if err != nil {
			continue
		}

		dataProcessingStrategy, ok := strategy.(DataProcessingStrategy)
		if !ok {
			continue
		}

		// Check if strategy supports the format
		supportedFormats := dataProcessingStrategy.GetSupportedFormats()
		for _, format := range supportedFormats {
			if format == "all" || format == "general" {
				supportedStrategies = append(supportedStrategies, dataProcessingStrategy)
				break
			}
		}
	}

	if len(supportedStrategies) == 0 {
		return nil, fmt.Errorf("no data processing strategy available")
	}

	// Select strategy based on priority (fastest processing time)
	sort.Slice(supportedStrategies, func(i, j int) bool {
		return supportedStrategies[i].GetProcessingTime() < supportedStrategies[j].GetProcessingTime()
	})

	return supportedStrategies[0], nil
}

// SelectStrategyByPriority selects strategy based on priority and criteria
func (ss *StrategySelectorImpl) SelectStrategyByPriority(ctx context.Context, selection StrategySelection) (interface{}, error) {
	var manager StrategyManager

	switch selection.StrategyType {
	case "payment":
		manager = ss.paymentManager
	case "notification":
		manager = ss.notificationManager
	case "pricing":
		manager = ss.pricingManager
	case "authentication":
		manager = ss.authManager
	case "caching":
		manager = ss.cachingManager
	case "logging":
		manager = ss.loggingManager
	case "data_processing":
		manager = ss.dataProcessingManager
	default:
		return nil, fmt.Errorf("unsupported strategy type: %s", selection.StrategyType)
	}

	// Get available strategies
	strategies := manager.GetAvailableStrategies()

	// Filter strategies based on criteria
	var supportedStrategies []interface{}
	for _, strategyName := range strategies {
		strategy, err := manager.GetStrategy(strategyName)
		if err != nil {
			continue
		}

		// Check if strategy meets criteria
		if ss.meetsCriteria(strategy, selection.Criteria) {
			supportedStrategies = append(supportedStrategies, strategy)
		}
	}

	if len(supportedStrategies) == 0 {
		return nil, fmt.Errorf("no strategy meets criteria for type: %s", selection.StrategyType)
	}

	// Select strategy based on priority
	sort.Slice(supportedStrategies, func(i, j int) bool {
		return ss.getStrategyPriority(supportedStrategies[i]) < ss.getStrategyPriority(supportedStrategies[j])
	})

	return supportedStrategies[0], nil
}

// meetsCriteria checks if strategy meets selection criteria
func (ss *StrategySelectorImpl) meetsCriteria(strategy interface{}, criteria map[string]string) bool {
	// Implement criteria matching logic
	// For now, return true for all strategies
	return true
}

// getStrategyPriority returns priority of strategy
func (ss *StrategySelectorImpl) getStrategyPriority(strategy interface{}) int {
	// Implement priority calculation logic
	// For now, return 0 for all strategies
	return 0
}
