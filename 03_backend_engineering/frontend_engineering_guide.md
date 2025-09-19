# 🎨 Frontend Engineering Complete Guide

> **Comprehensive frontend knowledge for backend engineers interviewing for full-stack roles**

## 🎯 **Overview**

Modern backend engineers are increasingly expected to understand frontend technologies, especially for senior and staff-level positions. This guide covers essential frontend concepts, frameworks, and best practices.

## 📚 **Core Technologies**

### **HTML5 & Semantic Web**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Payment processing dashboard">
    <title>Razorpay Dashboard</title>
    
    <!-- Progressive Web App -->
    <link rel="manifest" href="/manifest.json">
    <meta name="theme-color" content="#3b82f6">
    
    <!-- Performance -->
    <link rel="preload" href="/assets/main.css" as="style">
    <link rel="preload" href="/assets/main.js" as="script">
</head>
<body>
    <!-- Semantic HTML -->
    <header role="banner">
        <nav aria-label="Main navigation">
            <ul>
                <li><a href="/dashboard" aria-current="page">Dashboard</a></li>
                <li><a href="/transactions">Transactions</a></li>
                <li><a href="/analytics">Analytics</a></li>
            </ul>
        </nav>
    </header>
    
    <main role="main">
        <section aria-labelledby="dashboard-heading">
            <h1 id="dashboard-heading">Payment Dashboard</h1>
            <!-- Content -->
        </section>
    </main>
    
    <footer role="contentinfo">
        <!-- Footer content -->
    </footer>
</body>
</html>
```

### **Modern CSS & Styling**

```css
/* CSS Variables and Modern Features */
:root {
    --primary-color: #3b82f6;
    --secondary-color: #64748b;
    --success-color: #10b981;
    --error-color: #ef4444;
    --border-radius: 8px;
    --spacing-unit: 1rem;
    --font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* CSS Grid Layout */
.dashboard-layout {
    display: grid;
    grid-template-areas: 
        "sidebar header"
        "sidebar main"
        "sidebar footer";
    grid-template-columns: 250px 1fr;
    grid-template-rows: auto 1fr auto;
    min-height: 100vh;
    gap: var(--spacing-unit);
}

/* Flexbox Components */
.payment-card {
    display: flex;
    flex-direction: column;
    padding: var(--spacing-unit);
    border-radius: var(--border-radius);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s ease-in-out;
}

.payment-card:hover {
    transform: translateY(-2px);
}

/* Responsive Design */
@media (max-width: 768px) {
    .dashboard-layout {
        grid-template-areas: 
            "header"
            "main"
            "footer";
        grid-template-columns: 1fr;
    }
}

/* CSS Container Queries (Modern) */
@container (min-width: 400px) {
    .payment-card {
        flex-direction: row;
        align-items: center;
    }
}
```

### **JavaScript ES6+ Fundamentals**

```javascript
// Modern JavaScript Patterns
class PaymentService {
    #apiUrl = '/api/v1/payments';
    #headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    };

    constructor(token) {
        this.#headers.Authorization = `Bearer ${token}`;
    }

    // Async/Await with Error Handling
    async fetchPayments(filters = {}) {
        try {
            const params = new URLSearchParams(filters);
            const response = await fetch(`${this.#apiUrl}?${params}`, {
                headers: this.#headers
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            return { success: true, data };
        } catch (error) {
            console.error('Payment fetch error:', error);
            return { success: false, error: error.message };
        }
    }

    // Generator for Pagination
    async* paginatePayments(pageSize = 50) {
        let page = 1;
        let hasMore = true;

        while (hasMore) {
            const { success, data } = await this.fetchPayments({ 
                page, 
                limit: pageSize 
            });

            if (!success || !data.payments.length) {
                hasMore = false;
                break;
            }

            yield data.payments;
            page++;
            hasMore = data.hasMore;
        }
    }

    // Promise.allSettled for Batch Operations
    async batchUpdatePayments(paymentIds, updates) {
        const promises = paymentIds.map(id => 
            this.updatePayment(id, updates)
        );

        const results = await Promise.allSettled(promises);
        
        return {
            successful: results.filter(r => r.status === 'fulfilled').length,
            failed: results.filter(r => r.status === 'rejected').length,
            results
        };
    }
}

// Advanced JavaScript Patterns
const PaymentModule = (() => {
    // Private variables
    let payments = new Map();
    let observers = new Set();

    // Observer Pattern
    const notifyObservers = (event, data) => {
        observers.forEach(observer => observer(event, data));
    };

    // Public API
    return {
        addPayment(payment) {
            payments.set(payment.id, payment);
            notifyObservers('payment-added', payment);
        },

        subscribe(observer) {
            observers.add(observer);
            return () => observers.delete(observer);
        },

        // Memoization
        getPaymentSummary: (() => {
            const cache = new Map();
            return (dateRange) => {
                const key = `${dateRange.start}-${dateRange.end}`;
                if (cache.has(key)) {
                    return cache.get(key);
                }

                const summary = Array.from(payments.values())
                    .filter(p => p.createdAt >= dateRange.start && p.createdAt <= dateRange.end)
                    .reduce((acc, payment) => {
                        acc.total += payment.amount;
                        acc.count += 1;
                        return acc;
                    }, { total: 0, count: 0 });

                cache.set(key, summary);
                return summary;
            };
        })()
    };
})();
```

## 🚀 **Modern Frontend Frameworks**

### **React.js with Hooks**

```jsx
import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { ErrorBoundary } from 'react-error-boundary';

// Custom Hook for Payment Data
const usePayments = (filters) => {
    const [payments, setPayments] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const abortControllerRef = useRef();

    const fetchPayments = useCallback(async (filters) => {
        // Cancel previous request
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }

        abortControllerRef.current = new AbortController();
        setLoading(true);
        setError(null);

        try {
            const response = await fetch('/api/payments', {
                signal: abortControllerRef.current.signal,
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(filters)
            });

            if (!response.ok) throw new Error('Failed to fetch');
            
            const data = await response.json();
            setPayments(data.payments);
        } catch (err) {
            if (err.name !== 'AbortError') {
                setError(err.message);
            }
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchPayments(filters);
        
        return () => {
            if (abortControllerRef.current) {
                abortControllerRef.current.abort();
            }
        };
    }, [filters, fetchPayments]);

    return { payments, loading, error, refetch: () => fetchPayments(filters) };
};

// Payment Dashboard Component
const PaymentDashboard = () => {
    const [filters, setFilters] = useState({
        status: 'all',
        dateRange: { start: null, end: null }
    });

    const { payments, loading, error } = usePayments(filters);

    // Memoized calculations
    const paymentStats = useMemo(() => {
        return payments.reduce((stats, payment) => {
            stats.total += payment.amount;
            stats.count += 1;
            stats.avgAmount = stats.total / stats.count;
            return stats;
        }, { total: 0, count: 0, avgAmount: 0 });
    }, [payments]);

    // Error boundary fallback
    const ErrorFallback = ({ error, resetErrorBoundary }) => (
        <div className="error-boundary">
            <h2>Something went wrong</h2>
            <pre>{error.message}</pre>
            <button onClick={resetErrorBoundary}>Try again</button>
        </div>
    );

    if (loading) return <PaymentSkeleton />;
    if (error) return <ErrorMessage error={error} />;

    return (
        <ErrorBoundary FallbackComponent={ErrorFallback}>
            <div className="payment-dashboard">
                <PaymentFilters 
                    filters={filters} 
                    onChange={setFilters} 
                />
                
                <PaymentStats stats={paymentStats} />
                
                <PaymentList 
                    payments={payments}
                    onPaymentUpdate={(id, data) => {
                        // Optimistic update
                        setPayments(prev => 
                            prev.map(p => p.id === id ? { ...p, ...data } : p)
                        );
                    }}
                />
            </div>
        </ErrorBoundary>
    );
};

// Performance optimized component
const PaymentList = React.memo(({ payments, onPaymentUpdate }) => {
    return (
        <div className="payment-list">
            {payments.map(payment => (
                <PaymentCard 
                    key={payment.id} 
                    payment={payment}
                    onUpdate={onPaymentUpdate}
                />
            ))}
        </div>
    );
});
```

### **Vue.js 3 with Composition API**

```vue
<template>
  <div class="payment-dashboard">
    <PaymentFilters 
      v-model:filters="filters"
      @update:filters="handleFiltersChange"
    />
    
    <Suspense>
      <template #default>
        <PaymentList :payments="payments" />
      </template>
      <template #fallback>
        <PaymentSkeleton />
      </template>
    </Suspense>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { usePaymentStore } from '@/stores/payment'
import { useWebSocket } from '@/composables/useWebSocket'

// Props & Emits
const props = defineProps({
  customerId: String
})

const emit = defineEmits(['payment-updated'])

// Composables
const paymentStore = usePaymentStore()
const { connect, disconnect, send, isConnected } = useWebSocket()

// Reactive state
const filters = ref({
  status: 'all',
  dateRange: { start: null, end: null }
})

// Computed properties
const payments = computed(() => 
  paymentStore.getFilteredPayments(filters.value)
)

const totalAmount = computed(() =>
  payments.value.reduce((sum, payment) => sum + payment.amount, 0)
)

// Methods
const handleFiltersChange = (newFilters) => {
  filters.value = { ...filters.value, ...newFilters }
}

// Watchers
watch(filters, (newFilters) => {
  paymentStore.fetchPayments(newFilters)
}, { deep: true })

// Lifecycle
onMounted(async () => {
  await paymentStore.fetchPayments(filters.value)
  
  // WebSocket connection for real-time updates
  connect('/ws/payments')
})

onUnmounted(() => {
  disconnect()
})
</script>

<style scoped>
.payment-dashboard {
  display: grid;
  gap: 1rem;
  padding: 1rem;
}

@media (min-width: 768px) {
  .payment-dashboard {
    grid-template-columns: 300px 1fr;
  }
}
</style>
```

## 🏗️ **State Management**

### **Redux Toolkit (React)**

```javascript
// store/paymentSlice.js
import { createSlice, createAsyncThunk, createSelector } from '@reduxjs/toolkit';
import { PaymentService } from '../services/PaymentService';

// Async thunks
export const fetchPayments = createAsyncThunk(
    'payments/fetchPayments',
    async (filters, { rejectWithValue }) => {
        try {
            const response = await PaymentService.getPayments(filters);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.message);
        }
    }
);

export const updatePayment = createAsyncThunk(
    'payments/updatePayment',
    async ({ id, data }, { rejectWithValue }) => {
        try {
            const response = await PaymentService.updatePayment(id, data);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.message);
        }
    }
);

// Slice
const paymentSlice = createSlice({
    name: 'payments',
    initialState: {
        items: [],
        loading: false,
        error: null,
        filters: {
            status: 'all',
            dateRange: { start: null, end: null }
        }
    },
    reducers: {
        setFilters: (state, action) => {
            state.filters = { ...state.filters, ...action.payload };
        },
        clearError: (state) => {
            state.error = null;
        },
        // Optimistic update
        optimisticUpdate: (state, action) => {
            const { id, data } = action.payload;
            const payment = state.items.find(p => p.id === id);
            if (payment) {
                Object.assign(payment, data);
            }
        }
    },
    extraReducers: (builder) => {
        builder
            .addCase(fetchPayments.pending, (state) => {
                state.loading = true;
                state.error = null;
            })
            .addCase(fetchPayments.fulfilled, (state, action) => {
                state.loading = false;
                state.items = action.payload;
            })
            .addCase(fetchPayments.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
            })
            .addCase(updatePayment.fulfilled, (state, action) => {
                const index = state.items.findIndex(p => p.id === action.payload.id);
                if (index !== -1) {
                    state.items[index] = action.payload;
                }
            });
    }
});

// Selectors
export const selectPayments = (state) => state.payments.items;
export const selectPaymentsByStatus = createSelector(
    [selectPayments, (state, status) => status],
    (payments, status) => payments.filter(p => status === 'all' || p.status === status)
);

export const selectPaymentStats = createSelector(
    [selectPayments],
    (payments) => ({
        total: payments.reduce((sum, p) => sum + p.amount, 0),
        count: payments.length,
        avgAmount: payments.length ? payments.reduce((sum, p) => sum + p.amount, 0) / payments.length : 0
    })
);

export const { setFilters, clearError, optimisticUpdate } = paymentSlice.actions;
export default paymentSlice.reducer;
```

### **Pinia (Vue.js)**

```javascript
// stores/payment.js
import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { PaymentService } from '@/services/PaymentService'

export const usePaymentStore = defineStore('payment', () => {
    // State
    const payments = ref([])
    const loading = ref(false)
    const error = ref(null)
    const filters = ref({
        status: 'all',
        dateRange: { start: null, end: null }
    })

    // Getters (computed)
    const filteredPayments = computed(() => {
        return payments.value.filter(payment => {
            if (filters.value.status !== 'all' && payment.status !== filters.value.status) {
                return false
            }
            
            if (filters.value.dateRange.start && payment.createdAt < filters.value.dateRange.start) {
                return false
            }
            
            if (filters.value.dateRange.end && payment.createdAt > filters.value.dateRange.end) {
                return false
            }
            
            return true
        })
    })

    const paymentStats = computed(() => {
        const total = filteredPayments.value.reduce((sum, p) => sum + p.amount, 0)
        const count = filteredPayments.value.length
        return {
            total,
            count,
            avgAmount: count ? total / count : 0
        }
    })

    // Actions
    async function fetchPayments(newFilters = {}) {
        loading.value = true
        error.value = null
        
        try {
            const mergedFilters = { ...filters.value, ...newFilters }
            const response = await PaymentService.getPayments(mergedFilters)
            payments.value = response.data
            filters.value = mergedFilters
        } catch (err) {
            error.value = err.message
        } finally {
            loading.value = false
        }
    }

    async function updatePayment(id, data) {
        try {
            // Optimistic update
            const index = payments.value.findIndex(p => p.id === id)
            if (index !== -1) {
                const original = { ...payments.value[index] }
                payments.value[index] = { ...original, ...data }
                
                try {
                    const response = await PaymentService.updatePayment(id, data)
                    payments.value[index] = response.data
                } catch (err) {
                    // Revert on error
                    payments.value[index] = original
                    throw err
                }
            }
        } catch (err) {
            error.value = err.message
        }
    }

    function setFilters(newFilters) {
        filters.value = { ...filters.value, ...newFilters }
    }

    function clearError() {
        error.value = null
    }

    return {
        // State
        payments,
        loading,
        error,
        filters,
        // Getters
        filteredPayments,
        paymentStats,
        // Actions
        fetchPayments,
        updatePayment,
        setFilters,
        clearError
    }
})
```

## 🎨 **Advanced Frontend Concepts**

### **Performance Optimization**

```javascript
// Code Splitting with React.lazy
import { lazy, Suspense } from 'react';

const PaymentDashboard = lazy(() => import('./PaymentDashboard'));
const AnalyticsDashboard = lazy(() => import('./AnalyticsDashboard'));

function App() {
    return (
        <Suspense fallback={<div>Loading...</div>}>
            <PaymentDashboard />
        </Suspense>
    );
}

// Web Workers for Heavy Computations
class PaymentAnalytics {
    constructor() {
        this.worker = new Worker('/workers/analytics.js');
        this.worker.onmessage = this.handleWorkerMessage.bind(this);
    }

    calculateTrends(payments) {
        return new Promise((resolve, reject) => {
            const requestId = Date.now();
            this.pendingRequests = this.pendingRequests || new Map();
            this.pendingRequests.set(requestId, { resolve, reject });

            this.worker.postMessage({
                type: 'CALCULATE_TRENDS',
                data: payments,
                requestId
            });
        });
    }

    handleWorkerMessage(event) {
        const { type, data, requestId, error } = event.data;
        const request = this.pendingRequests.get(requestId);
        
        if (request) {
            if (error) {
                request.reject(new Error(error));
            } else {
                request.resolve(data);
            }
            this.pendingRequests.delete(requestId);
        }
    }
}

// Service Worker for Caching
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js')
        .then(registration => console.log('SW registered'))
        .catch(error => console.log('SW registration failed'));
}
```

### **Testing Strategies**

```javascript
// Jest + React Testing Library
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import PaymentDashboard from '../PaymentDashboard';
import paymentReducer from '../store/paymentSlice';

// Mock API
jest.mock('../services/PaymentService', () => ({
    getPayments: jest.fn(),
    updatePayment: jest.fn()
}));

describe('PaymentDashboard', () => {
    let store;
    
    beforeEach(() => {
        store = configureStore({
            reducer: {
                payments: paymentReducer
            }
        });
    });

    test('renders payment list', async () => {
        const mockPayments = [
            { id: '1', amount: 100, status: 'completed' },
            { id: '2', amount: 200, status: 'pending' }
        ];

        PaymentService.getPayments.mockResolvedValue({
            data: mockPayments
        });

        render(
            <Provider store={store}>
                <PaymentDashboard />
            </Provider>
        );

        await waitFor(() => {
            expect(screen.getByText('₹100')).toBeInTheDocument();
            expect(screen.getByText('₹200')).toBeInTheDocument();
        });
    });

    test('filters payments by status', async () => {
        // Test implementation
    });
});

// Cypress E2E Tests
describe('Payment Dashboard E2E', () => {
    beforeEach(() => {
        cy.intercept('GET', '/api/payments', { fixture: 'payments.json' });
        cy.visit('/dashboard');
    });

    it('should display payments and allow filtering', () => {
        cy.get('[data-testid="payment-list"]').should('be.visible');
        cy.get('[data-testid="payment-item"]').should('have.length.greaterThan', 0);
        
        // Test filtering
        cy.get('[data-testid="status-filter"]').select('completed');
        cy.get('[data-testid="payment-item"]').should('contain', 'Completed');
    });

    it('should handle payment updates', () => {
        cy.intercept('PUT', '/api/payments/*', { statusCode: 200 });
        
        cy.get('[data-testid="payment-item"]').first().click();
        cy.get('[data-testid="update-status"]').click();
        cy.get('[data-testid="status-updated"]').should('be.visible');
    });
});
```

## 🌐 **Modern Web Technologies**

### **Progressive Web App (PWA)**

```javascript
// manifest.json
{
    "name": "Razorpay Dashboard",
    "short_name": "RzpDashboard",
    "description": "Payment processing dashboard",
    "start_url": "/",
    "display": "standalone",
    "background_color": "#ffffff",
    "theme_color": "#3b82f6",
    "icons": [
        {
            "src": "/icons/icon-192.png",
            "sizes": "192x192",
            "type": "image/png"
        },
        {
            "src": "/icons/icon-512.png",
            "sizes": "512x512",
            "type": "image/png"
        }
    ]
}

// Service Worker
const CACHE_NAME = 'razorpay-dashboard-v1';
const urlsToCache = [
    '/',
    '/static/css/main.css',
    '/static/js/main.js',
    '/api/payments'
];

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => cache.addAll(urlsToCache))
    );
});

self.addEventListener('fetch', (event) => {
    event.respondWith(
        caches.match(event.request)
            .then((response) => {
                if (response) {
                    return response;
                }
                return fetch(event.request);
            })
    );
});
```

### **WebSockets & Real-time Updates**

```javascript
// WebSocket Hook
import { useEffect, useRef, useState } from 'react';

export const useWebSocket = (url) => {
    const ws = useRef(null);
    const [connectionStatus, setConnectionStatus] = useState('Disconnected');
    const [lastMessage, setLastMessage] = useState(null);

    useEffect(() => {
        ws.current = new WebSocket(url);
        
        ws.current.onopen = () => setConnectionStatus('Connected');
        ws.current.onclose = () => setConnectionStatus('Disconnected');
        ws.current.onmessage = (event) => {
            const data = JSON.parse(event.data);
            setLastMessage(data);
        };

        return () => {
            ws.current.close();
        };
    }, [url]);

    const sendMessage = (message) => {
        if (ws.current.readyState === WebSocket.OPEN) {
            ws.current.send(JSON.stringify(message));
        }
    };

    return { connectionStatus, lastMessage, sendMessage };
};

// Real-time Payment Updates
const PaymentDashboard = () => {
    const { lastMessage } = useWebSocket('wss://api.razorpay.com/ws/payments');
    const [payments, setPayments] = useState([]);

    useEffect(() => {
        if (lastMessage?.type === 'PAYMENT_UPDATE') {
            setPayments(prev => 
                prev.map(p => 
                    p.id === lastMessage.payment.id 
                        ? { ...p, ...lastMessage.payment }
                        : p
                )
            );
        }
    }, [lastMessage]);

    return (
        <div>
            {/* Component content */}
        </div>
    );
};
```

## 📱 **Mobile-First & Responsive Design**

```css
/* Mobile-First Approach */
.payment-dashboard {
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.payment-card {
    background: white;
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

/* Tablet (768px and up) */
@media (min-width: 768px) {
    .payment-dashboard {
        padding: 2rem;
        display: grid;
        grid-template-columns: 1fr 2fr;
        gap: 2rem;
    }
    
    .payment-card {
        padding: 1.5rem;
    }
}

/* Desktop (1024px and up) */
@media (min-width: 1024px) {
    .payment-dashboard {
        max-width: 1200px;
        margin: 0 auto;
        grid-template-columns: 300px 1fr 300px;
    }
}

/* Touch-friendly design */
@media (hover: none) and (pointer: coarse) {
    .button {
        min-height: 44px;
        min-width: 44px;
        padding: 12px 24px;
    }
}
```

## 🔧 **Build Tools & Optimization**

### **Webpack Configuration**

```javascript
// webpack.config.js
const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');
const TerserPlugin = require('terser-webpack-plugin');

module.exports = (env, argv) => {
    const isProduction = argv.mode === 'production';
    
    return {
        entry: './src/index.js',
        output: {
            path: path.resolve(__dirname, 'dist'),
            filename: isProduction 
                ? '[name].[contenthash].js' 
                : '[name].js',
            clean: true
        },
        
        module: {
            rules: [
                {
                    test: /\.(js|jsx)$/,
                    exclude: /node_modules/,
                    use: {
                        loader: 'babel-loader',
                        options: {
                            presets: ['@babel/preset-react']
                        }
                    }
                },
                {
                    test: /\.css$/,
                    use: [
                        isProduction ? MiniCssExtractPlugin.loader : 'style-loader',
                        'css-loader',
                        'postcss-loader'
                    ]
                }
            ]
        },
        
        plugins: [
            new HtmlWebpackPlugin({
                template: './public/index.html'
            }),
            ...(isProduction ? [
                new MiniCssExtractPlugin({
                    filename: '[name].[contenthash].css'
                })
            ] : [])
        ],
        
        optimization: isProduction ? {
            minimizer: [new TerserPlugin()],
            splitChunks: {
                chunks: 'all',
                cacheGroups: {
                    vendor: {
                        test: /[\\/]node_modules[\\/]/,
                        name: 'vendors',
                        chunks: 'all'
                    }
                }
            }
        } : {},
        
        devServer: {
            port: 3000,
            proxy: {
                '/api': 'http://localhost:8000'
            }
        }
    };
};
```

### **Vite Configuration**

```javascript
// vite.config.js
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';

export default defineConfig({
    plugins: [react()],
    
    resolve: {
        alias: {
            '@': resolve(__dirname, 'src'),
            '@components': resolve(__dirname, 'src/components'),
            '@utils': resolve(__dirname, 'src/utils')
        }
    },
    
    build: {
        rollupOptions: {
            output: {
                manualChunks: {
                    vendor: ['react', 'react-dom'],
                    ui: ['@mui/material', '@mui/icons-material']
                }
            }
        }
    },
    
    server: {
        proxy: {
            '/api': {
                target: 'http://localhost:8000',
                changeOrigin: true
            }
        }
    }
});
```

## 🎯 **Interview Questions & Scenarios**

### **Common Frontend Interview Questions**

**Q1: Explain the difference between client-side and server-side rendering.**

**Answer:**
- **Client-side rendering (CSR)**: HTML is generated in the browser using JavaScript
  - Pros: Rich interactions, better user experience after initial load
  - Cons: Slower initial load, SEO challenges
  
- **Server-side rendering (SSR)**: HTML is generated on the server
  - Pros: Faster initial load, better SEO
  - Cons: More server resources, less interactivity

**Q2: How would you optimize a React application's performance?**

**Answer:**
```javascript
// 1. Code splitting
const LazyComponent = React.lazy(() => import('./Component'));

// 2. Memoization
const ExpensiveComponent = React.memo(({ data }) => {
    const processedData = useMemo(() => 
        expensiveCalculation(data), [data]
    );
    
    return <div>{processedData}</div>;
});

// 3. Virtualization for large lists
import { FixedSizeList as List } from 'react-window';

// 4. Debouncing user input
const useDebounce = (value, delay) => {
    const [debouncedValue, setDebouncedValue] = useState(value);
    
    useEffect(() => {
        const handler = setTimeout(() => {
            setDebouncedValue(value);
        }, delay);
        
        return () => clearTimeout(handler);
    }, [value, delay]);
    
    return debouncedValue;
};
```

**Q3: Explain how you would implement state management in a large application.**

**Answer:**
```javascript
// Redux Toolkit approach
const store = configureStore({
    reducer: {
        payments: paymentSlice,
        auth: authSlice,
        ui: uiSlice
    },
    middleware: (getDefaultMiddleware) =>
        getDefaultMiddleware({
            serializableCheck: {
                ignoredActions: [FLUSH, REHYDRATE, PAUSE, PERSIST, PURGE, REGISTER]
            }
        }).concat(persistMiddleware)
});

// Context + Reducer for simpler state
const AppContext = createContext();

const appReducer = (state, action) => {
    switch (action.type) {
        case 'SET_USER':
            return { ...state, user: action.payload };
        default:
            return state;
    }
};

export const AppProvider = ({ children }) => {
    const [state, dispatch] = useReducer(appReducer, initialState);
    
    return (
        <AppContext.Provider value={{ state, dispatch }}>
            {children}
        </AppContext.Provider>
    );
};
```

## 📚 **Best Practices & Patterns**

### **Component Design Patterns**

```javascript
// 1. Compound Components
const Modal = ({ children, isOpen, onClose }) => {
    if (!isOpen) return null;
    
    return createPortal(
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content" onClick={e => e.stopPropagation()}>
                {children}
            </div>
        </div>,
        document.body
    );
};

Modal.Header = ({ children }) => <div className="modal-header">{children}</div>;
Modal.Body = ({ children }) => <div className="modal-body">{children}</div>;
Modal.Footer = ({ children }) => <div className="modal-footer">{children}</div>;

// Usage
<Modal isOpen={showModal} onClose={() => setShowModal(false)}>
    <Modal.Header>
        <h2>Payment Details</h2>
    </Modal.Header>
    <Modal.Body>
        <PaymentForm />
    </Modal.Body>
    <Modal.Footer>
        <button onClick={handleSubmit}>Submit</button>
    </Modal.Footer>
</Modal>

// 2. Render Props Pattern
const DataFetcher = ({ url, children }) => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        fetch(url)
            .then(res => res.json())
            .then(setData)
            .catch(setError)
            .finally(() => setLoading(false));
    }, [url]);

    return children({ data, loading, error });
};

// Usage
<DataFetcher url="/api/payments">
    {({ data, loading, error }) => {
        if (loading) return <Spinner />;
        if (error) return <Error message={error.message} />;
        return <PaymentList payments={data} />;
    }}
</DataFetcher>

// 3. Higher-Order Components
const withAuth = (WrappedComponent) => {
    return function WithAuthComponent(props) {
        const { user, loading } = useAuth();
        
        if (loading) return <Spinner />;
        if (!user) return <LoginForm />;
        
        return <WrappedComponent {...props} user={user} />;
    };
};

const ProtectedDashboard = withAuth(PaymentDashboard);
```

## 🔍 **Advanced Topics**

### **Micro-frontends Architecture**

```javascript
// Module Federation (Webpack 5)
const ModuleFederationPlugin = require('@module-federation/webpack');

module.exports = {
    plugins: [
        new ModuleFederationPlugin({
            name: 'payment_dashboard',
            filename: 'remoteEntry.js',
            exposes: {
                './PaymentDashboard': './src/components/PaymentDashboard',
                './PaymentForm': './src/components/PaymentForm'
            },
            shared: {
                react: { singleton: true },
                'react-dom': { singleton: true }
            }
        })
    ]
};

// Consuming micro-frontend
const PaymentDashboard = React.lazy(() => import('payment_dashboard/PaymentDashboard'));

function App() {
    return (
        <Suspense fallback={<div>Loading Payment Dashboard...</div>}>
            <PaymentDashboard />
        </Suspense>
    );
}
```

### **Advanced Browser APIs**

```javascript
// Intersection Observer for infinite scroll
const useInfiniteScroll = (callback) => {
    const targetRef = useRef();

    useEffect(() => {
        const observer = new IntersectionObserver(
            (entries) => {
                if (entries[0].isIntersecting) {
                    callback();
                }
            },
            { threshold: 1.0 }
        );

        if (targetRef.current) {
            observer.observe(targetRef.current);
        }

        return () => observer.disconnect();
    }, [callback]);

    return targetRef;
};

// Web Workers for background processing
const useWebWorker = (workerFunction) => {
    const workerRef = useRef();
    
    useEffect(() => {
        const worker = new Worker(
            URL.createObjectURL(
                new Blob([`(${workerFunction})()`], { type: 'application/javascript' })
            )
        );
        
        workerRef.current = worker;
        
        return () => {
            worker.terminate();
            URL.revokeObjectURL(worker.url);
        };
    }, []);

    const postMessage = (data) => {
        workerRef.current?.postMessage(data);
    };

    return { postMessage };
};
```

## 🚀 **Getting Started Checklist**

### **Frontend Development Setup**

- [ ] **Development Environment**
  - [ ] Node.js and npm/yarn installed
  - [ ] Code editor with extensions (VS Code, Prettier, ESLint)
  - [ ] Browser dev tools familiarity

- [ ] **Core Technologies**
  - [ ] HTML5 semantic elements
  - [ ] Modern CSS (Grid, Flexbox, Custom Properties)
  - [ ] JavaScript ES6+ features
  - [ ] TypeScript basics

- [ ] **Framework Proficiency**
  - [ ] React.js with Hooks or Vue.js 3
  - [ ] State management (Redux Toolkit/Pinia)
  - [ ] Routing (React Router/Vue Router)

- [ ] **Build Tools**
  - [ ] Webpack or Vite configuration
  - [ ] Package.json scripts
  - [ ] Environment variables

- [ ] **Testing**
  - [ ] Unit testing with Jest
  - [ ] Component testing with Testing Library
  - [ ] E2E testing with Cypress

- [ ] **Performance**
  - [ ] Code splitting and lazy loading
  - [ ] Bundle analysis and optimization
  - [ ] Performance monitoring

---

This comprehensive frontend guide provides the essential knowledge for backend engineers transitioning to full-stack roles or facing frontend questions in senior-level interviews.