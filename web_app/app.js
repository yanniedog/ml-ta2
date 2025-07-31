/**
 * ML-TA Trading Assistant - Main Application Script
 * Handles UI interactions and API communication
 */

// App configuration
const config = {
    apiBaseUrl: 'http://localhost:8000/api/v1',
    endpoints: {
        health: '/health',
        predict: '/predict',
        models: '/models',
        ready: '/ready',
        demoKeys: '/demo-keys'
    },
    defaultApiKey: null,
    localStorageKeys: {
        userMode: 'mlta-user-mode',
        predictionHistory: 'mlta-prediction-history',
        apiKey: 'mlta-api-key'
    }
};

// Main application class
class MLTAApp {
    constructor() {
        this.mode = localStorage.getItem(config.localStorageKeys.userMode) || 'basic';
        this.apiKey = localStorage.getItem(config.localStorageKeys.apiKey) || null;
        this.predictionHistory = JSON.parse(localStorage.getItem(config.localStorageKeys.predictionHistory) || '[]');
        this.systemStatus = { healthy: false, components: {} };
        this.currentPanel = null;
        this.charts = {};
        
        // Initialize the application
        this.init();
    }
    
    // Initialize the application
    async init() {
        // Set initial mode
        this.setMode(this.mode);
        
        // Check system health
        await this.checkSystemHealth();
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Initialize demo API key if needed
        if (!this.apiKey) {
            await this.getDemoApiKey();
        }
        
        // Load prediction history
        this.loadPredictionHistory();
        
        console.log('ML-TA Trading Assistant initialized');
    }
    
    // Set up event listeners for UI elements
    setupEventListeners() {
        // Mode switch
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.setMode(btn.dataset.mode);
            });
        });
        
        // Navigation items
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', () => {
                document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
                item.classList.add('active');
            });
        });
        
        // Feature cards
        document.querySelectorAll('.card').forEach(card => {
            card.addEventListener('click', () => {
                const panel = card.dataset.panel;
                this.openPanel(panel);
            });
        });
        
        // Panel close buttons
        document.querySelectorAll('.close-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.closePanel();
            });
        });
        
        // Advanced options toggle
        document.querySelectorAll('.toggle-advanced').forEach(btn => {
            btn.addEventListener('click', () => {
                const advancedOptions = btn.closest('.panel').querySelector('.advanced-options');
                advancedOptions.classList.toggle('hidden');
            });
        });
        
        // Prediction form submission
        const predictionForm = document.getElementById('prediction-form');
        if (predictionForm) {
            predictionForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                await this.submitPrediction();
            });
        }
    }
    
    // Set the user interface mode (basic or advanced)
    setMode(mode) {
        this.mode = mode;
        localStorage.setItem(config.localStorageKeys.userMode, mode);
        
        // Update UI
        document.body.classList.toggle('advanced-mode', mode === 'advanced');
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });
        
        console.log(`Mode switched to: ${mode}`);
    }
    
    // Open a panel
    openPanel(panelId) {
        // Close current panel if open
        if (this.currentPanel) {
            document.getElementById(this.currentPanel).classList.remove('active');
        }
        
        // Open new panel
        const panel = document.getElementById(panelId);
        if (panel) {
            panel.classList.add('active');
            this.currentPanel = panelId;
            
            // Special panel initializations
            if (panelId === 'history') {
                this.renderPredictionHistory();
            } else if (panelId === 'health') {
                this.updateSystemHealthDisplay();
            }
        }
    }
    
    // Close the current panel
    closePanel() {
        if (this.currentPanel) {
            document.getElementById(this.currentPanel).classList.remove('active');
            this.currentPanel = null;
        }
    }
    
    // API Calls
    
    // Check system health
    async checkSystemHealth() {
        try {
            const response = await this.apiRequest(config.endpoints.health);
            this.systemStatus = response;
            this.updateSystemStatus(response.healthy);
            return response;
        } catch (error) {
            console.error('Error checking system health:', error);
            this.updateSystemStatus(false);
            return { healthy: false, error: error.message };
        }
    }
    
    // Get a demo API key
    async getDemoApiKey() {
        try {
            const response = await this.apiRequest(config.endpoints.demoKeys);
            if (response && response.key) {
                this.apiKey = response.key;
                localStorage.setItem(config.localStorageKeys.apiKey, response.key);
                console.log('Demo API key obtained');
                return response.key;
            }
        } catch (error) {
            console.error('Error getting demo API key:', error);
            return null;
        }
    }
    
    // Submit a prediction request
    async submitPrediction() {
        const resultContainer = document.getElementById('prediction-result');
        resultContainer.innerHTML = '<div class="alert alert-info"><i class="fas fa-spinner fa-spin"></i> Generating prediction...</div>';
        resultContainer.classList.remove('hidden');
        
        const form = document.getElementById('prediction-form');
        const symbol = form.querySelector('#symbol').value;
        const timeframe = form.querySelector('#timeframe').value;
        
        // Optional advanced params
        const model = this.mode === 'advanced' && form.querySelector('#model') ? form.querySelector('#model').value : 'default';
        const horizon = this.mode === 'advanced' && form.querySelector('#horizon') ? form.querySelector('#horizon').value : 'default';
        
        try {
            const payload = {
                symbol: symbol,
                timeframe: timeframe,
                model: model !== 'default' ? model : undefined,
                horizon: horizon !== 'default' ? horizon : undefined
            };
            
            const response = await this.apiRequest(config.endpoints.predict, 'POST', payload);
            
            // Save to history
            this.savePrediction(response, symbol, timeframe);
            
            // Display result
            this.displayPredictionResult(response, symbol, timeframe);
            
        } catch (error) {
            console.error('Error submitting prediction:', error);
            resultContainer.innerHTML = `
                <div class="alert alert-error">
                    <i class="fas fa-exclamation-circle"></i>
                    <div>
                        <h4>Error</h4>
                        <p>${error.message || 'Failed to generate prediction'}</p>
                    </div>
                </div>
            `;
        }
    }
    
    // Helper function to make API requests
    async apiRequest(endpoint, method = 'GET', data = null) {
        const url = `${config.apiBaseUrl}${endpoint}`;
        
        const headers = {
            'Content-Type': 'application/json'
        };
        
        // Add API key if available
        if (this.apiKey) {
            headers['Authorization'] = `Bearer ${this.apiKey}`;
        }
        
        const options = {
            method,
            headers,
            credentials: 'include'
        };
        
        if (data && (method === 'POST' || method === 'PUT')) {
            options.body = JSON.stringify(data);
        }
        
        try {
            const response = await fetch(url, options);
            
            if (!response.ok) {
                throw new Error(`API Error: ${response.status} ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error(`API Request Error (${endpoint}):`, error);
            throw error;
        }
    }
    
    // UI Updates
    
    // Update system status indicator
    updateSystemStatus(healthy) {
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.querySelector('.status-text');
        
        if (statusDot && statusText) {
            if (healthy) {
                statusDot.className = 'status-dot healthy';
                statusText.textContent = 'System Online';
            } else {
                statusDot.className = 'status-dot error';
                statusText.textContent = 'System Offline';
            }
        }
    }
    
    // Update system health display in the health panel
    updateSystemHealthDisplay() {
        const healthPanel = document.getElementById('health');
        if (!healthPanel) return;
        
        const statusCards = healthPanel.querySelector('.status-cards');
        if (!statusCards) return;
        
        // Clear existing cards
        statusCards.innerHTML = '';
        
        // Default components if API doesn't provide them
        const components = this.systemStatus.components || {
            'API Service': this.systemStatus.healthy ? 'Online' : 'Offline',
            'Data Pipeline': this.systemStatus.healthy ? 'Running' : 'Stopped',
            'Model Service': this.systemStatus.healthy ? 'Operating' : 'Unavailable'
        };
        
        // Create status cards for each component
        Object.entries(components).forEach(([component, status]) => {
            const isHealthy = typeof status === 'object' ? status.status === 'healthy' : 
                              typeof status === 'string' ? status.toLowerCase().includes('online') || 
                              status.toLowerCase().includes('running') || 
                              status.toLowerCase().includes('operating') : false;
            
            const statusValue = typeof status === 'object' ? status.status : status;
            
            const card = document.createElement('div');
            card.className = `status-card ${isHealthy ? 'healthy' : 'error'}`;
            card.innerHTML = `
                <div class="status-icon"><i class="fas ${this.getIconForComponent(component)}"></i></div>
                <div class="status-details">
                    <h4>${component}</h4>
                    <p>${statusValue}</p>
                </div>
            `;
            
            statusCards.appendChild(card);
        });
    }
    
    // Get appropriate icon for component
    getIconForComponent(component) {
        const componentLower = component.toLowerCase();
        
        if (componentLower.includes('api') || componentLower.includes('service')) return 'fa-server';
        if (componentLower.includes('data') || componentLower.includes('pipeline')) return 'fa-database';
        if (componentLower.includes('model')) return 'fa-brain';
        if (componentLower.includes('monitor')) return 'fa-chart-line';
        if (componentLower.includes('test')) return 'fa-flask';
        
        return 'fa-cog';
    }
    
    // Display prediction result
    displayPredictionResult(result, symbol, timeframe) {
        const resultContainer = document.getElementById('prediction-result');
        if (!resultContainer) return;
        
        // Format prediction direction with icon and color
        const direction = result.prediction > 0 ? 'up' : 'down';
        const directionIcon = direction === 'up' ? 'fa-arrow-up' : 'fa-arrow-down';
        const directionClass = direction === 'up' ? 'success' : 'danger';
        
        // Format the prediction timestamp
        const predictionDate = new Date();
        const formattedDate = predictionDate.toLocaleString();
        
        // Create the prediction result HTML
        resultContainer.innerHTML = `
            <div class="prediction-result">
                <div class="prediction-header">
                    <h3>${symbol} ${timeframe} Prediction</h3>
                    <span>${formattedDate}</span>
                </div>
                
                <div class="prediction-summary">
                    <div class="prediction-detail">
                        <div class="prediction-label">Prediction</div>
                        <div class="prediction-value ${direction}">
                            <i class="fas ${directionIcon}"></i> ${direction.toUpperCase()}
                        </div>
                    </div>
                    
                    <div class="prediction-detail">
                        <div class="prediction-label">Confidence</div>
                        <div class="prediction-value">${(Math.abs(result.prediction) * 100).toFixed(2)}%</div>
                    </div>
                    
                    <div class="prediction-detail">
                        <div class="prediction-label">Time Horizon</div>
                        <div class="prediction-value">${result.horizon || timeframe}</div>
                    </div>
                    
                    <div class="prediction-detail">
                        <div class="prediction-label">Model</div>
                        <div class="prediction-value">${result.model || 'Default'}</div>
                    </div>
                </div>
                
                <div class="alert alert-${directionClass}">
                    <i class="fas ${directionIcon}"></i>
                    <div>
                        <h4>Prediction Summary</h4>
                        <p>
                            Our model predicts that ${symbol} will likely go 
                            <strong>${direction}</strong> in the next ${timeframe} with 
                            ${(Math.abs(result.prediction) * 100).toFixed(2)}% confidence.
                        </p>
                    </div>
                </div>
                
                <div class="prediction-chart">
                    <canvas id="prediction-chart"></canvas>
                </div>
            </div>
        `;
        
        // Initialize chart
        this.initPredictionChart(result, symbol);
    }
    
    // Initialize prediction chart
    initPredictionChart(result, symbol) {
        const ctx = document.getElementById('prediction-chart');
        if (!ctx) return;
        
        // Prepare chart data
        const labels = [];
        const data = [];
        
        // Generate some sample data for visualization
        // In a real app, this would come from the API
        const baseValue = 100;
        const now = new Date();
        
        // Historical data (past 7 days)
        for (let i = 7; i >= 1; i--) {
            const date = new Date(now);
            date.setDate(date.getDate() - i);
            labels.push(date.toLocaleDateString());
            
            // Generate some realistic looking historical data
            const randomFactor = 0.02; // 2% daily change max
            const change = (Math.random() * 2 - 1) * randomFactor;
            if (i === 7) {
                data.push(baseValue);
            } else {
                data.push(data[data.length - 1] * (1 + change));
            }
        }
        
        // Current day
        labels.push('Today');
        data.push(data[data.length - 1]);
        
        // Prediction (next day)
        labels.push('Prediction');
        const predictedChange = result.prediction * 0.03; // Scale the prediction for visualization
        data.push(data[data.length - 1] * (1 + predictedChange));
        
        // Destroy existing chart if it exists
        if (this.charts.prediction) {
            this.charts.prediction.destroy();
        }
        
        // Create new chart
        this.charts.prediction = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: symbol + ' Price',
                    data: data,
                    borderColor: result.prediction > 0 ? '#10b981' : '#ef4444',
                    backgroundColor: result.prediction > 0 ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                    borderWidth: 2,
                    pointRadius: 3,
                    tension: 0.2,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        }
                    },
                    x: {
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        }
                    }
                }
            }
        });
    }
    
    // Prediction History Management
    
    // Save prediction to history
    savePrediction(result, symbol, timeframe) {
        const prediction = {
            id: Date.now().toString(),
            date: new Date().toISOString(),
            symbol: symbol,
            timeframe: timeframe,
            prediction: result.prediction,
            confidence: Math.abs(result.prediction),
            model: result.model || 'Default',
            horizon: result.horizon || timeframe
        };
        
        this.predictionHistory.unshift(prediction);
        
        // Keep history at a reasonable size (max 50 entries)
        if (this.predictionHistory.length > 50) {
            this.predictionHistory = this.predictionHistory.slice(0, 50);
        }
        
        // Save to local storage
        localStorage.setItem(config.localStorageKeys.predictionHistory, JSON.stringify(this.predictionHistory));
    }
    
    // Load prediction history from local storage
    loadPredictionHistory() {
        const history = localStorage.getItem(config.localStorageKeys.predictionHistory);
        if (history) {
            this.predictionHistory = JSON.parse(history);
        }
    }
    
    // Render prediction history in the history panel
    renderPredictionHistory() {
        const historyTable = document.querySelector('#history-table tbody');
        if (!historyTable) return;
        
        // Clear existing rows
        historyTable.innerHTML = '';
        
        if (this.predictionHistory.length === 0) {
            const row = document.createElement('tr');
            row.innerHTML = '<td colspan="6" class="text-center">No prediction history yet</td>';
            historyTable.appendChild(row);
            return;
        }
        
        // Add history entries
        this.predictionHistory.forEach(entry => {
            const row = document.createElement('tr');
            const date = new Date(entry.date).toLocaleString();
            const direction = entry.prediction > 0 ? 'up' : 'down';
            const directionIcon = direction === 'up' ? 'fa-arrow-up' : 'fa-arrow-down';
            const directionClass = direction === 'up' ? 'success' : 'danger';
            
            row.innerHTML = `
                <td>${date}</td>
                <td>${entry.symbol}</td>
                <td>${entry.timeframe}</td>
                <td>
                    <span class="prediction-value ${direction}">
                        <i class="fas ${directionIcon}"></i> 
                        ${(Math.abs(entry.prediction) * 100).toFixed(2)}%
                    </span>
                </td>
                <td>Pending</td>
                <td>-</td>
            `;
            
            historyTable.appendChild(row);
        });
        
        // Initialize history chart
        this.initHistoryChart();
    }
    
    // Initialize history chart
    initHistoryChart() {
        const ctx = document.getElementById('history-chart');
        if (!ctx || this.predictionHistory.length === 0) return;
        
        // Prepare data
        const labels = [];
        const accuracyData = [];
        
        // Get last 10 predictions (from oldest to newest)
        const recentHistory = [...this.predictionHistory]
            .reverse()
            .slice(0, 10)
            .reverse();
        
        recentHistory.forEach(entry => {
            labels.push(new Date(entry.date).toLocaleDateString());
            
            // In a real app, accuracy would be calculated based on actual outcomes
            // Here we're using random data for demonstration
            const randomAccuracy = Math.random() * 0.4 + 0.6; // Random accuracy between 60% and 100%
            accuracyData.push(randomAccuracy * 100);
        });
        
        // Destroy existing chart if it exists
        if (this.charts.history) {
            this.charts.history.destroy();
        }
        
        // Create new chart
        this.charts.history = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Prediction Accuracy (%)',
                    data: accuracyData,
                    backgroundColor: 'rgba(79, 70, 229, 0.7)',
                    borderColor: 'rgba(79, 70, 229, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Accuracy: ${context.parsed.y.toFixed(1)}%`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Accuracy (%)'
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        }
                    },
                    x: {
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        }
                    }
                }
            }
        });
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.mlta = new MLTAApp();
});
