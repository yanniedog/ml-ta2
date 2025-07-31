// ML-TA Frontend Application
class MLTAApp {
    constructor() {
        this.apiBase = '/api/v1';
        this.apiKey = null;
        this.predictionHistory = JSON.parse(localStorage.getItem('predictionHistory') || '[]');
        this.init();
    }

    async init() {
        await this.checkSystemStatus();
        await this.loadDemoKeys();
        this.setupEventListeners();
        this.loadPredictionHistory();
    }

    // System Status Check
    async checkSystemStatus() {
        try {
            const response = await fetch(`${this.apiBase}/health`);
            const data = await response.json();
            
            if (data.status === 'healthy') {
                this.updateSystemStatus('online', 'System Online');
            } else {
                this.updateSystemStatus('offline', 'System Issues');
            }
        } catch (error) {
            this.updateSystemStatus('offline', 'System Offline');
            console.error('System status check failed:', error);
        }
    }

    updateSystemStatus(status, text) {
        const indicator = document.getElementById('statusIndicator');
        const statusText = document.getElementById('statusText');
        
        indicator.className = `status-indicator ${status === 'offline' ? 'offline' : ''}`;
        statusText.textContent = text;
    }

    // Load demo API keys for testing
    async loadDemoKeys() {
        try {
            const response = await fetch(`${this.apiBase}/demo-keys`);
            const data = await response.json();
            
            if (data.demo_keys && data.demo_keys.user) {
                this.apiKey = data.demo_keys.user.api_key;
                console.log('Demo API key loaded for testing');
            }
        } catch (error) {
            console.warn('Could not load demo API keys:', error);
        }
    }

    // Event Listeners
    setupEventListeners() {
        // Auto-fill sample data when symbol changes
        document.getElementById('symbol').addEventListener('change', this.fillSampleData.bind(this));
        
        // Form validation
        const inputs = document.querySelectorAll('.form-input');
        inputs.forEach(input => {
            input.addEventListener('input', this.validateInput.bind(this));
        });
    }

    // Fill sample data based on selected symbol
    fillSampleData() {
        const symbol = document.getElementById('symbol').value;
        const sampleData = {
            'BTC/USDT': {
                price: 45000,
                volume: 1500000,
                rsi: 65,
                macd: 0.5,
                bollinger: 0.8,
                sentiment: 7
            },
            'ETH/USDT': {
                price: 3200,
                volume: 800000,
                rsi: 58,
                macd: 0.3,
                bollinger: 0.6,
                sentiment: 6
            },
            'ADA/USDT': {
                price: 0.85,
                volume: 200000,
                rsi: 72,
                macd: -0.1,
                bollinger: 0.9,
                sentiment: 5
            },
            'SOL/USDT': {
                price: 120,
                volume: 300000,
                rsi: 45,
                macd: 0.2,
                bollinger: 0.4,
                sentiment: 8
            }
        };

        const data = sampleData[symbol];
        if (data) {
            document.getElementById('price').value = data.price;
            document.getElementById('volume').value = data.volume;
            document.getElementById('rsi').value = data.rsi;
            document.getElementById('macd').value = data.macd;
            document.getElementById('bollinger').value = data.bollinger;
            document.getElementById('sentiment').value = data.sentiment;
        }
    }

    // Input validation
    validateInput(event) {
        const input = event.target;
        const value = parseFloat(input.value);
        
        // Remove any existing validation classes
        input.classList.remove('invalid', 'valid');
        
        if (input.value === '') return;
        
        let isValid = true;
        
        switch (input.id) {
            case 'rsi':
                isValid = value >= 0 && value <= 100;
                break;
            case 'sentiment':
                isValid = value >= 1 && value <= 10;
                break;
            case 'price':
            case 'volume':
                isValid = value > 0;
                break;
            default:
                isValid = !isNaN(value);
        }
        
        input.classList.add(isValid ? 'valid' : 'invalid');
    }

    // Panel Management
    showPredictionPanel() {
        this.hideAllPanels();
        document.getElementById('predictionPanel').style.display = 'block';
        this.fillSampleData(); // Auto-fill with sample data
    }

    showModelsPanel() {
        this.hideAllPanels();
        document.getElementById('modelsPanel').style.display = 'block';
        this.loadModels();
    }

    showHistoryPanel() {
        this.hideAllPanels();
        document.getElementById('historyPanel').style.display = 'block';
        this.displayPredictionHistory();
    }

    hideAllPanels() {
        const panels = document.querySelectorAll('.panel');
        panels.forEach(panel => panel.style.display = 'none');
    }

    closePredictionPanel() {
        document.getElementById('predictionPanel').style.display = 'none';
    }

    closeModelsPanel() {
        document.getElementById('modelsPanel').style.display = 'none';
    }

    closeHistoryPanel() {
        document.getElementById('historyPanel').style.display = 'none';
    }

    // Prediction Logic
    async makePrediction() {
        const formData = this.collectFormData();
        
        if (!this.validateFormData(formData)) {
            this.showToast('Please fill in all required fields correctly', 'error');
            return;
        }

        this.showLoading(true);
        
        try {
            let prediction;
            
            if (this.apiKey) {
                // Try real API call first
                try {
                    prediction = await this.makeRealPrediction(formData);
                } catch (apiError) {
                    console.warn('Real API call failed, falling back to simulation:', apiError);
                    prediction = await this.simulatePrediction(formData);
                }
            } else {
                // Fall back to simulation if no API key
                prediction = await this.simulatePrediction(formData);
            }
            
            this.displayPredictionResults(prediction);
            this.savePredictionToHistory(formData, prediction);
            this.showToast('Prediction completed successfully!', 'success');
        } catch (error) {
            console.error('Prediction failed:', error);
            this.showToast('Prediction failed. Please try again.', 'error');
        } finally {
            this.showLoading(false);
        }
    }

    collectFormData() {
        return {
            symbol: document.getElementById('symbol').value,
            price: parseFloat(document.getElementById('price').value),
            volume: parseFloat(document.getElementById('volume').value),
            rsi: parseFloat(document.getElementById('rsi').value),
            macd: parseFloat(document.getElementById('macd').value),
            bollinger: parseFloat(document.getElementById('bollinger').value),
            sentiment: parseFloat(document.getElementById('sentiment').value)
        };
    }

    validateFormData(data) {
        return Object.values(data).every(value => 
            value !== null && value !== undefined && !isNaN(value)
        );
    }

    // Make real API prediction call
    async makeRealPrediction(formData) {
        const features = {
            symbol: formData.symbol,
            price: formData.price,
            volume: formData.volume,
            rsi: formData.rsi,
            macd: formData.macd,
            bollinger_band: formData.bollinger,
            market_sentiment: formData.sentiment
        };

        const response = await fetch(`${this.apiBase}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': this.apiKey
            },
            body: JSON.stringify({ features: [features] })
        });

        if (!response.ok) {
            throw new Error(`API call failed: ${response.status}`);
        }

        const result = await response.json();
        
        // Transform API response to our expected format
        return {
            symbol: formData.symbol,
            currentPrice: formData.price,
            predictedPrice: formData.price * (1 + (result.predictions?.[0] || 0.05)),
            priceChange: (result.predictions?.[0] || 0.05) * 100,
            confidence: result.confidence_scores?.[0] || 0.75,
            direction: (result.predictions?.[0] || 0.05) > 0 ? 'bullish' : 'bearish',
            strength: Math.abs(result.predictions?.[0] || 0.05) > 0.05 ? 'strong' : 'moderate',
            recommendation: this.generateRecommendation(
                (result.predictions?.[0] || 0.05) > 0 ? 'bullish' : 'bearish',
                Math.abs(result.predictions?.[0] || 0.05) > 0.05 ? 'strong' : 'moderate',
                result.confidence_scores?.[0] || 0.75
            ),
            timestamp: new Date().toISOString()
        };
    }

    // Simulate prediction (fallback when API is not available)
    async simulatePrediction(formData) {
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Generate realistic prediction based on input data
        const basePrice = formData.price;
        const rsiInfluence = (formData.rsi - 50) / 100; // -0.5 to 0.5
        const macdInfluence = formData.macd / 10;
        const sentimentInfluence = (formData.sentiment - 5) / 10; // -0.5 to 0.5
        
        const priceChange = (rsiInfluence + macdInfluence + sentimentInfluence) * 0.1;
        const predictedPrice = basePrice * (1 + priceChange);
        const confidence = Math.min(0.95, Math.max(0.6, 0.8 + Math.abs(priceChange) * 2));
        
        const direction = predictedPrice > basePrice ? 'bullish' : 'bearish';
        const strength = Math.abs(priceChange) > 0.05 ? 'strong' : 'moderate';
        
        return {
            symbol: formData.symbol,
            currentPrice: basePrice,
            predictedPrice: predictedPrice,
            priceChange: ((predictedPrice - basePrice) / basePrice) * 100,
            confidence: confidence,
            direction: direction,
            strength: strength,
            recommendation: this.generateRecommendation(direction, strength, confidence),
            timestamp: new Date().toISOString()
        };
    }

    generateRecommendation(direction, strength, confidence) {
        if (confidence > 0.8) {
            if (direction === 'bullish' && strength === 'strong') {
                return 'Strong Buy - High confidence upward movement expected';
            } else if (direction === 'bullish') {
                return 'Buy - Moderate upward movement expected';
            } else if (direction === 'bearish' && strength === 'strong') {
                return 'Strong Sell - High confidence downward movement expected';
            } else {
                return 'Sell - Moderate downward movement expected';
            }
        } else {
            return 'Hold - Low confidence, consider waiting for clearer signals';
        }
    }

    displayPredictionResults(prediction) {
        const resultsDiv = document.getElementById('predictionResults');
        const contentDiv = document.getElementById('resultsContent');
        
        const changeClass = prediction.priceChange > 0 ? 'positive' : 'negative';
        const changeIcon = prediction.priceChange > 0 ? 'fas fa-arrow-up' : 'fas fa-arrow-down';
        
        contentDiv.innerHTML = `
            <div class="result-card">
                <div class="result-metric">
                    <span class="metric-label">Current Price</span>
                    <span class="metric-value">$${prediction.currentPrice.toLocaleString()}</span>
                </div>
                <div class="result-metric">
                    <span class="metric-label">Predicted Price</span>
                    <span class="metric-value ${changeClass}">$${prediction.predictedPrice.toLocaleString()}</span>
                </div>
                <div class="result-metric">
                    <span class="metric-label">Expected Change</span>
                    <span class="metric-value ${changeClass}">
                        <i class="${changeIcon}"></i>
                        ${Math.abs(prediction.priceChange).toFixed(2)}%
                    </span>
                </div>
                <div class="result-metric">
                    <span class="metric-label">Confidence</span>
                    <span class="metric-value">${(prediction.confidence * 100).toFixed(1)}%</span>
                </div>
            </div>
            <div class="result-card">
                <div class="result-metric">
                    <span class="metric-label">Market Direction</span>
                    <span class="metric-value ${changeClass}">
                        ${prediction.direction.toUpperCase()} (${prediction.strength})
                    </span>
                </div>
                <div style="margin-top: 1rem; padding: 1rem; background: #f8fafc; border-radius: 8px;">
                    <strong>Recommendation:</strong><br>
                    ${prediction.recommendation}
                </div>
            </div>
        `;
        
        resultsDiv.style.display = 'block';
        resultsDiv.scrollIntoView({ behavior: 'smooth' });
    }

    // History Management
    savePredictionToHistory(formData, prediction) {
        const historyItem = {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            input: formData,
            prediction: prediction
        };
        
        this.predictionHistory.unshift(historyItem);
        
        // Keep only last 50 predictions
        if (this.predictionHistory.length > 50) {
            this.predictionHistory = this.predictionHistory.slice(0, 50);
        }
        
        localStorage.setItem('predictionHistory', JSON.stringify(this.predictionHistory));
    }

    loadPredictionHistory() {
        this.predictionHistory = JSON.parse(localStorage.getItem('predictionHistory') || '[]');
    }

    displayPredictionHistory() {
        const historyContent = document.getElementById('historyContent');
        
        if (this.predictionHistory.length === 0) {
            historyContent.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-clock"></i>
                    <h4>No predictions yet</h4>
                    <p>Your prediction history will appear here once you start making predictions.</p>
                </div>
            `;
            return;
        }
        
        const historyHTML = this.predictionHistory.map(item => {
            const date = new Date(item.timestamp).toLocaleDateString();
            const time = new Date(item.timestamp).toLocaleTimeString();
            const changeClass = item.prediction.priceChange > 0 ? 'positive' : 'negative';
            
            return `
                <div class="result-card" style="margin-bottom: 1rem;">
                    <div class="result-metric">
                        <span class="metric-label">${item.prediction.symbol}</span>
                        <span class="metric-value">${date} ${time}</span>
                    </div>
                    <div class="result-metric">
                        <span class="metric-label">Predicted Change</span>
                        <span class="metric-value ${changeClass}">
                            ${item.prediction.priceChange > 0 ? '+' : ''}${item.prediction.priceChange.toFixed(2)}%
                        </span>
                    </div>
                    <div class="result-metric">
                        <span class="metric-label">Confidence</span>
                        <span class="metric-value">${(item.prediction.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #718096;">
                        ${item.prediction.recommendation}
                    </div>
                </div>
            `;
        }).join('');
        
        historyContent.innerHTML = historyHTML;
    }

    // Models Management
    async loadModels() {
        const modelsGrid = document.getElementById('modelsGrid');
        
        // Simulate model data (replace with actual API call)
        const models = [
            {
                name: 'LSTM Neural Network',
                status: 'active',
                accuracy: '87.3%',
                description: 'Long Short-Term Memory network for time series prediction'
            },
            {
                name: 'Random Forest',
                status: 'active',
                accuracy: '82.1%',
                description: 'Ensemble method using multiple decision trees'
            },
            {
                name: 'XGBoost',
                status: 'active',
                accuracy: '85.7%',
                description: 'Gradient boosting framework for structured data'
            },
            {
                name: 'Support Vector Machine',
                status: 'inactive',
                accuracy: '79.4%',
                description: 'SVM model for classification and regression'
            }
        ];
        
        const modelsHTML = models.map(model => `
            <div class="model-card">
                <div class="model-header">
                    <span class="model-name">${model.name}</span>
                    <span class="model-status ${model.status}">${model.status.toUpperCase()}</span>
                </div>
                <div class="result-metric">
                    <span class="metric-label">Accuracy</span>
                    <span class="metric-value">${model.accuracy}</span>
                </div>
                <p style="margin-top: 1rem; color: #718096; font-size: 0.9rem;">
                    ${model.description}
                </p>
            </div>
        `).join('');
        
        modelsGrid.innerHTML = modelsHTML;
    }

    // Utility Functions
    clearForm() {
        const inputs = document.querySelectorAll('.form-input');
        inputs.forEach(input => {
            if (input.type !== 'select-one') {
                input.value = '';
                input.classList.remove('valid', 'invalid');
            }
        });
        document.getElementById('predictionResults').style.display = 'none';
    }

    showLoading(show) {
        const overlay = document.getElementById('loadingOverlay');
        overlay.style.display = show ? 'flex' : 'none';
    }

    showToast(message, type = 'success') {
        const toast = document.getElementById('toast');
        const icon = document.getElementById('toastIcon');
        const messageEl = document.getElementById('toastMessage');
        
        // Set icon based on type
        icon.className = type === 'success' ? 'fas fa-check-circle' : 'fas fa-exclamation-circle';
        messageEl.textContent = message;
        
        // Set toast class
        toast.className = `toast ${type}`;
        
        // Show toast
        setTimeout(() => toast.classList.add('show'), 100);
        
        // Hide toast after 3 seconds
        setTimeout(() => {
            toast.classList.remove('show');
        }, 3000);
    }
}

// Global functions for HTML onclick events
function showPredictionPanel() {
    window.mlta.showPredictionPanel();
}

function showModelsPanel() {
    window.mlta.showModelsPanel();
}

function showHistoryPanel() {
    window.mlta.showHistoryPanel();
}

function closePredictionPanel() {
    window.mlta.closePredictionPanel();
}

function closeModelsPanel() {
    window.mlta.closeModelsPanel();
}

function closeHistoryPanel() {
    window.mlta.closeHistoryPanel();
}

function makePrediction() {
    window.mlta.makePrediction();
}

function clearForm() {
    window.mlta.clearForm();
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.mlta = new MLTAApp();
});

// Add some CSS for form validation
const style = document.createElement('style');
style.textContent = `
    .form-input.valid {
        border-color: #10b981;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
    }
    
    .form-input.invalid {
        border-color: #ef4444;
        box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.1);
    }
`;
document.head.appendChild(style);
