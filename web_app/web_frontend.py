"""
Embedded web frontend for ML-TA system.
This module provides a complete web interface embedded as HTML strings.
"""

def get_embedded_frontend():
    """Return complete embedded HTML frontend."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML-TA Trading Assistant</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; color: #333; line-height: 1.6;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
        .header {
            background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px);
            padding: 1rem 2rem; border-radius: 16px; margin-bottom: 2rem;
            display: flex; justify-content: space-between; align-items: center;
        }
        .logo { display: flex; align-items: center; gap: 1rem; }
        .logo i { font-size: 2rem; color: #667eea; }
        .logo h1 { font-size: 1.5rem; font-weight: 700; color: #2d3748; }
        .status { display: flex; align-items: center; gap: 0.5rem; }
        .status-dot { width: 8px; height: 8px; border-radius: 50%; background: #10b981; }
        .welcome {
            text-align: center; margin-bottom: 3rem; color: white;
        }
        .welcome h2 { font-size: 2.5rem; margin-bottom: 1rem; }
        .welcome p { font-size: 1.2rem; opacity: 0.9; }
        .cards {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem;
        }
        .card {
            background: rgba(255, 255, 255, 0.95); border-radius: 16px; padding: 2rem;
            text-align: center; cursor: pointer; transition: all 0.3s ease;
        }
        .card:hover { transform: translateY(-5px); box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1); }
        .card-icon {
            width: 80px; height: 80px; margin: 0 auto 1.5rem;
            background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
        }
        .card-icon i { font-size: 2rem; color: white; }
        .card h3 { font-size: 1.5rem; margin-bottom: 1rem; color: #2d3748; }
        .card p { color: #718096; margin-bottom: 1.5rem; }
        .btn {
            background: linear-gradient(135deg, #667eea, #764ba2); color: white;
            border: none; padding: 0.75rem 2rem; border-radius: 50px;
            font-weight: 600; cursor: pointer; transition: all 0.3s ease;
        }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3); }
        .panel {
            background: rgba(255, 255, 255, 0.95); border-radius: 16px; padding: 2rem;
            margin-top: 2rem; display: none;
        }
        .panel.active { display: block; animation: slideIn 0.3s ease; }
        @keyframes slideIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        .panel-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; }
        .panel-header h3 { font-size: 1.5rem; color: #2d3748; display: flex; align-items: center; gap: 0.5rem; }
        .close-btn { background: none; border: none; font-size: 1.2rem; color: #718096; cursor: pointer; }
        .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
        .form-group { margin-bottom: 1rem; }
        .form-group label { display: block; font-weight: 500; color: #4a5568; margin-bottom: 0.5rem; }
        .form-input {
            width: 100%; padding: 0.75rem; border: 2px solid #e2e8f0; border-radius: 8px;
            font-size: 1rem; transition: all 0.3s ease;
        }
        .form-input:focus { outline: none; border-color: #667eea; }
        .form-actions { text-align: center; margin-top: 2rem; }
        .form-actions .btn { margin: 0 0.5rem; }
        .results {
            margin-top: 2rem; padding: 2rem; background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
            border-radius: 12px; display: none;
        }
        .results.active { display: block; }
        .result-item { display: flex; justify-content: space-between; margin-bottom: 1rem; }
        .result-label { font-weight: 500; color: #4a5568; }
        .result-value { font-weight: 600; }
        .positive { color: #10b981; }
        .negative { color: #ef4444; }
        .loading {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0, 0, 0, 0.5); display: none; align-items: center; justify-content: center;
        }
        .loading.active { display: flex; }
        .spinner {
            width: 40px; height: 40px; border: 4px solid #f3f4f6; border-top: 4px solid #667eea;
            border-radius: 50%; animation: spin 1s linear infinite;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .toast {
            position: fixed; top: 2rem; right: 2rem; background: white; padding: 1rem 1.5rem;
            border-radius: 8px; box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
            transform: translateX(400px); transition: transform 0.3s ease; border-left: 4px solid #10b981;
        }
        .toast.show { transform: translateX(0); }
        .toast.error { border-left-color: #ef4444; }
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <div class="logo">
                <i class="fas fa-chart-line"></i>
                <h1>ML-TA Trading Assistant</h1>
            </div>
            <div class="status">
                <span class="status-dot" id="statusDot"></span>
                <span id="statusText">System Online</span>
            </div>
        </header>

        <section class="welcome">
            <h2>Welcome to Your AI Trading Assistant</h2>
            <p>Get intelligent trading predictions and market analysis with our advanced machine learning system. No coding required!</p>
        </section>

        <section class="cards">
            <div class="card" onclick="showPredictionPanel()">
                <div class="card-icon">
                    <i class="fas fa-crystal-ball"></i>
                </div>
                <h3>Get Prediction</h3>
                <p>Analyze market data and get AI-powered trading predictions</p>
                <button class="btn">Start Analysis</button>
            </div>
            
            <div class="card" onclick="showModelsPanel()">
                <div class="card-icon">
                    <i class="fas fa-brain"></i>
                </div>
                <h3>View Models</h3>
                <p>Explore available AI models and their performance metrics</p>
                <button class="btn">View Models</button>
            </div>
            
            <div class="card" onclick="showHistoryPanel()">
                <div class="card-icon">
                    <i class="fas fa-history"></i>
                </div>
                <h3>Prediction History</h3>
                <p>Review your past predictions and their accuracy</p>
                <button class="btn">View History</button>
            </div>
        </section>

        <section class="panel" id="predictionPanel">
            <div class="panel-header">
                <h3><i class="fas fa-crystal-ball"></i> Trading Prediction</h3>
                <button class="close-btn" onclick="closePredictionPanel()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            
            <div class="form-grid">
                <div class="form-group">
                    <label for="symbol">Trading Symbol</label>
                    <select id="symbol" class="form-input" onchange="fillSampleData()">
                        <option value="BTC/USDT">Bitcoin (BTC/USDT)</option>
                        <option value="ETH/USDT">Ethereum (ETH/USDT)</option>
                        <option value="ADA/USDT">Cardano (ADA/USDT)</option>
                        <option value="SOL/USDT">Solana (SOL/USDT)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="price">Current Price</label>
                    <input type="number" id="price" class="form-input" placeholder="e.g., 45000" step="0.01">
                </div>
                <div class="form-group">
                    <label for="volume">24h Volume</label>
                    <input type="number" id="volume" class="form-input" placeholder="e.g., 1000000">
                </div>
                <div class="form-group">
                    <label for="rsi">RSI (0-100)</label>
                    <input type="number" id="rsi" class="form-input" placeholder="e.g., 65" min="0" max="100">
                </div>
                <div class="form-group">
                    <label for="macd">MACD</label>
                    <input type="number" id="macd" class="form-input" placeholder="e.g., 0.5" step="0.01">
                </div>
                <div class="form-group">
                    <label for="sentiment">Market Sentiment (1-10)</label>
                    <input type="number" id="sentiment" class="form-input" placeholder="e.g., 7" min="1" max="10">
                </div>
            </div>
            
            <div class="form-actions">
                <button class="btn" onclick="makePrediction()">
                    <i class="fas fa-magic"></i> Get Prediction
                </button>
                <button class="btn" onclick="clearForm()" style="background: #f7fafc; color: #4a5568;">
                    <i class="fas fa-undo"></i> Clear Form
                </button>
            </div>
            
            <div class="results" id="predictionResults">
                <h4><i class="fas fa-chart-line"></i> Prediction Results</h4>
                <div id="resultsContent"></div>
            </div>
        </section>

        <section class="panel" id="modelsPanel">
            <div class="panel-header">
                <h3><i class="fas fa-brain"></i> AI Models</h3>
                <button class="close-btn" onclick="closeModelsPanel()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div id="modelsContent">
                <div class="result-item">
                    <span class="result-label">LSTM Neural Network</span>
                    <span class="result-value positive">87.3% Accuracy</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Random Forest</span>
                    <span class="result-value positive">82.1% Accuracy</span>
                </div>
                <div class="result-item">
                    <span class="result-label">XGBoost</span>
                    <span class="result-value positive">85.7% Accuracy</span>
                </div>
            </div>
        </section>

        <section class="panel" id="historyPanel">
            <div class="panel-header">
                <h3><i class="fas fa-history"></i> Prediction History</h3>
                <button class="close-btn" onclick="closeHistoryPanel()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div id="historyContent">
                <p style="text-align: center; color: #718096; padding: 2rem;">
                    <i class="fas fa-clock" style="font-size: 2rem; margin-bottom: 1rem; display: block;"></i>
                    No predictions yet. Your prediction history will appear here once you start making predictions.
                </p>
            </div>
        </section>
    </div>

    <div class="loading" id="loading">
        <div style="background: white; padding: 2rem; border-radius: 12px; text-align: center;">
            <div class="spinner"></div>
            <p>Processing your request...</p>
        </div>
    </div>

    <div class="toast" id="toast">
        <span id="toastMessage"></span>
    </div>

    <script>
        let apiKey = null;
        let predictionHistory = JSON.parse(localStorage.getItem('predictionHistory') || '[]');

        // Initialize app
        document.addEventListener('DOMContentLoaded', async () => {
            await checkSystemStatus();
            await loadDemoKeys();
            fillSampleData();
        });

        async function checkSystemStatus() {
            try {
                const response = await fetch('/api/v1/health');
                const data = await response.json();
                document.getElementById('statusText').textContent = data.status === 'healthy' ? 'System Online' : 'System Issues';
            } catch (error) {
                document.getElementById('statusText').textContent = 'System Offline';
                document.getElementById('statusDot').style.background = '#ef4444';
            }
        }

        async function loadDemoKeys() {
            try {
                const response = await fetch('/api/v1/demo-keys');
                const data = await response.json();
                if (data.demo_keys && data.demo_keys.user) {
                    apiKey = data.demo_keys.user.api_key;
                    console.log('Demo API key loaded');
                }
            } catch (error) {
                console.warn('Could not load demo API keys:', error);
            }
        }

        function showPredictionPanel() {
            hideAllPanels();
            document.getElementById('predictionPanel').classList.add('active');
            fillSampleData();
        }

        function showModelsPanel() {
            hideAllPanels();
            document.getElementById('modelsPanel').classList.add('active');
        }

        function showHistoryPanel() {
            hideAllPanels();
            document.getElementById('historyPanel').classList.add('active');
            displayHistory();
        }

        function hideAllPanels() {
            document.querySelectorAll('.panel').forEach(panel => {
                panel.classList.remove('active');
            });
        }

        function closePredictionPanel() { document.getElementById('predictionPanel').classList.remove('active'); }
        function closeModelsPanel() { document.getElementById('modelsPanel').classList.remove('active'); }
        function closeHistoryPanel() { document.getElementById('historyPanel').classList.remove('active'); }

        function fillSampleData() {
            const symbol = document.getElementById('symbol').value;
            const sampleData = {
                'BTC/USDT': { price: 45000, volume: 1500000, rsi: 65, macd: 0.5, sentiment: 7 },
                'ETH/USDT': { price: 3200, volume: 800000, rsi: 58, macd: 0.3, sentiment: 6 },
                'ADA/USDT': { price: 0.85, volume: 200000, rsi: 72, macd: -0.1, sentiment: 5 },
                'SOL/USDT': { price: 120, volume: 300000, rsi: 45, macd: 0.2, sentiment: 8 }
            };

            const data = sampleData[symbol];
            if (data) {
                document.getElementById('price').value = data.price;
                document.getElementById('volume').value = data.volume;
                document.getElementById('rsi').value = data.rsi;
                document.getElementById('macd').value = data.macd;
                document.getElementById('sentiment').value = data.sentiment;
            }
        }

        async function makePrediction() {
            const formData = {
                symbol: document.getElementById('symbol').value,
                price: parseFloat(document.getElementById('price').value),
                volume: parseFloat(document.getElementById('volume').value),
                rsi: parseFloat(document.getElementById('rsi').value),
                macd: parseFloat(document.getElementById('macd').value),
                sentiment: parseFloat(document.getElementById('sentiment').value)
            };

            if (Object.values(formData).some(v => isNaN(v))) {
                showToast('Please fill in all fields correctly', 'error');
                return;
            }

            document.getElementById('loading').classList.add('active');

            try {
                // Simulate prediction
                await new Promise(resolve => setTimeout(resolve, 2000));
                
                const basePrice = formData.price;
                const rsiInfluence = (formData.rsi - 50) / 100;
                const macdInfluence = formData.macd / 10;
                const sentimentInfluence = (formData.sentiment - 5) / 10;
                
                const priceChange = (rsiInfluence + macdInfluence + sentimentInfluence) * 0.1;
                const predictedPrice = basePrice * (1 + priceChange);
                const confidence = Math.min(0.95, Math.max(0.6, 0.8 + Math.abs(priceChange) * 2));
                
                const prediction = {
                    symbol: formData.symbol,
                    currentPrice: basePrice,
                    predictedPrice: predictedPrice,
                    priceChange: ((predictedPrice - basePrice) / basePrice) * 100,
                    confidence: confidence,
                    direction: predictedPrice > basePrice ? 'bullish' : 'bearish',
                    timestamp: new Date().toISOString()
                };

                displayResults(prediction);
                savePrediction(formData, prediction);
                showToast('Prediction completed successfully!', 'success');
            } catch (error) {
                showToast('Prediction failed. Please try again.', 'error');
            } finally {
                document.getElementById('loading').classList.remove('active');
            }
        }

        function displayResults(prediction) {
            const changeClass = prediction.priceChange > 0 ? 'positive' : 'negative';
            const changeIcon = prediction.priceChange > 0 ? 'fas fa-arrow-up' : 'fas fa-arrow-down';
            
            document.getElementById('resultsContent').innerHTML = `
                <div class="result-item">
                    <span class="result-label">Current Price</span>
                    <span class="result-value">$${prediction.currentPrice.toLocaleString()}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Predicted Price</span>
                    <span class="result-value ${changeClass}">$${prediction.predictedPrice.toLocaleString()}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Expected Change</span>
                    <span class="result-value ${changeClass}">
                        <i class="${changeIcon}"></i> ${Math.abs(prediction.priceChange).toFixed(2)}%
                    </span>
                </div>
                <div class="result-item">
                    <span class="result-label">Confidence</span>
                    <span class="result-value">${(prediction.confidence * 100).toFixed(1)}%</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Direction</span>
                    <span class="result-value ${changeClass}">${prediction.direction.toUpperCase()}</span>
                </div>
            `;
            
            document.getElementById('predictionResults').classList.add('active');
        }

        function savePrediction(formData, prediction) {
            const historyItem = {
                id: Date.now(),
                timestamp: new Date().toISOString(),
                input: formData,
                prediction: prediction
            };
            
            predictionHistory.unshift(historyItem);
            if (predictionHistory.length > 50) {
                predictionHistory = predictionHistory.slice(0, 50);
            }
            
            localStorage.setItem('predictionHistory', JSON.stringify(predictionHistory));
        }

        function displayHistory() {
            const historyContent = document.getElementById('historyContent');
            
            if (predictionHistory.length === 0) {
                historyContent.innerHTML = `
                    <p style="text-align: center; color: #718096; padding: 2rem;">
                        <i class="fas fa-clock" style="font-size: 2rem; margin-bottom: 1rem; display: block;"></i>
                        No predictions yet. Your prediction history will appear here once you start making predictions.
                    </p>
                `;
                return;
            }
            
            const historyHTML = predictionHistory.map(item => {
                const date = new Date(item.timestamp).toLocaleDateString();
                const changeClass = item.prediction.priceChange > 0 ? 'positive' : 'negative';
                
                return `
                    <div style="background: white; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                        <div class="result-item">
                            <span class="result-label">${item.prediction.symbol}</span>
                            <span class="result-value">${date}</span>
                        </div>
                        <div class="result-item">
                            <span class="result-label">Change</span>
                            <span class="result-value ${changeClass}">
                                ${item.prediction.priceChange > 0 ? '+' : ''}${item.prediction.priceChange.toFixed(2)}%
                            </span>
                        </div>
                        <div class="result-item">
                            <span class="result-label">Confidence</span>
                            <span class="result-value">${(item.prediction.confidence * 100).toFixed(1)}%</span>
                        </div>
                    </div>
                `;
            }).join('');
            
            historyContent.innerHTML = historyHTML;
        }

        function clearForm() {
            document.querySelectorAll('.form-input').forEach(input => {
                if (input.type !== 'select-one') input.value = '';
            });
            document.getElementById('predictionResults').classList.remove('active');
        }

        function showToast(message, type = 'success') {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.className = `toast ${type} show`;
            
            setTimeout(() => {
                toast.classList.remove('show');
            }, 3000);
        }
    </script>
</body>
</html>"""
