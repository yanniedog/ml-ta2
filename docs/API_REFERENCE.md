# ML-TA API Documentation

## Base URL
http://localhost:8000/api/v1

## Authentication
API Key required for production. Development mode: no auth.

## Endpoints

### Health Check
```
GET /health
Response: {"status": "healthy", "components": {...}}
```

### Generate Prediction
```
POST /predict
Body: {"symbol": "BTCUSDT", "timeframe": "1h"}
Response: {"prediction": {...}, "confidence": 0.85}
```

### Get Historical Data
```
GET /data/{symbol}?interval=1h&limit=100
Response: {"data": [...], "count": 100}
```

### System Metrics
```
GET /metrics
Response: Prometheus metrics format
```

## Error Codes
- 400: Bad Request
- 404: Not Found  
- 500: Internal Error
- 503: Service Unavailable

## Rate Limits
- Default: 100 requests/minute
- Predictions: 10 requests/minute
