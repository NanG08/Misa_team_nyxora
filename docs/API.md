# API Documentation

Base URL: `http://localhost:5000/api`

## Authentication

Currently, the API does not require authentication. Future versions will implement API key authentication.

## Response Format

All API responses follow this structure:

**Success Response:**
```json
{
  "success": true,
  "data": { ... }
}
```

**Error Response:**
```json
{
  "success": false,
  "error": {
    "message": "Error description",
    "status": 400
  }
}
```

---

## Migration Endpoints

### Get All Species

Retrieve a list of all tracked species.

**Endpoint:** `GET /api/migration/species`

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": "monarch",
      "name": "Monarch Butterfly",
      "scientificName": "Danaus plexippus",
      "status": "endangered"
    }
  ]
}
```

### Get Migration Data

Get migration data for a specific species.

**Endpoint:** `GET /api/migration/species/:speciesId`

**Parameters:**
- `speciesId` (path) - Species identifier (e.g., "monarch")
- `startDate` (query, optional) - Start date (ISO 8601)
- `endDate` (query, optional) - End date (ISO 8601)

**Example:**
```
GET /api/migration/species/monarch?startDate=2024-01-01&endDate=2024-12-31
```

**Response:**
```json
{
  "success": true,
  "data": {
    "speciesId": "monarch",
    "points": [...],
    "metadata": {
      "source": "movebank",
      "lastUpdated": "2024-01-15T10:30:00Z"
    }
  }
}
```

### Get Migration Points

Retrieve migration points with filtering.

**Endpoint:** `GET /api/migration/points`

**Query Parameters:**
- `speciesId` (required) - Species identifier
- `startTime` (optional) - Start timestamp
- `endTime` (optional) - End timestamp

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": "monarch-1",
      "lat": 40.7128,
      "lon": -74.0060,
      "timestamp": "2024-01-15T10:00:00Z",
      "status": "active",
      "altitude": 500,
      "speed": 25
    }
  ]
}
```

### Get Risk Score

Calculate migration risk score for a species.

**Endpoint:** `GET /api/migration/risk-score/:speciesId`

**Response:**
```json
{
  "success": true,
  "data": {
    "score": 65,
    "factors": {
      "habitatFragmentation": 25,
      "climateStress": 30,
      "humanImpact": 10
    },
    "severity": "high",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

---

## Prediction Endpoints

### Predict Route

Predict future migration route using ML models.

**Endpoint:** `POST /api/predictions/route`

**Request Body:**
```json
{
  "speciesId": "monarch",
  "currentLocation": {
    "lat": 40.7128,
    "lon": -74.0060
  },
  "environmentalFactors": {
    "temperature": 22.5,
    "windSpeed": 15,
    "precipitation": 0
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "speciesId": "monarch",
    "currentLocation": {...},
    "predictedRoute": [
      { "lat": 40.8, "lon": -74.1, "day": 0 },
      { "lat": 41.0, "lon": -74.3, "day": 3 }
    ],
    "confidence": 0.85,
    "timeframe": "30-days"
  }
}
```

### Get Feature Attribution

Get SHAP-based feature importance for predictions.

**Endpoint:** `POST /api/predictions/attribution`

**Request Body:**
```json
{
  "speciesId": "monarch",
  "predictionData": {}
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "features": [
      {
        "name": "Temperature",
        "importance": 0.35,
        "value": "+2.5°C",
        "impact": "high"
      }
    ],
    "method": "SHAP"
  }
}
```

### Explain Prediction

Get AI-generated explanation for route prediction.

**Endpoint:** `POST /api/predictions/explain`

**Request Body:**
```json
{
  "speciesId": "monarch",
  "prediction": {},
  "factors": [
    { "name": "Temperature", "importance": 0.35 }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "explanation": "The predicted route shift is primarily driven by...",
    "factors": [
      {
        "name": "Temperature",
        "importance": 0.35,
        "detailedExplanation": "Rising temperatures affect...",
        "mitigationStrategies": ["Protect high-altitude refugia"]
      }
    ]
  }
}
```

---

## Scenario Endpoints

### Run Simulation

Execute a scenario simulation.

**Endpoint:** `POST /api/scenarios/simulate`

**Request Body:**
```json
{
  "speciesId": "monarch",
  "scenario": "warming-2c",
  "parameters": {
    "temperatureChange": 2
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "scenarioId": "warming-2c",
    "speciesId": "monarch",
    "results": {
      "baselineRisk": 45,
      "projectedRisk": 61,
      "riskChange": 16,
      "rangeShift": {
        "north": 300,
        "elevation": 400
      },
      "populationImpact": -10,
      "extinctionRisk": "moderate"
    },
    "narrative": "Under the +2°C warming scenario..."
  }
}
```

### Get Scenario Templates

Retrieve predefined scenario templates.

**Endpoint:** `GET /api/scenarios/templates`

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": "warming-2c",
      "name": "+2°C Global Warming",
      "description": "Simulate impact of 2°C temperature increase",
      "parameters": {
        "temperatureChange": 2,
        "timeframe": "2050"
      }
    }
  ]
}
```

### Compare Scenarios

Compare multiple scenarios side-by-side.

**Endpoint:** `POST /api/scenarios/compare`

**Request Body:**
```json
{
  "speciesId": "monarch",
  "scenarios": ["warming-2c", "habitat-loss-50"]
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "speciesId": "monarch",
    "scenarios": [...],
    "comparison": {
      "worstCase": {...},
      "bestCase": {...},
      "averageRisk": 58
    }
  }
}
```

---

## Environmental Data Endpoints

### Get Environmental Overlays

Retrieve environmental data layers for mapping.

**Endpoint:** `GET /api/data/environmental`

**Query Parameters:**
- `lat` (required) - Latitude
- `lon` (required) - Longitude
- `radius` (required) - Radius in degrees
- `layers` (optional) - Comma-separated layer names (temperature,forest,drought)

**Example:**
```
GET /api/data/environmental?lat=40.7&lon=-74.0&radius=5&layers=temperature,forest
```

**Response:**
```json
{
  "success": true,
  "data": {
    "location": { "lat": 40.7, "lon": -74.0, "radius": 5 },
    "layers": {
      "temperature": {
        "current": 22.5,
        "anomaly": 2.3,
        "trend": "increasing",
        "heatmapData": [...]
      },
      "forestCover": {
        "coverage": 65,
        "change": -5.2,
        "fragmentationIndex": 0.45
      }
    }
  }
}
```

### Get Climate Data

Fetch climate data from NOAA.

**Endpoint:** `GET /api/data/climate`

**Query Parameters:**
- `lat` (required) - Latitude
- `lon` (required) - Longitude
- `startDate` (optional) - Start date
- `endDate` (optional) - End date

**Response:**
```json
{
  "success": true,
  "data": {
    "location": { "lat": 40.7, "lon": -74.0 },
    "data": {
      "temperature": [
        { "date": "2024-01-01", "value": 20.5 }
      ],
      "precipitation": [...],
      "humidity": [...]
    },
    "source": "NOAA"
  }
}
```

### Get Forest Cover Data

Retrieve forest cover information.

**Endpoint:** `GET /api/data/forest-cover`

**Query Parameters:**
- `bounds` (required) - JSON object with north, south, east, west

**Example:**
```
GET /api/data/forest-cover?bounds={"north":45,"south":35,"east":-70,"west":-80}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "bounds": {...},
    "forestCover": {
      "total": 75.5,
      "loss": 12.3,
      "gain": 3.2,
      "netChange": -9.1
    },
    "deforestation": [
      { "year": 2020, "area": 1200, "lat": 40.5, "lon": -75.0 }
    ],
    "source": "Global Forest Watch"
  }
}
```

### Get Real-Time Events

Fetch real-time environmental events.

**Endpoint:** `GET /api/data/events`

**Query Parameters:**
- `eventTypes` (optional) - Comma-separated event types (wildfire,drought,flood,heatwave)
- `bounds` (optional) - Geographic bounds as JSON

**Response:**
```json
{
  "success": true,
  "data": {
    "events": [
      {
        "id": "wildfire-123",
        "type": "wildfire",
        "lat": 40.5,
        "lon": -75.0,
        "severity": "high",
        "startDate": "2024-01-10T00:00:00Z",
        "area": 5000,
        "description": "Wildfire event detected in the region"
      }
    ],
    "count": 1
  }
}
```

---

## Error Codes

| Code | Description |
|------|-------------|
| 200  | Success |
| 400  | Bad Request - Invalid parameters |
| 404  | Not Found - Resource doesn't exist |
| 500  | Internal Server Error |

## Rate Limiting

Currently no rate limiting is implemented. Future versions will include:
- 100 requests per minute per IP
- 1000 requests per hour per IP

## Versioning

API version is included in the base URL. Current version: `v1`

Future versions will use: `/api/v2/...`
