import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Migration API
export const migrationAPI = {
  getSpecies: async () => {
    const { data } = await api.get('/migration/species');
    return data.data;
  },

  getMigrationData: async (speciesId: string, startDate?: string, endDate?: string) => {
    const { data } = await api.get(`/migration/species/${speciesId}`, {
      params: { startDate, endDate }
    });
    return data.data;
  },

  getMigrationPoints: async (speciesId: string, startTime?: string, endTime?: string) => {
    const { data } = await api.get('/migration/points', {
      params: { speciesId, startTime, endTime }
    });
    return data.data;
  },

  getRiskScore: async (speciesId: string) => {
    const { data } = await api.get(`/migration/risk-score/${speciesId}`);
    return data.data;
  },
};

// Prediction API
export const predictionAPI = {
  predictRoute: async (speciesId: string, currentLocation: { lat: number; lon: number }, environmentalFactors: any) => {
    const { data } = await api.post('/predictions/route', {
      speciesId,
      currentLocation,
      environmentalFactors
    });
    return data.data;
  },

  getFeatureAttribution: async (speciesId: string, predictionData: any) => {
    const { data } = await api.post('/predictions/attribution', {
      speciesId,
      predictionData
    });
    return data.data;
  },

  explainPrediction: async (speciesId: string, prediction: any, factors: any[]) => {
    const { data } = await api.post('/predictions/explain', {
      speciesId,
      prediction,
      factors
    });
    return data.data;
  },
};

// Scenario API
export const scenarioAPI = {
  simulate: async (speciesId: string, scenarioId: string, parameters?: any) => {
    const { data } = await api.post('/scenarios/simulate', {
      speciesId,
      scenario: scenarioId,
      parameters
    });
    return data.data;
  },

  getTemplates: async () => {
    const { data } = await api.get('/scenarios/templates');
    return data.data;
  },

  compare: async (speciesId: string, scenarios: string[]) => {
    const { data } = await api.post('/scenarios/compare', {
      speciesId,
      scenarios
    });
    return data.data;
  },
};

// Environmental Data API
export const dataAPI = {
  getEnvironmentalData: async (lat: number, lon: number, radius: number, layers?: string[]) => {
    const { data } = await api.get('/data/environmental', {
      params: { lat, lon, radius, layers: layers?.join(',') }
    });
    return data.data;
  },

  getClimateData: async (lat: number, lon: number, startDate?: string, endDate?: string) => {
    const { data } = await api.get('/data/climate', {
      params: { lat, lon, startDate, endDate }
    });
    return data.data;
  },

  getForestCover: async (bounds: { north: number; south: number; east: number; west: number }) => {
    const { data } = await api.get('/data/forest-cover', {
      params: { bounds: JSON.stringify(bounds) }
    });
    return data.data;
  },

  getRealTimeEvents: async (eventTypes?: string[], bounds?: any) => {
    const { data } = await api.get('/data/events', {
      params: { 
        eventTypes: eventTypes?.join(','),
        bounds: bounds ? JSON.stringify(bounds) : undefined
      }
    });
    return data.data;
  },
};

export default api;
