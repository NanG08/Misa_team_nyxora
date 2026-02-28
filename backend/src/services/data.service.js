const axios = require('axios');

exports.fetchEnvironmentalOverlays = async (lat, lon, radius, layers = []) => {
  const data = {};
  
  if (layers.includes('temperature') || layers.length === 0) {
    data.temperature = await getTemperatureData(lat, lon, radius);
  }
  
  if (layers.includes('forest') || layers.length === 0) {
    data.forestCover = await getForestCoverData(lat, lon, radius);
  }
  
  if (layers.includes('drought') || layers.length === 0) {
    data.droughtIndex = await getDroughtData(lat, lon, radius);
  }
  
  return {
    location: { lat, lon, radius },
    layers: data,
    timestamp: new Date().toISOString()
  };
};

exports.fetchClimateData = async (lat, lon, startDate, endDate) => {
  // In production, use NOAA API
  // const response = await axios.get('https://www.ncdc.noaa.gov/cdo-web/api/v2/data', {
  //   headers: { token: process.env.NOAA_API_KEY },
  //   params: { datasetid: 'GHCND', locationid: `POINT(${lon} ${lat})` }
  // });
  
  return {
    location: { lat, lon },
    period: { startDate, endDate },
    data: {
      temperature: generateMockTimeSeries('temperature', startDate, endDate),
      precipitation: generateMockTimeSeries('precipitation', startDate, endDate),
      humidity: generateMockTimeSeries('humidity', startDate, endDate)
    },
    source: 'NOAA',
    timestamp: new Date().toISOString()
  };
};

exports.fetchForestCoverData = async (bounds) => {
  // In production, use Global Forest Watch API
  // const response = await axios.get('https://data-api.globalforestwatch.org/dataset/...');
  
  return {
    bounds,
    forestCover: {
      total: 75.5,
      loss: 12.3,
      gain: 3.2,
      netChange: -9.1
    },
    deforestation: [
      { year: 2020, area: 1200, lat: bounds.north - 0.5, lon: bounds.west + 0.5 },
      { year: 2021, area: 1500, lat: bounds.north - 1, lon: bounds.west + 1 },
      { year: 2022, area: 980, lat: bounds.north - 0.8, lon: bounds.west + 0.8 }
    ],
    source: 'Global Forest Watch',
    timestamp: new Date().toISOString()
  };
};

exports.fetchRealTimeEvents = async (eventTypes = [], bounds = null) => {
  const events = [];
  
  if (eventTypes.includes('wildfire') || eventTypes.length === 0) {
    events.push(...generateMockEvents('wildfire', bounds));
  }
  
  if (eventTypes.includes('drought') || eventTypes.length === 0) {
    events.push(...generateMockEvents('drought', bounds));
  }
  
  if (eventTypes.includes('flood') || eventTypes.length === 0) {
    events.push(...generateMockEvents('flood', bounds));
  }
  
  if (eventTypes.includes('heatwave') || eventTypes.length === 0) {
    events.push(...generateMockEvents('heatwave', bounds));
  }
  
  return {
    events,
    count: events.length,
    bounds,
    timestamp: new Date().toISOString()
  };
};

// Helper functions
async function getTemperatureData(lat, lon, radius) {
  return {
    current: 22.5 + (Math.random() - 0.5) * 10,
    anomaly: 2.3,
    trend: 'increasing',
    heatmapData: generateHeatmapGrid(lat, lon, radius, 'temperature')
  };
}

async function getForestCoverData(lat, lon, radius) {
  return {
    coverage: 65 + Math.random() * 30,
    change: -5.2,
    fragmentationIndex: 0.45,
    gridData: generateHeatmapGrid(lat, lon, radius, 'forest')
  };
}

async function getDroughtData(lat, lon, radius) {
  return {
    index: Math.random() * 5,
    severity: ['none', 'mild', 'moderate', 'severe', 'extreme'][Math.floor(Math.random() * 5)],
    duration: Math.floor(Math.random() * 180),
    gridData: generateHeatmapGrid(lat, lon, radius, 'drought')
  };
}

function generateHeatmapGrid(centerLat, centerLon, radius, type) {
  const grid = [];
  const gridSize = 10;
  const step = (radius * 2) / gridSize;
  
  for (let i = 0; i < gridSize; i++) {
    for (let j = 0; j < gridSize; j++) {
      const lat = centerLat - radius + i * step;
      const lon = centerLon - radius + j * step;
      const value = Math.random() * 100;
      
      grid.push({ lat, lon, value });
    }
  }
  
  return grid;
}

function generateMockTimeSeries(metric, startDate, endDate) {
  const series = [];
  const start = new Date(startDate || '2024-01-01');
  const end = new Date(endDate || '2024-12-31');
  const days = Math.floor((end - start) / (1000 * 60 * 60 * 24));
  
  for (let i = 0; i <= days; i += 7) {
    const date = new Date(start.getTime() + i * 24 * 60 * 60 * 1000);
    const baseValue = metric === 'temperature' ? 20 : metric === 'precipitation' ? 50 : 60;
    const value = baseValue + Math.sin(i / 30) * 10 + (Math.random() - 0.5) * 5;
    
    series.push({
      date: date.toISOString().split('T')[0],
      value: Math.round(value * 10) / 10
    });
  }
  
  return series;
}

function generateMockEvents(type, bounds) {
  const numEvents = Math.floor(Math.random() * 5) + 1;
  const events = [];
  
  const severityLevels = ['low', 'moderate', 'high', 'critical'];
  
  for (let i = 0; i < numEvents; i++) {
    const lat = bounds ? bounds.south + Math.random() * (bounds.north - bounds.south) : Math.random() * 180 - 90;
    const lon = bounds ? bounds.west + Math.random() * (bounds.east - bounds.west) : Math.random() * 360 - 180;
    
    events.push({
      id: `${type}-${Date.now()}-${i}`,
      type,
      lat,
      lon,
      severity: severityLevels[Math.floor(Math.random() * severityLevels.length)],
      startDate: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString(),
      area: Math.floor(Math.random() * 10000) + 100,
      description: `${type.charAt(0).toUpperCase() + type.slice(1)} event detected in the region`
    });
  }
  
  return events;
}
