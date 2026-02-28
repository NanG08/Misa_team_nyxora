const { GoogleGenerativeAI } = require('@google/generative-ai');

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

exports.predictMigrationRoute = async (speciesId, currentLocation, environmentalFactors) => {
  // Simulate LSTM/GNN prediction
  const predictedRoute = generatePredictedRoute(currentLocation);
  
  return {
    speciesId,
    currentLocation,
    predictedRoute,
    confidence: 0.85,
    timeframe: '30-days',
    timestamp: new Date().toISOString()
  };
};

exports.calculateFeatureAttribution = async (speciesId, predictionData) => {
  // Simulate SHAP values for feature importance
  const features = [
    { name: 'Temperature', importance: 0.35, value: '+2.5°C', impact: 'high' },
    { name: 'Habitat Loss', importance: 0.28, value: '15% reduction', impact: 'high' },
    { name: 'Food Availability', importance: 0.18, value: 'Moderate decline', impact: 'medium' },
    { name: 'Wind Patterns', importance: 0.12, value: 'Favorable', impact: 'low' },
    { name: 'Precipitation', importance: 0.07, value: 'Normal', impact: 'low' }
  ];
  
  return {
    features,
    method: 'SHAP',
    timestamp: new Date().toISOString()
  };
};

exports.generateAIExplanation = async (speciesId, prediction, factors) => {
  const model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });
  
  const prompt = `As an ecological AI expert, explain why ${speciesId} is predicted to shift its migration route. 
  
Environmental Factors:
${factors.map(f => `- ${f.name}: ${f.value} (${f.importance * 100}% importance)`).join('\n')}

Provide a detailed, user-friendly explanation of how each factor influences the migration, focusing on:
1. Biological mechanisms
2. Species-specific adaptations
3. Interconnected effects
4. Conservation implications

Keep the explanation scientific but accessible to the general public.`;

  try {
    const result = await model.generateContent(prompt);
    const response = await result.response;
    const text = response.text();
    
    return {
      explanation: text,
      factors: generateDetailedFactorAnalysis(factors, speciesId),
      timestamp: new Date().toISOString()
    };
  } catch (error) {
    console.error('Gemini API error:', error);
    return {
      explanation: generateFallbackExplanation(factors, speciesId),
      factors: generateDetailedFactorAnalysis(factors, speciesId),
      timestamp: new Date().toISOString()
    };
  }
};

function generatePredictedRoute(currentLocation) {
  const route = [];
  let { lat, lon } = currentLocation;
  
  for (let i = 0; i < 10; i++) {
    lat += (Math.random() - 0.5) * 2;
    lon += (Math.random() - 0.5) * 2;
    route.push({ lat, lon, day: i * 3 });
  }
  
  return route;
}

function generateDetailedFactorAnalysis(factors, speciesId) {
  return factors.map(factor => ({
    name: factor.name,
    importance: factor.importance,
    value: factor.value,
    impact: factor.impact,
    detailedExplanation: getFactorExplanation(factor.name, speciesId),
    mitigationStrategies: getMitigationStrategies(factor.name)
  }));
}

function getFactorExplanation(factorName, speciesId) {
  const explanations = {
    'Temperature': `Rising temperatures directly affect ${speciesId} by altering breeding cycles, energy expenditure during flight, and the timing of food availability. Warmer conditions can cause phenological mismatches where migration timing no longer aligns with peak food resources.`,
    'Habitat Loss': `Habitat fragmentation reduces critical stopover sites where ${speciesId} rests and refuels. Loss of these waypoints forces longer non-stop flights, increasing mortality risk and reducing reproductive success.`,
    'Food Availability': `Changes in prey/plant phenology mean ${speciesId} may arrive at breeding or stopover sites before or after peak food abundance. This mismatch reduces survival rates, especially for juveniles.`,
    'Wind Patterns': `Shifting wind patterns affect flight efficiency and energy costs. Favorable tailwinds reduce migration time and energy expenditure, while headwinds can delay arrival and deplete energy reserves.`,
    'Precipitation': `Altered rainfall patterns impact habitat quality at stopover and breeding sites. Drought reduces food availability while excessive rain can flood nesting areas or reduce insect populations.`
  };
  
  return explanations[factorName] || `${factorName} influences migration patterns through complex ecological interactions.`;
}

function getMitigationStrategies(factorName) {
  const strategies = {
    'Temperature': ['Protect high-altitude refugia', 'Maintain climate corridors', 'Preserve thermal diversity in habitats'],
    'Habitat Loss': ['Establish protected migration corridors', 'Restore degraded stopover sites', 'Implement wildlife-friendly land use'],
    'Food Availability': ['Preserve diverse plant communities', 'Reduce pesticide use', 'Maintain natural phenological cycles'],
    'Wind Patterns': ['Protect ridge lines and thermal updraft areas', 'Minimize wind farm impacts on routes'],
    'Precipitation': ['Restore wetlands', 'Implement water conservation', 'Maintain riparian buffers']
  };
  
  return strategies[factorName] || ['Monitor and adapt conservation strategies'];
}

function generateFallbackExplanation(factors, speciesId) {
  return `The predicted route shift for ${speciesId} is primarily driven by ${factors[0].name} (${Math.round(factors[0].importance * 100)}% importance) and ${factors[1].name} (${Math.round(factors[1].importance * 100)}% importance). These environmental changes are forcing the species to adapt its traditional migration patterns to find suitable habitat and resources.`;
}
