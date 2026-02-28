const { GoogleGenerativeAI } = require('@google/generative-ai');

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

const SCENARIO_TEMPLATES = [
  {
    id: 'warming-2c',
    name: '+2°C Global Warming',
    description: 'Simulate impact of 2°C temperature increase',
    parameters: { temperatureChange: 2, timeframe: '2050' }
  },
  {
    id: 'warming-4c',
    name: '+4°C Global Warming',
    description: 'Simulate impact of 4°C temperature increase',
    parameters: { temperatureChange: 4, timeframe: '2100' }
  },
  {
    id: 'habitat-loss-50',
    name: '50% Habitat Loss',
    description: 'Simulate 50% reduction in critical habitat',
    parameters: { habitatLoss: 0.5, affectedAreas: ['stopover', 'breeding'] }
  },
  {
    id: 'habitat-loss-25',
    name: '25% Habitat Loss',
    description: 'Simulate 25% reduction in critical habitat',
    parameters: { habitatLoss: 0.25, affectedAreas: ['stopover'] }
  },
  {
    id: 'drought-severe',
    name: 'Severe Drought',
    description: 'Simulate prolonged drought conditions',
    parameters: { precipitationChange: -0.4, duration: '5-years' }
  },
  {
    id: 'conservation-success',
    name: 'Conservation Success',
    description: 'Simulate successful habitat restoration',
    parameters: { habitatGain: 0.3, protectedAreas: 0.5 }
  }
];

exports.simulateScenario = async (speciesId, scenarioId, customParameters = {}) => {
  const template = SCENARIO_TEMPLATES.find(s => s.id === scenarioId);
  const parameters = { ...template?.parameters, ...customParameters };
  
  // Run simulation
  const baselineRisk = 45;
  const projectedRisk = calculateProjectedRisk(baselineRisk, parameters);
  const rangeShift = calculateRangeShift(parameters);
  const populationImpact = calculatePopulationImpact(parameters);
  
  // Generate AI narrative
  const narrative = await generateScenarioNarrative(speciesId, scenarioId, parameters, {
    baselineRisk,
    projectedRisk,
    rangeShift,
    populationImpact
  });
  
  return {
    scenarioId,
    speciesId,
    parameters,
    results: {
      baselineRisk,
      projectedRisk,
      riskChange: projectedRisk - baselineRisk,
      rangeShift,
      populationImpact,
      extinctionRisk: calculateExtinctionRisk(projectedRisk, populationImpact)
    },
    narrative,
    timestamp: new Date().toISOString()
  };
};

exports.getPredefinedScenarios = async () => {
  return SCENARIO_TEMPLATES;
};

exports.compareMultipleScenarios = async (speciesId, scenarioIds) => {
  const results = await Promise.all(
    scenarioIds.map(id => this.simulateScenario(speciesId, id))
  );
  
  return {
    speciesId,
    scenarios: results,
    comparison: {
      worstCase: results.reduce((worst, curr) => 
        curr.results.projectedRisk > worst.results.projectedRisk ? curr : worst
      ),
      bestCase: results.reduce((best, curr) => 
        curr.results.projectedRisk < best.results.projectedRisk ? curr : best
      ),
      averageRisk: results.reduce((sum, r) => sum + r.results.projectedRisk, 0) / results.length
    },
    timestamp: new Date().toISOString()
  };
};

function calculateProjectedRisk(baseline, parameters) {
  let risk = baseline;
  
  if (parameters.temperatureChange) {
    risk += parameters.temperatureChange * 8;
  }
  if (parameters.habitatLoss) {
    risk += parameters.habitatLoss * 60;
  }
  if (parameters.habitatGain) {
    risk -= parameters.habitatGain * 40;
  }
  if (parameters.precipitationChange) {
    risk += Math.abs(parameters.precipitationChange) * 30;
  }
  
  return Math.min(100, Math.max(0, risk));
}

function calculateRangeShift(parameters) {
  let shift = { north: 0, elevation: 0 };
  
  if (parameters.temperatureChange) {
    shift.north = parameters.temperatureChange * 150; // km per degree
    shift.elevation = parameters.temperatureChange * 200; // meters per degree
  }
  
  return shift;
}

function calculatePopulationImpact(parameters) {
  let impact = 0;
  
  if (parameters.temperatureChange) {
    impact -= parameters.temperatureChange * 5;
  }
  if (parameters.habitatLoss) {
    impact -= parameters.habitatLoss * 40;
  }
  if (parameters.habitatGain) {
    impact += parameters.habitatGain * 30;
  }
  
  return Math.max(-100, Math.min(100, impact));
}

function calculateExtinctionRisk(projectedRisk, populationImpact) {
  const risk = (projectedRisk * 0.6 + Math.abs(populationImpact) * 0.4) / 100;
  
  if (risk > 0.8) return 'critical';
  if (risk > 0.6) return 'high';
  if (risk > 0.4) return 'moderate';
  if (risk > 0.2) return 'low';
  return 'minimal';
}

async function generateScenarioNarrative(speciesId, scenarioId, parameters, results) {
  const model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });
  
  const prompt = `Generate a compelling narrative about how ${scenarioId} affects ${speciesId} migration patterns.

Scenario Parameters: ${JSON.stringify(parameters)}
Projected Results:
- Risk increases from ${results.baselineRisk} to ${results.projectedRisk}
- Range shift: ${results.rangeShift.north}km north, ${results.rangeShift.elevation}m elevation
- Population impact: ${results.populationImpact}%

Create a 2-3 paragraph story that:
1. Describes the environmental changes
2. Explains the biological consequences for the species
3. Highlights the conservation implications
4. Uses vivid, accessible language

Include relevant hashtags like #climatechange #conservation #migration`;

  try {
    const result = await model.generateContent(prompt);
    const response = await result.response;
    return response.text();
  } catch (error) {
    console.error('Gemini API error:', error);
    return `Under the ${scenarioId} scenario, ${speciesId} faces significant challenges. The projected risk score increases to ${results.projectedRisk}, indicating ${results.extinctionRisk} extinction risk. Conservation action is needed to protect critical habitats and migration corridors.`;
  }
}
