const scenarioService = require('../services/scenario.service');

exports.runSimulation = async (req, res, next) => {
  try {
    const { speciesId, scenario, parameters } = req.body;
    
    const results = await scenarioService.simulateScenario(speciesId, scenario, parameters);
    res.json({ success: true, data: results });
  } catch (error) {
    next(error);
  }
};

exports.getScenarioTemplates = async (req, res, next) => {
  try {
    const templates = await scenarioService.getPredefinedScenarios();
    res.json({ success: true, data: templates });
  } catch (error) {
    next(error);
  }
};

exports.compareScenarios = async (req, res, next) => {
  try {
    const { speciesId, scenarios } = req.body;
    
    const comparison = await scenarioService.compareMultipleScenarios(speciesId, scenarios);
    res.json({ success: true, data: comparison });
  } catch (error) {
    next(error);
  }
};
