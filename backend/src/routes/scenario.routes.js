const express = require('express');
const router = express.Router();
const scenarioController = require('../controllers/scenario.controller');

// Run scenario simulation
router.post('/simulate', scenarioController.runSimulation);

// Get predefined scenarios
router.get('/templates', scenarioController.getScenarioTemplates);

// Compare multiple scenarios
router.post('/compare', scenarioController.compareScenarios);

module.exports = router;
