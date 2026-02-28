const express = require('express');
const router = express.Router();
const predictionController = require('../controllers/prediction.controller');

// Get route prediction with XAI
router.post('/route', predictionController.predictRoute);

// Get feature attribution (SHAP values)
router.post('/attribution', predictionController.getFeatureAttribution);

// Get AI-generated explanation
router.post('/explain', predictionController.explainPrediction);

module.exports = router;
