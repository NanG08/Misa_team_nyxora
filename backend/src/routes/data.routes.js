const express = require('express');
const router = express.Router();
const dataController = require('../controllers/data.controller');

// Get environmental overlays
router.get('/environmental', dataController.getEnvironmentalData);

// Get climate data from NOAA
router.get('/climate', dataController.getClimateData);

// Get forest cover data
router.get('/forest-cover', dataController.getForestCoverData);

// Get real-time events (wildfires, droughts)
router.get('/events', dataController.getRealTimeEvents);

module.exports = router;
