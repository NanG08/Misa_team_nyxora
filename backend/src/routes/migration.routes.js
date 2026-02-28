const express = require('express');
const router = express.Router();
const migrationController = require('../controllers/migration.controller');

// Get migration data for a species
router.get('/species/:speciesId', migrationController.getMigrationData);

// Get migration points within time range
router.get('/points', migrationController.getMigrationPoints);

// Get migration risk score
router.get('/risk-score/:speciesId', migrationController.getRiskScore);

// Get all tracked species
router.get('/species', migrationController.getAllSpecies);

module.exports = router;
