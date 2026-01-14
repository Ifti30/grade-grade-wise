import express from 'express';
import fs from 'fs/promises';
import path from 'path';
import multer from 'multer';
import { authenticateToken } from '../auth.js';
import prisma from '../lib/prisma.js';
import { STORAGE_ROOT, toStaticPath, normalizePlotObject } from '../lib/storage.js';
import { createOrgUploadStorage, jsonFileFilter } from '../lib/uploads.js';
import { runPythonPredict } from '../utils/python-runner.js';

const router = express.Router();

function hydratePrediction(pred) {
  if (!pred) return pred;
  const inputStatic = toStaticPath(pred.inputPath);
  const outputStatic = toStaticPath(pred.outFile);
  const summary = pred.summary || {};
  const normalizedPlots = normalizePlotObject(summary.plots || pred.plots);
  const files = {
    input: summary.files?.input || inputStatic,
    output: summary.files?.output || outputStatic
  };

  return {
    ...pred,
    inputFileUrl: inputStatic,
    outFileUrl: outputStatic,
    plots: normalizedPlots,
    bestModel: summary.bestModel || pred.bestModel,
    summary: {
      ...summary,
      files,
      plots: normalizedPlots,
      bestModel: summary.bestModel || pred.bestModel
    }
  };
}

// Configure multer
const upload = multer({
  storage: createOrgUploadStorage('predictions', 'student'),
  fileFilter: jsonFileFilter
});

// Make prediction
router.post('/', authenticateToken, upload.single('studentFile'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'Student file is required' });
    }

    // Get latest succeeded model
    const lastSucceeded = await prisma.modelRun.findFirst({
      where: { 
        orgId: req.orgId,
        status: 'SUCCEEDED'
      },
      orderBy: { createdAt: 'desc' }
    });

    if (!lastSucceeded) {
      return res.status(400).json({ error: 'No trained model found. Please train a model first.' });
    }

    // Parse student data to get ID
    const studentData = JSON.parse(await fs.readFile(req.file.path, 'utf-8'));
    const rawStudentId = studentData.student_id ?? studentData.id ?? 'unknown';
    const studentId = typeof rawStudentId === 'string' ? rawStudentId : String(rawStudentId);

    // Prepare output file path
    const outFile = req.file.path.replace('.json', '_prediction.json');

    // Run prediction
    const creditHoursRaw = req.body?.creditHours ?? req.body?.credit_hours;
    const creditHours = creditHoursRaw !== undefined && creditHoursRaw !== null && creditHoursRaw !== ''
      ? Number(creditHoursRaw)
      : null;

    const result = await runPythonPredict(
      req.orgId,
      req.file.path,
      lastSucceeded.artifactsDir,
      outFile,
      creditHours
    );

    if (result.status !== 'ok') {
      return res.status(500).json({ error: result.error || 'Prediction failed' });
    }

    const predictionResults = result?.predictions && typeof result.predictions === 'object'
      ? result.predictions
      : {};
    const inputStaticPath = toStaticPath(req.file.path);
    const outputStaticPath = toStaticPath(outFile);
    const predictionPlots = normalizePlotObject(result.plots);
    const predictionSummary = {
      risk: result?.risk ?? null,
      current: result?.current ?? null,
      ensemble: predictionResults.ensemble,
      bestModel: result?.bestModel ?? null,
      creditHours: result?.creditHours ?? null,
      courseLoad: result?.courseLoad ?? null,
      loadAdjusted: result?.loadAdjusted ?? null,
      files: {
        input: inputStaticPath,
        output: outputStaticPath
      },
      plots: predictionPlots
    };

    // Save prediction record
    const prediction = await prisma.prediction.create({
      data: {
        orgId: req.orgId,
        studentId,
        inputPath: req.file.path,
        outFile,
        results: predictionResults || {},
        summary: predictionSummary
      }
    });

    const hydrated = hydratePrediction(prediction);
    res.json(hydrated);
  } catch (error) {
    console.error('Prediction error:', error);
    res.status(500).json({ error: 'Prediction failed: ' + error.message });
  }
});

// List predictions
router.get('/', authenticateToken, async (req, res) => {
  try {
    const predictions = await prisma.prediction.findMany({
      where: { orgId: req.orgId },
      orderBy: { createdAt: 'desc' },
      take: 50
    });

    res.json(predictions.map(hydratePrediction));
  } catch (error) {
    console.error('List predictions error:', error);
    res.status(500).json({ error: 'Failed to list predictions' });
  }
});

// Get single prediction
router.get('/:id', authenticateToken, async (req, res) => {
  try {
    const prediction = await prisma.prediction.findFirst({
      where: { 
        id: req.params.id,
        orgId: req.orgId
      }
    });

    if (!prediction) {
      return res.status(404).json({ error: 'Prediction not found' });
    }

    res.json(hydratePrediction(prediction));
  } catch (error) {
    console.error('Get prediction error:', error);
    res.status(500).json({ error: 'Failed to get prediction' });
  }
});

// Clear prediction history for org
router.delete('/', authenticateToken, async (req, res) => {
  try {
    const result = await prisma.prediction.deleteMany({
      where: { orgId: req.orgId }
    });

    const orgPredictionsDir = path.join(STORAGE_ROOT, 'predictions', req.orgId);
    try {
      await fs.rm(orgPredictionsDir, { recursive: true, force: true });
    } catch (error) {
      if (error.code !== 'ENOENT') {
        console.error('Failed to delete prediction files:', error);
      }
    }

    res.json({ deleted: result.count });
  } catch (error) {
    console.error('Clear predictions error:', error);
    res.status(500).json({ error: 'Failed to clear prediction history' });
  }
});

export default router;
