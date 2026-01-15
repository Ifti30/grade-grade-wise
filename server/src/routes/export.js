import express from 'express';
import path from 'path';
import fs from 'fs';
import { authenticateToken } from '../auth.js';
import { runPythonExport } from '../utils/python-runner.js';
import prisma from '../lib/prisma.js';

const router = express.Router();

// POST /export  -> runs python exporter, returns downloadUrl
router.post('/', authenticateToken, async (req, res) => {
  try {
    const lastSucceeded = await prisma.modelRun.findFirst({
      where: { orgId: req.orgId, status: 'SUCCEEDED' },
      orderBy: { createdAt: 'desc' }
    });

    if (!lastSucceeded) {
      return res.status(400).json({ error: 'No trained model found. Please train a model first.' });
    }

    const result = await runPythonExport(lastSucceeded.artifactsDir);

    // Python returns { status:"ok", zipPath:"/abs/path/to.zip", timestamp:"..." }
    const zipPath = result?.zipPath;
    if (!zipPath) {
      return res.status(500).json({ error: 'Export did not return a zipPath.' });
    }

    // Download URL points to GET /export/download?file=<filename>
    const filename = path.basename(zipPath);
    return res.json({ downloadUrl: `/export/download?file=${encodeURIComponent(filename)}` });
  } catch (error) {
    console.error('Export error:', error);
    res.status(500).json({ error: 'Failed to export results' });
  }
});

// GET /export/download?file=thesis_results_....zip
router.get('/download', authenticateToken, async (req, res) => {
  try {
    const file = String(req.query.file || '');
    if (!file || file.includes('..') || file.includes('/') || file.includes('\\')) {
      return res.status(400).json({ error: 'Invalid file name.' });
    }

    // Find the latest SUCCEEDED run again (scopes download to this org’s last run)
    const lastSucceeded = await prisma.modelRun.findFirst({
      where: { orgId: req.orgId, status: 'SUCCEEDED' },
      orderBy: { createdAt: 'desc' }
    });

    if (!lastSucceeded) {
      return res.status(400).json({ error: 'No trained model found.' });
    }

    // zip is saved into artifactsDir by your Python script
    const fullPath = path.join(lastSucceeded.artifactsDir, file);

    // Ensure file exists
    await fs.promises.access(fullPath, fs.constants.R_OK);

    res.setHeader('Content-Type', 'application/zip');
    res.setHeader('Content-Disposition', `attachment; filename="${file}"`);

    fs.createReadStream(fullPath).pipe(res);
  } catch (error) {
    console.error('Download error:', error);
    res.status(404).json({ error: 'File not found.' });
  }
});

export default router;
