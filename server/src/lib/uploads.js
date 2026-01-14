import multer from 'multer';
import path from 'path';
import fs from 'fs/promises';
import { STORAGE_ROOT } from './storage.js';

export function jsonFileFilter(req, file, cb) {
  if (file.mimetype === 'application/json' || file.originalname.endsWith('.json')) {
    cb(null, true);
  } else {
    cb(new Error('Only JSON files are allowed'));
  }
}

export function createOrgUploadStorage(subdir, filenamePrefix) {
  return multer.diskStorage({
    destination: async (req, file, cb) => {
      const uploadDir = path.join(STORAGE_ROOT, subdir, req.orgId);
      await fs.mkdir(uploadDir, { recursive: true });
      cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
      const timestamp = Date.now();
      cb(null, `${filenamePrefix}_${timestamp}.json`);
    }
  });
}
