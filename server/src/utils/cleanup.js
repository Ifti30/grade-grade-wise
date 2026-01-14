import fs from 'fs/promises';
import path from 'path';

async function safeStat(filePath) {
  try {
    return await fs.stat(filePath);
  } catch {
    return null;
  }
}

async function pruneFilesByCount(dir, keep, filterFn = () => true) {
  if (!dir || keep < 0) return;
  let entries = [];
  try {
    entries = await fs.readdir(dir);
  } catch {
    return;
  }
  const files = [];
  for (const name of entries) {
    if (!filterFn(name)) continue;
    const fullPath = path.join(dir, name);
    const stat = await safeStat(fullPath);
    if (!stat || !stat.isFile()) continue;
    files.push({ name, fullPath, mtimeMs: stat.mtimeMs });
  }
  if (files.length <= keep) return;
  files
    .sort((a, b) => a.mtimeMs - b.mtimeMs)
    .slice(0, files.length - keep)
    .forEach((file) => {
      fs.unlink(file.fullPath).catch(() => {});
    });
}

async function pruneUploads(storageRoot, keepLatestPerOrg) {
  const uploadsRoot = path.join(storageRoot, 'uploads');
  let orgDirs = [];
  try {
    orgDirs = await fs.readdir(uploadsRoot, { withFileTypes: true });
  } catch {
    return;
  }
  for (const dirent of orgDirs) {
    if (!dirent.isDirectory()) continue;
    const orgPath = path.join(uploadsRoot, dirent.name);
    await pruneFilesByCount(orgPath, keepLatestPerOrg, (name) => name.endsWith('.json'));
  }
}

export async function runCleanup({
  storageRoot,
  uploadsKeepLatest = 1,
  trainLogsDir,
  trainLogsKeep = 5
}) {
  await pruneUploads(storageRoot, uploadsKeepLatest);
  if (trainLogsDir) {
    await pruneFilesByCount(trainLogsDir, trainLogsKeep, (name) => name.endsWith('.log'));
  }
}
