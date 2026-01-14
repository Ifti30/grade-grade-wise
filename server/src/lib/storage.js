import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const STORAGE_ROOT = path.join(__dirname, '../../storage');

export function toStaticPath(absPath, { fallback = null } = {}) {
  if (!absPath) return fallback;

  if (typeof absPath === 'string') {
    if (absPath.startsWith('http://') || absPath.startsWith('https://')) return absPath;
    if (absPath.startsWith('/static/')) return absPath;
    if (absPath.startsWith('static/')) return `/${absPath}`;
  }

  const normalized = path.normalize(String(absPath));
  if (normalized.startsWith(STORAGE_ROOT)) {
    const rel = normalized.slice(STORAGE_ROOT.length).replace(/\\/g, '/');
    return `/static${rel}`;
  }

  return fallback;
}

export function normalizePlotList(plots) {
  if (!Array.isArray(plots)) return {};

  const entries = plots
    .filter(Boolean)
    .map((plotPath) => {
      const key = path.basename(plotPath, path.extname(plotPath)) || 'plot';
      return [key, toStaticPath(plotPath, { fallback: plotPath })];
    });

  return Object.fromEntries(entries);
}

export function normalizePlotObject(plots) {
  if (!plots || typeof plots !== 'object') return {};

  return Object.fromEntries(
    Object.entries(plots).map(([key, value]) => {
      if (typeof value === 'string') {
        return [key, toStaticPath(value, { fallback: value })];
      }
      return [key, value];
    })
  );
}
