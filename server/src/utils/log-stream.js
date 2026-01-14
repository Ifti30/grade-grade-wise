import fs from 'fs/promises';

export async function readLogChunk(logPath, offset) {
  const handle = await fs.open(logPath, 'r');
  try {
    const stats = await handle.stat();
    let nextOffset = offset;
    if (stats.size < nextOffset) {
      nextOffset = 0;
    }
    if (stats.size === nextOffset) {
      return { chunk: '', nextOffset, size: stats.size };
    }
    const length = stats.size - nextOffset;
    const buffer = Buffer.alloc(length);
    const { bytesRead } = await handle.read(buffer, 0, length, nextOffset);
    const chunk = buffer.slice(0, bytesRead).toString('utf8');
    return { chunk, nextOffset: nextOffset + bytesRead, size: stats.size };
  } finally {
    await handle.close();
  }
}
