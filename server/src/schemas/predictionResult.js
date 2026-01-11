import { z } from 'zod';

export const PredictionResultSchema = z.object({
  status: z.literal('ok'),
  predictions: z.any().optional(),
  loadAdjusted: z.any().optional(),
  creditHours: z.any().optional(),
  courseLoad: z.any().optional(),
  risk: z.any().optional(),
  current: z.any().optional(),
  outFile: z.string().optional(),
  max_gpa: z.any().optional(),
  plots: z.any().optional(),
  bestModel: z.any().optional()
}).passthrough();
