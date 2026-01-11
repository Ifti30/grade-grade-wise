import { z } from 'zod';

export const TrainingResultSchema = z.object({
  status: z.literal('ok'),
  modelId: z.string(),
  bestModel: z.string().optional(),
  rmse: z.number().optional(),
  r2: z.number().optional(),
  riskAccuracy: z.number().optional(),
  enabledModels: z.array(z.string()).optional(),
  artifactsDir: z.string().optional(),
  metrics: z.any().optional(),
  plots: z.any().optional(),
  resumed: z.boolean()
}).passthrough();
