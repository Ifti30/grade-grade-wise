import { z } from 'zod';

export const TrainingResultSchema = z.object({
    status: z.literal('ok'),
    modelId: z.string(),
    epochsCompleted: z.number().int().positive(),
    resumed: z.boolean(),
    metrics: z.record(z.number()),
    plots: z.array(z.any()).optional(),
});