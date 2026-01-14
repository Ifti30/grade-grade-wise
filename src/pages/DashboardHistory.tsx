import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import { Layout } from '@/components/Layout';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Loader2, Eye, History, Trash2 } from 'lucide-react';
import { toast } from 'sonner';
import PredictionDetails from '@/components/predictions/PredictionDetails';

export default function DashboardHistory() {
  const [predictions, setPredictions] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedPrediction, setSelectedPrediction] = useState<any>(null);
  const [clearing, setClearing] = useState(false);

  useEffect(() => {
    loadPredictions();
  }, []);

  const loadPredictions = async () => {
    try {
      const data = await api.getPredictions();
      setPredictions(data);
    } catch (error: any) {
      toast.error('Failed to load predictions');
    } finally {
      setLoading(false);
    }
  };

  const handleViewDetails = async (id: string) => {
    try {
      const data = await api.getPrediction(id);
      setSelectedPrediction(data);
    } catch (error: any) {
      toast.error('Failed to load prediction details');
    }
  };

  const handleClearHistory = async () => {
    if (!window.confirm('Clear all prediction history for this organization?')) {
      return;
    }
    setClearing(true);
    try {
      await api.clearPredictions();
      setPredictions([]);
      toast.success('Prediction history cleared');
    } catch (error: any) {
      toast.error(error?.message || 'Failed to clear prediction history');
    } finally {
      setClearing(false);
    }
  };

  if (loading) {
    return (
      <Layout>
        <div className="flex items-center justify-center py-20">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      </Layout>
    );
  }

  if (selectedPrediction) {
    return (
      <Layout>
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-3xl font-bold text-foreground">Prediction Details</h2>
            <Button variant="outline" onClick={() => setSelectedPrediction(null)}>
              Back to List
            </Button>
          </div>

          <Card className="p-8 bg-card/50 backdrop-blur-sm border-border/50">
            <PredictionDetails prediction={selectedPrediction} />
          </Card>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-3xl font-bold text-foreground mb-2">Prediction History</h2>
            <p className="text-muted-foreground">View all past predictions</p>
          </div>
          {predictions.length > 0 && (
            <Button
              variant="outline"
              size="sm"
              onClick={handleClearHistory}
              disabled={clearing}
              className="gap-2"
            >
              <Trash2 className="h-4 w-4" />
              {clearing ? 'Clearing...' : 'Clear history'}
            </Button>
          )}
        </div>

        {predictions.length === 0 ? (
          <Card className="p-12 text-center bg-card/50 backdrop-blur-sm">
            <History className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-foreground mb-2">No Predictions Yet</h3>
            <p className="text-muted-foreground">Make your first prediction to see history</p>
          </Card>
        ) : (
          <div className="space-y-3">
            {predictions.map((pred) => (
              <Card
                key={pred.id}
                className="p-6 bg-card/50 backdrop-blur-sm border-border/50 hover:border-primary/50 transition-all"
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <p className="text-lg font-semibold text-foreground">
                        Student: {pred.studentId}
                      </p>
                      {pred.results?.predicted_grade && (
                        <span className="px-3 py-1 rounded-full bg-primary/10 text-primary text-sm font-medium">
                          {pred.results.predicted_grade}
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-muted-foreground">
                      {new Date(pred.createdAt).toLocaleString()}
                    </p>
                  </div>
                  
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleViewDetails(pred.id)}
                    className="gap-2"
                  >
                    <Eye className="h-4 w-4" />
                    View
                  </Button>
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>
    </Layout>
  );
}
