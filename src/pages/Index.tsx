import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/lib/auth';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Brain, Zap, TrendingUp, Shield, ArrowRight } from 'lucide-react';

export default function Index() {
  const { user, loading } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    // Redirect authenticated users to dashboard
    if (!loading && user) {
      navigate('/dashboard/summary');
    }
  }, [user, loading, navigate]);

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Hero Section */}
      <section className="relative overflow-hidden py-20 px-6">
        <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-accent/10 to-background pointer-events-none" />
        
        <div className="container mx-auto max-w-6xl relative">
          <div className="text-center space-y-8 animate-fade-in">
            <div className="inline-flex items-center justify-center p-6 rounded-3xl bg-primary/10 backdrop-blur-sm animate-glow-pulse">
              <Brain className="h-20 w-20 text-primary" />
            </div>
            
            <div className="space-y-4">
              <h1 className="text-5xl md:text-6xl font-bold text-foreground bg-clip-text">
              Empirical Grade Prediction
              </h1>
              <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
              A systematic study on features influencing grade forecasting using multiple basic ML algorithms
              </p>
            </div>

            <div className="flex gap-4 justify-center">
              <Button
                size="lg"
                onClick={() => navigate('/signup')}
                className="gap-2 text-lg px-8 py-6"
              >
                Get Started
                <ArrowRight className="h-5 w-5" />
              </Button>
              <Button
                size="lg"
                variant="outline"
                onClick={() => navigate('/signin')}
                className="text-lg px-8 py-6"
              >
                Sign In
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-6">
        <div className="container mx-auto max-w-6xl">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-foreground mb-4">Powerful ML Capabilities</h2>
            <p className="text-xl text-muted-foreground">Everything you need for accurate grade prediction</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              {
                icon: Brain,
                title: 'Advanced ML',
                description: 'Ensemble models including Random Forest, LightGBM, MLP, and SVR'
              },
              {
                icon: Zap,
                title: 'Real-time Training',
                description: 'Stream logs and monitor your model training progress live'
              },
              {
                icon: TrendingUp,
                title: 'Risk Analysis',
                description: 'Identify at-risk students before performance declines'
              },
              {
                icon: Shield,
                title: 'Secure & Private',
                description: 'Your data stays in your infrastructure with JWT auth'
              }
            ].map((feature, index) => {
              const Icon = feature.icon;
              return (
                <Card
                  key={index}
                  className="p-6 bg-card/50 backdrop-blur-sm border-border/50 hover:border-primary/50 transition-all duration-300 hover:shadow-lg hover:shadow-primary/10 animate-slide-up"
                  style={{ animationDelay: `${index * 100}ms` }}
                >
                  <div className="space-y-3">
                    <div className="p-3 rounded-lg bg-primary/10 w-fit">
                      <Icon className="h-6 w-6 text-primary" />
                    </div>
                    <h3 className="text-lg font-semibold text-foreground">{feature.title}</h3>
                    <p className="text-sm text-muted-foreground">{feature.description}</p>
                  </div>
                </Card>
              );
            })}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-6">
        <div className="container mx-auto max-w-4xl">
          <Card className="p-12 text-center bg-gradient-to-br from-primary/10 to-accent/10 backdrop-blur-sm border-primary/20">
            <h2 className="text-4xl font-bold text-foreground mb-4">
              Ready to predict academic success?
            </h2>
            <p className="text-xl text-muted-foreground mb-8">
              Join now and start training your first model in minutes
            </p>
            <Button
              size="lg"
              onClick={() => navigate('/signup')}
              className="gap-2 text-lg px-8 py-6"
            >
              Create Free Account
              <ArrowRight className="h-5 w-5" />
            </Button>
          </Card>
        </div>
      </section>
    </div>
  );
}
