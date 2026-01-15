import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider, useAuth } from "@/lib/auth";

import Index from "./pages/Index";
import NotFound from "./pages/NotFound";
import SignIn from "./pages/SignIn";
import SignUp from "./pages/SignUp";
import TrainModel from "./pages/TrainModel";
import TrainModels from "./pages/TrainModels";
import TrainingComplete from "./pages/TrainingComplete";
import DashboardSummary from "./pages/DashboardSummary";
import DashboardPredict from "./pages/DashboardPredict";
import DashboardHistory from "./pages/DashboardHistory";
import DashboardRetrain from "./pages/DashboardRetrain";
import Diagram from "./pages/Diagram";
import { Layout } from "@/components/Layout";

const queryClient = new QueryClient();

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary" />
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/signin" replace />;
  }

  return <>{children}</>;
}

function PublicRoute({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary" />
      </div>
    );
  }

  if (user) {
    return <Navigate to="/dashboard/summary" replace />;
  }

  return <>{children}</>;
}

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <AuthProvider>
          <Routes>
            <Route path="/" element={<Index />} />
            
            {/* Public Routes */}
            <Route path="/signin" element={<PublicRoute><SignIn /></PublicRoute>} />
            <Route path="/signup" element={<PublicRoute><SignUp /></PublicRoute>} />
            
            {/* Protected Routes */}
            <Route path="/train-model" element={<ProtectedRoute><TrainModel /></ProtectedRoute>} />
            <Route path="/train-models" element={<ProtectedRoute><TrainModels /></ProtectedRoute>} />
            <Route path="/training-complete" element={<ProtectedRoute><TrainingComplete /></ProtectedRoute>} />
            
            {/* Dashboard Routes */}
            <Route path="/dashboard/summary" element={<ProtectedRoute><DashboardSummary /></ProtectedRoute>} />
            <Route path="/dashboard/predict" element={<ProtectedRoute><DashboardPredict /></ProtectedRoute>} />
            <Route path="/dashboard/history" element={<ProtectedRoute><DashboardHistory /></ProtectedRoute>} />
            <Route path="/dashboard/retrain" element={<ProtectedRoute><Layout><DashboardRetrain /></Layout></ProtectedRoute>} />
            <Route path="/diagram" element={<ProtectedRoute><Diagram /></ProtectedRoute>} />
            
            {/* Catch-all */}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </AuthProvider>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
