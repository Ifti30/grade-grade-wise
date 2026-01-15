import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { api } from './api';

interface User {
  id: string;
  email: string;
}

interface Org {
  id: string;
  name: string;
}

interface AuthContextType {
  user: User | null;
  org: Org | null;
  loading: boolean;
  signin: (email: string, password: string) => Promise<void>;
  signup: (orgName: string, email: string, password: string) => Promise<void>;
  signout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [org, setOrg] = useState<Org | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check if user is logged in
    const init = async () => {
      const token = await api.getValidToken();
      if (!token) {
        setLoading(false);
        return;
      }
      try {
        const data = await api.me();
        setUser(data.user);
        setOrg(data.org);
      } catch {
        api.clearToken();
      } finally {
        setLoading(false);
      }
    };
    init();
  }, []);

  const signin = async (email: string, password: string) => {
    const data = await api.signin(email, password);
    setUser(data.user);
    setOrg(data.org);
  };

  const signup = async (orgName: string, email: string, password: string) => {
    const data = await api.signup(orgName, email, password);
    setUser(data.user);
    setOrg(data.org);
  };

  const signout = () => {
    api.signout().catch(() => {});
    setUser(null);
    setOrg(null);
  };

  return (
    <AuthContext.Provider value={{ user, org, loading, signin, signup, signout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
