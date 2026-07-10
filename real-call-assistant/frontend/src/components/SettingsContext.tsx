import React, { createContext, useContext, useState, useEffect } from 'react';

export interface AppSettings {
  doNotSaveMeetings: boolean;
  autoScroll: boolean;
  opacity: number;
  sttModel: string;
}

const DEFAULT_SETTINGS: AppSettings = {
  doNotSaveMeetings: false,
  autoScroll: true,
  opacity: 80,
  sttModel: 'moonshine/base',
};

interface SettingsContextProps {
  settings: AppSettings;
  updateSetting: <K extends keyof AppSettings>(key: K, value: AppSettings[K]) => void;
  showSettings: boolean;
  setShowSettings: (show: boolean) => void;
}

const SettingsContext = createContext<SettingsContextProps | undefined>(undefined);

export const SettingsProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [settings, setSettings] = useState<AppSettings>(() => {
    const saved = localStorage.getItem('nexus-settings');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        delete parsed.theme;
        // Sanitize legacy remote GPU model selection
        if (parsed.sttModel && parsed.sttModel.startsWith('remote/')) {
          parsed.sttModel = 'moonshine/base';
        }
        return { ...DEFAULT_SETTINGS, ...parsed };
      } catch {
        return DEFAULT_SETTINGS;
      }
    }
    return DEFAULT_SETTINGS;
  });

  const [showSettings, setShowSettings] = useState(false);

  const updateSetting = <K extends keyof AppSettings>(key: K, value: AppSettings[K]) => {
    setSettings((prev) => {
      const next = { ...prev, [key]: value };
      localStorage.setItem('nexus-settings', JSON.stringify(next));
      return next;
    });
  };

  // Always apply dark theme
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', 'dark');
  }, []);

  // Keep window always fully opaque
  useEffect(() => {
    const resetWindowOpacity = async () => {
      if (typeof window !== 'undefined' && (window as any).__TAURI_INTERNALS__ !== undefined) {
        try {
          const { getCurrentWindow } = await import('@tauri-apps/api/window');
          const appWindow = getCurrentWindow();
          await (appWindow as any).setOpacity(1.0);
        } catch (e) {
          console.error('Failed to set window opacity via Tauri:', e);
        }
      }
    };
    resetWindowOpacity();
  }, []);




  return (
    <SettingsContext.Provider value={{ settings, updateSetting, showSettings, setShowSettings }}>
      {children}
    </SettingsContext.Provider>
  );
};

export const useSettings = () => {
  const context = useContext(SettingsContext);
  if (!context) {
    throw new Error('useSettings must be used within a SettingsProvider');
  }
  return context;
};
