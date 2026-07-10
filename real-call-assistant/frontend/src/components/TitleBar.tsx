import { useState, useEffect } from 'react';
import { Search, Settings, Minus, Square, X } from 'lucide-react';
import { useSettings } from './SettingsContext';

interface TitleBarProps {
  searchQuery: string;
  onSearchChange: (query: string) => void;
}

export const TitleBar: React.FC<TitleBarProps> = ({ searchQuery, onSearchChange }) => {
  const { setShowSettings } = useSettings();
  const [isTauri, setIsTauri] = useState(false);

  useEffect(() => {
    setIsTauri(typeof window !== 'undefined' && (window as any).__TAURI_INTERNALS__ !== undefined);
  }, []);

  const handleMinimize = async () => {
    if (isTauri) {
      const { getCurrentWindow } = await import('@tauri-apps/api/window');
      await getCurrentWindow().minimize();
    }
  };

  const handleMaximize = async () => {
    if (isTauri) {
      const { getCurrentWindow } = await import('@tauri-apps/api/window');
      await getCurrentWindow().toggleMaximize();
    }
  };

  const handleClose = async () => {
    if (isTauri) {
      const { getCurrentWindow } = await import('@tauri-apps/api/window');
      await getCurrentWindow().close();
    }
  };

  return (
    <div className="titlebar-container">
      {/* Background drag handle for window movement */}
      <div className="titlebar-drag-handle" data-tauri-drag-region="true" />

      {/* Title area on left */}
      <div className="titlebar-left">
        <div className="titlebar-logo-circle">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
            <path d="M4 4v16M20 4v16M4 4l16 16" />
          </svg>
        </div>
        <span className="titlebar-logo-text">Nexus AI</span>
      </div>

      {/* Centered Search Bar */}
      <div className="titlebar-center">
        <div className="titlebar-search-wrapper">
          <Search size={14} className="titlebar-search-icon" />
          <input
            type="text"
            placeholder="Search or ask anything..."
            className="titlebar-search-input"
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
          />
        </div>
      </div>

      {/* Right Controls */}
      <div className="titlebar-right">
        <button className="titlebar-btn titlebar-settings-btn" onClick={() => setShowSettings(true)} title="Settings">
          <Settings size={16} />
        </button>

        {isTauri && (
          <div className="window-controls">
            <button className="titlebar-btn control-btn minimize-btn" onClick={handleMinimize} title="Minimize">
              <Minus size={14} />
            </button>
            <button className="titlebar-btn control-btn maximize-btn" onClick={handleMaximize} title="Maximize">
              <Square size={10} />
            </button>
            <button className="titlebar-btn control-btn close-btn" onClick={handleClose} title="Close">
              <X size={14} />
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
