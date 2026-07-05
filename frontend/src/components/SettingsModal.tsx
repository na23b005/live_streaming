import React from 'react';
import { 
  Monitor, 
  Volume2, 
  Calendar, 
  Info, 
  LogOut, 
  X, 
  Shield, 
  ArrowDown, 
  Eye,
  ChevronUp,
  ChevronDown,
  Square,
  Edit2,
  Sliders,
  Search
} from 'lucide-react';
import { useSettings } from './SettingsContext';

export const SettingsModal: React.FC = () => {
  const { settings, updateSetting, showSettings, setShowSettings } = useSettings();
  const [activeTab, setActiveTab] = React.useState<'general' | 'audio' | 'calendar' | 'about'>('general');
  const [isPreviewingOpacity, setIsPreviewingOpacity] = React.useState(false);

  React.useEffect(() => {
    if (!isPreviewingOpacity) return;
    const handleMouseUp = () => {
      setIsPreviewingOpacity(false);
    };
    window.addEventListener('mouseup', handleMouseUp);
    return () => window.removeEventListener('mouseup', handleMouseUp);
  }, [isPreviewingOpacity]);

  if (!showSettings) return null;

  const handleQuit = async () => {
    if (confirm('Are you sure you want to quit Nexus?')) {
      const isTauri = typeof window !== 'undefined' && (window as any).__TAURI_INTERNALS__ !== undefined;
      if (isTauri) {
        try {
          const { getCurrentWindow } = await import('@tauri-apps/api/window');
          await getCurrentWindow().close();
        } catch (e) {
          console.error('Failed to close app window:', e);
        }
      } else {
        alert('Quit Nexus triggered (simulation outside desktop wrapper).');
      }
    }
  };

  return (
    <div className={`settings-overlay ${isPreviewingOpacity ? 'preview-mode' : ''}`} onClick={() => setShowSettings(false)}>
      {isPreviewingOpacity && (
        <div 
          className="live-overlay-container expanded"
          style={{ 
            background: `rgba(20, 20, 22, ${settings.opacity / 100})`,
            width: '626px',
            height: '420px',
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            pointerEvents: 'none',
            zIndex: 1
          }}
        >
          {/* Header pill container */}
          <div className="nexus-header-container">
            <div className="nexus-header-pill">
              <div className="nexus-logo-circle" title="Nexus AI">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M4 4v16M20 4v16M4 4l16 16" />
                </svg>
              </div>

              <div className="btn-nexus-hide" title="Hide Nexus AI" style={{ cursor: 'default' }}>
                <ChevronUp size={12} />
                <span>Hide</span>
              </div>

              <div className="btn-nexus-stop" title="Stop Session" style={{ cursor: 'default' }}>
                <Square size={10} fill="currentColor" stroke="none" />
              </div>
            </div>
          </div>

          {/* Glassmorphic scrolling live transcript feed */}
          <div className="live-transcript-panel">
            <div className="live-transcript-scroll" style={{ textAlign: 'left' }}>
              <div className="transcript-card">
                <div className="transcript-card-header">
                  <span className="speaker-name">Speaker 1</span>
                  <span className="timestamp">00:15</span>
                </div>
                <div className="transcript-card-body">
                  How can we optimize the database query latency for real-time dashboards?
                </div>
              </div>
              <div className="transcript-card">
                <div className="transcript-card-header">
                  <span className="speaker-name">Me</span>
                  <span className="timestamp">00:22</span>
                </div>
                <div className="transcript-card-body">
                  We should implement caching for static lookups and use connection pooling.
                </div>
              </div>
            </div>
          </div>

          {/* Bottom Footer Box */}
          <div className="nexus-footer-box">
            <div className="btn-nexus-footer-action" style={{ cursor: 'default' }}>
              <span>Ask suggestions</span>
            </div>
            <div className="btn-nexus-footer-action" style={{ cursor: 'default' }}>
              <span>Follow up questions</span>
            </div>
          </div>
        </div>
      )}
      <div className="settings-container" onClick={(e) => e.stopPropagation()}>
        {/* Left Sidebar */}
        <div className="settings-sidebar">
          <div className="sidebar-header">SETTINGS</div>
          
          <nav className="sidebar-nav">
            <button 
              className={`nav-item ${activeTab === 'general' ? 'active' : ''}`}
              onClick={() => setActiveTab('general')}
            >
              <Monitor size={16} />
              <span>General</span>
            </button>

            <button 
              className={`nav-item disabled ${activeTab === 'audio' ? 'active' : ''}`}
              onClick={() => {}}
              title="Audio settings (coming soon)"
            >
              <Volume2 size={16} />
              <span>Audio</span>
            </button>

            <button 
              className={`nav-item disabled ${activeTab === 'calendar' ? 'active' : ''}`}
              onClick={() => {}}
              title="Calendar settings (coming soon)"
            >
              <Calendar size={16} />
              <span>Calendar</span>
            </button>

            <button 
              className={`nav-item disabled ${activeTab === 'about' ? 'active' : ''}`}
              onClick={() => {}}
              title="About Nexus (coming soon)"
            >
              <Info size={16} />
              <span>About</span>
            </button>
          </nav>

          <div className="sidebar-footer">
            <button className="nav-item quit-item" onClick={handleQuit}>
              <LogOut size={16} />
              <span>Quit Nexus</span>
            </button>

            <button className="nav-item close-item" onClick={() => setShowSettings(false)}>
              <X size={16} />
              <span>Close</span>
            </button>
          </div>
        </div>

        {/* Right Content Panel */}
        <div className="settings-content">
          {activeTab === 'general' && (
            <div className="settings-pane">

              {/* Subheading: General settings */}
              <div className="settings-subheading">
                <h2>General settings</h2>
                <p>Customize how Nexus works for you</p>
              </div>

              {/* Grid of settings below subheader */}
              <div className="settings-grid">
                {/* Do not save meetings */}
                <div className="setting-row-item">
                  <div className="item-icon-badge bg-shield">
                    <Shield size={16} />
                  </div>
                  <div className="item-details">
                    <h3>Do not save meetings</h3>
                    <p>When enabled, live assistance works but transcripts, summaries, and history are discarded when the meeting ends</p>
                  </div>
                  <label className="toggle-switch">
                    <input 
                      type="checkbox" 
                      checked={settings.doNotSaveMeetings} 
                      onChange={(e) => updateSetting('doNotSaveMeetings', e.target.checked)} 
                    />
                    <span className="toggle-slider"></span>
                  </label>
                </div>

                {/* Auto Scroll */}
                <div className="setting-row-item">
                  <div className="item-icon-badge bg-scroll">
                    <ArrowDown size={16} />
                  </div>
                  <div className="item-details">
                    <h3>Auto Scroll</h3>
                    <p>Automatically scroll to the latest message as new responses arrive</p>
                  </div>
                  <label className="toggle-switch">
                    <input 
                      type="checkbox" 
                      checked={settings.autoScroll} 
                      onChange={(e) => updateSetting('autoScroll', e.target.checked)} 
                    />
                    <span className="toggle-slider"></span>
                  </label>
                </div>



                {/* Interface Opacity slider */}
                <div className="setting-slider-item">
                  <div className="slider-header-row">
                    <div className="slider-label-group">
                      <Eye size={16} className="slider-icon" />
                      <span>INTERFACE OPACITY</span>
                    </div>
                    <span className="opacity-badge">{settings.opacity}%</span>
                  </div>
                  
                  <div className="slider-input-container">
                    <input 
                      type="range" 
                      min="10" 
                      max="100" 
                      value={settings.opacity} 
                      onChange={(e) => updateSetting('opacity', parseInt(e.target.value))}
                      onPointerDown={() => setIsPreviewingOpacity(true)}
                      className="opacity-slider"
                    />
                    <div className="slider-labels">
                      <span>More Stealth</span>
                      <span>Fully Visible</span>
                    </div>
                  </div>

                  <p className="slider-caption">
                    Controls the visibility of the in-meeting overlay. Hold the slider to preview.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
