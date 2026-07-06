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
  Cpu,
  Trash2,
  Download,
  Check,
  Zap,
  HardDrive
} from 'lucide-react';
import { useSettings } from './SettingsContext';

interface STTModel {
  id: string;
  name: string;
  size: string;
  speed: string;
  accuracy: string;
  downloaded: boolean;
  downloading: boolean;
  is_recommended: boolean;
}

export const SettingsModal: React.FC = () => {
  const { settings, updateSetting, showSettings, setShowSettings } = useSettings();
  const [activeTab, setActiveTab] = React.useState<'general' | 'audio' | 'calendar' | 'about'>('general');
  const [isPreviewingOpacity, setIsPreviewingOpacity] = React.useState(false);
  const [isRecording, setIsRecording] = React.useState(false);
  const [isModelLoading, setIsModelLoading] = React.useState(false);
  const [activeBackendModel, setActiveBackendModel] = React.useState('');
  const [models, setModels] = React.useState<STTModel[]>([]);
  const [recommendedModelName, setRecommendedModelName] = React.useState('Moonshine Tiny');

  const fetchModels = async () => {
    try {
      const res = await fetch('/api/models');
      if (res.ok) {
        const data = await res.json();
        setModels(data);
        const recommended = data.find((m: any) => m.is_recommended);
        if (recommended) {
          setRecommendedModelName(recommended.name);
        }
      }
    } catch (e) {
      console.error('Failed to fetch models list:', e);
    }
  };

  const fetchBackendStatus = async () => {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        setIsRecording(data.recording);
        setIsModelLoading(data.loading);
        setActiveBackendModel(data.model);
      }
    } catch (e) {
      console.error('Failed to fetch backend status in modal:', e);
    }
  };

  React.useEffect(() => {
    if (!showSettings) return;

    fetchModels();
    fetchBackendStatus();

    const interval = setInterval(() => {
      fetchBackendStatus();
      fetchModels();
    }, 2000);
    return () => clearInterval(interval);
  }, [showSettings]);

  const handleInstallModel = async (modelId: string) => {
    setModels(prev => prev.map(m => m.id === modelId ? { ...m, downloading: true } : m));
    try {
      const res = await fetch('/api/models/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: modelId })
      });
      if (res.ok) {
        fetchModels();
      }
    } catch (e) {
      console.error('Failed to start model download:', e);
    }
  };

  const handleDeleteModel = async (modelId: string) => {
    if (!confirm('Are you sure you want to delete this model to free up space?')) return;
    try {
      const res = await fetch('/api/models/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: modelId })
      });
      if (res.ok) {
        fetchModels();
        fetchBackendStatus();
      }
    } catch (e) {
      console.error('Failed to delete model:', e);
    }
  };

  const handleModelChange = async (newModel: string) => {
    updateSetting('sttModel', newModel);
    setIsModelLoading(true);

    try {
      const res = await fetch('/api/config', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ model_size: newModel }),
      });

      if (!res.ok) {
        const errData = await res.json();
        console.error('Failed to update config on the backend:', errData.detail);
        setIsModelLoading(false);
      } else {
        const checkStatus = async () => {
          try {
            const statusRes = await fetch('/api/status');
            if (statusRes.ok) {
              const data = await statusRes.json();
              setIsModelLoading(data.loading);
              setActiveBackendModel(data.model);
              if (!data.loading) {
                clearInterval(pollInterval);
              }
            }
          } catch (e) {
            clearInterval(pollInterval);
            setIsModelLoading(false);
          }
        };
        const pollInterval = setInterval(checkStatus, 1000);
      }
    } catch (e) {
      console.error('Error changing model:', e);
      setIsModelLoading(false);
    }
  };

  React.useEffect(() => {
    if (!isPreviewingOpacity) return;
    const handleMouseUp = () => {
      setIsPreviewingOpacity(false);
    };
    window.addEventListener('mouseup', handleMouseUp);
    return () => window.removeEventListener('mouseup', handleMouseUp);
  }, [isPreviewingOpacity]);

  const downloadedModels = models.filter(m => m.downloaded);

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
        <div className="settings-sidebar" data-tauri-drag-region="true">
          <div className="sidebar-header" data-tauri-drag-region="true">SETTINGS</div>
          
          <nav className="sidebar-nav">
            <button 
              className={`nav-item ${activeTab === 'general' ? 'active' : ''}`}
              onClick={() => setActiveTab('general')}
            >
              <Monitor size={16} />
              <span>General</span>
            </button>

            <button 
              className={`nav-item ${activeTab === 'audio' ? 'active' : ''}`}
              onClick={() => setActiveTab('audio')}
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

          {activeTab === 'audio' && (
            <div className="settings-pane">
              <div className="settings-subheading">
                <h2>Audio settings</h2>
                <p>Configure speech-to-text models and hardware acceleration</p>
              </div>

              {/* Speech Provider Card */}
              <div className="speech-provider-card">
                <span className="card-label">Speech Provider</span>
                <div className="provider-selector-box">
                  <div className="provider-details-left">
                    <div className="provider-icon-wrapper">
                      <Cpu size={18} />
                    </div>
                    <div className="provider-texts">
                      <span className="provider-name">Local Whisper</span>
                      <span className="provider-desc">Privacy-first: runs 100% on your device</span>
                    </div>
                  </div>
                  <div className="provider-arrow">
                    <ChevronDown size={16} />
                  </div>
                </div>
              </div>

              {/* Local Engine Configuration */}
              <div className="engine-config-card">
                <h3 className="section-title">Local Engine Configuration</h3>
                <p className="section-subtitle">
                  Select the AI models you want to use for Speech-to-Text inference.
                </p>

                <div className="global-model-selector-group">
                  <span className="group-label">GLOBAL MODEL</span>
                  <div className="select-container">
                    <select
                      value={activeBackendModel || settings.sttModel || 'moonshine/base'}
                      onChange={(e) => handleModelChange(e.target.value)}
                      disabled={isRecording || isModelLoading || downloadedModels.length === 0}
                      className="model-select-dropdown"
                    >
                      {downloadedModels.length === 0 ? (
                        <option value="">No models downloaded. Install a model below.</option>
                      ) : (
                        downloadedModels.map(m => (
                          <option key={m.id} value={m.id}>{m.name}</option>
                        ))
                      )}
                    </select>
                  </div>
                </div>
              </div>

              {/* Model Manager Section */}
              <div className="model-manager-section">
                <div className="manager-header-row">
                  <h3 className="section-title">Model Manager</h3>
                  <div className="recommendation-badge-pill">
                    <span>Recommended for your PC: <strong>{recommendedModelName}</strong></span>
                  </div>
                </div>

                <div className="model-manager-list">
                  {models.map(m => (
                    <div key={m.id} className="model-manager-row">
                      <div className="model-info-left">
                        <div className="model-name-row">
                          <span className="model-name">{m.name}</span>
                          {m.is_recommended && (
                            <span className="recommended-tag">RECOMMENDED</span>
                          )}
                        </div>
                        <div className="model-meta-row">
                          <span className="meta-item">
                            <HardDrive size={13} />
                            <span>{m.size}</span>
                          </span>
                          <span className="meta-item">
                            <Zap size={13} />
                            <span>{m.speed}</span>
                          </span>
                          <span className="meta-item">
                            <Check size={13} />
                            <span>{m.accuracy}</span>
                          </span>
                        </div>
                      </div>

                      <div className="model-action-right">
                        {m.downloading ? (
                          <div className="downloading-status">
                            <div className="spinner-small"></div>
                            <span>Downloading...</span>
                          </div>
                        ) : m.downloaded ? (
                          <button 
                            className="btn-delete-model"
                            onClick={() => handleDeleteModel(m.id)}
                            disabled={isRecording || isModelLoading || activeBackendModel === m.id}
                            title={activeBackendModel === m.id ? "Cannot delete the active model" : "Delete model"}
                          >
                            <Trash2 size={16} />
                          </button>
                        ) : (
                          <button 
                            className="btn-install-model"
                            onClick={() => handleInstallModel(m.id)}
                            disabled={isRecording || isModelLoading}
                          >
                            <Download size={14} />
                            <span>Install</span>
                          </button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

            </div>
          )}
        </div>
      </div>
    </div>
  );
};
