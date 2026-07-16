import React from 'react';
import { 
  Monitor, 
  Volume2, 
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
  HardDrive,
  Mic
} from 'lucide-react';
import { useSettings } from './SettingsContext';
import { API_BASE } from '../types';

interface STTModel {
  id: string;
  name: string;
  size: string;
  speed: string;
  accuracy: string;
  downloaded: boolean;
  downloading: boolean;
  progress?: number;
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
  
  // Audio devices configuration states
  const [mics, setMics] = React.useState<{id: string, name: string}[]>([]);
  const [speakers, setSpeakers] = React.useState<{id: string, name: string}[]>([]);
  const [selectedMic, setSelectedMic] = React.useState('');
  const [selectedSpeaker, setSelectedSpeaker] = React.useState('');
  const [inputLevel, setInputLevel] = React.useState(0);
  const [isTestingSound, setIsTestingSound] = React.useState(false);

  const fetchModels = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/models`);
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
      const res = await fetch(`${API_BASE}/api/status`);
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

  const fetchAudioDevices = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/audio-devices`);
      if (res.ok) {
        const data = await res.json();
        setMics(data.mics || []);
        setSpeakers(data.speakers || []);
      }
    } catch (e) {
      console.error('Failed to fetch audio devices:', e);
    }
  };

  const fetchAudioConfig = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/config`);
      if (res.ok) {
        const data = await res.json();
        setSelectedMic(data.mic_device || '');
        setSelectedSpeaker(data.speaker_device || '');
      }
    } catch (e) {
      console.error('Failed to fetch audio config:', e);
    }
  };

  const handleDeviceChange = async (type: 'mic' | 'speaker', deviceId: string) => {
    try {
      const payload = type === 'mic' ? { mic_device: deviceId } : { speaker_device: deviceId };
      const res = await fetch(`${API_BASE}/api/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (res.ok) {
        if (type === 'mic') {
          setSelectedMic(deviceId);
        } else {
          setSelectedSpeaker(deviceId);
        }
      }
    } catch (e) {
      console.error('Failed to update config on the backend:', e);
    }
  };

  const handleTestSound = async () => {
    setIsTestingSound(true);
    try {
      await fetch(`${API_BASE}/api/test-sound`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ speaker_device: selectedSpeaker })
      });
    } catch (e) {
      console.error('Failed to test sound:', e);
    } finally {
      setTimeout(() => setIsTestingSound(false), 800);
    }
  };

  // Web Audio API mic level measurement for real-time visualization in Settings
  React.useEffect(() => {
    if (activeTab !== 'audio' || !showSettings) {
      setInputLevel(0);
      return;
    }
    
    let audioContext: AudioContext | null = null;
    let analyser: AnalyserNode | null = null;
    let microphone: MediaStreamAudioSourceNode | null = null;
    let javascriptNode: ScriptProcessorNode | null = null;
    let stream: MediaStream | null = null;
    
    // Find the browser device ID that matches the selected backend mic name
    const getConstraints = async () => {
      const constraints: MediaStreamConstraints = { video: false };
      
      try {
        // We first need temporary mic permission to fetch labels in enumerateDevices
        const tempStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        tempStream.getTracks().forEach(t => t.stop());
        
        const devices = await navigator.mediaDevices.enumerateDevices();
        const audioInputs = devices.filter(d => d.kind === 'audioinput');
        
        // Find matching backend mic in mics list
        const currentMicObj = mics.find(m => m.id === selectedMic);
        const micName = currentMicObj ? currentMicObj.name : '';
        
        if (micName) {
          const match = audioInputs.find(d => 
            d.label.toLowerCase().includes(micName.toLowerCase()) || 
            micName.toLowerCase().includes(d.label.toLowerCase())
          );
          if (match) {
            constraints.audio = { deviceId: { exact: match.deviceId } };
            return constraints;
          }
        }
      } catch (e) {
        console.warn('Enumerate devices failed or permission denied:', e);
      }
      
      constraints.audio = true; // Fallback to default
      return constraints;
    };
    
    getConstraints().then((constraints) => {
      navigator.mediaDevices.getUserMedia(constraints)
        .then((s) => {
          stream = s;
          audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
          analyser = audioContext.createAnalyser();
          microphone = audioContext.createMediaStreamSource(s);
          javascriptNode = audioContext.createScriptProcessor(2048, 1, 1);
          
          analyser.smoothingTimeConstant = 0.8;
          analyser.fftSize = 1024;
          
          microphone.connect(analyser);
          analyser.connect(javascriptNode);
          javascriptNode.connect(audioContext.destination);
          
          javascriptNode.onaudioprocess = () => {
            if (!analyser) return;
            const array = new Uint8Array(analyser.frequencyBinCount);
            analyser.getByteFrequencyData(array);
            let values = 0;
            const length = array.length;
            for (let i = 0; i < length; i++) {
              values += array[i];
            }
            const average = values / length;
            setInputLevel(Math.min(1.0, average / 48.0)); // normalize & cap
          };
        })
        .catch((err) => {
          console.error('Failed to get mic stream for visualization:', err);
        });
    });
      
    return () => {
      if (javascriptNode) javascriptNode.disconnect();
      if (microphone) microphone.disconnect();
      if (analyser) analyser.disconnect();
      if (audioContext) audioContext.close();
      if (stream) stream.getTracks().forEach(track => track.stop());
    };
  }, [activeTab, showSettings, selectedMic, mics]);

  React.useEffect(() => {
    if (!showSettings) return;

    fetchModels();
    fetchBackendStatus();
    fetchAudioDevices();
    fetchAudioConfig();

    const interval = setInterval(() => {
      fetchBackendStatus();
      fetchModels();
    }, 2000);
    return () => clearInterval(interval);
  }, [showSettings]);

  const handleInstallModel = async (modelId: string) => {
    setModels(prev => prev.map(m => m.id === modelId ? { ...m, downloading: true } : m));
    try {
      const res = await fetch(`${API_BASE}/api/models/download`, {
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
  const handleCancelDownload = async (modelId: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/models/cancel`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: modelId })
      });
      if (res.ok) {
        fetchModels();
      }
    } catch (e) {
      console.error('Failed to cancel model download:', e);
    }
  };

  const handleDeleteModel = async (modelId: string) => {
    if (!confirm('Are you sure you want to delete this model to free up space?')) return;
    try {
      const res = await fetch(`${API_BASE}/api/models/delete`, {
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
      const res = await fetch(`${API_BASE}/api/config`, {
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
            const statusRes = await fetch(`${API_BASE}/api/status`);
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

              {/* STT Engine Configuration Section */}
              <div className="settings-subheading" style={{ marginTop: '0.5rem' }}>
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
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <h3 className="section-title">Local Engine Configuration</h3>
                  {isModelLoading && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#6366f1', fontSize: '0.8rem', fontWeight: 500 }}>
                      <div className="spinner-small" style={{ width: '12px', height: '12px', borderWidth: '2px' }} />
                      <span>Loading Engine...</span>
                    </div>
                  )}
                </div>
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
              <div className="model-manager-section" style={{ marginBottom: '2.5rem' }}>
                <div className="manager-header-row">
                  <h3 className="section-title">Model Manager</h3>
                  <div className="recommendation-badge-pill">
                    <span>Recommended: <strong>{recommendedModelName}</strong></span>
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
                          <div className="downloading-container" style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '6px', minWidth: '150px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.8rem', color: '#6366f1', fontWeight: 500 }}>
                              <div className="spinner-small" style={{ width: '12px', height: '12px', borderWidth: '2px', animation: 'spin 1s linear infinite' }}></div>
                              <span>{m.progress !== undefined && m.progress >= 0 ? (m.progress === 99 ? 'Converting...' : `Downloading (${m.progress}%)`) : 'Starting...'}</span>
                            </div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', width: '100%' }}>
                              <div className="progress-bar-bg" style={{ flexGrow: 1, height: '4px', backgroundColor: '#2e2f3d', borderRadius: '2px', overflow: 'hidden' }}>
                                <div className="progress-bar-fill" style={{ width: `${m.progress || 0}%`, height: '100%', backgroundColor: '#6366f1', transition: 'width 0.3s ease' }}></div>
                              </div>
                              <button
                                className="btn-cancel-download"
                                onClick={() => handleCancelDownload(m.id)}
                                style={{
                                  background: 'none',
                                  border: 'none',
                                  color: '#ef4444',
                                  cursor: 'pointer',
                                  padding: '2px',
                                  display: 'flex',
                                  alignItems: 'center',
                                  justifyContent: 'center',
                                  opacity: 0.8,
                                  transition: 'opacity 0.2s, transform 0.2s',
                                }}
                                onMouseEnter={(e) => e.currentTarget.style.opacity = '1'}
                                onMouseLeave={(e) => e.currentTarget.style.opacity = '0.8'}
                                title="Cancel Download"
                              >
                                <X size={14} />
                              </button>
                            </div>
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

              {/* Audio Configuration Section */}
              <div className="settings-subheading" style={{ marginTop: '0.5rem', marginBottom: '1rem' }}>
                <h2>Audio Configuration</h2>
                <p>Manage input and output devices.</p>
              </div>

              <div className="settings-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1.25rem', marginBottom: '2rem' }}>
                {/* Input Device select */}
                <div className="setting-select-card">
                  <div className="card-header-row">
                    <Mic size={16} />
                    <span>INPUT DEVICE</span>
                  </div>
                  <div className="select-container">
                    <select 
                      value={selectedMic} 
                      onChange={(e) => handleDeviceChange('mic', e.target.value)}
                      className="model-select-dropdown"
                    >
                      <option value="">Default Microphone</option>
                      {mics.map(m => (
                        <option key={m.id} value={m.id}>{m.name}</option>
                      ))}
                    </select>
                  </div>
                  
                  {/* Real-time input level meter */}
                  <div className="input-level-meter-container" style={{ marginTop: '0.5rem' }}>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '0.35rem' }}>Input Level</span>
                    <div style={{ 
                      width: '100%', 
                      height: '6px', 
                      background: '#1f2029', 
                      borderRadius: '3px',
                      overflow: 'hidden'
                    }}>
                      <div style={{ 
                        width: `${inputLevel * 100}%`, 
                        height: '100%', 
                        background: 'linear-gradient(to right, #10b981, #34d399)',
                        transition: 'width 0.1s ease',
                        borderRadius: '3px'
                      }} />
                    </div>
                  </div>
                </div>

                {/* Output Device select */}
                <div className="setting-select-card">
                  <div className="card-header-row">
                    <Volume2 size={16} />
                    <span>OUTPUT DEVICE</span>
                  </div>
                  <div className="select-container">
                    <select 
                      value={selectedSpeaker} 
                      onChange={(e) => handleDeviceChange('speaker', e.target.value)}
                      className="model-select-dropdown"
                    >
                      <option value="">Default Speakers</option>
                      {speakers.map(s => (
                        <option key={s.id} value={s.id}>{s.name}</option>
                      ))}
                    </select>
                  </div>
                  
                  {/* Test sound button aligned right */}
                  <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '0.5rem' }}>
                    <button 
                      className="btn-secondary" 
                      onClick={handleTestSound}
                      disabled={isTestingSound}
                      style={{ fontSize: '0.8rem', padding: '0.4rem 0.8rem', display: 'flex', alignItems: 'center', gap: '6px' }}
                    >
                      <Volume2 size={12} />
                      <span>{isTestingSound ? 'Playing...' : 'Test Sound'}</span>
                    </button>
                  </div>
                </div>
              </div>

            </div>
          )}
        </div>
      </div>
    </div>
  );
};
