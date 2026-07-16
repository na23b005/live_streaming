import { useState, useEffect } from 'react';
import { Dashboard } from './components/Dashboard';
import { MeetingDetail } from './components/MeetingDetail';
import { LiveTranscribeOverlay } from './components/LiveTranscribeOverlay';
import { API_BASE } from './types';
import type { Meeting } from './types';

import { SettingsProvider, useSettings } from './components/SettingsContext';
import { TitleBar } from './components/TitleBar';
import { SettingsModal } from './components/SettingsModal';

type AppView = 'dashboard' | 'meeting-detail' | 'live-transcribing';

function AppContent() {
  const [view, setView] = useState<AppView>('dashboard');
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [selectedMeeting, setSelectedMeeting] = useState<Meeting | null>(null);
  const [backendOnline, setBackendOnline] = useState(false);
  const [backendLoading, setBackendLoading] = useState(false);
  const { settings, updateSetting } = useSettings();
  const [searchQuery, setSearchQuery] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const [recoverableMeeting, setRecoverableMeeting] = useState<any>(null);

  // Initialize and check status
  const checkStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/status`);
      if (res.ok) {
        setBackendOnline(true);
        const data = await res.json();
        setBackendLoading(!!data.loading);
        if (data.recording) {
          setView('live-transcribing');
          setRecoverableMeeting(null);
        } else if (data.has_recoverable && data.recoverable_meeting) {
          setRecoverableMeeting(data.recoverable_meeting);
        } else {
          setRecoverableMeeting(null);
        }

        // Auto-sync frontend settings to match backend's active config on startup
        if (data.model && settings.sttModel !== data.model) {
          updateSetting('sttModel', data.model);
        }
        if (data.stt_language !== undefined && settings.sttLanguage !== data.stt_language) {
          updateSetting('sttLanguage', data.stt_language);
        }
        if (data.stt_initial_prompt !== undefined && settings.sttInitialPrompt !== data.stt_initial_prompt) {
          updateSetting('sttInitialPrompt', data.stt_initial_prompt);
        }
      } else {
        setBackendOnline(false);
        setBackendLoading(false);
      }
    } catch (e) {
      setBackendOnline(false);
      setBackendLoading(false);
    }
  };

  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/history`);
      if (res.ok) {
        const data = await res.json();
        setMeetings(data);
      }
    } catch (e) {
      console.error('Failed to load meeting histories:', e);
    }
  };

  useEffect(() => {
    checkStatus();
    fetchHistory();
    // Poll status periodically (every 5 seconds)
    const interval = setInterval(checkStatus, 5000);
    return () => clearInterval(interval);
  }, [settings.sttModel]);

  const handleRefresh = () => {
    checkStatus();
    fetchHistory();
  };

  const handleSelectMeeting = async (id: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/history/${id}`);
      if (res.ok) {
        const data = await res.json();
        setSelectedMeeting(data);
        setView('meeting-detail');
      } else {
        alert('Failed to load meeting details.');
      }
    } catch (e) {
      console.error('Error fetching meeting details:', e);
      alert('Error fetching meeting details.');
    }
  };

  const handleRecoverMeeting = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/recover`, { method: 'POST' });
      if (res.ok) {
        setView('live-transcribing');
        setRecoverableMeeting(null);
      } else {
        alert('Could not recover the active meeting.');
      }
    } catch (e) {
      console.error('Error recovering meeting:', e);
      alert('Error recovering meeting.');
    }
  };

  const handleDiscardMeeting = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/discard`, { method: 'POST' });
      if (res.ok) {
        setRecoverableMeeting(null);
        fetchHistory();
      } else {
        alert('Could not discard the active meeting.');
      }
    } catch (e) {
      console.error('Error discarding meeting:', e);
      alert('Error discarding meeting.');
    }
  };

  const handleStartRecording = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/start`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        if (data.status === 'started' || data.status === 'already_recording') {
          setView('live-transcribing');
        }
      } else {
        const errData = await res.json().catch(() => ({}));
        alert(errData.detail || 'Could not start recording session.');
      }
    } catch (e) {
      console.error('Error starting recording:', e);
      alert('Error starting recording.');
    }
  };

  const handleStopRecording = async () => {
    setIsSaving(true);
    try {
      const res = await fetch(`${API_BASE}/api/stop`, { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ do_not_save: settings.doNotSaveMeetings })
      });
      if (res.ok) {
        const data = await res.json();
        if (data.status === 'stopped') {
          fetchHistory(); // Refresh history list in background
          if (data.meeting_id && !settings.doNotSaveMeetings) {
            await handleSelectMeeting(data.meeting_id);
          } else {
            setView('dashboard');
          }
        }
      } else {
        alert('Could not stop recording session.');
      }
    } catch (e) {
      console.error('Error stopping recording:', e);
      alert('Error stopping recording.');
    } finally {
      setIsSaving(false);
    }
  };

  const handleRenameMeeting = async (id: string, newTitle: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/history/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: newTitle })
      });
      if (res.ok) {
        const updated = await res.json();
        // Update meeting state
        setSelectedMeeting(updated);
        // Refresh list
        fetchHistory();
      }
    } catch (e) {
      console.error('Error renaming meeting:', e);
    }
  };

  const handleDeleteMeeting = async (id: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/history/${id}`, { method: 'DELETE' });
      if (res.ok) {
        setSelectedMeeting(null);
        setView('dashboard');
        fetchHistory();
      }
    } catch (e) {
      console.error('Error deleting meeting:', e);
    }
  };

  return (
    <div className="app-container">
      {/* Show TitleBar only on dashboard and detail view */}
      {view !== 'live-transcribing' && (
        <TitleBar 
          searchQuery={searchQuery}
          onSearchChange={setSearchQuery}
        />
      )}

      {view === 'dashboard' && (
        <Dashboard
          meetings={meetings.filter(m => 
            m.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
            (m.full_text && m.full_text.toLowerCase().includes(searchQuery.toLowerCase()))
          )}
          onSelectMeeting={handleSelectMeeting}
          onStartRecording={handleStartRecording}
          onRefresh={handleRefresh}
          backendOnline={backendOnline}
          backendLoading={backendLoading}
        />
      )}

      {view === 'meeting-detail' && selectedMeeting && (
        <MeetingDetail
          meeting={selectedMeeting}
          onBack={() => { setSelectedMeeting(null); setView('dashboard'); }}
          onRename={handleRenameMeeting}
          onDelete={handleDeleteMeeting}
          searchQuery={searchQuery}
        />
      )}

      {view === 'live-transcribing' && (
        <LiveTranscribeOverlay
          onStop={handleStopRecording}
        />
      )}

      <SettingsModal />

      {isSaving && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100vw',
          height: '100vh',
          background: 'rgba(10, 10, 12, 0.85)',
          backdropFilter: 'blur(10px)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 9999,
          color: '#ffffff',
          fontFamily: 'Inter, sans-serif'
        }}>
          <div style={{
            width: '40px',
            height: '40px',
            border: '3px solid rgba(255, 255, 255, 0.1)',
            borderTop: '3px solid #6366f1',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite',
            marginBottom: '1.2rem'
          }} />
          <style>{`
            @keyframes spin {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
            }
          `}</style>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 600, margin: 0, marginBottom: '0.4rem' }}>Reprocessing Transcript</h2>
          <p style={{ fontSize: '0.9rem', color: '#9ca3af', margin: 0 }}>Aligning diarization & running high-accuracy STT refinement...</p>
        </div>
      )}

      {recoverableMeeting && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100vw',
          height: '100vh',
          background: 'rgba(8, 8, 10, 0.75)',
          backdropFilter: 'blur(16px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 10000,
          fontFamily: 'Outfit, Inter, sans-serif'
        }}>
          <div style={{
            background: 'rgba(17, 17, 21, 0.9)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)',
            borderRadius: '16px',
            padding: '2.5rem',
            width: '100%',
            maxWidth: '480px',
            textAlign: 'center',
            color: '#f3f4f6'
          }}>
            <div style={{
              width: '56px',
              height: '56px',
              borderRadius: '12px',
              background: 'rgba(99, 102, 241, 0.1)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 1.5rem auto',
              color: '#6366f1'
            }}>
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/>
              </svg>
            </div>
            
            <h2 style={{
              fontSize: '1.5rem',
              fontWeight: 600,
              margin: '0 0 0.75rem 0',
              color: '#ffffff',
              letterSpacing: '-0.025em'
            }}>
              Recover Active Session?
            </h2>
            
            <p style={{
              fontSize: '0.925rem',
              color: '#9ca3af',
              margin: '0 0 2rem 0',
              lineHeight: '1.5'
            }}>
              We found an unsaved meeting session from <strong>{new Date(recoverableMeeting.date).toLocaleString()}</strong> containing <strong>{recoverableMeeting.segments?.length || 0} transcript segments</strong>. Would you like to recover it?
            </p>
            
            <div style={{
              display: 'flex',
              gap: '12px',
              justifyContent: 'center'
            }}>
              <button 
                onClick={handleDiscardMeeting}
                style={{
                  flex: 1,
                  background: 'rgba(239, 68, 68, 0.1)',
                  color: '#ef4444',
                  border: '1px solid rgba(239, 68, 68, 0.2)',
                  borderRadius: '10px',
                  padding: '0.75rem 1.25rem',
                  fontSize: '0.9rem',
                  fontWeight: 500,
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.background = 'rgba(239, 68, 68, 0.15)';
                  e.currentTarget.style.border = '1px solid rgba(239, 68, 68, 0.3)';
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.background = 'rgba(239, 68, 68, 0.1)';
                  e.currentTarget.style.border = '1px solid rgba(239, 68, 68, 0.2)';
                }}
              >
                Discard Session
              </button>
              
              <button 
                onClick={handleRecoverMeeting}
                style={{
                  flex: 1,
                  background: '#6366f1',
                  color: '#ffffff',
                  border: 'none',
                  borderRadius: '10px',
                  padding: '0.75rem 1.25rem',
                  fontSize: '0.9rem',
                  fontWeight: 500,
                  cursor: 'pointer',
                  boxShadow: '0 4px 12px rgba(99, 102, 241, 0.3)',
                  transition: 'all 0.2s'
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.background = '#4f46e5';
                  e.currentTarget.style.boxShadow = '0 6px 16px rgba(99, 102, 241, 0.4)';
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.background = '#6366f1';
                  e.currentTarget.style.boxShadow = '0 4px 12px rgba(99, 102, 241, 0.3)';
                }}
              >
                Recover Session
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function App() {
  return (
    <SettingsProvider>
      <AppContent />
    </SettingsProvider>
  );
}

export default App;
