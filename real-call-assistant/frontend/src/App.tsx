import { useState, useEffect } from 'react';
import { Dashboard } from './components/Dashboard';
import { MeetingDetail } from './components/MeetingDetail';
import { LiveTranscribeOverlay } from './components/LiveTranscribeOverlay';
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
  const { settings } = useSettings();
  const [searchQuery, setSearchQuery] = useState('');
  const [isSaving, setIsSaving] = useState(false);

  // Initialize and check status
  const checkStatus = async () => {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        if (data.loading) {
          setBackendOnline(false);
        } else {
          setBackendOnline(true);
        }
        if (data.recording) {
          setView('live-transcribing');
        }

        // Auto-sync model settings on startup
        if (settings.sttModel && data.model !== settings.sttModel && !data.recording && !data.loading && !data.error) {
          try {
            await fetch('/api/config', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ model_size: settings.sttModel })
            });
          } catch (err) {
            console.error('Failed to sync settings with backend:', err);
          }
        }
      } else {
        setBackendOnline(false);
      }
    } catch (e) {
      setBackendOnline(false);
    }
  };

  const fetchHistory = async () => {
    try {
      const res = await fetch('/api/history');
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
      const res = await fetch(`/api/history/${id}`);
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

  const handleStartRecording = async () => {
    try {
      const res = await fetch('/api/start', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        if (data.status === 'started' || data.status === 'already_recording') {
          setView('live-transcribing');
        }
      } else {
        alert('Could not start recording session.');
      }
    } catch (e) {
      console.error('Error starting recording:', e);
      alert('Error starting recording.');
    }
  };

  const handleStopRecording = async () => {
    setIsSaving(true);
    try {
      const res = await fetch('/api/stop', { 
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
      const res = await fetch(`/api/history/${id}`, {
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
      const res = await fetch(`/api/history/${id}`, { method: 'DELETE' });
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
