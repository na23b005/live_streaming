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
  const [hardwareInfo, setHardwareInfo] = useState('Loading hardware profile...');
  const { settings } = useSettings();
  const [searchQuery, setSearchQuery] = useState('');

  // Initialize and check status
  const checkStatus = async () => {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        setBackendOnline(true);
        setHardwareInfo(`Device: ${data.device} | Model: ${data.model}`);
        if (data.recording) {
          setView('live-transcribing');
        }
      } else {
        setBackendOnline(false);
        setHardwareInfo('Backend Connection Offline');
      }
    } catch (e) {
      setBackendOnline(false);
      setHardwareInfo('Backend Connection Offline');
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
  }, []);

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
    try {
      const res = await fetch('/api/stop', { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ do_not_save: settings.doNotSaveMeetings })
      });
      if (res.ok) {
        const data = await res.json();
        if (data.status === 'stopped') {
          setView('dashboard');
          fetchHistory();
        }
      } else {
        alert('Could not stop recording session.');
      }
    } catch (e) {
      console.error('Error stopping recording:', e);
      alert('Error stopping recording.');
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
          meetings={meetings.filter(m => m.title.toLowerCase().includes(searchQuery.toLowerCase()))}
          onSelectMeeting={handleSelectMeeting}
          onStartRecording={handleStartRecording}
          onRefresh={handleRefresh}
          backendOnline={backendOnline}
          hardwareInfo={hardwareInfo}
        />
      )}

      {view === 'meeting-detail' && selectedMeeting && (
        <MeetingDetail
          meeting={selectedMeeting}
          onBack={() => { setSelectedMeeting(null); setView('dashboard'); }}
          onRename={handleRenameMeeting}
          onDelete={handleDeleteMeeting}
        />
      )}

      {view === 'live-transcribing' && (
        <LiveTranscribeOverlay
          onStop={handleStopRecording}
        />
      )}

      <SettingsModal />
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
