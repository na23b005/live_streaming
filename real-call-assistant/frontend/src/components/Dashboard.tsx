import React from 'react';
import { RefreshCw, Play } from 'lucide-react';
import type { Meeting } from '../types';


interface DashboardProps {
  meetings: Meeting[];
  onSelectMeeting: (id: string) => void;
  onStartRecording: () => void;
  onRefresh: () => void;
  backendOnline: boolean;
}

export const Dashboard: React.FC<DashboardProps> = ({
  meetings,
  onSelectMeeting,
  onStartRecording,
  onRefresh,
  backendOnline
}) => {


  // Group meetings by Today, Yesterday, Older
  const today = new Date().toDateString();
  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);
  const yesterdayStr = yesterday.toDateString();

  const grouped = meetings.reduce(
    (acc, meeting) => {
      const mDate = new Date(meeting.date).toDateString();
      if (mDate === today) {
        acc.today.push(meeting);
      } else if (mDate === yesterdayStr) {
        acc.yesterday.push(meeting);
      } else {
        acc.older.push(meeting);
      }
      return acc;
    },
    { today: [] as Meeting[], yesterday: [] as Meeting[], older: [] as Meeting[] }
  );

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const formatTime = (isoString: string) => {
    try {
      const date = new Date(isoString);
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }).toLowerCase();
    } catch {
      return '';
    }
  };

  return (
    <div className="dashboard-view">
      {/* Header */}
      <header className="dashboard-header">
        <div className="dashboard-title-section">
          <h1>Nexus AI</h1>
          <button className="btn-refresh" onClick={onRefresh} title="Sync History">
            <RefreshCw size={16} />
          </button>
        </div>

        <div className="controls-right">
          <button 
            className="btn-start-natively" 
            onClick={onStartRecording}
            disabled={!backendOnline}
            style={{ opacity: backendOnline ? 1 : 0.6 }}
          >
            <Play size={16} fill="white" />
            Start Nexus
          </button>
        </div>
      </header>


      {/* Meeting History Section */}
      <section className="history-section">
        {meetings.length === 0 ? (
          <div className="history-empty">
            <span className="empty-icon">🎙️</span>
            <h3>No transcription history</h3>
            <p>Your meetings and audio sessions will appear here once saved.</p>
          </div>
        ) : (
          <>
            {/* Today Group */}
            {grouped.today.length > 0 && (
              <div>
                <h3 className="history-group-title">Today</h3>
                <div className="history-list">
                  {grouped.today.map((meeting) => (
                    <div 
                      key={meeting.id} 
                      className="history-item"
                      onClick={() => onSelectMeeting(meeting.id)}
                    >
                      <div className="meeting-info-left">
                        <span className="meeting-title">{meeting.title}</span>
                      </div>
                      <div className="meeting-meta-right">
                        <span className="duration-pill">{formatDuration(meeting.duration)}</span>
                        <span className="meeting-time">{formatTime(meeting.date)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Yesterday Group */}
            {grouped.yesterday.length > 0 && (
              <div style={{ marginTop: '1.5rem' }}>
                <h3 className="history-group-title">Yesterday</h3>
                <div className="history-list">
                  {grouped.yesterday.map((meeting) => (
                    <div 
                      key={meeting.id} 
                      className="history-item"
                      onClick={() => onSelectMeeting(meeting.id)}
                    >
                      <div className="meeting-info-left">
                        <span className="meeting-title">{meeting.title}</span>
                      </div>
                      <div className="meeting-meta-right">
                        <span className="duration-pill">{formatDuration(meeting.duration)}</span>
                        <span className="meeting-time">{formatTime(meeting.date)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Older Group */}
            {grouped.older.length > 0 && (
              <div style={{ marginTop: '1.5rem' }}>
                <h3 className="history-group-title">Older</h3>
                <div className="history-list">
                  {grouped.older.map((meeting) => (
                    <div 
                      key={meeting.id} 
                      className="history-item"
                      onClick={() => onSelectMeeting(meeting.id)}
                    >
                      <div className="meeting-info-left">
                        <span className="meeting-title">{meeting.title}</span>
                      </div>
                      <div className="meeting-meta-right">
                        <span className="duration-pill">{formatDuration(meeting.duration)}</span>
                        <span className="meeting-time">
                          {new Date(meeting.date).toLocaleDateString([], { month: 'short', day: 'numeric' })}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </section>
    </div>
  );
};
