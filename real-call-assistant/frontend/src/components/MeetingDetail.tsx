import React, { useState } from 'react';
import { ArrowLeft, Copy, Download, Trash2, Edit2, Check, X, HelpCircle } from 'lucide-react';
import type { Meeting } from '../types';
import { API_BASE } from '../types';

interface MeetingDetailProps {
  meeting: Meeting;
  onBack: () => void;
  onRename: (id: string, newTitle: string) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
  searchQuery?: string;
}

export const MeetingDetail: React.FC<MeetingDetailProps> = ({
  meeting,
  onBack,
  onRename,
  onDelete,
  searchQuery = ''
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editedTitle, setEditedTitle] = useState(meeting.title);
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [showRtfTooltip, setShowRtfTooltip] = useState(false);

  const highlightText = (text: string, search: string) => {
    if (!search.trim()) return text;
    const parts = text.split(new RegExp(`(${search.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&')})`, 'gi'));
    return (
      <>
        {parts.map((part, i) => 
          part.toLowerCase() === search.toLowerCase() ? (
            <mark key={i} style={{ backgroundColor: '#facc15', color: '#1e293b', borderRadius: '2px', padding: '0 2px' }}>
              {part}
            </mark>
          ) : (
            part
          )
        )}
      </>
    );
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}m ${secs}s`;
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const showToast = (msg: string) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 3000);
  };

  const handleSaveRename = async () => {
    if (!editedTitle.trim()) return;
    await onRename(meeting.id, editedTitle);
    setIsEditing(false);
    showToast('Meeting renamed');
  };

  const handleCopy = () => {
    if (!meeting.segments || meeting.segments.length === 0) {
      showToast('Transcript is empty');
      return;
    }
    const text = meeting.segments
      .map((s) => `[${formatTime(s.start_ts)}] ${s.speaker}: ${s.text}`)
      .join('\n');
    
    navigator.clipboard.writeText(text);
    showToast('Copied to clipboard!');
  };

  const handleExport = () => {
    if (!meeting.segments || meeting.segments.length === 0) {
      showToast('Transcript is empty');
      return;
    }
    const text = meeting.segments
      .map((s) => `[${formatTime(s.start_ts)}] ${s.speaker}: ${s.text}`)
      .join('\n');
    
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `${meeting.title.replace(/\s+/g, '_')}_transcript.txt`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    showToast('Transcript exported!');
  };

  const handleDeleteClick = async () => {
    if (window.confirm('Are you sure you want to delete this meeting?')) {
      await onDelete(meeting.id);
    }
  };

  const maxRtf = meeting.stats ? Math.max(meeting.stats.mic.rtf, meeting.stats.sys.rtf) : 0;

  return (
    <div className="meeting-detail-view">
      {/* Detail Header */}
      <div className="detail-header">
        <div className="detail-title-section">
          <div className="detail-back-row" onClick={onBack}>
            <ArrowLeft size={16} />
            <span>Back to Dashboard</span>
          </div>
          
          <div className="title-edit-container">
            {isEditing ? (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', width: '100%' }}>
                <input
                  type="text"
                  className="title-input-edit"
                  value={editedTitle}
                  onChange={(e) => setEditedTitle(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSaveRename()}
                  autoFocus
                />
                <button className="btn-icon-action" onClick={handleSaveRename}>
                  <Check size={20} style={{ color: '#10b981' }} />
                </button>
                <button className="btn-icon-action" onClick={() => { setIsEditing(false); setEditedTitle(meeting.title); }}>
                  <X size={20} style={{ color: '#ef4444' }} />
                </button>
              </div>
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <h2>{meeting.title}</h2>
                <button className="btn-icon-action" onClick={() => setIsEditing(true)}>
                  <Edit2 size={16} />
                </button>
              </div>
            )}
          </div>

          <div className="detail-meta-info">
            <span>{new Date(meeting.date).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })}</span>
            <span style={{ color: 'var(--text-muted)' }}>•</span>
            <span>Duration: {formatDuration(meeting.duration)}</span>
          </div>
        </div>

        <div className="detail-actions-panel">
          <button className="btn-secondary" onClick={handleCopy}>
            <Copy size={14} />
            Copy
          </button>
          <button className="btn-secondary" onClick={handleExport}>
            <Download size={14} />
            Save TXT
          </button>
          <button className="btn-danger-outline" onClick={handleDeleteClick}>
            <Trash2 size={14} />
            Delete
          </button>
        </div>
      </div>

      {/* Audio Player Card */}
      <div className="audio-player-card" style={{
        background: 'rgba(30, 30, 35, 0.6)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(255, 255, 255, 0.08)',
        borderRadius: '12px',
        padding: '1rem',
        marginBottom: '1.5rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.5rem'
      }}>
        <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)' }}>Meeting Audio Playback</div>
        <audio 
          src={`${API_BASE}/api/history/${meeting.id}/audio`} 
          controls 
          style={{ width: '100%', outline: 'none' }}
        />
      </div>

      {/* Stats Summary Card */}
      {meeting.stats && (
        <div className="stats-summary-card">
          <div className="stat-item">
            <span 
              className="stat-item-label" 
              style={{ position: 'relative', display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer' }}
              onMouseEnter={() => setShowRtfTooltip(true)}
              onMouseLeave={() => setShowRtfTooltip(false)}
            >
              Max Rtf
              <HelpCircle 
                size={14} 
                style={{ color: 'var(--text-muted)' }}
              />
              {showRtfTooltip && (
                <div style={{
                  position: 'absolute',
                  bottom: '100%',
                  left: '50%',
                  transform: 'translateX(-50%) translateY(-8px)',
                  background: 'rgba(20, 20, 24, 0.96)',
                  backdropFilter: 'blur(10px)',
                  border: '1px solid rgba(255, 255, 255, 0.08)',
                  borderRadius: '6px',
                  padding: '6px 10px',
                  color: '#9ca3af',
                  fontSize: '11px',
                  fontWeight: 400,
                  fontFamily: 'Inter, sans-serif',
                  lineHeight: '1.4',
                  width: '220px',
                  zIndex: 9999,
                  boxShadow: '0 4px 12px rgba(0, 0, 0, 0.5)',
                  pointerEvents: 'none',
                  whiteSpace: 'normal',
                  textAlign: 'left',
                  textTransform: 'none'
                }}>
                  Real-time factor (Rtf) measures transcription speed relative to audio length. For example, a 0.1 Rtf means 1 minute of audio is transcribed in 6 seconds.
                </div>
              )}
            </span>
            <span className="stat-item-value" style={{ color: maxRtf > 1.0 ? '#ef4444' : '#10b981' }}>
              {maxRtf.toFixed(3)}
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-item-label">Whisper Inference Time</span>
            <span className="stat-item-value">
              {(meeting.stats.mic.inference_time + meeting.stats.sys.inference_time).toFixed(1)}s
            </span>
          </div>
        </div>
      )}

      {/* Transcript Box */}
      <div className="transcript-box">
        <div className="transcript-box-header">Conversation Transcript</div>
        <div className="transcript-viewport">
          {!meeting.segments || meeting.segments.length === 0 ? (
            <div 
              style={{ 
                height: '100%', 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center',
                color: 'var(--text-muted)' 
              }}
            >
              No speech segments transcribed in this session.
            </div>
          ) : (
            meeting.segments.map((segment, idx) => {
              const isMe = segment.speaker === 'Speaker 1';
              return (
                <div key={idx} className={`speech-bubble ${isMe ? 'me' : 'other'}`}>
                  <div className="bubble-meta">
                    <span className="bubble-speaker">{segment.speaker}</span>
                    <span className="bubble-time">{formatTime(segment.start_ts)}</span>
                  </div>
                  <div className="bubble-text">{highlightText(segment.text, searchQuery)}</div>
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* Toast Notification */}
      {toastMessage && (
        <div className="toast-notification">
          <span>{toastMessage}</span>
        </div>
      )}
    </div>
  );
};
