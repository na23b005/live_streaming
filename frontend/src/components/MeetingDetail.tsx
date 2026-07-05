import React, { useState } from 'react';
import { ArrowLeft, Copy, Download, Trash2, Edit2, Check, X } from 'lucide-react';
import type { Meeting } from '../types';

interface MeetingDetailProps {
  meeting: Meeting;
  onBack: () => void;
  onRename: (id: string, newTitle: string) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
}

export const MeetingDetail: React.FC<MeetingDetailProps> = ({
  meeting,
  onBack,
  onRename,
  onDelete
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editedTitle, setEditedTitle] = useState(meeting.title);
  const [toastMessage, setToastMessage] = useState<string | null>(null);

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
      .map((s) => `[${formatTime(s.start_ts)}] ${s.speaker === 'Me' ? 'Me' : 'Speaker 1'}: ${s.text}`)
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
      .map((s) => `[${formatTime(s.start_ts)}] ${s.speaker === 'Me' ? 'Me' : 'Speaker 1'}: ${s.text}`)
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

      {/* Stats Summary Card */}
      {meeting.stats && (
        <div className="stats-summary-card">
          <div className="stat-item">
            <span className="stat-item-label">Diarization Summary</span>
            <span className="stat-item-value" style={{ fontSize: '1.05rem', color: 'var(--text-secondary)' }}>
              Me: {meeting.stats.mic.segments} segs | Speaker 1: {meeting.stats.sys.segments} segs
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-item-label">Max RTF (DirectML)</span>
            <span className="stat-item-value" style={{ color: maxRtf > 1.0 ? '#ef4444' : '#10b981' }}>
              {maxRtf.toFixed(3)}
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-item-label">Whisper Inference</span>
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
              const isMe = segment.speaker === 'Me';
              return (
                <div key={idx} className={`speech-bubble ${isMe ? 'me' : 'other'}`}>
                  <div className="bubble-meta">
                    <span className="bubble-speaker">{isMe ? 'Me' : 'Speaker 1'}</span>
                    <span className="bubble-time">{formatTime(segment.start_ts)}</span>
                  </div>
                  <div className="bubble-text">{segment.text}</div>
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
