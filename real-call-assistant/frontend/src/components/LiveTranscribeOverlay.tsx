import React, { useEffect, useState, useRef } from 'react';
import { Square, Mic, ChevronUp } from 'lucide-react';
import { WS_BASE } from '../types';
import type { Segment } from '../types';
import { useSettings } from './SettingsContext';

interface LiveTranscribeOverlayProps {
  onStop: () => void;
}

export const LiveTranscribeOverlay: React.FC<LiveTranscribeOverlayProps> = ({
  onStop
}) => {
  const { settings } = useSettings();
  const [isMinimized, setIsMinimized] = useState(false);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [suggestionsText, setSuggestionsText] = useState('Ask suggestions');
  const [followUpText, setFollowUpText] = useState('Follow up questions');
  const [showStopConfirm, setShowStopConfirm] = useState(false);

  const socketRef = useRef<WebSocket | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  // Check if we are running inside Tauri
  const isTauri = typeof window !== 'undefined' && (window as any).__TAURI_INTERNALS__ !== undefined;

  // Window adjustment helper
  const adjustTauriWindow = async (width: number, height: number, decorations: boolean, alwaysOnTop: boolean) => {
    if (!isTauri) return;
    try {
      const { getCurrentWindow, LogicalSize } = await import('@tauri-apps/api/window');
      const appWindow = getCurrentWindow();
      await appWindow.setDecorations(decorations);
      await appWindow.setAlwaysOnTop(alwaysOnTop);
      await appWindow.setSize(new LogicalSize(width, height));
    } catch (e) {
      console.error('Failed to adjust Tauri window:', e);
    }
  };

  // Adjust Tauri window and body styles when entering this view, and restore when exiting
  useEffect(() => {
    // Resize to live transcribing layout (expanded)
    adjustTauriWindow(650, 480, false, true);

    // Add transparent window body background helper class
    document.body.classList.add('live-transcribe-active');
    document.documentElement.classList.add('live-transcribe-active');

    return () => {
      // Restore to dashboard layout
      adjustTauriWindow(1024, 768, false, false);

      // Clean up transparent window class
      document.body.classList.remove('live-transcribe-active');
      document.documentElement.classList.remove('live-transcribe-active');
    };
  }, []);

  // Handle minimizing/restoring window size
  useEffect(() => {
    if (isMinimized) {
      // Minimized to logo circle size (36x36 to accommodate 32px circle + margin for scale/glow)
      adjustTauriWindow(36, 36, false, true);
    } else {
      // Expanded layout size
      adjustTauriWindow(650, 480, false, true);
    }
  }, [isMinimized]);

  // Connect WebSocket
  useEffect(() => {
    console.log(`Connecting to WebSocket: ${WS_BASE}`);
    const socket = new WebSocket(WS_BASE);
    socketRef.current = socket;

    socket.onmessage = (event) => {
      try {
        const segment: Segment = JSON.parse(event.data);
        setSegments((prev) => {
          const existsIdx = prev.findIndex((s) => s.start_ts === segment.start_ts && s.speaker === segment.speaker);
          if (existsIdx !== -1) {
            const next = [...prev];
            if (segment.text === "") {
              next.splice(existsIdx, 1);
            } else {
              const existing = prev[existsIdx];
              if (existing.is_final && !segment.is_final) {
                next[existsIdx] = {
                  ...existing,
                  text: existing.text || segment.text
                };
              } else {
                next[existsIdx] = segment;
              }
            }
            return next;
          } else {
            return segment.text === "" ? prev : [...prev, segment];
          }
        });
      } catch (err) {
        console.error('Failed to parse websocket message:', err);
      }
    };

    socket.onclose = () => {
      console.log('Websocket closed.');
    };

    socket.onerror = (err) => {
      console.error('Websocket error:', err);
    };

    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, []);

  // Auto-scroll transcripts
  useEffect(() => {
    if (settings.autoScroll && scrollRef.current) {
      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: 'smooth'
      });
    }
  }, [segments, isMinimized, settings.autoScroll]);

  const formatTimestamp = (ts: number) => {
    const mins = Math.floor(ts / 60);
    const secs = Math.floor(ts % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const handleAskSuggestions = () => {
    setSuggestionsText('Coming soon!');
    setTimeout(() => {
      setSuggestionsText('Ask suggestions');
    }, 2000);
  };

  const handleFollowUpQuestions = () => {
    setFollowUpText('Coming soon!');
    setTimeout(() => {
      setFollowUpText('Follow up questions');
    }, 2000);
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return; // Only left click

    const startX = e.clientX;
    const startY = e.clientY;
    const startTime = Date.now();
    let hasDragged = false;

    const handleMouseMove = async (moveEvent: MouseEvent) => {
      const dx = moveEvent.clientX - startX;
      const dy = moveEvent.clientY - startY;
      const distance = Math.sqrt(dx * dx + dy * dy);

      // If moved more than 4px or held down longer than 150ms, start dragging
      if (distance > 4 || (Date.now() - startTime) > 150) {
        hasDragged = true;
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
        
        if (isTauri) {
          try {
            const { getCurrentWindow } = await import('@tauri-apps/api/window');
            const appWindow = getCurrentWindow();
            await appWindow.startDragging();
          } catch (err) {
            console.error('Failed to start Tauri dragging:', err);
          }
        }
      }
    };

    const handleMouseUp = () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);

      if (!hasDragged) {
        // It's a click! Reopen the overlay
        setIsMinimized(false);
      }
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  };

  return (
    <div 
      className={`live-overlay-container ${isMinimized ? 'minimized' : 'expanded'}`}
      style={{ 
        background: isMinimized ? undefined : `rgba(20, 20, 22, ${settings.opacity / 100})`
      }}
    >
      {isMinimized ? (
        <button
          className="nexus-minimized-logo"
          onMouseDown={handleMouseDown}
          title="Reopen Nexus AI"
        >
          {/* Stylized N Logo */}
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
            <path d="M4 4v16M20 4v16M4 4l16 16" />
          </svg>
        </button>
      ) : (
        <>
          {/* Header pill container */}
          <div className="nexus-header-container" data-tauri-drag-region="true">
            <div className="nexus-header-pill">
              <div className="nexus-logo-circle" title="Nexus AI">
                {/* Stylized N Logo */}
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M4 4v16M20 4v16M4 4l16 16" />
                </svg>
              </div>

              <button
                className="btn-nexus-hide"
                onClick={() => setIsMinimized(true)}
                title="Hide Nexus AI"
              >
                <ChevronUp size={12} />
                <span>Hide</span>
              </button>

              <button
                className="btn-nexus-stop"
                onClick={() => setShowStopConfirm(true)}
                title="Stop Session"
              >
                <Square size={10} fill="currentColor" stroke="none" />
              </button>
            </div>
          </div>

          {/* Glassmorphic scrolling live transcript feed */}
          <div className="live-transcript-panel">
            <div className="live-transcript-scroll" ref={scrollRef}>
              {segments.length === 0 ? (
                <div className="live-empty-state">
                  <Mic size={20} className="empty-mic-icon" />
                  <h3>Listening...</h3>
                  <p>Waiting for audio</p>
                </div>
              ) : (
                segments.map((segment, index) => {
                  return (
                    <div key={index} className={`transcript-card ${segment.is_final === false ? 'partial' : ''}`}>
                      <div className="transcript-card-header">
                        <span className="speaker-name">
                          {segment.speaker}
                          {segment.is_final === false && (
                            <span className="typing-dot-indicator">...</span>
                          )}
                        </span>
                        <span className="timestamp">{formatTimestamp(segment.start_ts)}</span>
                      </div>
                      <div className="transcript-card-body">{segment.text}</div>
                    </div>
                  );
                })
              )}
            </div>
          </div>

          {/* Bottom Footer Box */}
          <div className="nexus-footer-box">
            <button
              className="btn-nexus-footer-action"
              onClick={handleAskSuggestions}
              title="Coming Soon"
            >
              <span>{suggestionsText}</span>
            </button>
            <button
              className="btn-nexus-footer-action"
              onClick={handleFollowUpQuestions}
              title="Coming Soon"
            >
              <span>{followUpText}</span>
            </button>
          </div>
        </>
      )}

      {showStopConfirm && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          background: 'rgba(8, 8, 10, 0.75)',
          backdropFilter: 'blur(12px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 99999,
          borderRadius: '12px'
        }}>
          <div style={{
            background: 'rgba(20, 20, 25, 0.95)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            boxShadow: '0 20px 40px -15px rgba(0, 0, 0, 0.5)',
            borderRadius: '14px',
            padding: '1.5rem',
            width: '90%',
            maxWidth: '300px',
            textAlign: 'center',
            color: '#f3f4f6',
            fontFamily: 'Inter, sans-serif'
          }}>
            <h3 style={{
              fontSize: '1.1rem',
              fontWeight: 600,
              margin: '0 0 0.5rem 0',
              color: '#ffffff'
            }}>Stop Session?</h3>
            <p style={{
              fontSize: '0.825rem',
              color: '#9ca3af',
              margin: '0 0 1.25rem 0',
              lineHeight: '1.4'
            }}>Are you sure you want to end this recording and process the final transcript?</p>
            <div style={{
              display: 'flex',
              gap: '10px',
              justifyContent: 'center'
            }}>
              <button 
                onClick={() => setShowStopConfirm(false)}
                style={{
                  flex: 1,
                  background: 'rgba(255, 255, 255, 0.05)',
                  color: '#e5e7eb',
                  border: '1px solid rgba(255, 255, 255, 0.08)',
                  borderRadius: '8px',
                  padding: '0.5rem',
                  fontSize: '0.85rem',
                  fontWeight: 500,
                  cursor: 'pointer'
                }}
              >
                Cancel
              </button>
              <button 
                onClick={onStop}
                style={{
                  flex: 1,
                  background: '#ef4444',
                  color: '#ffffff',
                  border: 'none',
                  borderRadius: '8px',
                  padding: '0.5rem',
                  fontSize: '0.85rem',
                  fontWeight: 500,
                  cursor: 'pointer'
                }}
              >
                Stop
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
