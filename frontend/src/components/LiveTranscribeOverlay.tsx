import React, { useEffect, useState, useRef } from 'react';
import { Square, Mic, ChevronUp } from 'lucide-react';
import type { Segment } from '../types';

interface LiveTranscribeOverlayProps {
  onStop: () => void;
}

export const LiveTranscribeOverlay: React.FC<LiveTranscribeOverlayProps> = ({
  onStop
}) => {
  const [isMinimized, setIsMinimized] = useState(false);
  const [isRequesting, setIsRequesting] = useState(false);
  const [segments, setSegments] = useState<Segment[]>([]);

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
      adjustTauriWindow(1024, 768, true, false);

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
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host || '127.0.0.1:8000';
    const wsUrl = `${protocol}//${host}/ws`;

    console.log(`Connecting to WebSocket: ${wsUrl}`);
    const socket = new WebSocket(wsUrl);
    socketRef.current = socket;

    socket.onmessage = (event) => {
      try {
        const segment: Segment = JSON.parse(event.data);
        setSegments((prev) => [...prev, segment]);
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
    if (scrollRef.current) {
      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: 'smooth'
      });
    }
  }, [segments, isMinimized]);

  const formatTimestamp = (ts: number) => {
    const mins = Math.floor(ts / 60);
    const secs = Math.floor(ts % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const handleAskSuggestions = async () => {
    setIsRequesting(true);
    console.log('Triggered Ask Suggestions endpoint (/api/suggestions)...');
    try {
      const response = await fetch('/api/suggestions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ segments })
      });
      if (response.ok) {
        const data = await response.json();
        console.log('Suggestions response data:', data);
      } else {
        console.warn(`Suggestions endpoint returned status ${response.status} (expected since it is dummy)`);
      }
    } catch (err) {
      console.error('Error in suggestions fetch (dummy endpoint):', err);
    } finally {
      setIsRequesting(false);
    }
  };

  const handleFollowUpQuestions = async () => {
    setIsRequesting(true);
    console.log('Triggered Follow Up Questions endpoint (/api/follow-up)...');
    try {
      const response = await fetch('/api/follow-up', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ segments })
      });
      if (response.ok) {
        const data = await response.json();
        console.log('Follow-up questions response data:', data);
      } else {
        console.warn(`Follow-up endpoint returned status ${response.status} (expected since it is dummy)`);
      }
    } catch (err) {
      console.error('Error in follow-up fetch (dummy endpoint):', err);
    } finally {
      setIsRequesting(false);
    }
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
    <div className={`live-overlay-container ${isMinimized ? 'minimized' : 'expanded'}`}>
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
                onClick={onStop}
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
                  const isMe = segment.speaker === 'Me';
                  return (
                    <div key={index} className="transcript-card">
                      <div className="transcript-card-header">
                        <span className="speaker-name">{isMe ? 'Me' : 'Speaker 1'}</span>
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
              disabled={isRequesting}
            >
              <span>{isRequesting ? 'Loading...' : 'Ask suggestions'}</span>
            </button>
            <button
              className="btn-nexus-footer-action"
              onClick={handleFollowUpQuestions}
              disabled={isRequesting}
            >
              <span>{isRequesting ? 'Loading...' : 'Follow up questions'}</span>
            </button>
          </div>
        </>
      )}
    </div>
  );
};
