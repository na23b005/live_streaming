export interface Segment {
  speaker: string;
  start_ts: number;
  end_ts: number;
  text: string;
  is_final?: boolean;
}

export interface ChannelStats {
  segments: number;
  duration: number;
  inference_time: number;
  rtf: number;
}

export interface PerformanceStats {
  duration: number;
  mic: ChannelStats;
  sys: ChannelStats;
}

export interface Meeting {
  id: string;
  title: string;
  date: string;
  duration: number;
  segments?: Segment[];
  stats?: PerformanceStats;
  num_segments?: number;
  full_text?: string;
}

export interface BackendStatus {
  recording: boolean;
  device: string;
  model: string;
}

export const API_BASE = import.meta.env.DEV ? '' : 'http://127.0.0.1:8000';
export const WS_BASE = import.meta.env.DEV 
  ? `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`
  : 'ws://127.0.0.1:8000/ws';
