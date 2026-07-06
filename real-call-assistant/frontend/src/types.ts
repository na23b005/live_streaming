export interface Segment {
  speaker: string;
  start_ts: number;
  end_ts: number;
  text: string;
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
}

export interface BackendStatus {
  recording: boolean;
  device: string;
  model: string;
}
