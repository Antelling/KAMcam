import type { Timeline } from '../shots/Timeline';

export interface OfflineFrameResult {
  frameIndex: number;
  data: unknown;
}

export interface OfflineRenderer {
  render(timeline: Timeline, onFrame: (result: OfflineFrameResult) => void): Promise<void>;
}
