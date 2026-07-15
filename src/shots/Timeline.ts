import { SNAPSHOT_VERSION } from './ViewSnapshot';
import type { ViewSnapshot } from './ViewSnapshot';

export type TransitionType = 'cut' | 'crossfade' | 'viewport-lerp';

export interface Transition {
  type: TransitionType;
  durationFrames: number;
  params?: Record<string, number>;
}

export interface Shot {
  snapshot: ViewSnapshot;
  durationFrames: number;
}

export interface Timeline {
  version: number;
  shots: Shot[];
  transitions: Transition[];
}
