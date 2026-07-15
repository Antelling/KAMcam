import { SNAPSHOT_VERSION } from './ViewSnapshot';
import type { ViewSnapshot } from './ViewSnapshot';
import type { Timeline } from './Timeline';

export function format(snapshot: ViewSnapshot): string;
export function format(timeline: Timeline): string;
export function format(value: ViewSnapshot | Timeline): string {
  return JSON.stringify(value, null, 2);
}

export function parse(text: string): ViewSnapshot | Timeline {
  const data = JSON.parse(text);
  if (!data || typeof data !== 'object') throw new Error('Invalid JSON');
  if (data.version !== undefined && data.version > SNAPSHOT_VERSION) {
    throw new Error(`Unsupported snapshot version: ${data.version}`);
  }
  if (data.shots && Array.isArray(data.shots)) {
    return data as Timeline;
  }
  return data as ViewSnapshot;
}
