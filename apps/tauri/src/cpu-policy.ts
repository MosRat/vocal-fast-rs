import type { CpuExecutionConfig } from './types';

export function validateCpuConfig(config: CpuExecutionConfig): string | null {
  if (!Number.isInteger(config.threads) || config.threads < 0 || config.threads > 256) return 'CPU threads must be Automatic or between 1 and 256.';
  if (!Number.isInteger(config.spinWaitUs) || config.spinWaitUs < 0 || config.spinWaitUs > 1000) return 'CPU spin wait must be between 0 and 1000 µs.';
  return null;
}

export function cpuStreamingTiming(profile: string) {
  return profile === 'realtime_44960'
    ? { chunkMs: 250, warmupMs: 3250 }
    : { chunkMs: 512, warmupMs: 1536 };
}
