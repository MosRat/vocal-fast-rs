import type { BootStatus } from './types';

/** A terminal boot event may beat the initial IPC reply; revisions prevent
 * that reply from regressing readiness. Startup has no timer/polling loop. */
export class BootGate {
  private status: BootStatus | null = null;
  private resolve!: (status: BootStatus) => void;
  private ready = new Promise<BootStatus>(resolve => { this.resolve = resolve; });

  observe(status: BootStatus): BootStatus {
    if (this.status && status.revision <= this.status.revision) return this.status;
    this.status = status;
    if (['ready', 'failed', 'shuttingDown'].includes(status.phase)) this.resolve(status);
    return status;
  }

  wait(initial: BootStatus): Promise<BootStatus> {
    this.observe(initial);
    return this.ready;
  }
}
