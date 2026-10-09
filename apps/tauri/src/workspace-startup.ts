import type { Snapshot, StartupResources } from './types';

interface WorkspaceStartup {
  subscribe(): Promise<void>;
  /** Hidden webviews can connect before device discovery finishes. */
  waitForShell(): Promise<void>;
  snapshot(): Promise<Snapshot>;
  resources(): Promise<StartupResources | null>;
  present(snapshot: Snapshot, resources: StartupResources | null): void;
  /** Resolves after a paint; true only for a visible main window. */
  afterPaint(): Promise<boolean>;
  restore(revision: number): Promise<void>;
  refreshLibrary(): Promise<void>;
  report(error: unknown): void;
}

/** Finish connection without waiting on model loading or a library disk scan. */
export async function initializeWorkspace(io: WorkspaceStartup): Promise<void> {
  await io.subscribe();
  await io.waitForShell();
  const [snapshot, resources] = await Promise.all([io.snapshot(), io.resources()]);
  io.present(snapshot, resources);
  const visibleMain = await io.afterPaint();
  if (visibleMain && snapshot.boot.phase === 'ready' && snapshot.modelRestorePending)
    void io.restore(snapshot.configRevision).catch(io.report);
  void io.refreshLibrary().catch(io.report);
}
