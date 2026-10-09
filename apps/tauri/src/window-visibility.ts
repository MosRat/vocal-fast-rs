interface VisibilitySource {
  isVisible(): Promise<boolean>;
  observe(callback: () => void): Promise<() => void>;
}

/** Register before checking to cover a Show/Focus event racing the IPC reply. */
export async function waitForVisible(source: VisibilitySource): Promise<void> {
  if (await source.isVisible()) return;
  await new Promise<void>((resolve, reject) => {
    let stop: (() => void) | undefined;
    let finished = false;
    const finish = (error?: unknown) => {
      if (finished) return;
      finished = true;
      stop?.();
      if (error) reject(error); else resolve();
    };
    const check = async () => {
      try { if (await source.isVisible()) finish(); } catch (error) { finish(error); }
    };
    source.observe(() => { void check(); }).then(unlisten => {
      stop = unlisten;
      if (finished) stop(); else void check();
    }).catch(finish);
  });
}
