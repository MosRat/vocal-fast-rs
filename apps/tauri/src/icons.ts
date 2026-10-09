export type IconName =
  | 'waveform'
  | 'play'
  | 'stop'
  | 'sparkle'
  | 'folder'
  | 'download'
  | 'refresh'
  | 'settings'
  | 'speaker'
  | 'sliders'
  | 'chart'
  | 'moon'
  | 'sun'
  | 'desktop'
  | 'minimize'
  | 'maximize'
  | 'restore'
  | 'close'
  | 'chevron'
  | 'check'
  | 'alert'
  | 'cpu'
  | 'database'
  | 'keyboard'
  | 'inspect';

const paths: Record<IconName, string> = {
  waveform: '<path d="M3 12h2l2.2-6 3.6 12 2.2-8 1.8 4H21"/>',
  play: '<path d="m8 5 11 7-11 7V5Z" fill="currentColor" stroke="none"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" stroke="none"/>',
  sparkle:
    '<path d="m12 3 1.4 5.6L19 10l-5.6 1.4L12 17l-1.4-5.6L5 10l5.6-1.4L12 3Z"/><path d="m19 15 .6 2.4L22 18l-2.4.6L19 21l-.6-2.4L16 18l2.4-.6L19 15Z"/>',
  folder:
    '<path d="M3 7.5A2.5 2.5 0 0 1 5.5 5h4l2 2H18.5A2.5 2.5 0 0 1 21 9.5v7A2.5 2.5 0 0 1 18.5 19h-13A2.5 2.5 0 0 1 3 16.5v-9Z"/>',
  download:'<path d="M12 3v12m-5-5 5 5 5-5M4 16v4h16v-4"/>',
  refresh:
    '<path d="M20 11a8 8 0 0 0-14.9-4L3 10"/><path d="M3 5v5h5"/><path d="M4 13a8 8 0 0 0 14.9 4L21 14"/><path d="M21 19v-5h-5"/>',
  settings:
    '<path d="M12 8.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7Z"/><path d="m19.4 15 .1.1a1.8 1.8 0 0 1-2.5 2.5l-.1-.1a1.8 1.8 0 0 0-3 .8v.2a1.8 1.8 0 0 1-3.6 0v-.2a1.8 1.8 0 0 0-3-.8l-.1.1a1.8 1.8 0 0 1-2.5-2.5l.1-.1a1.8 1.8 0 0 0-.8-3h-.2a1.8 1.8 0 0 1 0-3.6h.2a1.8 1.8 0 0 0 .8-3l-.1-.1a1.8 1.8 0 0 1 2.5-2.5l.1.1a1.8 1.8 0 0 0 3-.8v-.2a1.8 1.8 0 0 1 3.6 0v.2a1.8 1.8 0 0 0 3 .8l.1-.1a1.8 1.8 0 0 1 2.5 2.5l-.1.1a1.8 1.8 0 0 0 .8 3h.2a1.8 1.8 0 0 1 0 3.6h-.2a1.8 1.8 0 0 0-.8 3Z"/>',
  speaker: '<path d="M4 10v4h4l5 4V6L8 10H4Z"/><path d="M17 9a4.3 4.3 0 0 1 0 6M19.5 6.5a8 8 0 0 1 0 11"/>',
  sliders:
    '<path d="M4 6h16M4 12h16M4 18h16"/><circle cx="9" cy="6" r="2" fill="var(--surface-1)"/><circle cx="15" cy="12" r="2" fill="var(--surface-1)"/><circle cx="11" cy="18" r="2" fill="var(--surface-1)"/>',
  chart: '<path d="M4 19V5M4 19h17"/><path d="m7 15 3-4 3 2 5-7"/>',
  moon: '<path d="M20 15.5A8 8 0 0 1 8.5 4 8 8 0 1 0 20 15.5Z"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  desktop: '<rect x="3" y="4" width="18" height="13" rx="2"/><path d="M8 21h8M12 17v4"/>',
  minimize: '<path d="M5 12h14"/>',
  maximize: '<rect x="5" y="5" width="14" height="14" rx="1"/>',
  restore: '<path d="M8 8h11v11H8z"/><path d="M5 16V5h11"/>',
  close: '<path d="m6 6 12 12M18 6 6 18"/>',
  chevron: '<path d="m8 10 4 4 4-4"/>',
  check: '<path d="m5 12 4 4L19 6"/>',
  alert: '<path d="M12 3 2.8 20h18.4L12 3Z"/><path d="M12 9v4M12 16h.01"/>',
  cpu: '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 9h6v6H9zM9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
  database:
    '<ellipse cx="12" cy="5" rx="7" ry="3"/><path d="M5 5v7c0 1.7 3.1 3 7 3s7-1.3 7-3V5M5 12v7c0 1.7 3.1 3 7 3s7-1.3 7-3v-7"/>',
  keyboard:
    '<rect x="3" y="6" width="18" height="12" rx="2"/><path d="M6 10h.01M9 10h.01M12 10h.01M15 10h.01M18 10h.01M7 14h10"/>',
  inspect: '<circle cx="11" cy="11" r="6"/><path d="m16 16 4.5 4.5M11 8v6M8 11h6"/>',
};

export function icon(name: IconName, size = 18): string {
  return `<svg class="icon" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name]}</svg>`;
}
