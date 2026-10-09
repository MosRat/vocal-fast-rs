import { isWorkspaceRoute, type WorkspaceRoute } from './types';

export type ViewRoute = WorkspaceRoute | 'settings';

export function routeFromLocation(): ViewRoute {
  const route = window.location.hash.replace(/^#\/?/, '');
  return route === 'settings' || isWorkspaceRoute(route) ? route : 'studio';
}

export function navigate(route: ViewRoute, onNavigate: () => void): void {
  const nextHash = `#/${route}`;
  if (window.location.hash !== nextHash) window.history.pushState({ route }, '', nextHash);
  onNavigate();
  requestAnimationFrame(() => document.querySelector<HTMLElement>('.content-scroll')?.scrollTo({ top: 0, behavior: 'instant' }));
}

export function routeCopy(route: ViewRoute): { title: string; description: string } {
  return {
    models: {title:'A home for every voice.',description:'Edit model details, browse remote repositories and import complete portable models.'},
    studio: {
      title: 'Make every take sound like you.',
      description: 'Configure your voice chain, start the stream, and keep an eye on the audio path.',
    },
    audio: {
      title: 'Route sound with confidence.',
      description: 'Choose the capture and render endpoints that keep your live path stable.',
    },
    tuning: {
      title: 'Shape the voice in real time.',
      description: 'Tune the conversion controls while keeping the stream safe to adjust.',
    },
    diagnostics: {
      title: 'A clear view of your audio.',
      description: 'Live latency, worker performance and queue health — one sample at a time.',
    },
    settings: {
      title: 'Make the workspace yours.',
      description: 'Manage appearance, defaults, and the windows that stay close to your session.',
    },
  }[route];
}
