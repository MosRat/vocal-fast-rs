import morphdom from 'morphdom';

const renderedMarkup = new WeakMap<HTMLElement, string>();
const eventBindings = new WeakMap<HTMLElement, Map<string, EventListener>>();

/** Keep controls, focus, scroll positions and animations alive across snapshots. */
export function patchView(target: HTMLElement, next: HTMLElement): boolean {
  const markup = next.innerHTML;
  if (renderedMarkup.get(target) === markup) return false;
  morphdom(target, next, {
    childrenOnly: true,
    onBeforeElUpdated(from, to) {
      if (from instanceof HTMLDetailsElement && to instanceof HTMLDetailsElement) to.open = from.open;
      return !from.isEqualNode(to);
    },
  });
  renderedMarkup.set(target, markup);
  return true;
}

/** Replace a binding when a retained node is rendered again. */
export function on<K extends keyof HTMLElementEventMap>(
  target: HTMLElement | null,
  type: K,
  handler: (event: HTMLElementEventMap[K]) => unknown,
): void {
  if (!target) return;
  let bindings = eventBindings.get(target);
  if (!bindings) {
    bindings = new Map();
    eventBindings.set(target, bindings);
  }
  const previous = bindings.get(type);
  if (previous) target.removeEventListener(type, previous);
  const listener = handler as EventListener;
  target.addEventListener(type, listener);
  bindings.set(type, listener);
}
