// Factory replacing React's useShortcutSettings — same public API (spec §8.7).
// Must be called during component init (it registers a persist $effect).

export interface ShortcutMap {
  [className: string]: string; // class name -> key
}

export function buildDefaultShortcuts(classNames: string[]): ShortcutMap {
  const map: ShortcutMap = {};
  classNames.forEach((name, i) => {
    if (i < 9) map[name] = String(i + 1);
    else if (i === 9) map[name] = "0";
  });
  return map;
}

interface PersistedSettings {
  shortcuts: ShortcutMap;
  pinnedClasses: string[];
}

function storageKey(dbPath: string) {
  return `kumo:settings:${dbPath}`;
}

function loadPersistedSettings(dbPath: string | undefined): PersistedSettings | null {
  if (!dbPath) return null;
  try {
    const raw = localStorage.getItem(storageKey(dbPath));
    if (!raw) return null;
    return JSON.parse(raw) as PersistedSettings;
  } catch {
    return null;
  }
}

export function createShortcutSettings(classNames: () => string[], dbPath?: string) {
  const saved = loadPersistedSettings(dbPath);
  const initialNames = classNames();

  let shortcuts = $state<ShortcutMap>(saved?.shortcuts ?? buildDefaultShortcuts(initialNames));
  let pinnedClasses = $state<Set<string>>(
    saved?.pinnedClasses ? new Set(saved.pinnedClasses) : new Set(initialNames.slice(0, 5)),
  );

  // Only track classes seen at RUNTIME via syncClasses calls (not from
  // localStorage). This prevents syncClasses([]) on mount from wiping
  // saved shortcuts when classNames starts empty. Deliberately non-reactive.
  const knownClasses = new Set(initialNames);

  // Classes from persisted settings — used to distinguish "loaded from
  // localStorage" from "genuinely new class" during brand-new detection.
  const persistedClasses = new Set<string>();
  if (saved) {
    Object.keys(saved.shortcuts).forEach((cls) => persistedClasses.add(cls));
    saved.pinnedClasses.forEach((cls) => persistedClasses.add(cls));
  }

  // Persist on changes
  $effect(() => {
    if (!dbPath) return;
    const data: PersistedSettings = { shortcuts: { ...shortcuts }, pinnedClasses: Array.from(pinnedClasses) };
    localStorage.setItem(storageKey(dbPath), JSON.stringify(data));
  });

  function updateShortcut(className: string, key: string) {
    const next = { ...shortcuts };
    // Clear the key from any other class that currently holds it
    if (key) {
      for (const cls of Object.keys(next)) {
        if (cls !== className && next[cls] === key) {
          next[cls] = "";
        }
      }
    }
    next[className] = key;
    shortcuts = next;
  }

  function togglePinned(className: string) {
    const next = new Set(pinnedClasses);
    if (next.has(className)) next.delete(className);
    else next.add(className);
    pinnedClasses = next;
  }

  function resetDefaults() {
    const names = classNames();
    shortcuts = buildDefaultShortcuts(names);
    pinnedClasses = new Set(names.slice(0, 5));
  }

  // Sync new classes: add default shortcuts and pin state for classes that
  // don't have one. Only prune classes that were previously seen at runtime
  // but are now missing (i.e. explicitly removed by the user).
  function syncClasses(names: string[]) {
    // A class is "brand new" only if it was never seen at runtime AND
    // wasn't loaded from persisted settings.
    const brandNew = names.filter((n) => !knownClasses.has(n) && !persistedClasses.has(n));

    const next = { ...shortcuts };
    // Only prune classes that were seen at runtime but are now gone
    for (const key of Object.keys(next)) {
      if (knownClasses.has(key) && !names.includes(key)) delete next[key];
    }
    names.forEach((name, i) => {
      if (!(name in next)) {
        if (i < 9) next[name] = String(i + 1);
        else if (i === 9) next[name] = "0";
      }
    });
    shortcuts = next;

    const nextPinned = new Set(pinnedClasses);
    for (const cls of nextPinned) {
      if (knownClasses.has(cls) && !names.includes(cls)) nextPinned.delete(cls);
    }
    // Only auto-pin genuinely new classes
    for (const name of brandNew) nextPinned.add(name);
    pinnedClasses = nextPinned;

    // Mark all current names as seen at runtime
    for (const name of names) knownClasses.add(name);
  }

  // Reverse map: key -> className
  const keyToClass = $derived.by(() => {
    const map: Record<string, string> = {};
    for (const [cls, key] of Object.entries(shortcuts)) {
      if (key) map[key] = cls;
    }
    return map;
  });

  return {
    get shortcuts() { return shortcuts; },
    get keyToClass() { return keyToClass; },
    updateShortcut,
    resetDefaults,
    syncClasses,
    get pinnedClasses() { return pinnedClasses; },
    togglePinned,
  };
}
