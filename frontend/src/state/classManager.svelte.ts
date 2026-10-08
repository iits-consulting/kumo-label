// Factory replacing React's useClassManager — same public API (spec §8.7).
// No persistence, no API: classes are recovered from DB data by Explorer.

export const DEFAULT_PALETTE = [
  '#E3000F', '#2563EB', '#16A34A', '#D97706', '#7C3AED',
  '#0891B2', '#DC2626', '#65A30D', '#EC4899', '#F59E0B',
  '#6366F1', '#14B8A6', '#F97316', '#8B5CF6', '#10B981',
];

export interface ClassEntry {
  name: string;
  color: string;
}

const DEFAULT_CLASSES: ClassEntry[] = [
  { name: 'Cat', color: '#E3000F' },
  { name: 'Dog', color: '#2563EB' },
  { name: 'Car', color: '#16A34A' },
  { name: 'Truck', color: '#D97706' },
  { name: 'Bird', color: '#7C3AED' },
  { name: 'Plane', color: '#0891B2' },
  { name: 'Ship', color: '#DC2626' },
  { name: 'Frog', color: '#65A30D' },
];

export function createClassManager(initial?: ClassEntry[]) {
  let classes = $state<ClassEntry[]>(initial || DEFAULT_CLASSES);

  const classNames = $derived(classes.map((c) => c.name));
  const classColors = $derived.by(() => {
    const colors: Record<string, string> = {};
    for (const c of classes) colors[c.name] = c.color;
    // Always forced — checklist item 23.
    colors['Unlabeled'] = '#9CA3AF';
    return colors;
  });

  function addClass(name: string) {
    const trimmed = name.trim();
    if (!trimmed) return;
    if (classes.some((c) => c.name === trimmed)) return;
    // Palette color cycles by current class count — checklist item 23.
    const color = DEFAULT_PALETTE[classes.length % DEFAULT_PALETTE.length];
    classes = [...classes, { name: trimmed, color }];
  }

  function removeClass(name: string) {
    classes = classes.filter((c) => c.name !== name);
  }

  function renameClass(oldName: string, newName: string) {
    const trimmed = newName.trim();
    if (!trimmed) return;
    if (classes.some((c) => c.name === trimmed && c.name !== oldName)) return;
    classes = classes.map((c) => (c.name === oldName ? { ...c, name: trimmed } : c));
  }

  return {
    get classes() { return classes; },
    get classNames() { return classNames; },
    get classColors() { return classColors; },
    addClass,
    removeClass,
    renameClass,
  };
}
