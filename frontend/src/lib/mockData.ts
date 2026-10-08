// Mock data for the Active Learning Studio

export type TaskType = 'classification' | 'object-detection' | 'multilabel-classification';

export const CLASS_NAMES = ['Cat', 'Dog', 'Car', 'Truck', 'Bird', 'Plane', 'Ship', 'Frog'] as const;
export type ClassName = typeof CLASS_NAMES[number];

export interface BoundingBox {
  id: string;
  x: number;      // 0-1 relative to image width
  y: number;      // 0-1 relative to image height
  width: number;  // 0-1 relative
  height: number; // 0-1 relative
  label: string;
}

export const CLASS_COLORS: Record<string, string> = {
  Cat: '#E3000F',
  Dog: '#2563EB',
  Car: '#16A34A',
  Truck: '#D97706',
  Bird: '#7C3AED',
  Plane: '#0891B2',
  Ship: '#DC2626',
  Frog: '#65A30D',
  Unlabeled: '#9CA3AF',
};

export interface TopPrediction {
  label: string;
  confidence: number;
}

export type SplitName = 'train' | 'valid' | 'test';

export interface DataPoint {
  id: string;
  filename: string;
  x: number;
  y: number;
  groundTruth: string;
  label: string | null;
  predictedLabel: string | null;
  confidence: number;
  uncertainty: number;
  alScore: number;
  isUnlabeled: boolean;
  isWrong: boolean;
  classImbalanceScore: number;
  topPredictions: TopPrediction[];
  width: number;
  height: number;
  imageUrl: string;
  thumbnailUrl: string;
  cluster: number;
  ignored: boolean;
  noObjects: boolean;
  annotationCount: number;
  annotationClasses: string[];
  annotationClassCounts: Record<string, number>;
  // Multi-label classification: the class tags applied to this image.
  labels: string[];
  // Multi-label classification: the class tags the model predicted.
  predictedLabels: string[];
  // Folder-derived split from the dataset structure (train/valid/test/other).
  split: string;
  // User-applied override; if set, wins over `split` for training.
  splitOverride: SplitName | null;
}

// Seeded random for reproducibility
function seededRandom(seed: number) {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

function gaussianRandom(rng: () => number, mean: number, stdDev: number): number {
  const u1 = rng();
  const u2 = rng();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return z * stdDev + mean;
}

// Cluster centers for UMAP-like 2D embedding
const CLUSTER_CENTERS: [number, number][] = [
  [-3.2, 2.1],   // Cat
  [-2.8, -1.5],  // Dog
  [2.5, 3.0],    // Car
  [3.1, 1.2],    // Truck
  [-0.5, 4.2],   // Bird
  [1.8, -2.8],   // Plane
  [4.0, -1.0],   // Ship
  [-1.2, -3.5],  // Frog
];

export function generateMockData(count: number = 500): DataPoint[] {
  const rng = seededRandom(42);
  const points: DataPoint[] = [];

  for (let i = 0; i < count; i++) {
    const clusterIdx = Math.floor(rng() * CLASS_NAMES.length);
    const center = CLUSTER_CENTERS[clusterIdx];
    const className = CLASS_NAMES[clusterIdx];

    const x = gaussianRandom(rng, center[0], 0.6);
    const y = gaussianRandom(rng, center[1], 0.6);

    // ~70% labeled, 30% unlabeled
    const isLabeled = rng() < 0.7;
    // Some mislabels for realism
    const isMislabeled = isLabeled && rng() < 0.05;
    const label = isLabeled
      ? isMislabeled
        ? CLASS_NAMES[Math.floor(rng() * CLASS_NAMES.length)]
        : className
      : null;

    const confidence = 0.3 + rng() * 0.7;
    const uncertainty = 1 - confidence + (rng() * 0.2 - 0.1);

    // Generate top predictions
    const predScores = CLASS_NAMES.map((name) => ({
      label: name,
      confidence: name === className ? confidence : rng() * (1 - confidence) * 0.5,
    }));
    predScores.sort((a, b) => b.confidence - a.confidence);
    const topPredictions = predScores.slice(0, 3);
    // Normalize
    const totalConf = topPredictions.reduce((s, p) => s + p.confidence, 0);
    topPredictions.forEach((p) => (p.confidence = Math.round((p.confidence / totalConf) * 100) / 100));

    const seed = 100 + i;
    points.push({
      id: `img_${String(i).padStart(5, '0')}`,
      filename: `IMG_${String(1000 + i).padStart(5, '0')}.jpg`,
      x,
      y,
      groundTruth: className,
      label,
      predictedLabel: className,
      confidence: Math.round(confidence * 100) / 100,
      uncertainty: Math.max(0, Math.min(1, Math.round(uncertainty * 100) / 100)),
      alScore: 0,
      isUnlabeled: !isLabeled,
      isWrong: isLabeled && isMislabeled,
      classImbalanceScore: 0,
      topPredictions,
      width: 640 + Math.floor(rng() * 640),
      height: 480 + Math.floor(rng() * 480),
      imageUrl: `https://picsum.photos/seed/${seed}/800/600`,
      thumbnailUrl: `https://picsum.photos/seed/${seed}/200/200`,
      cluster: clusterIdx,
      ignored: false,
      noObjects: false,
      annotationCount: 0,
      annotationClasses: [],
      annotationClassCounts: {},
      labels: [],
      predictedLabels: [],
      split: 'train',
      splitOverride: null,
    });
  }

  return points;
}

export function effectiveSplit(d: Pick<DataPoint, 'split' | 'splitOverride'>): string {
  return d.splitOverride ?? d.split;
}

// Whether an image counts as labeled for the given task: boxes for detection,
// at least one tag for multi-label, a label (or folder ground truth) otherwise.
export const isLabeledPoint = (d: DataPoint, taskType: TaskType = 'classification'): boolean => {
  if (taskType === 'object-detection') return d.annotationCount > 0;
  if (taskType === 'multilabel-classification') return d.labels.length > 0;
  return d.label !== null || Boolean(d.groundTruth);
};

export function getDatasetStats(data: DataPoint[], taskType: TaskType = 'classification') {
  const total = data.length;
  const labeled = data.filter((d) => isLabeledPoint(d, taskType)).length;
  const unlabeled = total - labeled;

  const classCounts: Record<string, number> = {};
  if (taskType === 'object-detection') {
    data.forEach((d) => {
      if (d.annotationClasses.length > 0) {
        d.annotationClasses.forEach((cls) => {
          classCounts[cls] = (classCounts[cls] || 0) + 1;
        });
      }
    });
  } else if (taskType === 'multilabel-classification') {
    // Each tag counts once for its class; untagged images count under nothing.
    data.forEach((d) => {
      d.labels.forEach((cls) => {
        classCounts[cls] = (classCounts[cls] || 0) + 1;
      });
    });
  } else {
    data.forEach((d) => {
      const key = d.label || d.groundTruth || 'Unlabeled';
      classCounts[key] = (classCounts[key] || 0) + 1;
    });
  }

  return { total, labeled, unlabeled, classCounts };
}

export function getHighValueCount(data: DataPoint[]): number {
  return data.filter((d) => !d.ignored && d.alScore > 0.3).length;
}
