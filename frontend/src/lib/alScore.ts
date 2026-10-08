export interface ALWeights {
  uncertainty: number;
  unlabeled: number;
  wrongPrediction: number;
  inverseConfidence: number;
  classImbalance: number;
}

export const DEFAULT_AL_WEIGHTS: ALWeights = {
  uncertainty: 0.35,
  unlabeled: 0.25,
  wrongPrediction: 0.15,
  inverseConfidence: 0.10,
  classImbalance: 0.15,
};

export function computeALScore(
  point: {
    uncertainty: number;
    isUnlabeled: boolean;
    isWrong: boolean;
    confidence: number;
    classImbalanceScore: number;
  },
  weights: ALWeights,
): number {
  const totalWeight =
    weights.uncertainty +
    weights.unlabeled +
    weights.wrongPrediction +
    weights.inverseConfidence +
    weights.classImbalance;
  if (totalWeight === 0) return 0;

  const raw =
    weights.uncertainty * point.uncertainty +
    weights.unlabeled * (point.isUnlabeled ? 1 : 0) +
    weights.wrongPrediction * (point.isWrong ? 1 : 0) +
    weights.inverseConfidence * (1 - point.confidence) +
    weights.classImbalance * point.classImbalanceScore;

  return Math.min(raw / totalWeight, 1.0);
}
