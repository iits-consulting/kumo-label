import { describe, it, expect } from "vitest";
import { type DataPoint, getDatasetStats, isLabeledPoint } from "./mockData";

function makePoint(overrides: Partial<DataPoint> = {}): DataPoint {
  return {
    id: "1",
    filename: "a.jpg",
    x: 0,
    y: 0,
    groundTruth: "",
    label: null,
    predictedLabel: null,
    confidence: 0,
    uncertainty: 0,
    alScore: 0,
    isUnlabeled: true,
    isWrong: false,
    classImbalanceScore: 0,
    topPredictions: [],
    width: 0,
    height: 0,
    imageUrl: "",
    thumbnailUrl: "",
    cluster: 0,
    ignored: false,
    noObjects: false,
    annotationCount: 0,
    annotationClasses: [],
    annotationClassCounts: {},
    labels: [],
    predictedLabels: [],
    split: "train",
    splitOverride: null,
    ...overrides,
  };
}

describe("isLabeledPoint", () => {
  it("defaults to classification semantics when no task type is given", () => {
    expect(isLabeledPoint(makePoint({ label: "Cat" }))).toBe(true);
    expect(isLabeledPoint(makePoint({ groundTruth: "Dog" }))).toBe(true);
    expect(isLabeledPoint(makePoint())).toBe(false);
  });

  it("treats a classification point as labeled via label or ground truth", () => {
    expect(isLabeledPoint(makePoint({ label: "Cat" }), "classification")).toBe(true);
    expect(isLabeledPoint(makePoint({ groundTruth: "Dog" }), "classification")).toBe(true);
    expect(isLabeledPoint(makePoint({ groundTruth: "" }), "classification")).toBe(false);
  });

  it("ignores tags and boxes for classification", () => {
    expect(
      isLabeledPoint(makePoint({ labels: ["Cat"], annotationCount: 3 }), "classification"),
    ).toBe(false);
  });

  it("treats a detection point as labeled only when it has boxes", () => {
    expect(isLabeledPoint(makePoint({ annotationCount: 2 }), "object-detection")).toBe(true);
    expect(isLabeledPoint(makePoint({ annotationCount: 0 }), "object-detection")).toBe(false);
  });

  it("ignores label and ground truth for detection", () => {
    expect(
      isLabeledPoint(makePoint({ label: "Cat", groundTruth: "Cat" }), "object-detection"),
    ).toBe(false);
  });

  it("treats a multilabel point as labeled only when it has at least one tag", () => {
    expect(
      isLabeledPoint(makePoint({ labels: ["Cat"] }), "multilabel-classification"),
    ).toBe(true);
    expect(
      isLabeledPoint(makePoint({ labels: ["Cat", "Dog"] }), "multilabel-classification"),
    ).toBe(true);
    expect(isLabeledPoint(makePoint({ labels: [] }), "multilabel-classification")).toBe(false);
  });

  it("ignores label, ground truth and boxes for multilabel", () => {
    expect(
      isLabeledPoint(
        makePoint({ label: "Cat", groundTruth: "Cat", annotationCount: 4 }),
        "multilabel-classification",
      ),
    ).toBe(false);
  });
});

describe("getDatasetStats", () => {
  it("counts multilabel images as labeled once they carry a tag", () => {
    const data = [
      makePoint({ id: "1", labels: ["Cat", "Dog"] }),
      makePoint({ id: "2", labels: ["Cat"] }),
      makePoint({ id: "3", labels: [], groundTruth: "Cat" }),
    ];
    const stats = getDatasetStats(data, "multilabel-classification");
    expect(stats.total).toBe(3);
    expect(stats.labeled).toBe(2);
    expect(stats.unlabeled).toBe(1);
  });

  it("counts every multilabel tag of an image in its own class bucket", () => {
    const data = [
      makePoint({ id: "1", labels: ["Cat", "Dog"] }),
      makePoint({ id: "2", labels: ["Dog"] }),
    ];
    const stats = getDatasetStats(data, "multilabel-classification");
    expect(stats.classCounts).toEqual({ Cat: 1, Dog: 2 });
  });

  it("counts untagged multilabel images under no class", () => {
    const data = [
      makePoint({ id: "1", labels: ["Cat"] }),
      makePoint({ id: "2", labels: [], groundTruth: "Dog", label: "Dog" }),
    ];
    const stats = getDatasetStats(data, "multilabel-classification");
    expect(stats.classCounts).toEqual({ Cat: 1 });
  });

  it("keeps classification stats keyed by label, ground truth or Unlabeled", () => {
    const data = [
      makePoint({ id: "1", label: "Cat" }),
      makePoint({ id: "2", groundTruth: "Dog" }),
      makePoint({ id: "3" }),
    ];
    const stats = getDatasetStats(data, "classification");
    expect(stats.labeled).toBe(2);
    expect(stats.classCounts).toEqual({ Cat: 1, Dog: 1, Unlabeled: 1 });
  });

  it("keeps detection stats keyed by distinct annotation classes", () => {
    const data = [
      makePoint({ id: "1", annotationCount: 3, annotationClasses: ["Cat", "Dog"] }),
      makePoint({ id: "2", annotationCount: 0, annotationClasses: [] }),
    ];
    const stats = getDatasetStats(data, "object-detection");
    expect(stats.labeled).toBe(1);
    expect(stats.unlabeled).toBe(1);
    expect(stats.classCounts).toEqual({ Cat: 1, Dog: 1 });
  });
});
