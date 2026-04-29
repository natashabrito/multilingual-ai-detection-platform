import { useEffect, useState } from "react";

type Stats = {
  totalTexts: number;
  aiCount: number;
  humanCount: number;
  mostCommonLang: string | null;
  avgAiProb: number;
};

const STORAGE_KEY = "ml-ai-detector-stats-v1";

function readInitial(): Stats {
  if (typeof window === "undefined") {
    return {
      totalTexts: 0,
      aiCount: 0,
      humanCount: 0,
      mostCommonLang: null,
      avgAiProb: 0
    };
  }
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return {
        totalTexts: 0,
        aiCount: 0,
        humanCount: 0,
        mostCommonLang: null,
        avgAiProb: 0
      };
    }
    return JSON.parse(raw) as Stats;
  } catch {
    return {
      totalTexts: 0,
      aiCount: 0,
      humanCount: 0,
      mostCommonLang: null,
      avgAiProb: 0
    };
  }
}

export function useStatsStore() {
  const [stats, setStats] = useState<Stats>(() => readInitial());

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(stats));
  }, [stats]);

  const recordSample = (lang: string | null, isAi: boolean, probAi: number) => {
    setStats((prev) => {
      const totalTexts = prev.totalTexts + 1;
      const aiCount = prev.aiCount + (isAi ? 1 : 0);
      const humanCount = prev.humanCount + (!isAi ? 1 : 0);

      const weightPrev = prev.totalTexts;
      const avgAiProb =
        weightPrev === 0
          ? probAi
          : (prev.avgAiProb * weightPrev + probAi) / (weightPrev + 1);

      const langCounts: Record<string, number> = {};
      if (prev.mostCommonLang) {
        langCounts[prev.mostCommonLang] = totalTexts - 1;
      }
      if (lang) {
        langCounts[lang] = (langCounts[lang] || 0) + 1;
      }
      const mostCommonLang =
        lang && lang !== "" ? lang : prev.mostCommonLang ?? null;

      return {
        totalTexts,
        aiCount,
        humanCount,
        avgAiProb,
        mostCommonLang
      };
    });
  };

  const clearStats = () => {
    const defaultStats = {
      totalTexts: 0,
      aiCount: 0,
      humanCount: 0,
      mostCommonLang: null,
      avgAiProb: 0
    };
    setStats(defaultStats);
    if (typeof window !== "undefined") {
      window.localStorage.removeItem(STORAGE_KEY);
    }
  };

  return { stats, recordSample, clearStats };
}

