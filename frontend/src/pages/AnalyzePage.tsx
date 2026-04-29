import { motion } from "framer-motion";
import { FormEvent, useState } from "react";
import { useStatsStore } from "../hooks/useStatsStore";

type FeatureCardProps = {
  title: string;
  color: string;
  score?: number | null;
  explanation?: string | null;
};
function safeText(val) {
  return val ? val.toString() : "";
}
const FeatureCard = ({ title, color, score, explanation }: FeatureCardProps) => {
  const pct = score != null ? Math.max(0, Math.min(score, 1)) * 100 : 0;
  return (
    <div className="glass p-4 card-tilt flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium">{title}</p>
        <span className="text-xs text-slate-400">
          {score != null ? score.toFixed(3) : "–"}
        </span>
      </div>
      <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <p className="text-xs text-slate-400 line-clamp-2">
        {explanation || "Run analysis to see details."}
      </p>
    </div>
  );
};

type AnalyzeResponse = {
  prob_ai?: number;
  ai_probability?: number;
  pred_label?: number;
  is_ai_generated?: boolean;
  pred_lang?: string;
  language_detected?: string;
  features?: {
    burstiness?: { score: number; explanation: string };
    entropy?: { score: number; explanation: string };
    syntax_depth?: { score: number; explanation: string };
    semantic_drift?: { score: number; explanation: string };
  };
};

const API_BASE = "/api/detect";

function AnalyzePage() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const { recordSample } = useStatsStore();

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    const trimmed = text.trim();
    if (!trimmed) {
      setError("Please paste or type some text first.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: trimmed, language: "auto" })
      });
      const data = (await res.json()) as AnalyzeResponse;
      if (!res.ok || (data as any).error) {
        setError((data as any).error || "Analysis failed.");
        setResult(null);
      } else {
        setResult(data);
        const prob =
          typeof data.prob_ai === "number"
            ? data.prob_ai
            : typeof data.ai_probability === "number"
            ? data.ai_probability
            : 0;
        const isAi =
          typeof data.pred_label === "number"
            ? data.pred_label === 1
            : data.is_ai_generated ?? prob > 0.5;
        const lang = data.pred_lang ?? data.language_detected ?? null;
        recordSample(lang, isAi, prob);
      }
    } catch (err: any) {
      setError(err.message || "Network error.");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const prob =
    result &&
    (typeof result.prob_ai === "number"
      ? result.prob_ai
      : typeof result.ai_probability === "number"
      ? result.ai_probability
      : 0);
  const isAi =
    result &&
    (typeof result.pred_label === "number"
      ? result.pred_label === 1
      : result.is_ai_generated ?? (prob ?? 0) > 0.5);
  const lang = result?.pred_lang ?? result?.language_detected ?? "–";

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,1.4fr)]">
      <motion.form
        onSubmit={onSubmit}
        className="glass p-5 sm:p-6 space-y-4 card-tilt"
        initial={{ opacity: 0, x: -16 }}
        animate={{ opacity: 1, x: 0 }}
      >
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold mb-1">Analyze a text</h2>
            <p className="text-xs text-slate-400">
              Paste any English / Hindi / Hinglish paragraph and get an
              AI-vs-human prediction with structural fingerprinting.
            </p>
          </div>
        </div>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste or type text here. Example: Yeh ek Hinglish sample hai jo AI detection ke liye test ho raha hai."
          className="w-full min-h-[180px] rounded-xl bg-black/40 border border-white/10 px-3 py-2 text-sm outline-none focus:border-accent resize-vertical"
        />
        <div className="flex flex-wrap gap-3 items-center justify-between">
          <button
            type="submit"
            className="px-4 py-2 rounded-xl bg-accent text-sm font-semibold hover:bg-purple-500 transition-colors disabled:opacity-60"
            disabled={loading}
          >
            {loading ? "Analyzing…" : "Run analysis"}
          </button>
          {error && (
            <p className="text-xs text-red-400 max-w-xs">{error}</p>
          )}
        </div>
      </motion.form>

      <motion.div
        className="space-y-4"
        initial={{ opacity: 0, x: 16 }}
        animate={{ opacity: 1, x: 0 }}
      >
        <div className="glass p-4 sm:p-5 card-tilt space-y-2">
          <p className="text-xs uppercase tracking-wide text-slate-400">
            Prediction
          </p>
          <p className="text-sm text-slate-300">
            {result ? (isAi ? "AI-generated" : "Human-written") : "–"}
          </p>
          <p className="text-xs text-slate-400">
            AI probability:{" "}
            {result ? `${((prob ?? 0) * 100).toFixed(1)}%` : "–"}
          </p>
          <p className="text-xs text-slate-400">Detected language: {lang}</p>
        </div>

        <div className="glass p-4 sm:p-5 space-y-3 card-tilt">
          <div className="flex items-center justify-between">
            <p className="text-sm font-semibold">Explainability cards</p>
            <p className="text-xs text-slate-400">Structural fingerprint</p>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <FeatureCard
              title="Burstiness"
              color="bg-emerald-400"
              score={result?.features?.burstiness?.score}
              explanation={result?.features?.burstiness?.explanation}
            />
            <FeatureCard
              title="Entropy"
              color="bg-sky-400"
              score={result?.features?.entropy?.score}
              explanation={result?.features?.entropy?.explanation}
            />
            <FeatureCard
              title="Syntax depth"
              color="bg-amber-400"
              score={result?.features?.syntax_depth?.score}
              explanation={result?.features?.syntax_depth?.explanation}
            />
            <FeatureCard
              title="Semantic drift"
              color="bg-pink-400"
              score={result?.features?.semantic_drift?.score}
              explanation={result?.features?.semantic_drift?.explanation}
            />
          </div>
        </div>
      </motion.div>
    </div>
  );
}

export default AnalyzePage;

