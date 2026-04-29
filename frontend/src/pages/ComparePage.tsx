import { FormEvent, useState } from "react";

type CompareResult = {
  a?: { prob: number; isAi: boolean };
  b?: { prob: number; isAi: boolean };
};

const API_BASE = "/api/detect";
function safeText(val) {
  return val ? val.toString() : "";
}
function ComparePage() {
  const [textA, setTextA] = useState("");
  const [textB, setTextB] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<CompareResult | null>(null);

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    const a = textA.trim();
    const b = textB.trim();
    if (!a || !b) {
      setError("Please provide both Text A and Text B.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const [resA, resB] = await Promise.all([
        fetch(`${API_BASE}/analyze`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: a, language: "auto" })
        }),
        fetch(`${API_BASE}/analyze`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: b, language: "auto" })
        })
      ]);
      const dataA = await resA.json();
      const dataB = await resB.json();
      const probA =
        typeof dataA.prob_ai === "number"
          ? dataA.prob_ai
          : typeof dataA.ai_probability === "number"
          ? dataA.ai_probability
          : 0;
      const isAiA =
        typeof dataA.pred_label === "number"
          ? dataA.pred_label === 1
          : dataA.is_ai_generated ?? probA > 0.5;

      const probB =
        typeof dataB.prob_ai === "number"
          ? dataB.prob_ai
          : typeof dataB.ai_probability === "number"
          ? dataB.ai_probability
          : 0;
      const isAiB =
        typeof dataB.pred_label === "number"
          ? dataB.pred_label === 1
          : dataB.is_ai_generated ?? probB > 0.5;

      setResult({
        a: { prob: probA, isAi: isAiA },
        b: { prob: probB, isAi: isAiB }
      });
    } catch (err: any) {
      setError(err.message || "Network error.");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const probA = result?.a?.prob ?? 0;
  const probB = result?.b?.prob ?? 0;

  return (
    <div className="space-y-5">
      <form
        onSubmit={onSubmit}
        className="glass p-4 sm:p-5 grid gap-4 lg:grid-cols-2 card-tilt"
      >
        <div className="space-y-2">
          <p className="text-xs uppercase tracking-wide text-slate-400">
            Text A
          </p>
          <textarea
            value={textA}
            onChange={(e) => setTextA(e.target.value)}
            placeholder="Paste Text A (e.g., student answer version 1)"
            className="w-full min-h-[150px] rounded-xl bg-black/40 border border-white/10 px-3 py-2 text-sm outline-none focus:border-accent resize-vertical"
          />
        </div>
        <div className="space-y-2">
          <p className="text-xs uppercase tracking-wide text-slate-400">
            Text B
          </p>
          <textarea
            value={textB}
            onChange={(e) => setTextB(e.target.value)}
            placeholder="Paste Text B (e.g., paraphrased or translated version)"
            className="w-full min-h-[150px] rounded-xl bg-black/40 border border-white/10 px-3 py-2 text-sm outline-none focus:border-accent resize-vertical"
          />
        </div>
        <div className="lg:col-span-2 flex items-center justify-between gap-3 flex-wrap">
          <button
            type="submit"
            className="px-4 py-2 rounded-xl bg-accent text-sm font-semibold hover:bg-purple-500 transition-colors disabled:opacity-60"
            disabled={loading}
          >
            {loading ? "Comparing…" : "Compare predictions"}
          </button>
          {error && <p className="text-xs text-red-400">{error}</p>}
        </div>
      </form>

      {result && (
        <div className="glass p-4 sm:p-5 card-tilt space-y-4">
          <p className="text-sm font-semibold mb-1">
            Comparison summary (higher bar = higher AI probability)
          </p>
          <div className="grid grid-cols-[auto,1fr,1fr] gap-3 text-xs sm:text-sm items-center">
            <span className="text-slate-400">Metric</span>
            <span className="text-slate-300">Text A</span>
            <span className="text-slate-300">Text B</span>

            <span className="text-slate-400">Prediction</span>
            <span>{result.a?.isAi ? "AI-generated" : "Human-written"}</span>
            <span>{result.b?.isAi ? "AI-generated" : "Human-written"}</span>

            <span className="text-slate-400">AI probability</span>
            <div className="flex items-center gap-2">
              <span>{(probA * 100).toFixed(1)}%</span>
              <div className="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
                <div
                  className="h-full bg-ai rounded-full"
                  style={{ width: `${probA * 100}%` }}
                />
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span>{(probB * 100).toFixed(1)}%</span>
              <div className="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
                <div
                  className="h-full bg-ai rounded-full"
                  style={{ width: `${probB * 100}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default ComparePage;

