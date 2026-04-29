import { useState } from "react";
function safeText(val) {
  return val ? val.toString() : "";
}
function SettingsPage() {
  const [academicMode, setAcademicMode] = useState(false);

  return (
    <div className="glass p-4 sm:p-5 card-tilt space-y-4">
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div>
          <h2 className="text-lg font-semibold mb-1">Modes & presentation</h2>
          <p className="text-xs text-slate-400 max-w-xl">
            Toggle academic integrity mode to simulate a clean “report-style”
            view suitable for integrity dashboards and classroom analytics.
          </p>
        </div>
        <label className="flex items-center gap-2 text-xs sm:text-sm cursor-pointer select-none">
          <span className="text-slate-300">🎓 Academic mode</span>
          <span
            className={`w-9 h-5 rounded-full bg-white/10 flex items-center px-0.5 transition-colors ${
              academicMode ? "bg-emerald-500/80" : ""
            }`}
          >
            <span
              className={`w-4 h-4 rounded-full bg-white shadow-sm transform transition-transform ${
                academicMode ? "translate-x-4" : ""
              }`}
            />
          </span>
          <input
            type="checkbox"
            className="hidden"
            checked={academicMode}
            onChange={() => setAcademicMode((v) => !v)}
          />
        </label>
      </div>

      <div
        className={`p-4 rounded-2xl border text-xs sm:text-sm ${
          academicMode
            ? "bg-slate-900 border-emerald-500/50"
            : "bg-black/30 border-white/10"
        }`}
      >
        <p className="font-semibold mb-2">
          Academic integrity report (mock layout)
        </p>
        <ul className="list-disc list-inside space-y-1 text-slate-300">
          <li>Fabrication risk: <span className="font-semibold">Low–Moderate</span></li>
          <li>Cross-lingual stability: <span className="font-semibold">High</span></li>
          <li>Structural uniformity: <span className="font-semibold">Medium</span></li>
          <li>Recommended action: <span className="font-semibold">Manual review on edge cases</span></li>
        </ul>
        <p className="mt-3 text-slate-500">
          This panel demonstrates how the underlying APIs could be surfaced in a
          university-facing integrity portal, with neutral styling and
          report-style language.
        </p>
      </div>
    </div>
  );
}

export default SettingsPage;

