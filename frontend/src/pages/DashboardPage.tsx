import { motion } from "framer-motion";
import { useStatsStore } from "../hooks/useStatsStore";

const cardVariants = {
  hidden: { opacity: 0, y: 16 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: 0.05 * i }
  })
};

function safeText(val) {
  return val ? val.toString() : "";
}
function DashboardPage() {
  const { stats, clearStats } = useStatsStore();

  return (
    <div className="space-y-6">
      <motion.div
        className="glass p-5 sm:p-6 card-tilt relative"
        initial={{ opacity: 0, scale: 0.96 }}
        animate={{ opacity: 1, scale: 1 }}
      >
        <div className="absolute top-5 right-5 sm:top-6 sm:right-6">
          <button 
            onClick={clearStats} 
            className="px-3 py-1.5 text-xs bg-white/10 hover:bg-white/20 rounded-md transition-colors text-slate-300"
          >
            Clear Stats
          </button>
        </div>
        <p className="text-sm text-slate-300 mb-1">Welcome</p>
        <h2 className="text-xl font-semibold mb-2">
          VeriText AI Console
        </h2>
        <p className="text-sm text-slate-400">
          Paste text, compare variants, and probe cross-lingual robustness for
          English, Hindi, and Hinglish content. This dashboard simulates an
          analytics-style experience with local session statistics.
        </p>
      </motion.div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          {
            label: "Total texts analyzed",
            value: stats.totalTexts,
            accent: "from-accent to-violet-500"
          },
          {
            label: "AI vs Human ratio",
            value:
              stats.totalTexts === 0
                ? "–"
                : `${stats.aiCount}/${stats.humanCount}`,
            accent: "from-ai to-rose-500"
          },
          {
            label: "Most common language",
            value: stats.mostCommonLang ?? "–",
            accent: "from-emerald-400 to-teal-500"
          },
          {
            label: "Avg. AI probability",
            value:
              stats.totalTexts === 0
                ? "–"
                : `${(stats.avgAiProb * 100).toFixed(1)}%`,
            accent: "from-sky-400 to-cyan-500"
          }
        ].map((item, idx) => (
          <motion.div
            key={item.label}
            className="glass p-4 card-tilt"
            custom={idx}
            variants={cardVariants}
            initial="hidden"
            animate="visible"
          >
            <p className="text-xs uppercase tracking-wide text-slate-400 mb-1">
              {item.label}
            </p>
            <p className="text-2xl font-semibold mb-2">{item.value}</p>
            <div
              className={`h-1.5 rounded-full bg-gradient-to-r ${item.accent}`}
            />
          </motion.div>
        ))}
      </div>
    </div>
  );
}

export default DashboardPage;

