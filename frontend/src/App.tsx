import { motion } from "framer-motion";
import { NavLink, Route, Routes } from "react-router-dom";
import AnalyzePage from "./pages/AnalyzePage";
import ComparePage from "./pages/ComparePage";
import DashboardPage from "./pages/DashboardPage";
import SettingsPage from "./pages/SettingsPage";

function safeText(val) {
  return val ? val.toString() : "";
}
const navLinkClass =
  "px-3 py-1.5 rounded-full text-sm font-medium transition-colors";

function App() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-bg to-black text-slate-100">
      <header className="px-4 sm:px-8 pt-5 pb-4 border-b border-white/10">
        <div className="max-w-6xl mx-auto flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="space-y-1">
            <motion.h1
              className="text-2xl sm:text-3xl font-bold tracking-tight"
              initial={{ opacity: 0, y: -12 }}
              animate={{ opacity: 1, y: 0 }}
            >
              VeriText AI
            </motion.h1>
            <motion.p
              className="text-xs sm:text-sm text-slate-400"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.15 }}
            >
              Multilingual, paraphrase-resistant AI-generated text detection with
              cross-lingual drift and robustness analytics.
            </motion.p>
          </div>
          <nav className="flex flex-wrap gap-2">
            <NavLink
              to="/"
              end
              className={({ isActive }) =>
                `${navLinkClass} ${
                  isActive
                    ? "bg-accent text-white"
                    : "bg-white/5 text-slate-300 hover:bg-white/10"
                }`
              }
            >
              Overview
            </NavLink>
            <NavLink
              to="/analyze"
              className={({ isActive }) =>
                `${navLinkClass} ${
                  isActive
                    ? "bg-accent text-white"
                    : "bg-white/5 text-slate-300 hover:bg-white/10"
                }`
              }
            >
              Analyze
            </NavLink>
            <NavLink
              to="/compare"
              className={({ isActive }) =>
                `${navLinkClass} ${
                  isActive
                    ? "bg-accent text-white"
                    : "bg-white/5 text-slate-300 hover:bg-white/10"
                }`
              }
            >
              Compare
            </NavLink>
            <NavLink
              to="/settings"
              className={({ isActive }) =>
                `${navLinkClass} ${
                  isActive
                    ? "bg-accent text-white"
                    : "bg-white/5 text-slate-300 hover:bg-white/10"
                }`
              }
            >
              Modes
            </NavLink>
          </nav>
        </div>
      </header>

      <main className="px-4 sm:px-8 pb-10 pt-6">
        <div className="max-w-6xl mx-auto">
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/analyze" element={<AnalyzePage />} />
            <Route path="/compare" element={<ComparePage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}

export default App;

