import { useState, useRef, useCallback } from "react";
import {
  BarChart3,
  Brain,
  Database,
  GitBranch,
  Search,
  Zap,
  Send,
  Clock,
  FileText,
  Activity,
  ChevronDown,
  ChevronRight,
  AlertCircle,
  Layers,
  Target,
  Cpu,
  ArrowRight,
  MessageSquare,
  Sparkles,
  Network,
} from "lucide-react";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
} from "recharts";

import "./App.css";

const API_BASE = "http://localhost:8000";

const SAMPLE_QUESTIONS = [
  "What is the relationship between transformers and attention mechanisms?",
  "How do knowledge graphs improve information retrieval?",
  "What are the key differences between RAG and GraphRAG?",
  "Explain multi-hop reasoning in question answering systems.",
  "What role do embeddings play in semantic search?",
];

const PIPELINES = {
  rag: { label: "RAG", color: "#3b82f6", gradient: "linear-gradient(135deg, #3b82f6, #06b6d4)" },
  graphrag: { label: "GraphRAG", color: "#8b5cf6", gradient: "linear-gradient(135deg, #8b5cf6, #d946ef)" },
  agentic: { label: "Agentic GraphRAG", color: "#f59e0b", gradient: "linear-gradient(135deg, #f59e0b, #f43f5e)" },
};

function App() {
  const [activeTab, setActiveTab] = useState("ask");
  const [question, setQuestion] = useState("");
  const [activePipelines, setActivePipelines] = useState(new Set(["rag", "graphrag", "agentic"]));
  const [results, setResults] = useState({});
  const [loading, setLoading] = useState(new Set());
  const [expandedTrace, setExpandedTrace] = useState(new Set());
  const [expandedAnswers, setExpandedAnswers] = useState(new Set());
  const [expandedSources, setExpandedSources] = useState(new Set());
  const [history, setHistory] = useState([]);
  const inputRef = useRef(null);

  const togglePipeline = useCallback((pip) => {
    setActivePipelines((prev) => {
      const next = new Set(prev);
      if (next.has(pip)) {
        if (next.size > 1) next.delete(pip);
      } else {
        next.add(pip);
      }
      return next;
    });
  }, []);

  const runPipeline = useCallback(async (pipeline, q) => {
    setLoading((prev) => new Set(prev).add(pipeline));
    try {
      const res = await fetch(`${API_BASE}/ask/${pipeline}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });
      const data = await res.json();
      setResults((prev) => ({ ...prev, [pipeline]: data }));
    } catch (err) {
      setResults((prev) => ({
        ...prev,
        [pipeline]: { success: false, error: err.message, pipeline },
      }));
    } finally {
      setLoading((prev) => {
        const next = new Set(prev);
        next.delete(pipeline);
        return next;
      });
    }
  }, []);

  const handleAsk = useCallback(
    (e) => {
      e?.preventDefault();
      const q = question.trim();
      if (!q) return;

      setResults({});
      setExpandedTrace(new Set());
      setExpandedAnswers(new Set());

      for (const pip of activePipelines) {
        runPipeline(pip, q);
      }

      setHistory((prev) => [{ question: q, timestamp: new Date().toISOString() }, ...prev.slice(0, 9)]);
    },
    [question, activePipelines, runPipeline]
  );

  const handleSampleClick = useCallback(
    (q) => {
      setQuestion(q);
      setTimeout(() => {
        setResults({});
        for (const pip of activePipelines) {
          runPipeline(pip, q);
        }
        setHistory((prev) => [{ question: q, timestamp: new Date().toISOString() }, ...prev.slice(0, 9)]);
      }, 100);
    },
    [activePipelines, runPipeline]
  );

  const toggleTraceStep = useCallback((stepKey) => {
    setExpandedTrace((prev) => {
      const next = new Set(prev);
      next.has(stepKey) ? next.delete(stepKey) : next.add(stepKey);
      return next;
    });
  }, []);

  const toggleAnswer = useCallback((pip) => {
    setExpandedAnswers((prev) => {
      const next = new Set(prev);
      next.has(pip) ? next.delete(pip) : next.add(pip);
      return next;
    });
  }, []);

  const toggleSource = useCallback((key) => {
    setExpandedSources((prev) => {
      const next = new Set(prev);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });
  }, []);

  const isLoading = loading.size > 0;
  const hasResults = Object.keys(results).length > 0;

  return (
    <div className="app">
      {/* ========== SIDEBAR ========== */}
      <aside className="sidebar">
        <div className="logo">
          <div className="logo-icon">
            <Zap size={20} />
          </div>
          <div>
            <h2>AgentRAG</h2>
            <span>Investigation Lab</span>
          </div>
        </div>

        <nav>
          <button className={`nav-item ${activeTab === "ask" ? "active" : ""}`} onClick={() => setActiveTab("ask")}>
            <MessageSquare size={17} />
            <span>Ask & Compare</span>
          </button>

          <button className={`nav-item ${activeTab === "dashboard" ? "active" : ""}`} onClick={() => setActiveTab("dashboard")}>
            <BarChart3 size={17} />
            <span>Dashboard</span>
          </button>

          <button className={`nav-item ${activeTab === "architecture" ? "active" : ""}`} onClick={() => setActiveTab("architecture")}>
            <Layers size={17} />
            <span>Architecture</span>
          </button>

          <div className="nav-divider" />

          <button className={`nav-item ${activeTab === "trace" ? "active" : ""}`} onClick={() => setActiveTab("trace")}>
            <Activity size={17} />
            <span>Agent Trace</span>
          </button>

          <button className={`nav-item ${activeTab === "evidence" ? "active" : ""}`} onClick={() => setActiveTab("evidence")}>
            <FileText size={17} />
            <span>Evidence</span>
          </button>
        </nav>

        <div className="sidebar-bottom">
          <div className="sidebar-info">
            <div className="sidebar-info-icon">
              <Database size={16} />
            </div>
            <div>
              <strong>Corpus Loaded</strong>
              <span>2,951 documents</span>
            </div>
            <span className="status-dot" />
          </div>
        </div>
      </aside>

      {/* ========== MAIN ========== */}
      <main className="main">
        <header className="topbar">
          <div>
            <p className="eyebrow">TigerGraph Agentic GraphRAG Hackathon</p>
            <h1>Agentic GraphRAG — Investigation Lab</h1>
            <p className="subtitle">Compare RAG, GraphRAG, and Agentic GraphRAG pipelines side by side</p>
          </div>
          <div className="header-badges">
            <div className="badge">
              <span className="status-dot" />
              Live Environment
            </div>
            <div className="badge">
              <Cpu size={13} />
              3 Pipelines
            </div>
          </div>
        </header>

        <div className="page-content">
          {/* ============================== ASK & COMPARE ============================== */}
          {activeTab === "ask" && (
            <div className="ask-section">
              {/* Pipeline Selector */}
              <div className="pipeline-selector">
                {Object.entries(PIPELINES).map(([key, cfg]) => (
                  <button
                    key={key}
                    className={`pipeline-btn ${activePipelines.has(key) ? `active ${key}` : ""}`}
                    onClick={() => togglePipeline(key)}
                  >
                    <span className={`pip-dot ${key}`} />
                    {cfg.label}
                  </button>
                ))}
              </div>

              {/* Question Input */}
              <form onSubmit={handleAsk}>
                <div className="question-input-wrapper">
                  <div className="question-input-container">
                    <div style={{ position: "relative", flex: 1 }}>
                      <Search size={18} className="input-icon" />
                      <input
                        ref={inputRef}
                        type="text"
                        className="question-input"
                        placeholder="Ask a question across all pipelines..."
                        value={question}
                        onChange={(e) => setQuestion(e.target.value)}
                        disabled={isLoading}
                      />
                    </div>
                    <button type="submit" className="ask-btn" disabled={isLoading || !question.trim()}>
                      {isLoading ? <span className="spinner" /> : <Send size={16} />}
                      {isLoading ? "Investigating..." : "Run Pipelines"}
                    </button>
                  </div>
                </div>
              </form>

              {/* Sample Questions */}
              {!hasResults && (
                <div className="sample-questions">
                  <div className="sample-questions-label">Try a sample question</div>
                  <div className="sample-list">
                    {SAMPLE_QUESTIONS.map((sq) => (
                      <button key={sq} className="sample-q" onClick={() => handleSampleClick(sq)}>
                        {sq.length > 60 ? sq.slice(0, 57) + "..." : sq}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Results */}
              {(hasResults || isLoading) && (
                <div className="results-section">
                  {/* Comparison metrics bar */}
                  {hasResults && !isLoading && Object.keys(results).length > 1 && (
                    <ComparisonBar results={results} />
                  )}

                  {/* Pipeline result cards */}
                  <div className="results-grid">
                    {[...activePipelines].map((pip) => (
                      <ResultCard
                        key={pip}
                        pipeline={pip}
                        result={results[pip]}
                        isLoading={loading.has(pip)}
                        expanded={expandedAnswers.has(pip)}
                        onToggleExpand={() => toggleAnswer(pip)}
                        expandedSources={expandedSources}
                        onToggleSource={toggleSource}
                      />
                    ))}
                  </div>

                  {/* Agent Trace (inline for agentic) */}
                  {results.agentic?.trace && results.agentic.trace.length > 0 && (
                    <TraceTimeline
                      trace={results.agentic.trace}
                      expandedSteps={expandedTrace}
                      onToggleStep={toggleTraceStep}
                    />
                  )}
                </div>
              )}

              {/* Empty state */}
              {!hasResults && !isLoading && (
                <div className="empty-state">
                  <div className="empty-state-icon">
                    <Sparkles size={28} />
                  </div>
                  <h3>Ready to investigate</h3>
                  <p>Ask a question and watch all three pipelines race to find the answer — each using a different strategy.</p>
                </div>
              )}
            </div>
          )}

          {/* ============================== DASHBOARD ============================== */}
          {activeTab === "dashboard" && (
            <DashboardTab results={results} />
          )}

          {/* ============================== ARCHITECTURE ============================== */}
          {activeTab === "architecture" && <ArchitectureTab />}

          {/* ============================== AGENT TRACE (full) ============================== */}
          {activeTab === "trace" && (
            <div style={{ maxWidth: 900, margin: "0 auto" }}>
              {results.agentic?.trace && results.agentic.trace.length > 0 ? (
                <TraceTimeline
                  trace={results.agentic.trace}
                  expandedSteps={expandedTrace}
                  onToggleStep={toggleTraceStep}
                  metadata={results.agentic?.metadata}
                />
              ) : (
                <div className="empty-state">
                  <div className="empty-state-icon"><Activity size={28} /></div>
                  <h3>No trace data yet</h3>
                  <p>Run an Agentic GraphRAG query from the "Ask & Compare" tab to see the step-by-step investigation trace here.</p>
                </div>
              )}
            </div>
          )}

          {/* ============================== EVIDENCE ============================== */}
          {activeTab === "evidence" && (
            <EvidenceTab results={results} expandedSources={expandedSources} onToggleSource={toggleSource} />
          )}
        </div>
      </main>
    </div>
  );
}

/* ============================================================
   RESULT CARD
   ============================================================ */
function ResultCard({ pipeline, result, isLoading, expanded, onToggleExpand, expandedSources, onToggleSource }) {
  const cfg = PIPELINES[pipeline];

  if (isLoading) {
    return (
      <div className={`result-card ${pipeline} loading`}>
        <div className="result-header">
          <div className="result-label">
            <span className={`label-dot ${pipeline}`} />
            <h3>{cfg.label}</h3>
          </div>
        </div>
        <div className="typing-indicator">
          <span className="dot" />
          <span className="dot" />
          <span className="dot" />
        </div>
        <div className="result-skeleton">
          <div className="skeleton-line" />
          <div className="skeleton-line" />
          <div className="skeleton-line" />
          <div className="skeleton-line" />
        </div>
      </div>
    );
  }

  if (!result) return null;

  const meta = result.metadata || {};
  const sources = result.sources || [];

  return (
    <div className={`result-card ${pipeline}`}>
      <div className="result-header">
        <div className="result-label">
          <span className={`label-dot ${pipeline}`} />
          <h3>{cfg.label}</h3>
        </div>
        {meta.execution_time_s != null && (
          <span className="result-time">
            <Clock size={10} style={{ marginRight: 4, verticalAlign: "middle" }} />
            {meta.execution_time_s}s
          </span>
        )}
      </div>

      <div className="result-body">
        {result.success === false ? (
          <div className="result-error">
            <AlertCircle size={16} />
            <span>{result.error || "Pipeline failed"}</span>
          </div>
        ) : (
          <>
            <div className={`result-answer ${expanded ? "expanded" : ""}`}>
              {result.answer || "No answer generated."}
            </div>
            {result.answer && result.answer.length > 300 && (
              <button className="expand-btn" onClick={onToggleExpand}>
                {expanded ? "Show less" : "Show more"} {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
              </button>
            )}
          </>
        )}

        {/* Meta chips */}
        <div className="result-meta">
          {meta.chunks_retrieved != null && (
            <span className="meta-chip"><FileText size={12} /> {meta.chunks_retrieved} chunks</span>
          )}
          {meta.entities_found != null && (
            <span className="meta-chip"><Target size={12} /> {meta.entities_found} entities</span>
          )}
          {meta.relationships_found != null && (
            <span className="meta-chip"><Network size={12} /> {meta.relationships_found} rels</span>
          )}
          {meta.steps_taken != null && (
            <span className="meta-chip"><Activity size={12} /> {meta.steps_taken} steps</span>
          )}
          {meta.total_tokens_approx != null && (
            <span className="meta-chip"><Cpu size={12} /> ~{meta.total_tokens_approx} tokens</span>
          )}
          {meta.stopped_reason && (
            <span className="meta-chip"><Zap size={12} /> {meta.stopped_reason}</span>
          )}
          {sources.length > 0 && (
            <span className="meta-chip"><Database size={12} /> {sources.length} sources</span>
          )}
        </div>

        {/* Sources preview */}
        {sources.length > 0 && (
          <div className="sources-section">
            <h4>Sources</h4>
            {sources.slice(0, 3).map((src, i) => {
              const key = `${pipeline}-${i}`;
              const isExp = expandedSources.has(key);
              return (
                <div key={key} className="source-card" onClick={() => onToggleSource(key)}>
                  <div className="source-id">{src.chunk_id || src.doc_id || `Source ${i + 1}`}</div>
                  <div className={`source-text ${isExp ? "expanded" : ""}`}>
                    {src.text || "(no preview)"}
                  </div>
                </div>
              );
            })}
            {sources.length > 3 && (
              <div style={{ fontSize: 11, color: "var(--text-dim)", marginTop: 6 }}>
                + {sources.length - 3} more sources
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/* ============================================================
   COMPARISON BAR (tokens, time, sources)
   ============================================================ */
function ComparisonBar({ results }) {
  const rows = Object.entries(results)
    .filter(([, r]) => r && r.success !== false)
    .map(([pip, r]) => {
      const meta = r.metadata || {};
      return {
        pipeline: pip,
        label: PIPELINES[pip]?.label || pip,
        color: PIPELINES[pip]?.color || "#666",
        time: meta.execution_time_s ?? "–",
        tokens: meta.total_tokens_approx ?? "–",
        sources: (r.sources || []).length,
        steps: meta.steps_taken ?? "–",
        entities: meta.entities_found ?? "–",
      };
    });

  if (rows.length === 0) return null;

  return (
    <div className="comparison-panel">
      <h3>Pipeline Comparison</h3>
      <table className="comparison-table">
        <thead>
          <tr>
            <th>Pipeline</th>
            <th>Time</th>
            <th>Tokens</th>
            <th>Sources</th>
            <th>Steps</th>
            <th>Entities</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.pipeline}>
              <td>
                <div className="pipeline-name">
                  <span className={`label-dot ${r.pipeline}`} style={{ width: 8, height: 8, borderRadius: "50%", background: r.color, display: "inline-block" }} />
                  {r.label}
                </div>
              </td>
              <td>{typeof r.time === "number" ? `${r.time}s` : r.time}</td>
              <td>{typeof r.tokens === "number" ? `~${r.tokens.toLocaleString()}` : r.tokens}</td>
              <td>{r.sources}</td>
              <td>{r.steps}</td>
              <td>{r.entities}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ============================================================
   TRACE TIMELINE
   ============================================================ */
function TraceTimeline({ trace, expandedSteps, onToggleStep, metadata }) {
  return (
    <div className="trace-section">
      <div className="trace-header">
        <Zap size={20} style={{ color: "var(--accent-amber)" }} />
        <h3>Agentic Investigation Trace</h3>
        <span className="trace-count">{trace.length} steps</span>
        {metadata?.stopped_reason && (
          <span className="trace-count" style={{ background: "var(--accent-emerald-glow)", color: "var(--accent-emerald)" }}>
            {metadata.stopped_reason}
          </span>
        )}
      </div>

      <div className="trace-timeline">
        {trace.map((step, i) => {
          const key = `step-${i}`;
          const isExpanded = expandedSteps.has(key);
          return (
            <div key={key} className="trace-step" style={{ animationDelay: `${i * 0.05}s` }}>
              <div className="trace-step-card" onClick={() => onToggleStep(key)}>
                <div className="trace-step-top">
                  <span className="trace-tool-name">{step.tool}</span>
                  <span className="trace-step-num">STEP {step.step}</span>
                </div>
                <div className="trace-reasoning">{step.reasoning}</div>

                {isExpanded && (
                  <div className="trace-details">
                    <div className="trace-detail-row">
                      <span>Tokens</span>
                      <span>~{step.tokens_approx || 0}</span>
                    </div>
                    <div className="trace-detail-row">
                      <span>Output</span>
                      <span>{step.output_summary || "—"}</span>
                    </div>
                    {step.inputs && Object.keys(step.inputs).length > 0 && (
                      <div className="trace-detail-row">
                        <span>Inputs</span>
                        <span>{JSON.stringify(step.inputs).slice(0, 80)}</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ============================================================
   DASHBOARD TAB
   ============================================================ */
function DashboardTab({ results }) {
  const hasData = Object.keys(results).length > 0;

  const chartData = Object.entries(results)
    .filter(([, r]) => r?.success !== false)
    .map(([pip, r]) => ({
      name: PIPELINES[pip]?.label || pip,
      time: r.metadata?.execution_time_s || 0,
      tokens: r.metadata?.total_tokens_approx || 0,
      sources: (r.sources || []).length,
      steps: r.metadata?.steps_taken || 0,
    }));

  return (
    <div className="dashboard-section">
      {/* Quick metrics */}
      <div className="metrics-grid">
        <MetricCard icon={<FileText size={18} />} colorClass="blue" label="Evaluation Questions" value="100" sub="Public benchmark" />
        <MetricCard icon={<Search size={18} />} colorClass="purple" label="Corpus Documents" value="2,951" sub="Wikipedia-derived" />
        <MetricCard icon={<GitBranch size={18} />} colorClass="emerald" label="Pipelines Active" value="3" sub="RAG · GraphRAG · Agentic" />
        <MetricCard icon={<Zap size={18} />} colorClass="amber" label="Questions Answered" value={Object.keys(results).length > 0 ? "✓" : "–"} sub={hasData ? "Latest run complete" : "Awaiting first run"} />
      </div>

      <div className="chart-grid">
        {/* Chart */}
        <div className="chart-panel">
          <h3>Pipeline Performance</h3>
          <p>Execution time and token usage comparison</p>
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="name" stroke="#64748b" fontSize={12} />
                <YAxis stroke="#64748b" fontSize={11} />
                <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8, color: "#f1f5f9", fontSize: 12 }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="time" name="Time (s)" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="sources" name="Sources" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="steps" name="Steps" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="empty-state" style={{ padding: "40px 0" }}>
              <p>Run a query from the Ask tab to see performance data</p>
            </div>
          )}
        </div>

        {/* Pipeline status */}
        <div className="chart-panel">
          <h3>System Status</h3>
          <p>Pipeline readiness</p>
          <div className="pipeline-status-list">
            <PipelineStatusItem icon={<Search size={16} />} colorClass="blue" name="RAG Pipeline" desc="FAISS vector search + LLM generation" status="ready" />
            <PipelineStatusItem icon={<GitBranch size={16} />} colorClass="purple" name="GraphRAG Pipeline" desc="Entity extraction + graph traversal + LLM" status="ready" />
            <PipelineStatusItem icon={<Zap size={16} />} colorClass="amber" name="Agentic GraphRAG" desc="Autonomous orchestrator + multi-tool agent" status="ready" />
          </div>
        </div>
      </div>

      {/* Key finding */}
      <div className="key-finding">
        <div className="finding-icon"><Brain size={22} /></div>
        <div>
          <span className="finding-label">Key Insight</span>
          <h4>Graph reasoning significantly improves multi-hop questions</h4>
          <p>
            RAG struggles with multi-hop reasoning questions that require connecting entities across documents.
            GraphRAG achieves this through structured graph traversal, while Agentic GraphRAG adds autonomous planning
            to dynamically choose the optimal retrieval strategy for each question.
          </p>
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   ARCHITECTURE TAB
   ============================================================ */
function ArchitectureTab() {
  return (
    <div className="architecture-section">
      <div className="arch-card">
        <h3><Search size={18} style={{ color: "var(--accent-blue)" }} /> Pipeline 1: RAG</h3>
        <p>Standard Retrieval-Augmented Generation. Embeds the query, retrieves similar text chunks from a FAISS vector index, and generates an answer using an LLM with the retrieved context.</p>
        <div className="arch-flow">
          <span className="arch-node">Query</span>
          <span className="arch-arrow">→</span>
          <span className="arch-node">Embed Query</span>
          <span className="arch-arrow">→</span>
          <span className="arch-node">FAISS Search</span>
          <span className="arch-arrow">→</span>
          <span className="arch-node">Top-K Chunks</span>
          <span className="arch-arrow">→</span>
          <span className="arch-node">LLM Generate</span>
          <span className="arch-arrow">→</span>
          <span className="arch-node">Answer</span>
        </div>
      </div>

      <div className="arch-card">
        <h3><GitBranch size={18} style={{ color: "var(--accent-purple)" }} /> Pipeline 2: GraphRAG</h3>
        <p>Extracts entities from the question, queries a knowledge graph for matching nodes and 1-hop relationships, retrieves linked source chunks, and generates an answer grounded in both graph structure and text.</p>
        <div className="arch-flow">
          <span className="arch-node">Query</span>
          <span className="arch-arrow">→</span>
          <span className="arch-node">Entity Extraction</span>
          <span className="arch-arrow">→</span>
          <span className="arch-node">Graph Traversal</span>
          <span className="arch-arrow">→</span>
          <span className="arch-node">Context Building</span>
          <span className="arch-arrow">→</span>
          <span className="arch-node">LLM Generate</span>
          <span className="arch-arrow">→</span>
          <span className="arch-node">Answer</span>
        </div>
      </div>

      <div className="arch-card">
        <h3><Zap size={18} style={{ color: "var(--accent-amber)" }} /> Pipeline 3: Agentic GraphRAG</h3>
        <p>An autonomous orchestrator agent that dynamically selects retrieval tools — vector search, entity search, graph traversal, document retrieval, and evidence evaluation — based on the question and intermediate results. It plans its own investigation, evaluates evidence quality, identifies gaps, and adapts its strategy.</p>
        <div className="arch-flow">
          <span className="arch-node">Query</span>
          <span className="arch-arrow">→</span>
          <span className="arch-node">Orchestrator</span>
          <span className="arch-arrow">→</span>
          <span className="arch-node">Tool Selection</span>
          <span className="arch-arrow">→</span>
          <span className="arch-node">Execute & Evaluate</span>
          <span className="arch-arrow">→</span>
          <span className="arch-node">Loop / Stop</span>
          <span className="arch-arrow">→</span>
          <span className="arch-node">Final Answer</span>
        </div>
        <div style={{ marginTop: 16 }}>
          <p style={{ fontSize: 12, color: "var(--text-dim)" }}><strong>Available Tools:</strong></p>
          <div className="arch-flow" style={{ marginTop: 8 }}>
            <span className="arch-node">vector_search</span>
            <span className="arch-node">entity_search</span>
            <span className="arch-node">graph_search</span>
            <span className="arch-node">document_search</span>
            <span className="arch-node">evaluate_evidence</span>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   EVIDENCE TAB
   ============================================================ */
function EvidenceTab({ results, expandedSources, onToggleSource }) {
  const pipelines = Object.entries(results).filter(([, r]) => r?.sources?.length > 0);

  if (pipelines.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon"><FileText size={28} /></div>
        <h3>No evidence collected yet</h3>
        <p>Run a query from the Ask tab to see source evidence from each pipeline.</p>
      </div>
    );
  }

  return (
    <div style={{ maxWidth: 900, margin: "0 auto" }}>
      {pipelines.map(([pip, r]) => (
        <div key={pip} style={{ marginBottom: 32 }}>
          <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 12, display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ width: 10, height: 10, borderRadius: "50%", background: PIPELINES[pip]?.color, display: "inline-block" }} />
            {PIPELINES[pip]?.label} — {r.sources.length} Sources
          </h3>
          {r.sources.map((src, i) => {
            const key = `ev-${pip}-${i}`;
            const isExp = expandedSources.has(key);
            return (
              <div key={key} className="source-card" onClick={() => onToggleSource(key)}>
                <div className="source-id">
                  {src.chunk_id || src.doc_id || `Source ${i + 1}`}
                  {src.score != null && <span style={{ marginLeft: 8, color: "var(--text-dim)" }}>score: {src.score.toFixed(3)}</span>}
                </div>
                <div className={`source-text ${isExp ? "expanded" : ""}`}>
                  {src.text || "(no preview available)"}
                </div>
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}

/* ============================================================
   HELPER COMPONENTS
   ============================================================ */
function MetricCard({ icon, colorClass, label, value, sub }) {
  return (
    <div className="metric-card">
      <div className={`metric-icon ${colorClass}`}>{icon}</div>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      <div className="metric-sub">{sub}</div>
    </div>
  );
}

function PipelineStatusItem({ icon, colorClass, name, desc, status }) {
  return (
    <div className="pipeline-status-item">
      <div className={`pipeline-status-icon ${colorClass}`}>{icon}</div>
      <div className="pipeline-status-info">
        <strong>{name}</strong>
        <span>{desc}</span>
      </div>
      <span className={`status-badge ${status}`}>{status}</span>
    </div>
  );
}

export default App;