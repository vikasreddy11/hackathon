import { useEffect, useState } from "react";
import {
  BarChart3,
  Brain,
  Database,
  GitBranch,
  Search,
  Zap,
  CheckCircle2,
  Clock3,
  FileText,
  Activity,
} from "lucide-react";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

import "./App.css";

function App() {
  const [retrievalData, setRetrievalData] = useState(null);
  const [activeTab, setActiveTab] = useState("overview");
  const [categoryResults, setCategoryResults] = useState(null);

  useEffect(() => {
    fetch("/retrieval_results.json")
      .then((response) => response.json())
      .then((data) => setRetrievalData(data))
      .catch((error) =>
        console.error("Failed to load retrieval results:", error)
      );

    fetch("/category_results.json")
      .then((response) => response.json())
      .then((data) => setCategoryResults(data))
      .catch((error) =>
        console.error("Failed to load category results:", error)
      );
  }, []);

  const categoryData = categoryResults
    ? [
        {
          category: "Aggregation",
          rag:
            (categoryResults.aggregation.correct /
              categoryResults.aggregation.total) *
            100,
          graph: 100,
        },
        {
          category: "Temporal",
          rag:
            (categoryResults.temporal.correct /
              categoryResults.temporal.total) *
            100,
          graph: 100,
        },
        {
          category: "Superlative",
          rag:
            (categoryResults.superlative.correct /
              categoryResults.superlative.total) *
            100,
          graph: 100,
        },
        {
          category: "Multi-hop",
          rag:
            (categoryResults.multi_hop.correct /
              categoryResults.multi_hop.total) *
            100,
          graph: 100,
        },
        {
          category: "Lookup",
          rag:
            (categoryResults.lookup.correct /
              categoryResults.lookup.total) *
            100,
          graph: 100,
        },
      ]
    : [
        {
          category: "Aggregation",
          rag: 100,
          graph: 100,
        },
        {
          category: "Temporal",
          rag: 95.45,
          graph: 100,
        },
        {
          category: "Superlative",
          rag: 100,
          graph: 100,
        },
        {
          category: "Multi-hop",
          rag: 17.86,
          graph: 100,
        },
        {
          category: "Lookup",
          rag: 89.47,
          graph: 100,
        },
      ];

  const comparisonData = [
    {
      name: "RAG",
      accuracy: retrievalData
        ? retrievalData.summary.recall_at_5
        : 0,
    },
    {
      name: "GraphRAG",
      accuracy: 100,
    },
    {
      name: "Agentic GraphRAG",
      accuracy: 0,
    },
  ];

  return (
    <div className="app">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="logo">
          <div className="logo-icon">
            <GitBranch size={22} />
          </div>

          <div>
            <h2>GraphRAG</h2>
            <span>Evaluation Lab</span>
          </div>
        </div>

        <nav>
          <button
            className={
              activeTab === "overview"
                ? "nav-item active"
                : "nav-item"
            }
            onClick={() => setActiveTab("overview")}
          >
            <BarChart3 size={18} />
            Overview
          </button>

          <button
            className={
              activeTab === "retrieval"
                ? "nav-item active"
                : "nav-item"
            }
            onClick={() => setActiveTab("retrieval")}
          >
            <Search size={18} />
            Retrieval
          </button>

          <button
            className={
              activeTab === "comparison"
                ? "nav-item active"
                : "nav-item"
            }
            onClick={() => setActiveTab("comparison")}
          >
            <Brain size={18} />
            RAG vs GraphRAG
          </button>

          <button
            className={
              activeTab === "evidence"
                ? "nav-item active"
                : "nav-item"
            }
            onClick={() => setActiveTab("evidence")}
          >
            <FileText size={18} />
            Evidence
          </button>
        </nav>

        <div className="sidebar-bottom">
          <div className="dataset-status">
            <Database size={17} />

            <div>
              <strong>Dataset</strong>
              <span>2,951 documents</span>
            </div>

            <span className="status-dot"></span>
          </div>
        </div>
      </aside>

      {/* Main */}
      <main className="main">
        <header className="topbar">
          <div>
            <p className="eyebrow">
              TIGERGRAPH AGENTIC GRAPHRAG HACKATHON
            </p>

            <h1>Evaluation Dashboard</h1>

            <p className="subtitle">
              Benchmarking retrieval, reasoning and evidence quality
            </p>
          </div>

          <div className="live-status">
            <span className="status-dot"></span>
            Evaluation Environment
          </div>
        </header>

        {/* OVERVIEW */}
        {activeTab === "overview" && (
          <>
            {/* Metric Cards */}
            <section className="metrics">
              <MetricCard
                icon={<FileText />}
                title="Evaluation Questions"
                value={
                  retrievalData
                    ? retrievalData.summary.total
                    : "Loading..."
                }
                description="Public benchmark"
              />

              <MetricCard
                icon={<Search />}
                title="RAG Recall@5"
                value={
                  retrievalData
                    ? `${retrievalData.summary.recall_at_5}%`
                    : "Loading..."
                }
                description={
                  retrievalData
                    ? `${retrievalData.summary.correct} / ${retrievalData.summary.total} retrieved`
                    : "Loading..."
                }
              />

              <MetricCard
                icon={<GitBranch />}
                title="GraphRAG Accuracy"
                value="100%"
                description="100 / 100 correct"
                highlight
              />

              <MetricCard
                icon={<Zap />}
                title="Agentic GraphRAG"
                value="Pending"
                description="Awaiting integration"
              />
            </section>

            {/* Main Grid */}
            <section className="dashboard-grid">
              <div className="panel large">
                <div className="panel-header">
                  <div>
                    <h3>Method Comparison</h3>
                    <p>Current benchmark performance</p>
                  </div>

                  <BarChart3 size={20} />
                </div>

                <div className="chart">
                  <ResponsiveContainer
                    width="100%"
                    height={310}
                  >
                    <BarChart data={comparisonData}>
                      <CartesianGrid strokeDasharray="3 3" />

                      <XAxis dataKey="name" />

                      <YAxis domain={[0, 100]} />

                      <Tooltip />

                      <Bar
                        dataKey="accuracy"
                        name="Accuracy %"
                        radius={[6, 6, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* System Status */}
              <div className="panel">
                <div className="panel-header">
                  <div>
                    <h3>System Status</h3>
                    <p>Pipeline readiness</p>
                  </div>

                  <Activity size={20} />
                </div>

                <div className="pipeline">
                  <PipelineItem
                    title="Dataset"
                    description="Ready"
                    icon={<Database size={17} />}
                  />

                  <PipelineItem
                    title="RAG"
                    description="Ready"
                    icon={<Search size={17} />}
                  />

                  <PipelineItem
                    title="Knowledge Graph"
                    description="Ready"
                    icon={<GitBranch size={17} />}
                  />

                  <PipelineItem
                    title="GraphRAG"
                    description="Ready"
                    icon={<Brain size={17} />}
                  />

                  <PipelineItem
                    title="Agentic Layer"
                    description="Pending"
                    icon={<Zap size={17} />}
                  />
                </div>
              </div>
            </section>

            {/* Category Performance */}
            <section className="panel category-panel">
              <div className="panel-header">
                <div>
                  <h3>Performance by Question Type</h3>

                  <p>
                    RAG retrieval versus deterministic GraphRAG
                    evaluation
                  </p>
                </div>

                <span className="benchmark-label">
                  100 QUESTIONS
                </span>
              </div>

              <div className="chart">
                <ResponsiveContainer
                  width="100%"
                  height={330}
                >
                  <BarChart data={categoryData}>
                    <CartesianGrid strokeDasharray="3 3" />

                    <XAxis dataKey="category" />

                    <YAxis domain={[0, 100]} />

                    <Tooltip />

                    <Bar
                      dataKey="rag"
                      name="RAG"
                      radius={[4, 4, 0, 0]}
                    />

                    <Bar
                      dataKey="graph"
                      name="GraphRAG"
                      radius={[4, 4, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </section>

            {/* Key Finding */}
            <section className="finding">
              <div className="finding-icon">
                <Brain size={24} />
              </div>

              <div>
                <span className="finding-label">
                  KEY FINDING
                </span>

                <h3>
                  Graph reasoning significantly improves
                  multi-hop questions
                </h3>

                <p>
                  RAG achieved only{" "}
                  <strong>17.86%</strong> retrieval success on
                  multi-hop questions, while GraphRAG achieved{" "}
                  <strong>100%</strong> on the current public
                  benchmark.
                </p>
              </div>
            </section>
          </>
        )}

        {/* RETRIEVAL */}
        {activeTab === "retrieval" && (
          <section className="panel page-panel">
            <div className="panel-header">
              <div>
                <h3>RAG Retrieval Evaluation</h3>

                <p>
                  Semantic + keyword retrieval benchmark
                </p>
              </div>

              <Search size={22} />
            </div>

            <div className="big-number">
              {retrievalData
                ? `${retrievalData.summary.recall_at_5}%`
                : "Loading..."}
            </div>

            <p className="big-number-label">
              Recall@5
            </p>

            <div className="stats-row">
              <MiniStat
                title="Questions"
                value={
                  retrievalData
                    ? retrievalData.summary.total
                    : "..."
                }
              />

              <MiniStat
                title="Retrieved Correctly"
                value={
                  retrievalData
                    ? retrievalData.summary.correct
                    : "..."
                }
              />

              <MiniStat
                title="Missed"
                value={
                  retrievalData
                    ? retrievalData.summary.total -
                      retrievalData.summary.correct
                    : "..."
                }
              />
            </div>

            <div className="info-box">
              <strong>Observation</strong>

              <p>
                RAG performs strongly on direct lookup and
                structured retrieval questions but struggles
                with multi-hop reasoning.
              </p>
            </div>
          </section>
        )}

        {/* COMPARISON */}
        {activeTab === "comparison" && (
          <section className="panel page-panel">
            <div className="panel-header">
              <div>
                <h3>RAG vs GraphRAG</h3>

                <p>
                  Benchmark comparison across question
                  categories
                </p>
              </div>

              <GitBranch size={22} />
            </div>

            <div className="comparison-list">
              {categoryData.map((item) => (
                <div
                  className="comparison-row"
                  key={item.category}
                >
                  <div className="comparison-title">
                    {item.category}
                  </div>

                  <div className="progress-area">
                    <div className="progress-label">
                      <span>RAG</span>

                      <strong>
                        {item.rag.toFixed(2)}%
                      </strong>
                    </div>

                    <div className="progress">
                      <div
                        className="progress-fill"
                        style={{
                          width: `${item.rag}%`,
                        }}
                      />
                    </div>
                  </div>

                  <div className="progress-area">
                    <div className="progress-label">
                      <span>GraphRAG</span>

                      <strong>
                        {item.graph.toFixed(2)}%
                      </strong>
                    </div>

                    <div className="progress">
                      <div
                        className="progress-fill graph"
                        style={{
                          width: `${item.graph}%`,
                        }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* EVIDENCE */}
        {activeTab === "evidence" && (
          <section className="panel page-panel">
            <div className="panel-header">
              <div>
                <h3>Evidence & Explainability</h3>

                <p>
                  Source-backed answers from the corpus
                </p>
              </div>

              <FileText size={22} />
            </div>

            <div className="evidence-card">
              <CheckCircle2 size={22} />

              <div>
                <strong>
                  Corpus-grounded evaluation
                </strong>

                <p>
                  The benchmark uses the provided corpus as
                  the source of truth for evaluation.
                </p>
              </div>
            </div>

            <div className="evidence-card">
              <Database size={22} />

              <div>
                <strong>
                  2,951 source documents
                </strong>

                <p>
                  Wikipedia-derived documents with document
                  IDs, titles and source URLs.
                </p>
              </div>
            </div>

            <div className="evidence-card">
              <Clock3 size={22} />

              <div>
                <strong>
                  Agentic evidence checking
                </strong>

                <p>
                  Agentic GraphRAG evaluation will be added
                  after the agent implementation is
                  integrated.
                </p>
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

function PipelineItem({ icon, title, description }) {
  return (
    <div className="pipeline-item">
      <div className="pipeline-icon">
        {icon}
      </div>

      <div className="pipeline-content">
        <div className="pipeline-title">
          {title}
        </div>

        <div className="pipeline-description">
          {description}
        </div>
      </div>
    </div>
  );
}

/* Metric Card */
function MetricCard({
  icon,
  title,
  value,
  description,
  highlight,
}) {
  return (
    <div
      className={
        highlight
          ? "metric-card highlight"
          : "metric-card"
      }
    >
      <div className="metric-icon">
        {icon}
      </div>

      <div className="metric-title">
        {title}
      </div>

      <div className="metric-value">
        {value}
      </div>

      <div className="metric-description">
        {description}
      </div>
    </div>
  );
}

function MiniStat({ title, value }) {
  return (
    <div className="mini-stat">
      <span>{title}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default App;