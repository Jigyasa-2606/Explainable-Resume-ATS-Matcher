import React, { useMemo, useState, useRef } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

const ROLE_OPTIONS = [
  "Software Engineer", "Frontend Developer", "Backend Developer", "Full Stack Developer",
  "Python Developer", "Java Developer", "React Developer", "Node.js Developer",
  "Mobile App Developer", "Android Developer", "iOS Developer", "DevOps Engineer",
  "Cloud Engineer", "AWS Cloud Engineer", "Azure Cloud Engineer", "Site Reliability Engineer",
  "Data Analyst", "Business Analyst", "Business Intelligence Analyst", "Power BI Developer",
  "Tableau Developer", "Data Scientist", "Machine Learning Engineer", "AI Engineer",
  "NLP Engineer", "Computer Vision Engineer", "MLOps Engineer", "Data Engineer",
  "Database Administrator", "SQL Developer", "QA Engineer", "Automation Tester",
  "Manual Tester", "Cybersecurity Analyst", "Security Engineer", "Network Engineer",
  "System Administrator", "Product Manager", "Project Manager", "Scrum Master",
  "UI UX Designer", "Graphic Designer", "Digital Marketing Executive", "SEO Analyst",
  "Content Writer", "Technical Writer", "HR Executive", "Recruiter",
  "Finance Analyst", "Operations Analyst", "Customer Support Executive", "Sales Executive",
  "Data Analyst Intern", "Software Engineer Intern", "Web Developer Intern",
  "Machine Learning Intern", "Data Science Intern", "Business Analyst Intern",
  "Digital Marketing Intern", "HR Intern",
];

/* ── Icons ── */
function UploadIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  );
}

function ExternalLinkIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
      <polyline points="15 3 21 3 21 9" />
      <line x1="10" y1="14" x2="21" y2="3" />
    </svg>
  );
}

function MapPinIcon() {
  return (
    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" />
      <circle cx="12" cy="10" r="3" />
    </svg>
  );
}

function BriefcaseIcon() {
  return (
    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2" y="7" width="20" height="14" rx="2" ry="2" />
      <path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16" />
    </svg>
  );
}

function SourceIcon() {
  return (
    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <line x1="2" y1="12" x2="22" y2="12" />
      <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
    </svg>
  );
}

/* ── Loading skeleton ── */
function SkeletonCard() {
  return (
    <div className="skeleton-card">
      <div style={{ display: "flex", gap: 14, marginBottom: 18 }}>
        <div className="skel" style={{ width: 36, height: 36, borderRadius: 8, flexShrink: 0 }} />
        <div style={{ flex: 1 }}>
          <div className="skel skel-h1" />
          <div className="skel skel-h2" />
        </div>
      </div>
      <div className="skel-metrics">
        <div className="skel skel-metric" />
        <div className="skel skel-metric" />
        <div className="skel skel-metric" />
      </div>
      <div className="skel skel-p" />
      <div className="skel skel-p" />
      <div className="skel skel-p" style={{ width: "68%" }} />
    </div>
  );
}

function LoadingState({ count = 5 }) {
  return (
    <div className="skeleton-grid">
      {Array.from({ length: count }, (_, i) => (
        <SkeletonCard key={i} />
      ))}
    </div>
  );
}

/* ── App ── */
function App() {
  const [resumeFile, setResumeFile] = useState(null);
  const [resumeText, setResumeText] = useState("");
  const [query, setQuery] = useState("");
  const [selectedRoles, setSelectedRoles] = useState([]);
  const [provider, setProvider] = useState("jsearch");
  const [country, setCountry] = useState("in");
  const [datePosted, setDatePosted] = useState("month");
  const [opportunityType, setOpportunityType] = useState("jobs");
  const [resultLimit, setResultLimit] = useState(25);
  const [topN, setTopN] = useState(10);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const resultsRef = useRef(null);

  const suggestedQuery = useMemo(() => {
    if (!result?.suggested_queries?.length) return "software developer";
    return result.suggested_queries[0];
  }, [result]);

  async function findJobs(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    if (resumeFile) formData.append("resume_file", resumeFile);
    formData.append("resume_text", resumeText);
    formData.append("query", query);
    formData.append("selected_roles", JSON.stringify(selectedRoles));
    formData.append("provider", provider);
    formData.append("country", country);
    formData.append("date_posted", datePosted);
    formData.append("opportunity_type", opportunityType);
    formData.append("result_limit", String(resultLimit));
    formData.append("top_n", String(topN));

    try {
      const response = await fetch(`${API_BASE}/api/live-jobs`, {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Could not fetch matching jobs.");
      setResult(payload);
      if (!query && payload.query_used) setQuery(payload.query_used);
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function updateSelectedRoles(event) {
    setSelectedRoles(Array.from(event.target.selectedOptions, (o) => o.value));
  }

  return (
    <main>
      {/* Hero */}
      <section className="hero">
        <div className="hero-text">
          <span className="eyebrow">AI Resume Matching</span>
          <h1>Find the best live jobs<br /><em>for your resume</em></h1>
          <p className="hero-subtitle">
            Upload your resume, search from JSearch, Adzuna, or Remotive, and get ranked job matches with explainable fit scores.
          </p>
        </div>
        <div className="hero-stats">
          <div className="stat-card">
            <strong>{result?.jobs?.length ?? 0}</strong>
            <span>Ranked matches</span>
          </div>
          <div className="stat-card">
            <strong>{result?.jobs_fetched ?? "—"}</strong>
            <span>Jobs fetched</span>
          </div>
        </div>
      </section>

      {/* Form */}
      <form className="layout" onSubmit={findJobs}>
        {/* Resume panel */}
        <section className="panel">
          <div className="section-heading">
            <div className="step-badge">1</div>
            <div>
              <h2>Your Resume</h2>
              <p>Upload PDF, DOCX, TXT — or paste your resume text directly.</p>
            </div>
          </div>

          <label
            className={`upload-box${resumeFile ? " has-file" : ""}`}
            style={{ cursor: "pointer" }}
          >
            <input
              type="file"
              accept=".pdf,.docx,.txt"
              onChange={(e) => setResumeFile(e.target.files?.[0] || null)}
            />
            <div className="upload-icon">
              <UploadIcon />
            </div>
            <strong>{resumeFile ? resumeFile.name : "Click to upload resume"}</strong>
            <small>PDF, DOCX, or TXT supported</small>
          </label>

          <div className="or-divider"><span>or paste below</span></div>

          <textarea
            value={resumeText}
            onChange={(e) => setResumeText(e.target.value)}
            placeholder="Paste your resume text here..."
          />
        </section>

        {/* Settings panel */}
        <aside className="panel">
          <div className="section-heading">
            <div className="step-badge">2</div>
            <div>
              <h2>Job Search</h2>
              <p>Configure your search preferences.</p>
            </div>
          </div>

          <label>
            Looking For
            <select value={opportunityType} onChange={(e) => setOpportunityType(e.target.value)}>
              <option value="jobs">Full-time Jobs</option>
              <option value="internships">Internships</option>
              <option value="both">Both</option>
            </select>
          </label>

          <label>
            Job Provider
            <select value={provider} onChange={(e) => setProvider(e.target.value)}>
              <option value="all">All Providers</option>
              <option value="jsearch">JSearch via RapidAPI</option>
              <option value="adzuna">Adzuna</option>
              <option value="serpapi">SerpAPI Google Jobs</option>
              <option value="jooble">Jooble</option>
              <option value="remotive">Remotive (no key needed)</option>
            </select>
          </label>

          {(provider === "jsearch" || provider === "adzuna" || provider === "serpapi" || provider === "jooble" || provider === "all") && (
            <>
              <p className="hint">
                {provider === "all"
                  ? "🔒 Searches JSearch, Adzuna, SerpAPI, Jooble, and Remotive together. API keys stay in the backend."
                  : provider === "jsearch"
                    ? "🔒 JSearch API key is configured in the backend."
                    : provider === "adzuna"
                      ? "🔒 Adzuna credentials are configured in the backend."
                      : provider === "serpapi"
                        ? "🔒 SerpAPI key is configured in the backend."
                        : "🔒 Jooble API key is configured in the backend."}
              </p>
              <div className="two-col">
                <label>
                  Country
                  <select value={country} onChange={(e) => setCountry(e.target.value)}>
                    <option value="in">India</option>
                    <option value="us">United States</option>
                    <option value="gb">United Kingdom</option>
                    <option value="ca">Canada</option>
                    <option value="au">Australia</option>
                    <option value="sg">Singapore</option>
                  </select>
                </label>
                <label>
                  Date Posted
                  <select value={datePosted} onChange={(e) => setDatePosted(e.target.value)}>
                    <option value="all">Any time</option>
                    <option value="today">Today</option>
                    <option value="3days">3 days</option>
                    <option value="week">This week</option>
                    <option value="month">This month</option>
                  </select>
                </label>
              </div>
            </>
          )}

          <label>
            Select Role(s)
            <select multiple value={selectedRoles} onChange={updateSelectedRoles} className="multi-select">
              {ROLE_OPTIONS.map((role) => (
                <option key={role} value={role}>{role}</option>
              ))}
            </select>
            <small style={{ color: "var(--ink-3)", fontWeight: 300 }}>
              Hold Cmd/Ctrl to pick multiple. Manual query below overrides.
            </small>
          </label>

          {selectedRoles.length > 0 && (
            <div className="selected-roles">
              {selectedRoles.map((role) => (
                <span className="role-pill" key={role}>
                  {role}
                  <button
                    type="button"
                    onClick={() => setSelectedRoles(selectedRoles.filter((r) => r !== role))}
                    aria-label={`Remove ${role}`}
                  >
                    ×
                  </button>
                </span>
              ))}
            </div>
          )}

          <label>
            Search Query
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={selectedRoles.length ? selectedRoles.join(" OR ") : suggestedQuery}
            />
          </label>

          <div className="two-col">
            <label>
              Fetch limit
              <input
                type="number"
                min="5"
                max="100"
                value={resultLimit}
                onChange={(e) => setResultLimit(Number(e.target.value))}
              />
            </label>
            <label>
              Show top
              <input
                type="number"
                min="1"
                max="25"
                value={topN}
                onChange={(e) => setTopN(Number(e.target.value))}
              />
            </label>
          </div>

          <button type="submit" className={`btn-primary${loading ? " loading" : ""}`} disabled={loading}>
            {loading ? "Finding matches…" : "Find Matching Jobs"}
          </button>

          {error && <div className="error">⚠ {error}</div>}
        </aside>
      </form>

      {/* Loading skeletons */}
      {loading && <LoadingState count={Math.min(topN, 5)} />}

      {/* Results */}
      {result && !loading && (
        <section className="results" ref={resultsRef}>
          <div className="results-header">
            <div>
              <span className="eyebrow">Search complete</span>
              <h2>Best matches for "{result.query_used}"</h2>
              <p>
                Fetched {result.jobs_fetched} {result.opportunity_type} · Showing top {result.jobs.length} ranked matches via {result.provider}
              </p>
            </div>
            {result.suggested_queries?.length > 0 && (
              <div className="suggestions">
                <div style={{ width: "100%", fontSize: "0.72rem", fontWeight: 600, color: "var(--ink-3)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 6 }}>
                  Try also
                </div>
                {result.suggested_queries.map((item) => (
                  <button key={item} className="suggestion-btn" type="button" onClick={() => setQuery(item)}>
                    {item}
                  </button>
                ))}
              </div>
            )}
          </div>

          {result.provider_warnings?.length > 0 && (
            <div className="provider-warnings">
              <strong>Provider warnings</strong>
              <ul>
                {result.provider_warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            </div>
          )}

          <div className="job-grid">
            {result.jobs.map((job) => (
              <JobCard key={`${job.rank}-${job.url || job.title}`} job={job} />
            ))}
          </div>
        </section>
      )}
    </main>
  );
}

/* ── JobCard ── */
function JobCard({ job }) {
  const isTop = job.rank <= 3;

  return (
    <article className="job-card">
      {/* Header */}
      <div className="job-top">
        <div className={`rank-badge${isTop ? " top" : ""}`}>#{job.rank}</div>
        <div className="job-title-row" style={{ flex: 1 }}>
          <h3>{job.title}</h3>
          <div className="job-meta">
            {job.company && (
              <span className="meta-pill">
                <BriefcaseIcon />
                {job.company}
              </span>
            )}
            {job.location && (
              <span className="meta-pill">
                <MapPinIcon />
                {job.location}
              </span>
            )}
            {job.source && (
              <span className="meta-pill">
                <SourceIcon />
                {job.source}
              </span>
            )}
            {job.job_type && (
              <span className="meta-pill">{job.job_type}</span>
            )}
          </div>
        </div>
      </div>

      {/* Fit score bar */}
      <div className="score-bar-row">
        <div>
          <div className="fit-score">{job.final_score}</div>
          <div className="fit-label">/ 100 fit</div>
        </div>
        <div style={{ flex: 1 }}>
          <div className="bar-track">
            <div className="bar-fill" style={{ width: `${job.final_score}%` }} />
          </div>
        </div>
      </div>

      {/* Metric tiles */}
      <div className="metrics">
        {job.learning_to_rank_score !== null && job.learning_to_rank_score !== undefined && (
          <Metric label="LTR rank" value={`${Math.round(job.learning_to_rank_score)}%`} />
        )}
        <Metric label="Semantic" value={`${job.semantic_score}%`} />
        <Metric label="Skill match" value={`${job.skill_score}%`} />
        <Metric label="ATS score" value={`${job.ats_score}`} />
        {job.resume_intelligence?.weighted_section_score !== undefined && (
          <Metric label="Section fit" value={`${Math.round(job.resume_intelligence.weighted_section_score)}%`} />
        )}
      </div>

      {/* Summary */}
      <p className="summary">{job.summary}</p>

      {/* Experience warnings */}
      {job.experience_fit?.warning && (
        <div className="warning">⚠ {job.experience_fit.warning}</div>
      )}
      {job.experience_fit && (job.experience_fit.resume_years !== null && job.experience_fit.resume_years !== undefined) && (
        <p className="experience-line">
          Your experience: <strong>{formatYears(job.experience_fit.resume_years)}</strong>
          {" · "}Required: <strong>{formatExperienceRange(job.experience_fit)}</strong>
        </p>
      )}

      {job.resume_intelligence && <ResumeIntelligence intelligence={job.resume_intelligence} />}

      {/* Skills */}
      <SkillRow title="Strong Evidence" values={job.strong_evidence_skills} tone="good" />
      <SkillRow title="Weak Evidence" values={job.weak_evidence_skills} tone="warn" />
      <SkillRow title="Matched Skills" values={job.matched_skills} tone="good" />
      <SkillRow title="Related Skill Bridges" values={job.skill_graph_explanations} tone="neutral" />
      <SkillRow title="Missing Skills" values={job.missing_skills} tone="bad" />

      {/* Accordion */}
      <details>
        <summary>Score explanation &amp; improvements</summary>
        <div className="details-body">
          <p>{job.analysis.overall_explanation}</p>
          {job.analysis.improvements?.length > 0 && (
            <>
              <strong>Improvements</strong>
              <ul>
                {job.analysis.improvements.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </>
          )}
          {job.analysis.ats_suggestions?.length > 0 && (
            <>
              <strong>ATS Suggestions</strong>
              <ul>
                {job.analysis.ats_suggestions.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </>
          )}
        </div>
      </details>

      {/* Apply link */}
      {job.url && (
        <a className="apply" href={job.url} target="_blank" rel="noreferrer">
          Open Job Posting <ExternalLinkIcon />
        </a>
      )}
    </article>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
}

function ResumeIntelligence({ intelligence }) {
  const impacts = intelligence.project_impacts || [];
  const toolDepths = Object.entries(intelligence.tool_depths || {}).slice(0, 8);

  return (
    <div className="resume-intel">
      <div className="intel-head">
        <span>{String(intelligence.candidate_stage || "candidate").replace("_", " ")} profile</span>
        <span>Weighted sections: {Math.round(intelligence.weighted_section_score || 0)}%</span>
      </div>
      {impacts.length > 0 && (
        <div className="intel-block">
          <strong>Impact signals</strong>
          <ul>
            {impacts.slice(0, 2).map((impact) => (
              <li key={impact}>{impact}</li>
            ))}
          </ul>
        </div>
      )}
      {toolDepths.length > 0 && (
        <div className="intel-depths">
          {toolDepths.map(([skill, depth]) => (
            <span className={`depth-pill depth-${depth}`} key={skill}>
              {skill}: {depth}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function SkillRow({ title, values, tone }) {
  if (!values?.length) return null;
  return (
    <div className="skills">
      <div className="skills-label">{title}</div>
      <div>
        {values.slice(0, 8).map((item) => (
          <span className={`chip ${tone}`} key={item}>{item}</span>
        ))}
      </div>
    </div>
  );
}

function formatYears(value) {
  if (value === null || value === undefined) return "unknown";
  const n = Number(value);
  return `${n} year${n === 1 ? "" : "s"}`;
}

function formatExperienceRange(exp) {
  const min = exp.required_min_years;
  const max = exp.required_max_years;
  if (min === null || min === undefined) return "not specified";
  if (max === null || max === undefined) return `${formatYears(min)}+`;
  return `${formatYears(min)} – ${formatYears(max)}`;
}

createRoot(document.getElementById("root")).render(<App />);
