import fs from "fs";
import path from "path";
import { ClusterData, Initiative, InitiativeSummary } from "./types";

const DATA_DIR = path.join(
  process.cwd(),
  "..",
  "data",
  "scrape",
  "initiative_details"
);

const CLUSTERING_DIR = path.join(process.cwd(), "..", "data", "clustering");

let cachedIndex: InitiativeSummary[] | null = null;
let cachedAt = 0;
const CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes

function extractMetadataFromContent(
  content: string,
  filename: string
): InitiativeSummary | null {
  try {
    // Key fields are near the top of the JSON — extract via regex for speed
    // (avoids parsing multi-MB files as JSON).
    const getStr = (key: string): string => {
      const m = content.match(new RegExp(`"${key}"\\s*:\\s*"([^"]*?)"`));
      return m ? m[1] : "";
    };
    const getNum = (key: string): number => {
      const m = content.match(new RegExp(`"${key}"\\s*:\\s*(\\d+)`));
      return m ? parseInt(m[1], 10) : 0;
    };
    const getArr = (key: string): string[] => {
      const m = content.match(new RegExp(`"${key}"\\s*:\\s*\\[([^\\]]*?)\\]`));
      if (!m) return [];
      const items: string[] = [];
      const re = /"([^"]*?)"/g;
      let match;
      while ((match = re.exec(m[1])) !== null) {
        items.push(match[1]);
      }
      return items;
    };

    const id = getNum("id");
    if (!id) {
      const fid = parseInt(path.basename(filename, ".json"), 10);
      if (isNaN(fid)) return null;
    }

    // Sum total_feedback from publications
    let totalFeedback = 0;
    const fbRe = /"total_feedback"\s*:\s*(\d+)/g;
    let fbMatch;
    while ((fbMatch = fbRe.exec(content)) !== null) {
      totalFeedback += parseInt(fbMatch[1], 10);
    }

    // Check for open feedback publications
    const fsRe = /"feedback_status"\s*:\s*"([^"]+)"/g;
    let has_open_feedback = false;
    let fsm;
    while ((fsm = fsRe.exec(content)) !== null) {
      if (fsm[1] === "OPEN") {
        has_open_feedback = true;
        break;
      }
    }

    // Extract country and user_type counts from feedback objects
    const country_counts: Record<string, number> = {};
    const user_type_counts: Record<string, number> = {};

    const utRe = /"user_type"\s*:\s*"([^"]+)"/g;
    let m;
    while ((m = utRe.exec(content)) !== null) {
      const ut = m[1];
      user_type_counts[ut] = (user_type_counts[ut] || 0) + 1;
    }

    const cRe = /"country"\s*:\s*"([^"]+)"/g;
    while ((m = cRe.exec(content)) !== null) {
      const c = m[1];
      country_counts[c] = (country_counts[c] || 0) + 1;
    }

    // Extract feedback dates and compute timeline histogram
    const TIMELINE_BUCKETS = 20;
    const publishedDate = getStr("published_date");
    const startMs = new Date(publishedDate.replace(/\//g, "-")).getTime();
    const feedbackDates: number[] = [];
    const dateRe = /"date"\s*:\s*"(\d{4}\/\d{2}\/\d{2}\s+\d{2}:\d{2}:\d{2})"/g;
    while ((m = dateRe.exec(content)) !== null) {
      const t = new Date(m[1].replace(/\//g, "-")).getTime();
      if (!isNaN(t)) feedbackDates.push(t);
    }
    let feedback_timeline: number[] = [];
    let lastFeedbackMs = 0;
    if (feedbackDates.length > 0) {
      let endMs = startMs;
      for (const t of feedbackDates) {
        if (t > endMs) endMs = t;
      }
      lastFeedbackMs = endMs;
      if (endMs > startMs) {
        const bucketWidth = (endMs - startMs) / TIMELINE_BUCKETS;
        feedback_timeline = new Array(TIMELINE_BUCKETS).fill(0);
        for (const t of feedbackDates) {
          const idx = Math.min(
            Math.floor((t - startMs) / bucketWidth),
            TIMELINE_BUCKETS - 1
          );
          if (idx >= 0) feedback_timeline[idx]++;
        }
      }
    }
    const last_feedback_date = lastFeedbackMs
      ? new Date(lastFeedbackMs).toISOString()
      : "";

    return {
      id: id || parseInt(path.basename(filename, ".json"), 10),
      short_title: getStr("short_title"),
      department: getStr("department"),
      stage: getStr("stage"),
      status: getStr("status"),
      topics: getArr("topics"),
      policy_areas: getArr("policy_areas"),
      published_date: publishedDate,
      last_cached_at: getStr("last_cached_at"),
      type_of_act: getStr("type_of_act"),
      reference: getStr("reference"),
      total_feedback: totalFeedback,
      country_counts,
      user_type_counts,
      feedback_timeline,
      last_feedback_date,
      has_open_feedback,
    };
  } catch {
    return null;
  }
}

export async function getInitiativeIndex(): Promise<InitiativeSummary[]> {
  const now = Date.now();
  if (cachedIndex && now - cachedAt < CACHE_TTL_MS) {
    return cachedIndex;
  }

  const files = fs.readdirSync(DATA_DIR).filter((f) => f.endsWith(".json"));
  const summaries: InitiativeSummary[] = [];

  for (const file of files) {
    const filePath = path.join(DATA_DIR, file);
    const content = fs.readFileSync(filePath, "utf-8");

    const summary = extractMetadataFromContent(content, file);
    if (summary) {
      summaries.push(summary);
    }
  }

  // Default sort: most recently updated
  summaries.sort(
    (a, b) =>
      new Date(b.last_cached_at).getTime() -
      new Date(a.last_cached_at).getTime()
  );

  cachedIndex = summaries;
  cachedAt = now;
  return summaries;
}

export async function getInitiativeDetail(
  id: string
): Promise<Initiative | null> {
  const filePath = path.join(DATA_DIR, `${id}.json`);
  if (!fs.existsSync(filePath)) return null;

  const content = fs.readFileSync(filePath, "utf-8");
  return JSON.parse(content) as Initiative;
}

/** List clustering schemes that have data for a given initiative */
export async function getClusteringSchemesForInitiative(
  id: string
): Promise<string[]> {
  if (!fs.existsSync(CLUSTERING_DIR)) return [];

  const schemes: string[] = [];
  for (const scheme of fs.readdirSync(CLUSTERING_DIR)) {
    const schemeDir = path.join(CLUSTERING_DIR, scheme);
    if (!fs.statSync(schemeDir).isDirectory()) continue;
    const filePath = path.join(schemeDir, `${id}.json`);
    if (fs.existsSync(filePath)) {
      schemes.push(scheme);
    }
  }
  return schemes;
}

/** Load cluster data for a specific initiative + scheme */
export async function getClusterData(
  id: string,
  scheme: string
): Promise<ClusterData | null> {
  const filePath = path.join(CLUSTERING_DIR, scheme, `${id}.json`);
  if (!fs.existsSync(filePath)) return null;

  const content = fs.readFileSync(filePath, "utf-8");
  const data = JSON.parse(content);

  return {
    cluster_model: data.cluster_model,
    cluster_algorithm: data.cluster_algorithm,
    cluster_params: data.cluster_params || {},
    cluster_n_clusters: data.cluster_n_clusters,
    cluster_noise_count: data.cluster_noise_count,
    cluster_silhouette: data.cluster_silhouette ?? null,
    cluster_assignments: data.cluster_assignments || {},
  };
}
