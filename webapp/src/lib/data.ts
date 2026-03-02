import fs from "fs";
import path from "path";
import { ClusterData, ClusterSummaries, CountryStats, GlobalStats, Initiative, InitiativeSummary } from "./types";

const WEBAPP_DATA_DIR = path.join(process.cwd(), "..", "data", "webapp");

const DETAILS_DIR = path.join(WEBAPP_DATA_DIR, "initiative_details");

const INDEX_PATH = path.join(WEBAPP_DATA_DIR, "initiative_index.json");

const STATS_PATH = path.join(WEBAPP_DATA_DIR, "global_stats.json");

const COUNTRY_STATS_PATH = path.join(WEBAPP_DATA_DIR, "country_stats.json");

const CLUSTERING_DIR = path.join(process.cwd(), "..", "data", "clustering");

const CLUSTER_SUMMARIES_DIR = path.join(process.cwd(), "..", "data", "cluster_summaries");

let cachedIndex: InitiativeSummary[] | null = null;
let cachedIndexAt = 0;
let cachedStats: GlobalStats | null = null;
let cachedStatsAt = 0;
let cachedCountryStats: CountryStats | null = null;
let cachedCountryStatsAt = 0;
const CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes

export async function getInitiativeIndex(): Promise<InitiativeSummary[]> {
  const now = Date.now();
  if (cachedIndex && now - cachedIndexAt < CACHE_TTL_MS) {
    return cachedIndex;
  }

  const content = fs.readFileSync(INDEX_PATH, "utf-8");
  const summaries: InitiativeSummary[] = JSON.parse(content);

  cachedIndex = summaries;
  cachedIndexAt = now;
  return summaries;
}

export async function getGlobalStats(): Promise<GlobalStats> {
  const now = Date.now();
  if (cachedStats && now - cachedStatsAt < CACHE_TTL_MS) {
    return cachedStats;
  }

  const content = fs.readFileSync(STATS_PATH, "utf-8");
  const stats: GlobalStats = JSON.parse(content);

  cachedStats = stats;
  cachedStatsAt = now;
  return stats;
}

export async function getCountryStats(): Promise<CountryStats> {
  const now = Date.now();
  if (cachedCountryStats && now - cachedCountryStatsAt < CACHE_TTL_MS) {
    return cachedCountryStats;
  }

  const content = fs.readFileSync(COUNTRY_STATS_PATH, "utf-8");
  const stats: CountryStats = JSON.parse(content);

  cachedCountryStats = stats;
  cachedCountryStatsAt = now;
  return stats;
}

export async function getInitiativeDetail(
  id: string
): Promise<Initiative | null> {
  const filePath = path.join(DETAILS_DIR, `${id}.json`);
  if (!fs.existsSync(filePath)) return null;

  const content = fs.readFileSync(filePath, "utf-8");
  return JSON.parse(content) as Initiative;
}

/** Find the clustering JSON file for an initiative within a scheme directory.
 *  Files are named either `{id}.json` or `{id}_{params}.json`. */
function findClusteringFile(schemeDir: string, id: string): string | null {
  const simple = path.join(schemeDir, `${id}.json`);
  if (fs.existsSync(simple)) return simple;

  const prefix = `${id}_`;
  try {
    for (const entry of fs.readdirSync(schemeDir)) {
      if (entry.startsWith(prefix) && entry.endsWith(".json")) {
        return path.join(schemeDir, entry);
      }
    }
  } catch {
    // directory unreadable
  }
  return null;
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
    if (findClusteringFile(schemeDir, id)) {
      schemes.push(scheme);
    }
  }
  return schemes;
}

/** Load cluster summaries for a specific initiative + scheme */
function getClusterSummaries(
  id: string,
  scheme: string
): ClusterSummaries | null {
  const filePath = path.join(CLUSTER_SUMMARIES_DIR, scheme, `${id}.json`);
  if (!fs.existsSync(filePath)) return null;

  try {
    const content = fs.readFileSync(filePath, "utf-8");
    const data = JSON.parse(content);
    return {
      policy_summary: data.policy_summary ?? null,
      cluster_summaries: data.cluster_summaries ?? {},
    };
  } catch {
    return null;
  }
}

/** Load cluster data for a specific initiative + scheme */
export async function getClusterData(
  id: string,
  scheme: string
): Promise<ClusterData | null> {
  const schemeDir = path.join(CLUSTERING_DIR, scheme);
  const filePath = findClusteringFile(schemeDir, id);
  if (!filePath) return null;

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
    cluster_summaries: getClusterSummaries(id, scheme),
  };
}
