import fs from "fs";
import path from "path";
import { ClusterData, Initiative, InitiativeSummary } from "./types";

const WEBAPP_DATA_DIR = path.join(process.cwd(), "..", "data", "webapp");

const DETAILS_DIR = path.join(WEBAPP_DATA_DIR, "initiative_details");

const INDEX_PATH = path.join(WEBAPP_DATA_DIR, "initiative_index.json");

const CLUSTERING_DIR = path.join(process.cwd(), "..", "data", "clustering");

let cachedIndex: InitiativeSummary[] | null = null;
let cachedAt = 0;
const CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes

export async function getInitiativeIndex(): Promise<InitiativeSummary[]> {
  const now = Date.now();
  if (cachedIndex && now - cachedAt < CACHE_TTL_MS) {
    return cachedIndex;
  }

  const content = fs.readFileSync(INDEX_PATH, "utf-8");
  const summaries: InitiativeSummary[] = JSON.parse(content);

  cachedIndex = summaries;
  cachedAt = now;
  return summaries;
}

export async function getInitiativeDetail(
  id: string
): Promise<Initiative | null> {
  const filePath = path.join(DETAILS_DIR, `${id}.json`);
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
