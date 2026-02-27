"use client";

import { useMemo, useState } from "react";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  ClusterData,
  ClusterNode,
  Feedback,
  buildClusterTree,
} from "@/lib/types";
import { ClusterNodeComponent } from "@/components/cluster-node";

type ClusterSort = "size-desc" | "size-asc" | "id-asc";

export interface TimeRange {
  startMs: number;
  endMs: number;
  buckets: number;
}

interface ClusterViewProps {
  clusterData: ClusterData;
  allFeedback: Feedback[];
  schemes: string[];
  currentScheme: string;
  onSchemeChange: (scheme: string) => void;
  publishedDate: string;
}

function formatSchemeName(scheme: string): string {
  // Extract key info from the long scheme name
  const parts = scheme.split("_");
  const algorithm = parts[0];
  // Get model name (second segment, possibly hyphenated)
  const rest = scheme.slice(algorithm.length + 1);
  const paramStart = rest.indexOf("_distance_threshold=") !== -1
    ? rest.indexOf("_distance_threshold=")
    : rest.indexOf("_min_cluster_size=") !== -1
      ? rest.indexOf("_min_cluster_size=")
      : rest.indexOf("_linkage=") !== -1
        ? rest.indexOf("_linkage=")
        : rest.indexOf("_n_clusters=");
  const model = paramStart > 0 ? rest.slice(0, paramStart) : rest;
  const params = paramStart > 0 ? rest.slice(paramStart + 1) : "";

  const shortParams = params
    .split("_")
    .map((p) => {
      const [k, v] = p.split("=");
      if (!v) return p;
      // Abbreviate common param names
      const abbrev: Record<string, string> = {
        distance_threshold: "dt",
        linkage: "link",
        max_cluster_size: "max",
        max_depth: "depth",
        sub_cluster_scale: "scale",
        sub_cluster_step: "step",
        min_cluster_size: "min",
        min_samples: "samples",
        n_clusters: "k",
      };
      return `${abbrev[k] || k}=${v}`;
    })
    .join(" ");

  return `${algorithm} / ${model}${shortParams ? ` (${shortParams})` : ""}`;
}

function parseFeedbackDate(d: string): number {
  // "2016/11/24 22:27:09" → ms timestamp
  return new Date(d.replace(/\//g, "-")).getTime();
}

const SPARKLINE_BUCKETS = 20;

export function ClusterView({
  clusterData,
  allFeedback,
  schemes,
  currentScheme,
  onSchemeChange,
  publishedDate,
}: ClusterViewProps) {
  const [sort, setSort] = useState<ClusterSort>("size-desc");
  const [search, setSearch] = useState("");
  const [minSize, setMinSize] = useState(1);

  // Build feedback lookup
  const feedbackLookup = useMemo(() => {
    const map = new Map<string, Feedback>();
    for (const fb of allFeedback) {
      map.set(String(fb.id), fb);
    }
    return map;
  }, [allFeedback]);

  // Compute global time range for sparklines
  const timeRange = useMemo<TimeRange>(() => {
    const startMs = new Date(publishedDate.replace(/\//g, "-")).getTime();
    let endMs = startMs;
    for (const fb of allFeedback) {
      if (fb.date) {
        const t = parseFeedbackDate(fb.date);
        if (t > endMs) endMs = t;
      }
    }
    if (endMs <= startMs) endMs = startMs + 1;
    return { startMs, endMs, buckets: SPARKLINE_BUCKETS };
  }, [allFeedback, publishedDate]);

  // Build cluster tree
  const allClusters = useMemo(
    () => buildClusterTree(clusterData.cluster_assignments, feedbackLookup),
    [clusterData.cluster_assignments, feedbackLookup]
  );

  // Filter and sort
  const filteredClusters = useMemo(() => {
    let result = allClusters;

    if (minSize > 1) {
      result = result.filter((c) => c.allItems.length >= minSize);
    }

    if (search) {
      const q = search.toLowerCase();
      result = result.filter((c) =>
        c.allItems.some((fb) => {
          const hay = [
            fb.feedback_text,
            fb.organization,
            fb.country,
            fb.first_name,
            fb.surname,
            fb.combined_feedback_summary,
          ]
            .filter(Boolean)
            .join(" ")
            .toLowerCase();
          return hay.includes(q);
        })
      );
    }

    result = [...result];
    switch (sort) {
      case "size-desc":
        result.sort((a, b) => b.allItems.length - a.allItems.length);
        break;
      case "size-asc":
        result.sort((a, b) => a.allItems.length - b.allItems.length);
        break;
      case "id-asc":
        result.sort((a, b) =>
          a.label.localeCompare(b.label, undefined, { numeric: true })
        );
        break;
    }

    return result;
  }, [allClusters, sort, search, minSize]);

  const totalMatching = filteredClusters.reduce(
    (s, c) => s + c.allItems.length,
    0
  );

  return (
    <div>
      {/* Cluster info bar */}
      <div className="rounded-lg border bg-muted/20 p-4 mb-4">
        <div className="flex flex-wrap gap-x-6 gap-y-1 text-sm">
          <InfoItem label="Model" value={clusterData.cluster_model} />
          <InfoItem label="Algorithm" value={clusterData.cluster_algorithm} />
          <InfoItem
            label="Parameters"
            value={Object.entries(clusterData.cluster_params)
              .map(([k, v]) => `${k}=${v}`)
              .join(", ")}
          />
          <InfoItem
            label="Clusters"
            value={String(clusterData.cluster_n_clusters)}
          />
          <InfoItem
            label="Noise"
            value={String(clusterData.cluster_noise_count)}
          />
          <InfoItem
            label="Silhouette"
            value={
              clusterData.cluster_silhouette != null
                ? clusterData.cluster_silhouette.toFixed(4)
                : "N/A"
            }
          />
        </div>
      </div>

      {/* Controls */}
      <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:flex-wrap">
        {/* Scheme selector */}
        {schemes.length > 1 && (
          <Select value={currentScheme} onValueChange={onSchemeChange}>
            <SelectTrigger className="w-full sm:w-[320px]">
              <SelectValue placeholder="Clustering scheme" />
            </SelectTrigger>
            <SelectContent>
              {schemes.map((s) => (
                <SelectItem key={s} value={s}>
                  {formatSchemeName(s)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}

        <Select
          value={sort}
          onValueChange={(v) => setSort(v as ClusterSort)}
        >
          <SelectTrigger className="w-full sm:w-[220px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="size-desc">Largest first</SelectItem>
            <SelectItem value="size-asc">Smallest first</SelectItem>
            <SelectItem value="id-asc">Cluster ID</SelectItem>
          </SelectContent>
        </Select>

        <Input
          placeholder="Search feedback..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full sm:w-[240px]"
        />

        <div className="flex items-center gap-2">
          <label className="text-xs font-semibold text-muted-foreground uppercase">
            Min size
          </label>
          <Input
            type="number"
            min={1}
            value={minSize}
            onChange={(e) => setMinSize(Math.max(1, parseInt(e.target.value) || 1))}
            className="w-[70px]"
          />
        </div>
      </div>

      {/* Count */}
      <p className="text-sm text-muted-foreground mb-4">
        {filteredClusters.length} clusters &middot;{" "}
        {totalMatching.toLocaleString()} feedback items
      </p>

      {/* Cluster list */}
      <div className="space-y-3">
        {filteredClusters.map((cluster) => (
          <ClusterNodeComponent key={cluster.label} node={cluster} timeRange={timeRange} />
        ))}
      </div>
    </div>
  );
}

function InfoItem({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span className="text-[10px] uppercase tracking-wider font-semibold text-muted-foreground">
        {label}
      </span>
      <span className="ml-1.5 text-sm font-medium">{value}</span>
    </div>
  );
}
