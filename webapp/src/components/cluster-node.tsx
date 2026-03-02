"use client";

import { useMemo, useState } from "react";
import { ClusterNode, ClusterSummaryEntry, Feedback, computeClusterStats } from "@/lib/types";
import { FeedbackCard } from "@/components/feedback-card";
import {
  CountryBar,
  UserTypeBar,
  ClusterDetailStats,
} from "@/components/cluster-stats-bar";
import { ExpandableText } from "@/components/expandable-text";
import { Button } from "@/components/ui/button";
import { truncateText } from "@/lib/utils";
import { TimeRange } from "@/components/cluster-view";

const INITIAL_SHOW = 5;

interface DedupedFeedback {
  feedback: Feedback;
  duplicateCount: number;
}

/** Group feedback items with identical feedback_text, keeping one representative per group. */
function deduplicateFeedback(items: Feedback[]): DedupedFeedback[] {
  const groups = new Map<string, Feedback[]>();
  const noText: Feedback[] = [];

  for (const fb of items) {
    const text = fb.feedback_text;
    if (!text) {
      noText.push(fb);
    } else {
      const existing = groups.get(text);
      if (existing) existing.push(fb);
      else groups.set(text, [fb]);
    }
  }

  const result: DedupedFeedback[] = [];
  for (const [, group] of groups) {
    result.push({ feedback: group[0], duplicateCount: group.length });
  }
  for (const fb of noText) {
    result.push({ feedback: fb, duplicateCount: 1 });
  }
  return result;
}

function parseFeedbackDate(d: string): number {
  return new Date(d.replace(/\//g, "-")).getTime();
}

/** Tiny inline SVG sparkline showing feedback volume over time */
function Sparkline({
  items,
  timeRange,
}: {
  items: Feedback[];
  timeRange: TimeRange;
}) {
  const { startMs, endMs, buckets } = timeRange;
  const bucketWidth = (endMs - startMs) / buckets;

  const counts = useMemo(() => {
    const c = new Array(buckets).fill(0);
    for (const fb of items) {
      if (!fb.date) continue;
      const t = parseFeedbackDate(fb.date);
      const idx = Math.min(
        Math.floor((t - startMs) / bucketWidth),
        buckets - 1
      );
      if (idx >= 0) c[idx]++;
    }
    return c;
  }, [items, startMs, bucketWidth, buckets]);

  const max = Math.max(...counts, 1);
  const w = 80;
  const h = 18;
  const pad = 1;
  const stepX = w / (buckets - 1);

  const points = counts.map((c, i) => {
    const x = i * stepX;
    const y = pad + (1 - c / max) * (h - 2 * pad);
    return `${x},${y}`;
  });

  // Area fill path: line across top, then close along bottom
  const areaPath = `M0,${h} L${points.map((p) => `L${p}`).join(" ")} L${w},${h} Z`;

  return (
    <svg
      width={w}
      height={h}
      className="shrink-0"
      viewBox={`0 0 ${w} ${h}`}
    >
      <path d={areaPath} fill="#6366f1" opacity={0.15} />
      <polyline
        points={points.join(" ")}
        fill="none"
        stroke="#6366f1"
        strokeWidth={1.5}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

interface ClusterNodeComponentProps {
  node: ClusterNode;
  isSubCluster?: boolean;
  timeRange: TimeRange;
  summaryLookup?: Record<string, ClusterSummaryEntry>;
}

export function ClusterNodeComponent({
  node,
  isSubCluster = false,
  timeRange,
  summaryLookup,
}: ClusterNodeComponentProps) {
  const [open, setOpen] = useState(false);
  const [visibleCount, setVisibleCount] = useState(INITIAL_SHOW);

  const items = node.allItems;
  const hasChildren = node.children.length > 0;
  const stats = computeClusterStats(items);
  const total = items.length;

  // Compute organizations for detail stats
  const orgCounts = new Map<string, number>();
  for (const fb of items) {
    if (fb.organization) {
      orgCounts.set(fb.organization, (orgCounts.get(fb.organization) || 0) + 1);
    }
  }
  const organizations = Array.from(orgCounts.entries()).sort(
    (a, b) => b[1] - a[1]
  );

  // Cluster summary entry for this node
  const entry = summaryLookup?.[node.label];
  const summaryTitle = entry?.title?.replace(/\*\*/g, "") ?? null;

  // Preview text: use summary title, fall back to first non-empty feedback
  let preview = "";
  if (summaryTitle) {
    preview = truncateText(summaryTitle, 150);
  } else {
    for (const fb of items) {
      const t = fb.combined_feedback_summary || fb.feedback_text;
      if (t) {
        preview = truncateText(t, 150);
        break;
      }
    }
  }

  // Leaf items to render (direct items if available, otherwise all), deduped
  const rawLeafItems =
    node.directItems.length > 0 ? node.directItems : node.allItems;
  const leafItems = useMemo(() => deduplicateFeedback(rawLeafItems), [rawLeafItems]);
  const visibleItems = leafItems.slice(0, visibleCount);

  return (
    <div
      className={
        isSubCluster
          ? "ml-5 border-l-2 border-gray-200"
          : "rounded-lg border overflow-hidden"
      }
    >
      {/* Header */}
      <button
        className={`w-full text-left cursor-pointer hover:bg-accent/50 transition-colors ${
          isSubCluster ? "px-3 py-2" : "px-4 py-3"
        }`}
        onClick={() => setOpen(!open)}
      >
        <div className="flex items-start gap-3">
          <span
            className={`text-xs text-muted-foreground shrink-0 mt-0.5 transition-transform ${
              open ? "rotate-90" : ""
            }`}
          >
            &#9654;
          </span>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <Sparkline items={items} timeRange={timeRange} />
              <CountryBar
                sortedCountries={stats.sortedCountries}
                total={total}
              />
              <UserTypeBar sortedTypes={stats.sortedTypes} total={total} />
            </div>
            {(preview || hasChildren) && (
              <div className="flex items-center gap-2 mt-1">
                <span className="text-[11px] font-bold bg-primary text-primary-foreground px-2 py-px rounded-full shrink-0">
                  {items.length}
                </span>
                {hasChildren && (
                  <span className="text-[10px] text-muted-foreground bg-muted px-1.5 py-px rounded-lg shrink-0">
                    {node.children.length} sub-clusters
                  </span>
                )}
                {preview && (
                  <p className="text-xs text-muted-foreground truncate min-w-0">
                    {preview}
                  </p>
                )}
              </div>
            )}
          </div>
        </div>
      </button>

      {/* Body */}
      {open && (
        <div className={isSubCluster ? "" : "border-t"}>
          {entry?.summary && (
            <div className="px-4 pt-3">
              <ExpandableText
                text={entry.summary}
                maxPreviewChars={500}
                isMarkdown
                label="Cluster summary"
              />
            </div>
          )}
          <ClusterDetailStats
            sortedCountries={stats.sortedCountries}
            sortedTypes={stats.sortedTypes}
            total={total}
            organizations={organizations}
          />

          {hasChildren ? (
            <div className="p-2">
              {/* All sub-clusters (including promoted direct items) sorted by size */}
              {[...node.children, ...node.directItems.map((fb): ClusterNode => ({
                  label: `${node.label}.unclustered.${fb.id}`,
                  directItems: [fb],
                  children: [],
                  allItems: [fb],
                }))]
                .sort((a, b) => b.allItems.length - a.allItems.length)
                .map((child) => (
                  <ClusterNodeComponent
                    key={child.label}
                    node={child}
                    isSubCluster
                    timeRange={timeRange}
                    summaryLookup={summaryLookup}
                  />
                ))}
            </div>
          ) : (
            <div className="space-y-2 p-3">
              {visibleItems.map(({ feedback: fb, duplicateCount }) => (
                <FeedbackCard key={fb.id} feedback={fb} duplicateCount={duplicateCount} />
              ))}
              {visibleCount < leafItems.length && (
                <div className="flex justify-center">
                  <Button
                    variant="ghost"
                    className="w-full text-sm font-semibold text-primary"
                    onClick={(e) => {
                      e.stopPropagation();
                      setVisibleCount(leafItems.length);
                    }}
                  >
                    Show all {leafItems.length} unique items (
                    {leafItems.length - visibleCount} more)
                  </Button>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

