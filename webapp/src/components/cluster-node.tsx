"use client";

import { useMemo, useState } from "react";
import { ClusterNode, Feedback, computeClusterStats } from "@/lib/types";
import { FeedbackCard } from "@/components/feedback-card";
import {
  CountryBar,
  UserTypeBar,
  ClusterDetailStats,
} from "@/components/cluster-stats-bar";
import { Button } from "@/components/ui/button";
import { truncateText } from "@/lib/utils";
import { TimeRange } from "@/components/cluster-view";

const INITIAL_SHOW = 5;

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
}

export function ClusterNodeComponent({
  node,
  isSubCluster = false,
  timeRange,
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

  // Preview text from first non-empty feedback
  let preview = "";
  for (const fb of items) {
    const t = fb.combined_feedback_summary || fb.feedback_text;
    if (t) {
      preview = truncateText(t, 150);
      break;
    }
  }

  // Leaf items to render (direct items if available, otherwise all)
  const leafItems =
    node.directItems.length > 0 ? node.directItems : node.allItems;
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
                {node.directItems.length > 0 && hasChildren && (
                  <span className="text-[10px] text-muted-foreground bg-muted px-1.5 py-px rounded-lg shrink-0">
                    {node.directItems.length} unclustered
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
          <ClusterDetailStats
            sortedCountries={stats.sortedCountries}
            sortedTypes={stats.sortedTypes}
            total={total}
            organizations={organizations}
          />

          {hasChildren ? (
            <div className="p-2">
              {/* Unclustered direct items */}
              {node.directItems.length > 0 && (
                <UnclusteredSection items={node.directItems} />
              )}
              {/* Sub-clusters sorted by size */}
              {[...node.children]
                .sort((a, b) => b.allItems.length - a.allItems.length)
                .map((child) => (
                  <ClusterNodeComponent
                    key={child.label}
                    node={child}
                    isSubCluster
                    timeRange={timeRange}
                  />
                ))}
            </div>
          ) : (
            <div className="space-y-2 p-3">
              {visibleItems.map((fb) => (
                <FeedbackCard key={fb.id} feedback={fb} />
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
                    Show all {leafItems.length} items (
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

/** Noise / unclustered items within a parent cluster */
function UnclusteredSection({ items }: { items: Feedback[] }) {
  const [open, setOpen] = useState(false);
  const [visibleCount, setVisibleCount] = useState(3);
  const visible = items.slice(0, visibleCount);

  return (
    <div className="ml-5 border-l-2 border-red-300">
      <button
        className="w-full text-left px-3 py-2 hover:bg-accent/50 transition-colors cursor-pointer"
        onClick={() => setOpen(!open)}
      >
        <div className="flex items-center gap-2">
          <span
            className={`text-xs text-muted-foreground transition-transform ${
              open ? "rotate-90" : ""
            }`}
          >
            &#9654;
          </span>
          <span className="text-[11px] text-red-500 font-medium">
            Unclustered
          </span>
          <span className="text-[11px] font-bold bg-primary text-primary-foreground px-2 py-px rounded-full">
            {items.length}
          </span>
        </div>
      </button>
      {open && (
        <div className="space-y-2 p-3">
          {visible.map((fb) => (
            <FeedbackCard key={fb.id} feedback={fb} />
          ))}
          {visibleCount < items.length && (
            <Button
              variant="ghost"
              className="w-full text-sm font-semibold text-primary"
              onClick={(e) => {
                e.stopPropagation();
                setVisibleCount(items.length);
              }}
            >
              Show all {items.length} items ({items.length - visibleCount}{" "}
              more)
            </Button>
          )}
        </div>
      )}
    </div>
  );
}
