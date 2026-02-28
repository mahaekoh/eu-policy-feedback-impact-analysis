"use client";

import { useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Publication } from "@/lib/types";
import { formatDate } from "@/lib/utils";
import { DocumentCard } from "@/components/document-card";
import { FeedbackList } from "@/components/feedback-list";
import { CountryBar, UserTypeBar } from "@/components/cluster-stats-bar";

const TIMELINE_BUCKETS = 20;

function TimelineSparkline({ counts }: { counts: number[] }) {
  if (counts.length === 0) return null;
  const max = Math.max(...counts, 1);
  const w = 60;
  const h = 14;
  const pad = 1;
  const stepX = w / (counts.length - 1 || 1);
  const points = counts.map((c, i) => {
    const x = i * stepX;
    const y = pad + (1 - c / max) * (h - 2 * pad);
    return `${x},${y}`;
  });
  const areaPath = `M0,${h} ${points.map((p) => `L${p}`).join(" ")} L${w},${h} Z`;
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} className="shrink-0">
      <path d={areaPath} fill="#6366f1" opacity={0.12} />
      <polyline
        points={points.join(" ")}
        fill="none"
        stroke="#6366f1"
        strokeWidth={1.2}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

interface PublicationSectionProps {
  publication: Publication;
  defaultOpen?: boolean;
  showBars?: boolean;
  timelineStartMs?: number;
  timelineEndMs?: number;
  initiativeUrl?: string;
}

export function PublicationSection({
  publication: pub,
  defaultOpen = false,
  showBars = false,
  timelineStartMs = 0,
  timelineEndMs = 0,
  initiativeUrl = "",
}: PublicationSectionProps) {
  const hasDocs = pub.documents.length > 0;
  const hasFeedback = pub.feedback.length > 0;
  const isStub = !hasDocs && !hasFeedback && pub.total_feedback > 0;
  const [open, setOpen] = useState(isStub ? false : defaultOpen);

  const sortedCountries = useMemo(() => {
    if (!showBars || !hasFeedback) return [];
    const counts: Record<string, number> = {};
    for (const fb of pub.feedback) {
      if (fb.country) counts[fb.country] = (counts[fb.country] || 0) + 1;
    }
    return Object.entries(counts).sort((a, b) => b[1] - a[1]) as [string, number][];
  }, [pub.feedback, showBars, hasFeedback]);

  const sortedTypes = useMemo(() => {
    if (!showBars || !hasFeedback) return [];
    const counts: Record<string, number> = {};
    for (const fb of pub.feedback) {
      if (fb.user_type) counts[fb.user_type] = (counts[fb.user_type] || 0) + 1;
    }
    return Object.entries(counts).sort((a, b) => b[1] - a[1]) as [string, number][];
  }, [pub.feedback, showBars, hasFeedback]);

  const barTotal = useMemo(
    () => sortedTypes.reduce((s, e) => s + e[1], 0),
    [sortedTypes]
  );

  const sparkline = useMemo(() => {
    if (!showBars || !hasFeedback || timelineEndMs <= timelineStartMs) return [];
    const bucketWidth = (timelineEndMs - timelineStartMs) / TIMELINE_BUCKETS;
    const buckets = new Array(TIMELINE_BUCKETS).fill(0);
    for (const fb of pub.feedback) {
      if (fb.date) {
        const t = new Date(fb.date.replace(/\//g, "-")).getTime();
        if (!isNaN(t)) {
          const idx = Math.min(Math.floor((t - timelineStartMs) / bucketWidth), TIMELINE_BUCKETS - 1);
          if (idx >= 0) buckets[idx]++;
        }
      }
    }
    return buckets;
  }, [pub.feedback, showBars, hasFeedback, timelineStartMs, timelineEndMs]);
  const [activeTab, setActiveTab] = useState<"documents" | "feedback">(
    hasDocs ? "documents" : "feedback"
  );

  const header = (
    <div className="min-w-0 flex-1">
      <div className="flex items-center gap-2 mb-1">
        <span className="font-semibold text-sm">
          {pub.section_label || pub.type}
        </span>
        <Badge
          variant={
            pub.feedback_status === "OPEN" ? "default" : "secondary"
          }
          className="text-xs"
        >
          {pub.feedback_status}
        </Badge>
      </div>
      {pub.reference && (
        <p className="text-xs text-muted-foreground">{pub.reference}</p>
      )}
      <div className="mt-1 flex flex-wrap gap-3 text-xs text-muted-foreground">
        <span>Published: {formatDate(pub.published_date)}</span>
        {pub.feedback_end_date && (
          <span>
            Feedback until: {formatDate(pub.feedback_end_date)}
          </span>
        )}
        {!isStub && <span>{pub.documents.length} documents</span>}
        <span>{pub.total_feedback} feedback</span>
      </div>
      {sparkline.length > 0 && (
        <div className="mt-1">
          <TimelineSparkline counts={sparkline} />
        </div>
      )}
      {barTotal > 0 && (
        <div className="mt-1.5 flex flex-col gap-1 max-w-sm">
          {sortedCountries.length > 0 && (
            <CountryBar sortedCountries={sortedCountries} total={barTotal} />
          )}
          {sortedTypes.length > 0 && (
            <UserTypeBar sortedTypes={sortedTypes} total={barTotal} />
          )}
        </div>
      )}
    </div>
  );

  if (isStub) {
    const isOpc = pub.type === "OPC_LAUNCHED";
    const stubUrl = isOpc && initiativeUrl
      ? `${initiativeUrl.replace(/_en$/, "")}/public-consultation`
      : initiativeUrl;
    return (
      <div className="rounded-lg border">
        <a
          href={stubUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="block p-4 hover:bg-accent/50 transition-colors"
        >
          <div className="flex items-start justify-between gap-3">
            {header}
            <span className="text-sm text-primary shrink-0">&rarr;</span>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">
            {isOpc
              ? "Feedback collected via public consultation — view on EU portal"
              : "Feedback not available for download — view on EU portal"}
          </p>
        </a>
      </div>
    );
  }

  return (
    <div className="rounded-lg border">
      {/* Publication header */}
      <button
        className="w-full text-left p-4 hover:bg-accent/50 transition-colors cursor-pointer"
        onClick={() => setOpen(!open)}
      >
        <div className="flex items-start justify-between gap-3">
          {header}
          <span className="text-muted-foreground shrink-0">
            {open ? "\u25B2" : "\u25BC"}
          </span>
        </div>
      </button>

      {/* Publication content */}
      {open && (
        <div className="border-t p-4">
          {/* Tabs */}
          <div className="mb-4 flex gap-1">
            {hasDocs && (
              <Button
                variant={activeTab === "documents" ? "default" : "ghost"}
                size="sm"
                onClick={() => setActiveTab("documents")}
              >
                Documents ({pub.documents.length})
              </Button>
            )}
            {hasFeedback && (
              <Button
                variant={activeTab === "feedback" ? "default" : "ghost"}
                size="sm"
                onClick={() => setActiveTab("feedback")}
              >
                Feedback ({pub.total_feedback})
              </Button>
            )}
          </div>

          {activeTab === "documents" && (
            <div className="space-y-3">
              {pub.documents.length === 0 ? (
                <p className="text-sm text-muted-foreground">No documents</p>
              ) : (
                pub.documents.map((doc, i) => (
                  <DocumentCard key={i} document={doc} />
                ))
              )}
            </div>
          )}

          {activeTab === "feedback" && (
            <>
              {pub.feedback.length === 0 ? (
                <p className="text-sm text-muted-foreground">No feedback</p>
              ) : (
                <FeedbackList feedback={pub.feedback} />
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
