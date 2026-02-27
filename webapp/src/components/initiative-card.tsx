"use client";

import { useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { InitiativeSummary } from "@/lib/types";
import { formatDate } from "@/lib/utils";
import { CountryBar, UserTypeBar } from "@/components/cluster-stats-bar";

const STAGE_LABELS: Record<string, string> = {
  PLANNING_WORKFLOW: "Planning",
  ISC_WORKFLOW: "Inter-service consultation",
  ADOPTION_WORKFLOW: "Adoption",
};

function formatStage(stage: string): string {
  return STAGE_LABELS[stage] || stage.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

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

interface InitiativeCardProps {
  initiative: InitiativeSummary;
}

export function InitiativeCard({ initiative }: InitiativeCardProps) {
  const [showAllTopics, setShowAllTopics] = useState(false);
  const sortedCountries = Object.entries(initiative.country_counts)
    .sort((a, b) => b[1] - a[1]) as [string, number][];
  const sortedTypes = Object.entries(initiative.user_type_counts)
    .sort((a, b) => b[1] - a[1]) as [string, number][];
  const total = initiative.total_feedback;
  const timeline = initiative.feedback_timeline;
  const topics = initiative.topics;
  const stage = formatStage(initiative.stage);

  return (
    <Link href={`/initiative/${initiative.id}`}>
      <Card className="relative h-full transition-colors hover:bg-accent/50">
        {initiative.has_open_feedback && (
          <Badge className="absolute top-2 right-2 bg-green-600 text-white text-[10px] px-1.5 py-0 opacity-50">
            Open
          </Badge>
        )}
        <CardHeader className="pb-3">
          <CardTitle className="line-clamp-2 text-base leading-snug">
            {initiative.short_title}
          </CardTitle>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            {timeline.length > 0 && <TimelineSparkline counts={timeline} />}
            <span>
              {topics.length === 0 && stage}
              {topics.length === 1 && `${topics[0]} (${stage})`}
              {topics.length > 1 && !showAllTopics && (
                <>
                  {topics[0]}{" "}
                  <span
                    role="button"
                    className="text-primary hover:underline"
                    onClick={(e) => { e.preventDefault(); e.stopPropagation(); setShowAllTopics(true); }}
                  >
                    +{topics.length - 1} more
                  </span>
                  {` (${stage})`}
                </>
              )}
              {topics.length > 1 && showAllTopics && (
                <>
                  {topics.join(", ")}{" "}
                  <span
                    role="button"
                    className="text-primary hover:underline"
                    onClick={(e) => { e.preventDefault(); e.stopPropagation(); setShowAllTopics(false); }}
                  >
                    less
                  </span>
                  {` (${stage})`}
                </>
              )}
            </span>
          </div>
        </CardHeader>
        <CardContent className="pt-0">
          {total > 0 && (sortedCountries.length > 0 || sortedTypes.length > 0) && (
            <div className="flex flex-col gap-1 mb-3">
              {sortedCountries.length > 0 && (
                <CountryBar sortedCountries={sortedCountries} total={total} />
              )}
              {sortedTypes.length > 0 && (
                <UserTypeBar sortedTypes={sortedTypes} total={total} />
              )}
            </div>
          )}
          <div className="flex items-center justify-between text-sm text-muted-foreground">
            <span>
              {initiative.total_feedback.toLocaleString()} feedback
              {initiative.total_feedback !== 1 ? "s" : ""}
            </span>
            <span>{formatDate(initiative.published_date)}</span>
          </div>
        </CardContent>
      </Card>
    </Link>
  );
}
