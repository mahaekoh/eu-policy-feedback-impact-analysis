"use client";

import { useCallback, useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ClusterData, Feedback, Initiative } from "@/lib/types";
import { formatDate } from "@/lib/utils";
import { ExpandableText } from "@/components/expandable-text";
import { PublicationSection } from "@/components/publication-section";
import { ClusterView } from "@/components/cluster-view";
import { CountryBar, UserTypeBar } from "@/components/cluster-stats-bar";

type ViewMode = "publications" | "clusters";

function TimelineSparkline({ counts }: { counts: number[] }) {
  if (counts.length === 0) return null;
  const max = Math.max(...counts, 1);
  const w = 100;
  const h = 14;
  const pad = 1;
  const stepX = w / (counts.length - 1 || 1);
  const points = counts.map((c, i) => {
    const x = i * stepX;
    const y = pad + (1 - c / max) * (h - 2 * pad);
    return `${x},${y}`;
  });
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} className="shrink-0">
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

interface InitiativeDetailProps {
  initiative: Initiative;
  clusterSchemes?: string[];
  initialClusterData?: ClusterData | null;
  initialScheme?: string | null;
  feedbackTimeline?: number[];
  timelineStartMs?: number;
  timelineEndMs?: number;
  countryCounts?: Record<string, number>;
  userTypeCounts?: Record<string, number>;
}

export function InitiativeDetail({
  initiative,
  clusterSchemes = [],
  initialClusterData = null,
  initialScheme = null,
  feedbackTimeline = [],
  timelineStartMs = 0,
  timelineEndMs = 0,
  countryCounts = {},
  userTypeCounts = {},
}: InitiativeDetailProps) {
  const hasClusters = clusterSchemes.length > 0;
  const [viewMode, setViewMode] = useState<ViewMode>(
    hasClusters ? "clusters" : "publications"
  );
  const [currentScheme, setCurrentScheme] = useState<string>(
    initialScheme ?? ""
  );
  const [clusterData, setClusterData] = useState<ClusterData | null>(
    initialClusterData
  );
  const [loadingScheme, setLoadingScheme] = useState(false);

  const [activePublications, emptyPublications] = useMemo(() => {
    const active: typeof initiative.publications = [];
    const empty: typeof initiative.publications = [];
    for (const p of initiative.publications) {
      if (
        p.feedback_status !== "OPEN" &&
        p.documents.length === 0 &&
        p.feedback.length === 0 &&
        p.total_feedback === 0
      ) {
        empty.push(p);
      } else {
        active.push(p);
      }
    }
    return [active, empty];
  }, [initiative.publications]);

  const multiplePubsWithFeedback = useMemo(
    () => activePublications.filter((p) => p.feedback.length > 0).length > 1,
    [activePublications]
  );

  const totalFeedback = initiative.publications.reduce(
    (sum, p) => sum + p.total_feedback,
    0
  );
  const fetchedFeedback = initiative.publications.reduce(
    (sum, p) => sum + p.feedback.length,
    0
  );

  const sortedCountries = useMemo(
    () =>
      Object.entries(countryCounts).sort((a, b) => b[1] - a[1]) as [string, number][],
    [countryCounts]
  );
  const sortedTypes = useMemo(
    () =>
      Object.entries(userTypeCounts).sort((a, b) => b[1] - a[1]) as [string, number][],
    [userTypeCounts]
  );
  const scrapedFeedbackCount = useMemo(
    () => sortedTypes.reduce((s, e) => s + e[1], 0),
    [sortedTypes]
  );

  // Collect all feedback items for cluster view
  const allFeedback = useMemo(() => {
    // Prefer middle_feedback (from unit_summaries) which has combined_feedback_summary
    if (initiative.middle_feedback && initiative.middle_feedback.length > 0) {
      return initiative.middle_feedback;
    }
    // Fallback to publications -> feedback
    const items: Feedback[] = [];
    for (const pub of initiative.publications) {
      for (const fb of pub.feedback || []) {
        items.push(fb);
      }
    }
    return items;
  }, [initiative]);

  const handleSchemeChange = useCallback(
    async (scheme: string) => {
      setCurrentScheme(scheme);
      setLoadingScheme(true);
      try {
        const res = await fetch(
          `/api/clusters/${initiative.id}?scheme=${encodeURIComponent(scheme)}`
        );
        if (res.ok) {
          const data = await res.json();
          setClusterData(data);
        }
      } finally {
        setLoadingScheme(false);
      }
    },
    [initiative.id]
  );

  const [showMeta, setShowMeta] = useState(false);

  return (
    <div>
      {/* Header / metadata panel */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold tracking-tight mb-1">
          {initiative.short_title}
        </h1>

        <div className="text-sm text-muted-foreground mb-1">
          Published {formatDate(initiative.published_date)}
          {initiative.last_cached_at && ` · Updated ${formatDate(initiative.last_cached_at)}`}
        </div>

        {feedbackTimeline.length > 0 && (
          <div className="mb-1">
            <TimelineSparkline counts={feedbackTimeline} />
          </div>
        )}

        {scrapedFeedbackCount > 0 && (sortedCountries.length > 0 || sortedTypes.length > 0) && (
          <div className="flex flex-col gap-1 max-w-md mb-2">
            {sortedCountries.length > 0 && (
              <CountryBar sortedCountries={sortedCountries} total={scrapedFeedbackCount} />
            )}
            {sortedTypes.length > 0 && (
              <UserTypeBar sortedTypes={sortedTypes} total={scrapedFeedbackCount} />
            )}
          </div>
        )}

        {initiative.url && (
          <a
            href={initiative.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-primary hover:underline"
          >
            View on EU portal &rarr;
          </a>
        )}

        <div className="mb-3">
          <button
            className="text-sm text-primary hover:underline cursor-pointer"
            onClick={() => setShowMeta(!showMeta)}
          >
            {showMeta ? "Hide details" : "Show details"}
          </button>

          {showMeta && (
            <div className="mt-3 rounded-lg border p-4 space-y-4">
              <div className="flex flex-wrap gap-2">
                <Badge variant="secondary">{initiative.department}</Badge>
                <Badge variant="outline">
                  {initiative.stage.replace(/_/g, " ")}
                </Badge>
                <Badge variant="outline">{initiative.status}</Badge>
                {initiative.type_of_act && (
                  <Badge variant="outline">{initiative.type_of_act}</Badge>
                )}
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-x-8 gap-y-2 text-sm">
                <div>
                  <span className="text-muted-foreground">ID: </span>
                  <span className="font-medium">{initiative.id}</span>
                </div>
                {initiative.reference && (
                  <div>
                    <span className="text-muted-foreground">Reference: </span>
                    <span className="font-medium">{initiative.reference}</span>
                  </div>
                )}
                <div>
                  <span className="text-muted-foreground">Total feedback: </span>
                  <span className="font-medium">
                    {totalFeedback.toLocaleString()}
                  </span>
                </div>
                <div>
                  <span className="text-muted-foreground">Publications: </span>
                  <span className="font-medium">
                    {initiative.publications.length}
                  </span>
                </div>
              </div>

              {fetchedFeedback !== totalFeedback && (
                <p className="text-xs text-muted-foreground">
                  The API reports {totalFeedback.toLocaleString()} feedback
                  {" "}but only {fetchedFeedback.toLocaleString()} were available
                  {" "}for download.
                  {totalFeedback - fetchedFeedback > 1000 &&
                    " This is common for large public consultations where responses are collected via a separate survey tool."}
                </p>
              )}

              {initiative.topics.length > 0 && (
                <div>
                  <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide mr-2">
                    Topics
                  </span>
                  <div className="inline-flex flex-wrap gap-1">
                    {initiative.topics.map((t) => (
                      <Badge
                        key={t}
                        variant="outline"
                        className="text-xs font-normal"
                      >
                        {t}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {initiative.policy_areas.length > 0 && (
                <div>
                  <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide mr-2">
                    Policy areas
                  </span>
                  <div className="inline-flex flex-wrap gap-1">
                    {initiative.policy_areas.map((p) => (
                      <Badge
                        key={p}
                        variant="outline"
                        className="text-xs font-normal"
                      >
                        {p}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Summaries */}
        {initiative.summary && (
          <div className="mt-4">
            <ExpandableText
              text={initiative.summary}
              label="Summary"
              isMarkdown
              maxPreviewChars={600}
            />
          </div>
        )}

        {initiative.change_summary && (
          <div className="mt-4">
            <ExpandableText
              text={initiative.change_summary}
              label="Changes after feedback"
              isMarkdown
              maxPreviewChars={600}
            />
          </div>
        )}

        {initiative.before_feedback_summary && (
          <div className="mt-4">
            <ExpandableText
              text={initiative.before_feedback_summary}
              label="Before feedback summary"
              isMarkdown
              maxPreviewChars={400}
            />
          </div>
        )}

        {initiative.after_feedback_summary && (
          <div className="mt-4">
            <ExpandableText
              text={initiative.after_feedback_summary}
              label="After feedback summary"
              isMarkdown
              maxPreviewChars={400}
            />
          </div>
        )}
      </div>

      {/* View mode tabs */}
      <div className="flex items-center gap-1 mb-4 border-b pb-2">
        <Button
          variant={viewMode === "publications" ? "default" : "ghost"}
          size="sm"
          onClick={() => setViewMode("publications")}
        >
          Publications ({activePublications.length})
        </Button>
        {hasClusters && (
          <Button
            variant={viewMode === "clusters" ? "default" : "ghost"}
            size="sm"
            onClick={() => setViewMode("clusters")}
          >
            Clustered Feedback
          </Button>
        )}
      </div>

      {/* Publications view */}
      {viewMode === "publications" && (
        <div className="space-y-4">
          {activePublications.map((pub, i) => (
            <PublicationSection
              key={pub.publication_id}
              publication={pub}
              defaultOpen={i === 0}
              showBars={multiplePubsWithFeedback}
              timelineStartMs={timelineStartMs}
              timelineEndMs={timelineEndMs}
              initiativeUrl={initiative.url}
            />
          ))}
          {emptyPublications.length > 0 && (
            <div className="rounded-lg border border-dashed p-4">
              <p className="text-xs font-medium text-muted-foreground mb-2">
                {emptyPublications.length} empty{" "}
                {emptyPublications.length === 1
                  ? "publication"
                  : "publications"}
              </p>
              <div className="flex flex-wrap gap-x-4 gap-y-1">
                {emptyPublications.map((pub) => (
                  <a
                    key={pub.publication_id}
                    href={initiative.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-muted-foreground hover:text-primary transition-colors"
                  >
                    {pub.section_label || pub.type.replace(/_/g, " ")}{" "}
                    <span className="opacity-60">
                      ({pub.feedback_status.toLowerCase()})
                    </span>{" "}
                    &rarr;
                  </a>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Cluster view */}
      {viewMode === "clusters" && (
        <>
          {loadingScheme && (
            <p className="text-sm text-muted-foreground">
              Loading clustering data...
            </p>
          )}
          {!loadingScheme && clusterData && (
            <ClusterView
              clusterData={clusterData}
              allFeedback={allFeedback}
              schemes={clusterSchemes}
              currentScheme={currentScheme}
              onSchemeChange={handleSchemeChange}
              publishedDate={initiative.published_date}
            />
          )}
          {!loadingScheme && !clusterData && (
            <p className="text-sm text-muted-foreground">
              No clustering data available.
            </p>
          )}
        </>
      )}
    </div>
  );
}
