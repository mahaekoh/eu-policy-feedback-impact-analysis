import Link from "next/link";
import { notFound } from "next/navigation";
import {
  getInitiativeDetail,
  getClusteringSchemesForInitiative,
  getClusterData,
} from "@/lib/data";
import { InitiativeDetail } from "@/components/initiative-detail";

export const dynamic = "force-dynamic";

interface PageProps {
  params: Promise<{ id: string }>;
}

export default async function InitiativePage({ params }: PageProps) {
  const { id } = await params;
  const initiative = await getInitiativeDetail(id);

  if (!initiative) {
    notFound();
  }

  // Compute feedback timeline sparkline
  const TIMELINE_BUCKETS = 20;
  const startMs = new Date(initiative.published_date.replace(/\//g, "-")).getTime();
  const feedbackDates: number[] = [];
  for (const pub of initiative.publications) {
    for (const fb of pub.feedback || []) {
      if (fb.date) {
        const t = new Date(fb.date.replace(/\//g, "-")).getTime();
        if (!isNaN(t)) feedbackDates.push(t);
      }
    }
  }
  const countryCounts: Record<string, number> = {};
  const userTypeCounts: Record<string, number> = {};
  for (const pub of initiative.publications) {
    for (const fb of pub.feedback || []) {
      if (fb.country) {
        countryCounts[fb.country] = (countryCounts[fb.country] || 0) + 1;
      }
      if (fb.user_type) {
        userTypeCounts[fb.user_type] = (userTypeCounts[fb.user_type] || 0) + 1;
      }
    }
  }

  let feedbackTimeline: number[] = [];
  let timelineStartMs = startMs;
  let timelineEndMs = startMs;
  if (feedbackDates.length > 0) {
    for (const t of feedbackDates) {
      if (t > timelineEndMs) timelineEndMs = t;
    }
    if (timelineEndMs > startMs) {
      const bucketWidth = (timelineEndMs - startMs) / TIMELINE_BUCKETS;
      feedbackTimeline = new Array(TIMELINE_BUCKETS).fill(0);
      for (const t of feedbackDates) {
        const idx = Math.min(Math.floor((t - startMs) / bucketWidth), TIMELINE_BUCKETS - 1);
        if (idx >= 0) feedbackTimeline[idx]++;
      }
    }
  }

  const clusterSchemes = await getClusteringSchemesForInitiative(id);

  // Pre-load the first scheme's cluster data if available
  const initialClusterData =
    clusterSchemes.length > 0
      ? await getClusterData(id, clusterSchemes[0])
      : null;

  return (
    <main className="min-h-screen bg-background">
      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <Link
          href="/"
          className="mb-6 inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          &larr; Back to initiatives
        </Link>
        <InitiativeDetail
          initiative={initiative}
          clusterSchemes={clusterSchemes}
          initialClusterData={initialClusterData}
          initialScheme={clusterSchemes[0] ?? null}
          feedbackTimeline={feedbackTimeline}
          timelineStartMs={timelineStartMs}
          timelineEndMs={timelineEndMs}
          countryCounts={countryCounts}
          userTypeCounts={userTypeCounts}
        />
      </div>
    </main>
  );
}
