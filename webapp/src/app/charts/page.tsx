import { getCountryStats, getGlobalStats } from "@/lib/data";
import { Charts } from "@/components/charts";

export const dynamic = "force-dynamic";

export default async function ChartsPage() {
  const [stats, countryStats] = await Promise.all([
    getGlobalStats(),
    getCountryStats(),
  ]);

  return (
    <main className="min-h-screen bg-background">
      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold tracking-tight">
            Feedback Statistics
          </h1>
          <p className="mt-2 text-muted-foreground">
            Aggregate statistics across{" "}
            {stats.total_initiatives.toLocaleString()} initiatives and{" "}
            {stats.total_feedback.toLocaleString()} feedback submissions
          </p>
        </div>
        <Charts stats={stats} countryStats={countryStats} />
      </div>
    </main>
  );
}
