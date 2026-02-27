import { Suspense } from "react";
import { getInitiativeIndex } from "@/lib/data";
import { InitiativeList } from "@/components/initiative-list";

export const dynamic = "force-dynamic";

export default async function HomePage() {
  const initiatives = await getInitiativeIndex();

  return (
    <main className="min-h-screen bg-background">
      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <div className="mb-8">
          <div className="flex items-center gap-3">
            <h1 className="text-3xl font-bold tracking-tight">
              EU Have Your Say — Initiatives
            </h1>
            <span className="inline-flex items-center rounded-full bg-primary px-3 py-1 text-sm font-medium text-primary-foreground">
              {initiatives.length.toLocaleString()}
            </span>
          </div>
          <p className="mt-2 text-muted-foreground">
            Browse and explore EU Better Regulation initiatives and public feedback
          </p>
        </div>
        <Suspense>
          <InitiativeList initiatives={initiatives} />
        </Suspense>
      </div>
    </main>
  );
}
