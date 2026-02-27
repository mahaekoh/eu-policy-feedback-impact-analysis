"use client";

import { useCallback, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { InitiativeCard } from "@/components/initiative-card";
import { InitiativeSummary, SortOption } from "@/lib/types";

const ITEMS_PER_PAGE = 50;

interface InitiativeListProps {
  initiatives: InitiativeSummary[];
}

export function InitiativeList({ initiatives }: InitiativeListProps) {
  const router = useRouter();
  const searchParams = useSearchParams();

  // Read state from URL search params
  const search = searchParams.get("q") ?? "";
  const sort = (searchParams.get("sort") ?? "recently_discussed") as SortOption;
  const stageFilter = searchParams.get("stage") ?? "all";
  const deptFilter = searchParams.get("dept") ?? "all";
  const topicFilter = searchParams.get("topic") ?? "all";
  const page = Math.max(1, parseInt(searchParams.get("page") ?? "1", 10) || 1);
  const [showAdvanced, setShowAdvanced] = useState(
    stageFilter !== "all" || deptFilter !== "all" || topicFilter !== "all"
  );

  // Update URL search params (replaces current history entry for filter changes,
  // pushes for page changes so back button works across pages)
  const setParams = useCallback(
    (updates: Record<string, string>, push = false) => {
      const params = new URLSearchParams(searchParams.toString());
      for (const [k, v] of Object.entries(updates)) {
        if (v === "" || v === "all" || (k === "page" && v === "1") || (k === "sort" && v === "recently_discussed")) {
          params.delete(k);
        } else {
          params.set(k, v);
        }
      }
      const qs = params.toString();
      const url = qs ? `?${qs}` : "/";
      if (push) {
        router.push(url, { scroll: false });
      } else {
        router.replace(url, { scroll: false });
      }
    },
    [router, searchParams]
  );

  const setPage = useCallback(
    (p: number) => setParams({ page: String(p) }, true),
    [setParams]
  );

  const stages = useMemo(() => {
    const set = new Set(initiatives.map((i) => i.stage).filter(Boolean));
    return Array.from(set).sort();
  }, [initiatives]);

  const departments = useMemo(() => {
    const set = new Set(initiatives.map((i) => i.department).filter(Boolean));
    return Array.from(set).sort();
  }, [initiatives]);

  const topics = useMemo(() => {
    const set = new Set(initiatives.flatMap((i) => i.topics));
    return Array.from(set).sort();
  }, [initiatives]);

  const filtered = useMemo(() => {
    let result = initiatives;

    if (search) {
      const q = search.toLowerCase();
      result = result.filter(
        (i) =>
          i.short_title.toLowerCase().includes(q) ||
          i.reference.toLowerCase().includes(q) ||
          i.department.toLowerCase().includes(q) ||
          String(i.id).includes(q)
      );
    }

    if (stageFilter !== "all") {
      result = result.filter((i) => i.stage === stageFilter);
    }
    if (deptFilter !== "all") {
      result = result.filter((i) => i.department === deptFilter);
    }
    if (topicFilter !== "all") {
      result = result.filter((i) => i.topics.includes(topicFilter));
    }

    switch (sort) {
      case "most_discussed":
        result = [...result].sort(
          (a, b) => b.total_feedback - a.total_feedback
        );
        break;
      case "recently_discussed":
        result = [...result].sort((a, b) => {
          const aDate = a.last_feedback_date ? new Date(a.last_feedback_date).getTime() : 0;
          const bDate = b.last_feedback_date ? new Date(b.last_feedback_date).getTime() : 0;
          return bDate - aDate;
        });
        break;
      case "newest":
        result = [...result].sort(
          (a, b) =>
            new Date(b.published_date.replace(/\//g, "-")).getTime() -
            new Date(a.published_date.replace(/\//g, "-")).getTime()
        );
        break;
    }

    return result;
  }, [initiatives, search, sort, stageFilter, deptFilter, topicFilter]);

  const totalPages = Math.ceil(filtered.length / ITEMS_PER_PAGE);
  const safePage = Math.min(page, totalPages || 1);
  const paginated = filtered.slice(
    (safePage - 1) * ITEMS_PER_PAGE,
    safePage * ITEMS_PER_PAGE
  );

  return (
    <div>
      {/* Controls bar */}
      <div className="mb-6 space-y-3">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:flex-wrap">
          <Select
            value={sort}
            onValueChange={(v) => setParams({ sort: v, page: "1" })}
          >
            <SelectTrigger className="w-full sm:w-[200px]">
              <SelectValue placeholder="Sort by" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="recently_discussed">Recently discussed</SelectItem>
              <SelectItem value="newest">Newest</SelectItem>
              <SelectItem value="most_discussed">Most discussed</SelectItem>
            </SelectContent>
          </Select>

          <Input
            placeholder="Search initiatives..."
            value={search}
            onChange={(e) => setParams({ q: e.target.value, page: "1" })}
            className="w-full sm:w-[280px]"
          />

          <button
            className="text-sm text-primary hover:underline cursor-pointer"
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            {showAdvanced ? "Hide filters" : "Advanced search"}
          </button>
        </div>

        {showAdvanced && (
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:flex-wrap">
            <Select
              value={stageFilter}
              onValueChange={(v) => setParams({ stage: v, page: "1" })}
            >
              <SelectTrigger className="w-full sm:w-[200px]">
                <SelectValue placeholder="Stage" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All stages</SelectItem>
                {stages.map((s) => (
                  <SelectItem key={s} value={s}>
                    {s.replace(/_/g, " ")}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Select
              value={deptFilter}
              onValueChange={(v) => setParams({ dept: v, page: "1" })}
            >
              <SelectTrigger className="w-full sm:w-[160px]">
                <SelectValue placeholder="Department" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All departments</SelectItem>
                {departments.map((d) => (
                  <SelectItem key={d} value={d}>
                    {d}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Select
              value={topicFilter}
              onValueChange={(v) => setParams({ topic: v, page: "1" })}
            >
              <SelectTrigger className="w-full sm:w-[200px]">
                <SelectValue placeholder="Topic" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All topics</SelectItem>
                {topics.map((t) => (
                  <SelectItem key={t} value={t}>
                    {t}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}
      </div>

      {/* Results count */}
      <p className="mb-4 text-sm text-muted-foreground">
        Showing {paginated.length} of {filtered.length.toLocaleString()}{" "}
        initiatives
      </p>

      {/* Card grid */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
        {paginated.map((initiative) => (
          <InitiativeCard key={initiative.id} initiative={initiative} />
        ))}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="mt-8 flex items-center justify-center gap-2">
          <Button
            variant="outline"
            size="sm"
            disabled={safePage === 1}
            onClick={() => setPage(safePage - 1)}
          >
            Previous
          </Button>
          <span className="px-4 text-sm text-muted-foreground">
            Page {safePage} of {totalPages}
          </span>
          <Button
            variant="outline"
            size="sm"
            disabled={safePage === totalPages}
            onClick={() => setPage(safePage + 1)}
          >
            Next
          </Button>
        </div>
      )}
    </div>
  );
}
