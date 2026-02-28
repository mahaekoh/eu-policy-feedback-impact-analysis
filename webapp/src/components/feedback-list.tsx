"use client";

import { useMemo, useState } from "react";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { FeedbackCard } from "@/components/feedback-card";
import { Feedback, getUserTypeColor, formatUserType } from "@/lib/types";

const CHUNK_SIZE = 50;

interface FeedbackListProps {
  feedback: Feedback[];
}

export function FeedbackList({ feedback }: FeedbackListProps) {
  const [search, setSearch] = useState("");
  const [activeTypes, setActiveTypes] = useState<Set<string>>(new Set());
  const [hideEmpty, setHideEmpty] = useState(true);
  const [visibleCount, setVisibleCount] = useState(CHUNK_SIZE);

  const userTypes = useMemo(() => {
    const counts = new Map<string, number>();
    for (const fb of feedback) {
      counts.set(fb.user_type, (counts.get(fb.user_type) || 0) + 1);
    }
    return Array.from(counts.entries()).sort((a, b) => b[1] - a[1]);
  }, [feedback]);

  const emptyCount = useMemo(
    () =>
      feedback.filter(
        (f) =>
          (!f.feedback_text || f.feedback_text.trim().length === 0) &&
          (!f.attachments || f.attachments.length === 0)
      ).length,
    [feedback]
  );

  const filtered = useMemo(() => {
    let result = feedback;

    if (activeTypes.size > 0) {
      result = result.filter((f) => activeTypes.has(f.user_type));
    }

    if (hideEmpty) {
      result = result.filter(
        (f) =>
          (f.feedback_text && f.feedback_text.trim().length > 0) ||
          (f.attachments && f.attachments.length > 0)
      );
    }

    if (search) {
      const q = search.toLowerCase();
      result = result.filter(
        (f) =>
          f.feedback_text?.toLowerCase().includes(q) ||
          f.organization?.toLowerCase().includes(q) ||
          f.first_name?.toLowerCase().includes(q) ||
          f.surname?.toLowerCase().includes(q) ||
          f.country?.toLowerCase().includes(q)
      );
    }

    return result;
  }, [feedback, activeTypes, hideEmpty, search]);

  const visible = filtered.slice(0, visibleCount);

  const toggleType = (type: string) => {
    setActiveTypes((prev) => {
      const next = new Set(prev);
      if (next.has(type)) {
        next.delete(type);
      } else {
        next.add(type);
      }
      return next;
    });
    setVisibleCount(CHUNK_SIZE);
  };

  return (
    <div>
      {/* Filter controls */}
      <div className="mb-4 space-y-3">
        <div className="flex flex-wrap gap-1.5">
          {userTypes.map(([type, count]) => {
            const colors = getUserTypeColor(type);
            const isActive = activeTypes.has(type);
            return (
              <button
                key={type}
                onClick={() => toggleType(type)}
                className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium transition-colors cursor-pointer ${
                  isActive
                    ? `${colors.bg} ${colors.text} ${colors.border}`
                    : "bg-background text-muted-foreground hover:bg-accent"
                }`}
              >
                {formatUserType(type)}
                <span className="opacity-60">({count})</span>
              </button>
            );
          })}
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <Input
            placeholder="Search feedback..."
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setVisibleCount(CHUNK_SIZE);
            }}
            className="w-full sm:w-[280px]"
          />
          {emptyCount > 0 && (
            <Button
              variant={hideEmpty ? "default" : "outline"}
              size="sm"
              onClick={() => {
                setHideEmpty(!hideEmpty);
                setVisibleCount(CHUNK_SIZE);
              }}
            >
              {hideEmpty ? `Show ${emptyCount} empty` : "Hide empty"}
            </Button>
          )}
          <span className="text-sm text-muted-foreground">
            {filtered.length} of {feedback.length} feedback
            {feedback.length !== 1 ? "s" : ""}
          </span>
        </div>
      </div>

      {/* Feedback cards */}
      <div className="space-y-3">
        {visible.map((fb) => (
          <FeedbackCard key={fb.id} feedback={fb} />
        ))}
      </div>

      {/* Load more */}
      {visibleCount < filtered.length && (
        <div className="mt-4 flex justify-center">
          <Button
            variant="outline"
            onClick={() =>
              setVisibleCount((prev) =>
                Math.min(prev + CHUNK_SIZE, filtered.length)
              )
            }
          >
            Load more ({filtered.length - visibleCount} remaining)
          </Button>
        </div>
      )}
    </div>
  );
}
