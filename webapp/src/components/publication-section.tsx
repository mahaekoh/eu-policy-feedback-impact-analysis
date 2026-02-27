"use client";

import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Publication } from "@/lib/types";
import { formatDate } from "@/lib/utils";
import { DocumentCard } from "@/components/document-card";
import { FeedbackList } from "@/components/feedback-list";

interface PublicationSectionProps {
  publication: Publication;
  defaultOpen?: boolean;
}

export function PublicationSection({
  publication: pub,
  defaultOpen = false,
}: PublicationSectionProps) {
  const [open, setOpen] = useState(defaultOpen);
  const [activeTab, setActiveTab] = useState<"documents" | "feedback">(
    "documents"
  );

  return (
    <div className="rounded-lg border">
      {/* Publication header */}
      <button
        className="w-full text-left p-4 hover:bg-accent/50 transition-colors cursor-pointer"
        onClick={() => setOpen(!open)}
      >
        <div className="flex items-start justify-between gap-3">
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
              <span>{pub.documents.length} documents</span>
              <span>{pub.total_feedback} feedback</span>
            </div>
          </div>
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
            <Button
              variant={activeTab === "documents" ? "default" : "ghost"}
              size="sm"
              onClick={() => setActiveTab("documents")}
            >
              Documents ({pub.documents.length})
            </Button>
            <Button
              variant={activeTab === "feedback" ? "default" : "ghost"}
              size="sm"
              onClick={() => setActiveTab("feedback")}
            >
              Feedback ({pub.total_feedback})
            </Button>
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
