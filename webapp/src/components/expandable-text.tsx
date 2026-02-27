"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Button } from "@/components/ui/button";

interface ExpandableTextProps {
  text: string;
  maxPreviewChars?: number;
  isMarkdown?: boolean;
  label?: string;
}

export function ExpandableText({
  text,
  maxPreviewChars = 300,
  isMarkdown = false,
  label,
}: ExpandableTextProps) {
  const [expanded, setExpanded] = useState(false);

  if (!text) return null;

  const needsTruncation = text.length > maxPreviewChars;
  const displayText = expanded || !needsTruncation
    ? text
    : text.slice(0, maxPreviewChars) + "...";

  return (
    <div>
      {label && (
        <p className="mb-1 text-xs font-medium text-muted-foreground uppercase tracking-wide">
          {label}
        </p>
      )}
      <div className="rounded-md border bg-muted/30 p-3 text-sm">
        {isMarkdown ? (
          <div className="prose prose-sm max-w-none dark:prose-invert">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {displayText}
            </ReactMarkdown>
          </div>
        ) : (
          <pre className="whitespace-pre-wrap font-sans">{displayText}</pre>
        )}
        {needsTruncation && (
          <Button
            variant="ghost"
            size="sm"
            className="mt-2 h-auto p-0 text-xs text-muted-foreground hover:text-foreground"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? "Show less" : "Show more"}
          </Button>
        )}
      </div>
    </div>
  );
}
