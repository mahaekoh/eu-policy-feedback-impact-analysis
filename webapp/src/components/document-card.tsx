import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Document } from "@/lib/types";
import { formatFileSize } from "@/lib/utils";
import { ExpandableText } from "@/components/expandable-text";

interface DocumentCardProps {
  document: Document;
}

export function DocumentCard({ document: doc }: DocumentCardProps) {
  return (
    <div className="rounded-lg border p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <p className="font-medium text-sm truncate" title={doc.title || doc.filename}>
            {doc.title || doc.filename}
          </p>
          <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            {doc.filename && <span>{doc.filename}</span>}
            {doc.pages != null && <span>{doc.pages} pages</span>}
            {doc.size_bytes > 0 && <span>{formatFileSize(doc.size_bytes)}</span>}
            {doc.doc_type && (
              <Badge variant="outline" className="text-xs">
                {doc.doc_type}
              </Badge>
            )}
          </div>
        </div>
        {doc.download_url && (
          <Button variant="outline" size="sm" asChild>
            <a
              href={doc.download_url}
              target="_blank"
              rel="noopener noreferrer"
            >
              Download
            </a>
          </Button>
        )}
      </div>

      {doc.summary && (
        <div className="mt-3">
          <ExpandableText
            text={doc.summary}
            label="Summary"
            isMarkdown
            maxPreviewChars={500}
          />
        </div>
      )}

      {doc.extracted_text && (
        <div className="mt-3">
          <ExpandableText
            text={doc.extracted_text}
            label="Extracted text"
            maxPreviewChars={300}
          />
        </div>
      )}
    </div>
  );
}
