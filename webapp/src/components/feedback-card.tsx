import { Badge } from "@/components/ui/badge";
import { Feedback, getUserTypeColor, formatUserType } from "@/lib/types";
import { formatDate, formatFileSize } from "@/lib/utils";
import { ExpandableText } from "@/components/expandable-text";

interface FeedbackCardProps {
  feedback: Feedback;
}

export function FeedbackCard({ feedback }: FeedbackCardProps) {
  const colors = getUserTypeColor(feedback.user_type);

  const submitter = [feedback.organization, feedback.first_name, feedback.surname]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={`rounded-lg border-l-4 ${colors.border} border border-border p-4`}>
      <div className="flex flex-wrap items-center gap-2 mb-2">
        <Badge className={`${colors.bg} ${colors.text} border-0`}>
          {formatUserType(feedback.user_type)}
        </Badge>
        {feedback.country && (
          <span className="text-xs text-muted-foreground">
            {feedback.country}
          </span>
        )}
        {feedback.language && (
          <span className="text-xs text-muted-foreground uppercase">
            {feedback.language}
          </span>
        )}
        <span className="text-xs text-muted-foreground ml-auto flex items-center gap-2">
          {formatDate(feedback.date)}
          {feedback.url && (
            <a
              href={feedback.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary hover:underline"
            >
              EU Portal
            </a>
          )}
        </span>
      </div>

      {submitter && (
        <p className="text-sm font-medium mb-2">{submitter}</p>
      )}

      {feedback.feedback_text && (
        <ExpandableText
          text={feedback.feedback_text}
          maxPreviewChars={400}
        />
      )}

      {feedback.attachments && feedback.attachments.length > 0 && (
        <div className="mt-3 space-y-2">
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
            Attachments ({feedback.attachments.length})
          </p>
          {feedback.attachments.map((att) => (
            <div key={att.id} className="rounded border p-3 text-sm">
              <div className="flex items-center justify-between gap-2 mb-1">
                <span className="truncate font-medium text-xs">
                  {att.filename}
                </span>
                <div className="flex items-center gap-2 shrink-0">
                  {att.size_bytes > 0 && (
                    <span className="text-xs text-muted-foreground">
                      {formatFileSize(att.size_bytes)}
                    </span>
                  )}
                  {att.download_url && (
                    <a
                      href={att.download_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs text-primary hover:underline"
                    >
                      Download
                    </a>
                  )}
                </div>
              </div>
              {att.summary && (
                <ExpandableText
                  text={att.summary}
                  label="Summary"
                  isMarkdown
                  maxPreviewChars={300}
                />
              )}
              {att.extracted_text && !att.summary && (
                <ExpandableText
                  text={att.extracted_text}
                  maxPreviewChars={200}
                />
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
