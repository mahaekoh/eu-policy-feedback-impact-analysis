"use client";

import { useCallback, useMemo, useState } from "react";
import Link from "next/link";
import { useSearchParams, useRouter } from "next/navigation";
import {
  CountryStats,
  CountryStatsEntry,
  GlobalStats,
  TimeSeriesGroup,
  USER_TYPE_BAR_COLORS,
  COUNTRY_BAR_COLORS,
  countryToFlag,
  formatUserType,
} from "@/lib/types";

const TOPIC_COLORS = [
  "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
  "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
  "#86bcb6", "#8cd17d", "#b6992d", "#499894", "#d37295",
  "#a0cbe8", "#ffbe7d", "#d4a6c8", "#fabfd2", "#d7b5a6",
];

const STAGE_LABELS: Record<string, string> = {
  PLANNING_WORKFLOW: "Planning",
  ISC_WORKFLOW: "Inter-service consultation",
  ADOPTION_WORKFLOW: "Adoption",
};

function formatStage(stage: string): string {
  return STAGE_LABELS[stage] || stage.replace(/_/g, " ");
}

// --- Reusable horizontal bar row ---

interface BarRowProps {
  label: React.ReactNode;
  count: number;
  maxCount: number;
  color: string;
}

function BarRow({ label, count, maxCount, color }: BarRowProps) {
  const pct = maxCount > 0 ? (count / maxCount) * 100 : 0;
  return (
    <div className="flex items-center gap-2 text-sm">
      <span className="w-40 truncate shrink-0 text-right">{label}</span>
      <div className="flex-1 h-5 bg-gray-100 rounded overflow-hidden">
        <div
          className="h-full rounded opacity-80"
          style={{ width: `${pct.toFixed(1)}%`, backgroundColor: color }}
        />
      </div>
      <span className="w-16 text-right text-muted-foreground tabular-nums shrink-0">
        {count.toLocaleString()}
      </span>
    </div>
  );
}

// --- Section wrapper ---

interface SectionProps {
  title: string;
  children: React.ReactNode;
}

function Section({ title, children }: SectionProps) {
  return (
    <section>
      <h2 className="text-lg font-semibold mb-3">{title}</h2>
      {children}
    </section>
  );
}

// --- Timeline area chart ---

function TimelineChart({ data }: { data: [string, number][] }) {
  if (data.length === 0) return null;
  const counts = data.map(([, c]) => c);
  const labels = data.map(([m]) => m);
  const max = Math.max(...counts, 1);
  const w = 800;
  const h = 120;
  const pad = 2;
  const plotH = h - 20; // leave room for labels
  const stepX = (w - 2 * pad) / (counts.length - 1 || 1);

  const points = counts.map((c, i) => {
    const x = pad + i * stepX;
    const y = pad + (1 - c / max) * (plotH - 2 * pad);
    return [x, y] as const;
  });

  const polyline = points.map(([x, y]) => `${x},${y}`).join(" ");
  const areaPath = `M${pad},${plotH} ${points.map(([x, y]) => `L${x},${y}`).join(" ")} L${pad + (counts.length - 1) * stepX},${plotH} Z`;

  // Pick ~8 evenly spaced label positions
  const labelStep = Math.max(1, Math.floor(labels.length / 8));

  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      className="w-full max-w-4xl"
      preserveAspectRatio="xMidYMid meet"
    >
      <path d={areaPath} fill="#6366f1" opacity={0.15} />
      <polyline
        points={polyline}
        fill="none"
        stroke="#6366f1"
        strokeWidth={1.5}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      {labels.map((label, i) =>
        i % labelStep === 0 ? (
          <text
            key={label}
            x={pad + i * stepX}
            y={h - 2}
            textAnchor="middle"
            className="fill-gray-400"
            fontSize={9}
          >
            {label}
          </text>
        ) : null
      )}
    </svg>
  );
}

// --- Multi-series line chart ---

interface MultiSeriesChartProps {
  group: TimeSeriesGroup;
  colors: Record<string, string>;
  formatLabel?: (key: string) => string;
}

function MultiSeriesChart({ group, colors, formatLabel }: MultiSeriesChartProps) {
  const { months, series } = group;
  const keys = Object.keys(series);
  if (months.length === 0 || keys.length === 0) return null;

  // Compute global max across all series
  const globalMax = keys.reduce((m, k) => {
    const seriesMax = Math.max(...series[k]);
    return Math.max(m, seriesMax);
  }, 1);

  const w = 800;
  const h = 160;
  const pad = 2;
  const plotH = h - 20;
  const stepX = (w - 2 * pad) / (months.length - 1 || 1);
  const labelStep = Math.max(1, Math.floor(months.length / 8));

  return (
    <div>
      <svg
        viewBox={`0 0 ${w} ${h}`}
        className="w-full max-w-4xl"
        preserveAspectRatio="xMidYMid meet"
      >
        {keys.map((key) => {
          const values = series[key];
          const points = values.map((c, i) => {
            const x = pad + i * stepX;
            const y = pad + (1 - c / globalMax) * (plotH - 2 * pad);
            return `${x},${y}`;
          });
          return (
            <polyline
              key={key}
              points={points.join(" ")}
              fill="none"
              stroke={colors[key] || "#95a5a6"}
              strokeWidth={1.5}
              strokeLinejoin="round"
              strokeLinecap="round"
              opacity={0.8}
            />
          );
        })}
        {months.map((label, i) =>
          i % labelStep === 0 ? (
            <text
              key={label}
              x={pad + i * stepX}
              y={h - 2}
              textAnchor="middle"
              className="fill-gray-400"
              fontSize={9}
            >
              {label}
            </text>
          ) : null
        )}
      </svg>
      {/* Legend */}
      <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2 text-xs text-muted-foreground">
        {keys.map((key) => (
          <span key={key} className="flex items-center gap-1">
            <span
              className="inline-block w-2.5 h-2.5 rounded-sm shrink-0"
              style={{ backgroundColor: colors[key] || "#95a5a6" }}
            />
            {formatLabel ? formatLabel(key) : key}
          </span>
        ))}
      </div>
    </div>
  );
}

// --- Cross-tab expandable section ---

interface CrossTabProps {
  data: Record<string, [string, number][]>;
  outerLabel: (key: string) => React.ReactNode;
  innerLabel: (key: string) => React.ReactNode;
  limit: number;
  defaultExpanded?: boolean;
}

function CrossTab({ data, outerLabel, innerLabel, limit, defaultExpanded }: CrossTabProps) {
  const keys = Object.keys(data).slice(0, limit);
  const [expanded, setExpanded] = useState<Set<string>>(
    defaultExpanded ? new Set(keys) : new Set()
  );

  return (
    <div className="space-y-1">
      {keys.map((key) => {
        const items = data[key];
        const isOpen = expanded.has(key);
        const total = items.reduce((s, [, c]) => s + c, 0);
        return (
          <div key={key}>
            <button
              className="flex items-center gap-2 w-full text-left text-sm hover:bg-accent/50 rounded px-2 py-1 cursor-pointer"
              onClick={() => {
                const next = new Set(expanded);
                if (isOpen) next.delete(key);
                else next.add(key);
                setExpanded(next);
              }}
            >
              <span className="text-xs text-muted-foreground w-4">
                {isOpen ? "▾" : "▸"}
              </span>
              <span className="font-medium">{outerLabel(key)}</span>
              <span className="text-muted-foreground ml-auto tabular-nums">
                {total.toLocaleString()}
              </span>
            </button>
            {isOpen && (
              <div className="ml-8 mt-1 mb-2 space-y-0.5">
                {items.map(([inner, count]) => {
                  const pct = total > 0 ? (count / total) * 100 : 0;
                  return (
                    <div
                      key={inner}
                      className="flex items-center gap-2 text-sm"
                    >
                      <span className="w-32 truncate text-right shrink-0">
                        {innerLabel(inner)}
                      </span>
                      <div className="flex-1 h-4 bg-gray-100 rounded overflow-hidden">
                        <div
                          className="h-full rounded opacity-80"
                          style={{
                            width: `${pct.toFixed(1)}%`,
                            backgroundColor: "#6366f1",
                          }}
                        />
                      </div>
                      <span className="w-14 text-right text-muted-foreground tabular-nums text-xs shrink-0">
                        {count.toLocaleString()}
                      </span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// --- Truncated bar list with "see more" ---

interface TruncatedBarListProps {
  items: [string, number][];
  maxCount: number;
  limit: number;
}

function TruncatedBarList({ items, maxCount, limit }: TruncatedBarListProps) {
  const [showAll, setShowAll] = useState(false);
  const visible = showAll ? items : items.slice(0, limit);
  const hasMore = items.length > limit;

  return (
    <div className="space-y-1">
      {visible.map(([label, count], idx) => (
        <BarRow
          key={label}
          label={label}
          count={count}
          maxCount={maxCount}
          color={COUNTRY_BAR_COLORS[idx % COUNTRY_BAR_COLORS.length]}
        />
      ))}
      {hasMore && (
        <button
          className="text-sm text-primary hover:underline cursor-pointer mt-1"
          onClick={() => setShowAll(!showAll)}
        >
          {showAll ? "Show less" : `Show all ${items.length} topics`}
        </button>
      )}
    </div>
  );
}

// --- Stacked topic bars by country ---

interface TopicsByCountryChartProps {
  data: Record<string, [string, number][]>;
  topicColorMap: Record<string, string>;
}

function TopicsByCountryChart({ data, topicColorMap }: TopicsByCountryChartProps) {
  const keys = Object.keys(data).slice(0, 30);
  const globalMax = keys.reduce((m, k) => {
    const total = data[k].reduce((s, [, c]) => s + c, 0);
    return Math.max(m, total);
  }, 0);

  return (
    <div className="space-y-1.5">
      {/* Legend */}
      <div className="flex flex-wrap gap-x-4 gap-y-1 mb-2 text-xs text-muted-foreground">
        {Object.entries(topicColorMap).map(([topic, color]) => (
          <span key={topic} className="flex items-center gap-1">
            <span
              className="inline-block w-2.5 h-2.5 rounded-sm shrink-0"
              style={{ backgroundColor: color }}
            />
            {topic}
          </span>
        ))}
      </div>
      {keys.map((code) => {
        const items = data[code];
        const total = items.reduce((s, [, c]) => s + c, 0);
        const barPct = globalMax > 0 ? (total / globalMax) * 100 : 0;
        return (
          <div key={code} className="flex items-center gap-2 text-sm">
            <span className="w-20 truncate shrink-0 text-right">
              {countryToFlag(code)} {code}
            </span>
            <div className="flex-1 h-5 bg-gray-100 rounded overflow-hidden">
              <div
                className="h-full flex rounded overflow-hidden opacity-80"
                style={{ width: `${barPct.toFixed(1)}%` }}
              >
                {items.map(([topic, count]) => {
                  const segPct = total > 0 ? (count / total) * 100 : 0;
                  return (
                    <div
                      key={topic}
                      className="group relative h-full"
                      style={{
                        width: `${segPct.toFixed(2)}%`,
                        backgroundColor: topicColorMap[topic] || "#95a5a6",
                        minWidth: 0,
                      }}
                    >
                      <div className="pointer-events-none absolute bottom-full left-1/2 z-10 mb-1.5 hidden -translate-x-1/2 whitespace-nowrap rounded bg-gray-900 px-2 py-0.5 text-[11px] text-white group-hover:block">
                        {topic}: {count.toLocaleString()}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
            <span className="w-16 text-right text-muted-foreground tabular-nums shrink-0">
              {total.toLocaleString()}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// --- Country drill-down ---

interface CountryDrillDownProps {
  entry: CountryStatsEntry;
  topicColorMap: Record<string, string>;
}

function CountryDrillDown({ entry, topicColorMap }: CountryDrillDownProps) {
  const topicMax = entry.by_topic[0]?.[1] ?? 0;
  const userTypeMax = entry.by_user_type[0]?.[1] ?? 0;

  return (
    <div className="space-y-10">
      <div className="text-lg">
        <span className="font-semibold">{entry.total_feedback.toLocaleString()}</span>{" "}
        feedback submissions
      </div>

      <Section title="Top topics">
        <div className="space-y-1">
          {entry.by_topic.map(([topic, count], idx) => (
            <BarRow
              key={topic}
              label={topic}
              count={count}
              maxCount={topicMax}
              color={topicColorMap[topic] || COUNTRY_BAR_COLORS[idx % COUNTRY_BAR_COLORS.length]}
            />
          ))}
        </div>
      </Section>

      <Section title="Feedback by respondent type">
        <div className="space-y-1">
          {entry.by_user_type.map(([ut, count]) => (
            <BarRow
              key={ut}
              label={formatUserType(ut)}
              count={count}
              maxCount={userTypeMax}
              color={USER_TYPE_BAR_COLORS[ut] || USER_TYPE_BAR_COLORS.OTHER}
            />
          ))}
        </div>
      </Section>

      <Section title="Top initiatives">
        <div className="space-y-1">
          {entry.top_initiatives.map((init, idx) => (
            <div key={init.id} className="flex items-center gap-2 text-sm">
              <span className="w-8 text-right text-muted-foreground shrink-0">
                {idx + 1}.
              </span>
              <Link
                href={`/initiative/${init.id}`}
                className="flex-1 truncate text-primary hover:underline"
              >
                {init.short_title}
              </Link>
              <span className="w-16 text-right text-muted-foreground tabular-nums shrink-0">
                {init.count.toLocaleString()}
              </span>
            </div>
          ))}
        </div>
      </Section>

      {entry.topic_timeline.months.length > 0 && (
        <Section title="Topic timeline">
          <MultiSeriesChart
            group={entry.topic_timeline}
            colors={topicColorMap}
          />
        </Section>
      )}

      <Section title="Recent feedback">
        <div className="space-y-2">
          {entry.recent_feedback.map((fb, idx) => (
            <div
              key={idx}
              className="border rounded-lg p-3 text-sm space-y-1"
            >
              <div className="flex items-center gap-2 flex-wrap text-muted-foreground text-xs">
                <span>{fb.date?.slice(0, 10)}</span>
                <span className="font-medium text-foreground">
                  {formatUserType(fb.user_type)}
                </span>
                {fb.organization && (
                  <span>{fb.organization}</span>
                )}
                {!fb.organization && fb.first_name && (
                  <span>{fb.first_name} {fb.surname}</span>
                )}
              </div>
              <div className="flex items-center gap-2">
                <Link
                  href={`/initiative/${fb.initiative_id}`}
                  className="text-primary hover:underline text-xs"
                >
                  {fb.initiative_title}
                </Link>
                {fb.url && (
                  <a
                    href={fb.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-muted-foreground hover:text-primary hover:underline shrink-0"
                  >
                    view on portal
                  </a>
                )}
              </div>
              {fb.feedback_text && (
                <p className="text-muted-foreground text-xs line-clamp-2">
                  {fb.feedback_text}
                </p>
              )}
              {fb.attachments?.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {fb.attachments.map((att) => (
                    <a
                      key={att.download_url}
                      href={att.download_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
                    >
                      <svg className="w-3 h-3 shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828L18 9.828a4 4 0 00-5.656-5.656L5.757 10.757a6 6 0 008.486 8.486L20.5 13" /></svg>
                      {att.filename}
                    </a>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </Section>
    </div>
  );
}

// --- Main Charts component ---

interface ChartsProps {
  stats: GlobalStats;
  countryStats: CountryStats;
}

export function Charts({ stats, countryStats }: ChartsProps) {
  const searchParams = useSearchParams();
  const router = useRouter();
  const selectedCountry = searchParams.get("country") ?? "";

  const setSelectedCountry = useCallback((code: string) => {
    const params = new URLSearchParams(searchParams);
    if (code) {
      params.set("country", code);
    } else {
      params.delete("country");
    }
    const qs = params.toString();
    router.replace(qs ? `?${qs}` : "/charts", { scroll: false });
  }, [searchParams, router]);

  // Stable topic → color mapping based on global topic ranking
  const topicColorMap = useMemo(() => {
    const map: Record<string, string> = {};
    stats.by_topic.forEach(([topic], i) => {
      map[topic] = TOPIC_COLORS[i % TOPIC_COLORS.length];
    });
    return map;
  }, [stats.by_topic]);

  const countryColorMap = useMemo(() => {
    const map: Record<string, string> = {};
    stats.by_country.forEach(([code], i) => {
      map[code] = COUNTRY_BAR_COLORS[i % COUNTRY_BAR_COLORS.length];
    });
    return map;
  }, [stats.by_country]);

  const countryMax = stats.by_country[0]?.[1] ?? 0;
  const topicMax = stats.by_topic[0]?.[1] ?? 0;
  const userTypeMax = stats.by_user_type[0]?.[1] ?? 0;
  const stageMax = stats.by_stage[0]?.[1] ?? 0;

  const selectedEntry = selectedCountry ? countryStats[selectedCountry] : null;

  return (
    <div className="space-y-10">
      {/* Country selector */}
      <div className="flex items-center gap-3">
        <label htmlFor="country-select" className="text-sm font-medium shrink-0">
          Country
        </label>
        <select
          id="country-select"
          value={selectedCountry}
          onChange={(e) => setSelectedCountry(e.target.value)}
          className="border rounded-md px-3 py-1.5 text-sm bg-background w-64"
        >
          <option value="">All countries</option>
          {stats.by_country.map(([code, count]) => (
            <option key={code} value={code}>
              {countryToFlag(code)} {code} ({count.toLocaleString()})
            </option>
          ))}
        </select>
      </div>

      {selectedEntry ? (
        <CountryDrillDown entry={selectedEntry} topicColorMap={topicColorMap} />
      ) : (
      <>
      {/* Feedback over time */}
      <Section title="Feedback over time">
        <TimelineChart data={stats.feedback_by_month} />
      </Section>

      <Section title="Feedback by topic over time">
        <MultiSeriesChart
          group={stats.feedback_by_month_by_topic}
          colors={topicColorMap}
        />
      </Section>

      <Section title="Feedback by country over time">
        <MultiSeriesChart
          group={stats.feedback_by_month_by_country}
          colors={countryColorMap}
          formatLabel={(code) => `${countryToFlag(code)} ${code}`}
        />
      </Section>

      {/* By Country */}
      <Section title="Feedback by country">
        <div className="space-y-1">
          {stats.by_country.slice(0, 30).map(([code, count], idx) => (
            <BarRow
              key={code}
              label={
                <span>
                  {countryToFlag(code)} {code}
                </span>
              }
              count={count}
              maxCount={countryMax}
              color={COUNTRY_BAR_COLORS[idx % COUNTRY_BAR_COLORS.length]}
            />
          ))}
        </div>
      </Section>

      {/* By User Type */}
      <Section title="Feedback by respondent type">
        <div className="space-y-1">
          {stats.by_user_type.map(([ut, count]) => (
            <BarRow
              key={ut}
              label={formatUserType(ut)}
              count={count}
              maxCount={userTypeMax}
              color={USER_TYPE_BAR_COLORS[ut] || USER_TYPE_BAR_COLORS.OTHER}
            />
          ))}
        </div>
      </Section>

      {/* By Topic */}
      <Section title="Feedback by topic">
        <TruncatedBarList
          items={stats.by_topic}
          maxCount={topicMax}
          limit={10}
        />
      </Section>

      {/* Initiatives by Topic */}
      <Section title="Initiatives by topic">
        <TruncatedBarList
          items={stats.initiatives_by_topic}
          maxCount={stats.initiatives_by_topic[0]?.[1] ?? 0}
          limit={10}
        />
      </Section>

      {/* By Stage */}
      <Section title="Initiatives by stage">
        <div className="space-y-1">
          {stats.by_stage.map(([stage, count], idx) => (
            <BarRow
              key={stage}
              label={formatStage(stage)}
              count={count}
              maxCount={stageMax}
              color={COUNTRY_BAR_COLORS[idx % COUNTRY_BAR_COLORS.length]}
            />
          ))}
        </div>
      </Section>

      {/* Cross-tabs */}
      <Section title="Top topics by country">
        <TopicsByCountryChart data={stats.top_topics_by_country} topicColorMap={topicColorMap} />
      </Section>

      <Section title="Top countries by topic">
        <CrossTab
          data={stats.top_countries_by_topic}
          outerLabel={(topic) => topic}
          innerLabel={(code) => (
            <span>
              {countryToFlag(code)} {code}
            </span>
          )}
          limit={20}
          defaultExpanded
        />
      </Section>
      </>
      )}
    </div>
  );
}
