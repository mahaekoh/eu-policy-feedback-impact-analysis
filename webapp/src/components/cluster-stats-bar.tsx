"use client";

import {
  countryToFlag,
  formatUserType,
  USER_TYPE_BAR_COLORS,
  USER_TYPE_SHORT,
  COUNTRY_BAR_COLORS,
} from "@/lib/types";

interface BarSegmentProps {
  label: string;
  tooltip: string;
  pct: number;
  bgColor: string;
  textColor?: string;
  small?: boolean;
}

function BarSegment({
  label,
  tooltip,
  pct,
  bgColor,
  textColor = "white",
  small,
}: BarSegmentProps) {
  return (
    <div
      className="group relative flex items-center justify-center overflow-hidden transition-opacity hover:opacity-80"
      style={{
        width: `${pct.toFixed(2)}%`,
        backgroundColor: bgColor,
        color: textColor,
        minWidth: 0,
      }}
    >
      <span
        className={`truncate leading-none ${small ? "text-[10px] font-semibold" : "text-[13px]"}`}
      >
        {label}
      </span>
      <div className="pointer-events-none absolute bottom-full left-1/2 z-10 mb-1.5 hidden -translate-x-1/2 whitespace-nowrap rounded bg-gray-900 px-2 py-0.5 text-[11px] text-white group-hover:block">
        {tooltip}
      </div>
    </div>
  );
}

interface CountryBarProps {
  sortedCountries: [string, number][];
  total: number;
}

export function CountryBar({ sortedCountries, total }: CountryBarProps) {
  const countryTotal = sortedCountries.reduce((s, e) => s + e[1], 0);
  const hasUnknown = countryTotal < total;

  return (
    <div className="flex-1 min-w-0">
      <div className="flex h-[22px] overflow-hidden rounded bg-gray-100">
        {sortedCountries.map(([code, count], idx) => (
          <BarSegment
            key={code}
            label={countryToFlag(code)}
            tooltip={`${countryToFlag(code)} ${code}: ${count}`}
            pct={(count / total) * 100}
            bgColor={COUNTRY_BAR_COLORS[idx % COUNTRY_BAR_COLORS.length]}
          />
        ))}
        {hasUnknown && (
          <BarSegment
            label="?"
            tooltip={`Unknown: ${total - countryTotal}`}
            pct={((total - countryTotal) / total) * 100}
            bgColor="#edeff1"
            textColor="#878a8c"
          />
        )}
      </div>
    </div>
  );
}

interface UserTypeBarProps {
  sortedTypes: [string, number][];
  total: number;
}

export function UserTypeBar({ sortedTypes, total }: UserTypeBarProps) {
  return (
    <div className="flex-1 min-w-0">
      <div className="flex h-[22px] overflow-hidden rounded bg-gray-100">
        {sortedTypes.map(([ut, count]) => (
          <BarSegment
            key={ut}
            label={USER_TYPE_SHORT[ut] || ut}
            tooltip={`${formatUserType(ut)}: ${count}`}
            pct={(count / total) * 100}
            bgColor={USER_TYPE_BAR_COLORS[ut] || USER_TYPE_BAR_COLORS.OTHER}
            small
          />
        ))}
      </div>
    </div>
  );
}

interface ClusterDetailStatsProps {
  sortedCountries: [string, number][];
  sortedTypes: [string, number][];
  total: number;
  organizations: [string, number][];
}

export function ClusterDetailStats({
  sortedCountries,
  sortedTypes,
  total,
  organizations,
}: ClusterDetailStatsProps) {
  return (
    <div className="flex gap-4 border-b bg-muted/30 p-4 flex-col md:flex-row">
      {/* Country column */}
      <div className="flex-1 min-w-0">
        <p className="text-[10px] uppercase tracking-wider font-semibold text-muted-foreground mb-1.5">
          Countries ({sortedCountries.length})
        </p>
        <div className="space-y-0.5">
          {sortedCountries.slice(0, 8).map(([code, count], idx) => (
            <StatRow
              key={code}
              icon={countryToFlag(code)}
              label={code}
              count={count}
              pct={(count / total) * 100}
              barColor={
                COUNTRY_BAR_COLORS[idx % COUNTRY_BAR_COLORS.length]
              }
            />
          ))}
          {sortedCountries.length > 8 && (
            <p className="text-[11px] text-muted-foreground">
              +{sortedCountries.length - 8} more
            </p>
          )}
        </div>
      </div>

      {/* User type column */}
      <div className="flex-1 min-w-0">
        <p className="text-[10px] uppercase tracking-wider font-semibold text-muted-foreground mb-1.5">
          Respondent types ({sortedTypes.length})
        </p>
        <div className="space-y-0.5">
          {sortedTypes.map(([ut, count]) => (
            <StatRow
              key={ut}
              icon={
                <span
                  className="inline-block w-2.5 h-2.5 rounded-full shrink-0"
                  style={{
                    backgroundColor:
                      USER_TYPE_BAR_COLORS[ut] ||
                      USER_TYPE_BAR_COLORS.OTHER,
                  }}
                />
              }
              label={formatUserType(ut)}
              count={count}
              pct={(count / total) * 100}
              barColor={
                USER_TYPE_BAR_COLORS[ut] || USER_TYPE_BAR_COLORS.OTHER
              }
            />
          ))}
        </div>

        {/* Organizations */}
        {organizations.length > 0 && (
          <div className="mt-2">
            <p className="text-[10px] uppercase tracking-wider font-semibold text-muted-foreground">
              Top organizations
            </p>
            <p className="text-[11px] text-muted-foreground mt-0.5">
              {organizations.slice(0, 8).map(([org, count], i) => (
                <span key={org}>
                  {i > 0 && (
                    <span className="text-gray-300"> &middot; </span>
                  )}
                  {org}{" "}
                  <span className="text-gray-400">({count})</span>
                </span>
              ))}
              {organizations.length > 8 && (
                <span className="text-gray-400">
                  {" "}
                  +{organizations.length - 8} more
                </span>
              )}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

interface StatRowProps {
  icon: React.ReactNode;
  label: string;
  count: number;
  pct: number;
  barColor: string;
}

function StatRow({ icon, label, count, pct, barColor }: StatRowProps) {
  return (
    <div className="flex items-center gap-1.5 text-xs">
      <span className="w-5 text-center shrink-0 text-sm">{icon}</span>
      <span className="flex-1 min-w-0 truncate">{label}</span>
      <span className="font-bold shrink-0">{count}</span>
      <span className="text-[11px] text-muted-foreground w-9 text-right shrink-0">
        {pct.toFixed(1)}%
      </span>
      <div className="w-14 h-2 bg-gray-100 rounded overflow-hidden shrink-0">
        <div
          className="h-full rounded"
          style={{ width: `${pct.toFixed(1)}%`, backgroundColor: barColor }}
        />
      </div>
    </div>
  );
}
