# EU Policy Feedback Impact Analysis — Web Application

Interactive web application for browsing EU "Have Your Say" initiatives, feedback, and analysis results. Built with Next.js 16, React 19, and Tailwind CSS 4.

## Prerequisites

- **Node.js** >= 18
- **Pipeline data** — Run `build_webapp_index.py` before starting the webapp (see [root README](../README.md))
- **Google OAuth credentials** (optional) — See [AUTH.md](AUTH.md) for setup

## Quick Start

```bash
cd webapp
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The app is fully accessible without signing in.

## Authentication (Optional)

Google sign-in via [Auth.js v5](https://authjs.dev/). JWT-based sessions, no database required.

Create `webapp/.env.local`:

```bash
npx auth secret                  # Generates AUTH_SECRET
# Then add:
AUTH_GOOGLE_ID=your-client-id
AUTH_GOOGLE_SECRET=your-client-secret
```

See [AUTH.md](AUTH.md) for full Google Cloud Console setup instructions.

## Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| [Next.js](https://nextjs.org/) | 16.1.6 | React framework (App Router, server components) |
| [React](https://react.dev/) | 19.2.3 | UI library |
| [Tailwind CSS](https://tailwindcss.com/) | 4 | Utility-first CSS |
| [shadcn/ui](https://ui.shadcn.com/) | — | Component library (Radix UI 1.4.3 + Lucide React 0.575.0) |
| [next-auth](https://authjs.dev/) | 5.0.0-beta.30 | Google OAuth, JWT sessions |
| [react-markdown](https://github.com/remarkjs/react-markdown) | 10.1.0 | Markdown rendering (+ remark-gfm 4.0.1) |
| TypeScript | 5 | Type safety |

## npm Scripts

| Script | Command | Description |
|---|---|---|
| `dev` | `next dev` | Start development server with hot reload |
| `build` | `next build` | Create production build |
| `start` | `next start` | Run production server |
| `lint` | `eslint` | Run ESLint checks |

## Pages

### `/` — Initiative Index

Browse all initiatives with:
- **Search** — Filter by title text
- **Sort** — By title, feedback count, date, status
- **Filters** — By initiative status, feedback status, topic
- **Pagination** — 50 initiatives per page
- Deduplicates initiatives sharing identical feedback ID sets

### `/initiative/[id]` — Initiative Detail

Two-tab view for each initiative:

**Publications tab:**
- Expandable publication sections showing documents and feedback
- Document cards with extracted text, summaries, and download links
- Feedback list with infinite scroll, filtering by user type/search/empty
- Country and user type distribution counts
- Feedback timeline histogram

**Cluster tab:**
- Hierarchical cluster tree visualization
- Per-cluster country and user type distribution bars
- Cluster summaries (policy summary + per-cluster aggregate summaries)
- Scheme selector dropdown (when multiple clustering schemes are available)

### `/charts` — Aggregate Statistics

**Global view:**
- Feedback volume time series (monthly)
- Top countries, topics, user types
- Department and initiative stage breakdowns
- Cross-tabulations

**Country drill-down** (via `?country=CODE` URL parameter):
- Top 20 topics for that country
- User type breakdown
- Top 20 initiatives with feedback from that country
- 20 most recent feedback items with attachment links
- Top-5 topic timeline

## API Routes

| Route | Method | Description |
|---|---|---|
| `/api/auth/[...nextauth]` | GET/POST | NextAuth OAuth handlers (sign-in, callback, sign-out) |
| `/api/clusters/[id]` | GET | Fetch clustering data for an initiative. Query param: `?scheme=<scheme_name>`. Returns `ClusterData` JSON or 404. |

## Project Structure

```
webapp/
├── src/
│   ├── app/                           # Next.js App Router
│   │   ├── page.tsx                   # / — Initiative index
│   │   ├── layout.tsx                 # Root layout (SessionProvider)
│   │   ├── initiative/
│   │   │   └── [id]/
│   │   │       ├── page.tsx           # /initiative/[id] — Detail page
│   │   │       └── not-found.tsx      # 404 for missing initiatives
│   │   ├── charts/
│   │   │   └── page.tsx               # /charts — Statistics page
│   │   └── api/
│   │       ├── auth/[...nextauth]/
│   │       │   └── route.ts           # Auth endpoints
│   │       └── clusters/[id]/
│   │           └── route.ts           # Cluster data endpoint
│   ├── components/                    # React components
│   │   ├── header.tsx                 # Navigation header with user menu
│   │   ├── user-menu.tsx              # Sign-in/sign-out menu
│   │   ├── initiative-list.tsx        # Initiative table (search, sort, filter, pagination)
│   │   ├── initiative-card.tsx        # Single initiative card
│   │   ├── initiative-detail.tsx      # Main detail view (publications + clusters tabs)
│   │   ├── publication-section.tsx    # Expandable publication section
│   │   ├── document-card.tsx          # Publication document display
│   │   ├── feedback-card.tsx          # Single feedback item with attachments
│   │   ├── feedback-list.tsx          # Feedback list with filters and infinite scroll
│   │   ├── charts.tsx                 # Statistical charts and visualizations
│   │   ├── cluster-view.tsx           # Cluster tree visualization
│   │   ├── cluster-node.tsx           # Individual cluster node in tree
│   │   ├── cluster-stats-bar.tsx      # Country/user-type distribution bars
│   │   ├── expandable-text.tsx        # Collapsible text blocks
│   │   └── ui/                        # shadcn/ui primitives
│   │       ├── badge.tsx
│   │       ├── button.tsx
│   │       ├── card.tsx
│   │       ├── input.tsx
│   │       └── select.tsx
│   ├── lib/
│   │   ├── data.ts                    # Server-side data loading (cached)
│   │   ├── types.ts                   # TypeScript interfaces and utilities
│   │   └── utils.ts                   # CSS class utilities (cn)
│   ├── auth.ts                        # Auth.js config (Google provider)
│   └── proxy.ts                       # Session cookie refresh middleware
├── public/                            # Static assets
├── AUTH.md                            # Authentication setup guide
├── package.json
├── next.config.ts
├── tsconfig.json
└── components.json                    # shadcn/ui registry config
```

## Data Loading (`src/lib/data.ts`)

Server-side data loading with a **5-minute in-memory cache** (`CACHE_TTL_MS = 300,000`).

| Function | Returns | Cached | Description |
|---|---|---|---|
| `getInitiativeIndex()` | `InitiativeSummary[]` | Yes | Full initiative list for index page |
| `getGlobalStats()` | `GlobalStats` | Yes | Aggregate cross-initiative statistics |
| `getCountryStats()` | `CountryStats` | Yes | Per-country drill-down data |
| `getInitiativeDetail(id)` | `Initiative \| null` | No | Single initiative with full detail |
| `getClusteringSchemesForInitiative(id)` | `string[]` | No | Available clustering scheme names |
| `getClusterData(id, scheme)` | `ClusterData \| null` | No | Cluster assignments + summaries |

### Data paths (relative to `webapp/`)

| Path | Produced by | Used by | Overwrite behavior |
|---|---|---|---|
| `../data/webapp/initiative_index.json` | `build_webapp_index.py` | `getInitiativeIndex()` | Regenerated every run |
| `../data/webapp/global_stats.json` | `build_webapp_index.py` | `getGlobalStats()` | Regenerated every run |
| `../data/webapp/country_stats.json` | `build_webapp_index.py` | `getCountryStats()` | Regenerated every run |
| `../data/webapp/initiative_details/*.json` | `build_webapp_index.py` | `getInitiativeDetail()`, `getClusterData()` | Regenerated every run. Stripped copies with `extracted_text`, `extracted_text_without_ocr`, `extracted_text_before_translation` removed from feedback attachments. Cluster summaries (`cluster_policy_summary`, `cluster_summaries`) are present if `merge_cluster_feedback_summaries.py` was run before index building. |
| `../data/clustering/<scheme>/*.json` | `cluster_all_initiatives.py` | `getClusterData()` | Overwritten every clustering run (pipeline.sh does not pass `--skip-existing`). Filenames encode algorithm, model, and parameters: `{id}_{algo}_{model}_{params}.json`. |

All webapp data files are populated by running `build_webapp_index.py` (see the [root README](../README.md) for full pipeline details). To update the webapp after a pipeline re-run, re-run `build_webapp_index.py` and restart the dev server (or wait for the 5-minute cache to expire).

## TypeScript Interfaces (`src/lib/types.ts`)

### Core data types

| Interface | Description |
|---|---|
| `Attachment` | File attachment with id, filename, extracted_text, optional summary, OCR/translation variants |
| `Feedback` | Feedback item: user_type, date, status, country, organization, language, feedback_text, attachments |
| `Document` | Publication document: label, title, pages, extracted_text, optional summary |
| `Publication` | Publication: id, type, reference, feedback_end_date, documents[], feedback[] |
| `Initiative` | Top-level: id, url, title, department, status, stage, topics, publications[], derived summaries |

### Index and statistics types

| Interface | Description |
|---|---|
| `InitiativeSummary` | Lightweight entry for index page: metadata + country_counts, user_type_counts, feedback_timeline |
| `GlobalStats` | Aggregate stats: totals, by_country, by_topic, by_user_type, by_department, by_stage, feedback_by_month |
| `CountryStatsEntry` | Per-country: top topics, user types, top initiatives, recent feedback, topic timeline |
| `CountryStats` | `Record<string, CountryStatsEntry>` |

### Clustering types

| Interface | Description |
|---|---|
| `ClusterData` | Cluster metadata, assignments, silhouette score, optional summaries |
| `ClusterNode` | Hierarchical tree node: label, directItems, children, allItems |
| `ClusterSummaryEntry` | Cluster summary: title + summary text |

### Utility exports

| Export | Description |
|---|---|
| `USER_TYPE_COLORS` | User type → CSS color mapping |
| `USER_TYPE_BAR_COLORS` | User type → bar chart color palette |
| `USER_TYPE_SHORT` | User type → short label mapping |
| `COUNTRY_BAR_COLORS` | Color palette for country bar segments |
| `ISO3_TO_ISO2` | ISO 3166 alpha-3 → alpha-2 country code mapping |
| `countryToFlag(code)` | Convert country code to flag emoji |
| `getUserTypeColor(type)` | Get CSS color for a user type |
| `formatUserType(type)` | Get display label for a user type |
| `buildClusterTree(data)` | Build hierarchical `ClusterNode` tree from flat assignments |
| `computeClusterStats(node, data)` | Compute country/user-type stats for a cluster node |

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `AUTH_SECRET` | For auth | JWT signing key (generate: `npx auth secret`) |
| `AUTH_GOOGLE_ID` | For auth | Google OAuth 2.0 Client ID |
| `AUTH_GOOGLE_SECRET` | For auth | Google OAuth 2.0 Client Secret |

Authentication is entirely optional. All pages work without it.
