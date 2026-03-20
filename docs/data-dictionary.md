# Data Dictionary

This is the comprehensive data dictionary for the EU Policy Feedback Transparency Platform. It documents every data file, JSON schema, field definition, and derived field lifecycle in the system.

## Table of Contents

- [Data Directory Structure](#data-directory-structure)
- [Initiative JSON Schema](#initiative-json-schema)
- [Cluster Summaries Schema](#cluster-summaries-schema)
- [Cluster Rewrite Output Schema](#cluster-rewrite-output-schema)
- [Clustering Output Schema](#clustering-output-schema)
- [Cluster Summary Output Schema](#cluster-summary-output-schema)
- [Before/After Analysis Schema](#beforeafter-analysis-schema)
- [Unit Summaries Schema](#unit-summaries-schema)
- [Classification Output Schema](#classification-output-schema)
- [Webapp Data Schemas](#webapp-data-schemas)
- [Respondent Types](#respondent-types)
- [Derived Field Lifecycle](#derived-field-lifecycle)

---

## Data Directory Structure

All pipeline data lives under `data/`. The directory is gitignored entirely.

```
data/
  scrape/                                      # Scraped raw data (source of truth)
    eu_initiatives.csv                         # Flat CSV of all initiatives (overwritten every scrape)
    eu_initiatives_raw.json                    # Full API response data (overwritten every scrape)
    initiative_details/                        # Per-initiative JSON files (mutated in-place by merges)
      {id}.json                                # One file per initiative (e.g. 12970.json)
    doc_cache/                                 # Cached downloaded publication document files
      {init_id}/                               # Per-initiative subdirectory
        pub{pub_id}_doc{doc_id}_{filename}     # Cached file (never deleted, reused across runs)

  ocr/                                         # OCR pipeline input/output
    short_pdf_report.json                      # Report of PDFs with short extracted text
    pdfs/                                      # Downloaded PDF files for OCR processing
      {filename}                               # Attachment PDFs identified as needing OCR
    short_pdf_report_ocr.json                  # OCR results (overwritten every OCR run)

  translation/                                 # Translation pipeline input/output
    non_english_attachments.json               # List of non-English attachments to translate
    non_english_attachments_translated.json     # Combined translation output (overwritten)
    non_english_attachments_translated_batches/ # Per-batch translation results (append-only)
      batch_0000.json                          # Individual batch result files
      batch_0001.json

  analysis/                                    # Analysis and summarization output
    before_after/                              # initiative_stats.py output (overwritten every run)
      {id}.json                                # Initiatives with before/after feedback structure
    summaries/                                 # summarize_documents.py output (immutable per file)
      {id}.json                                # Initiatives with summary fields on docs/attachments
      _batches_pass1/                          # Crash recovery files (auto-cleaned)
        group_0000/
          batch_0000.json
      _batches_pass2/                          # Crash recovery for combining pass
    unit_summaries/                            # build_unit_summaries.py output (overwritten)
      {id}.json                                # Initiatives with unified summary fields
    change_summaries/                          # summarize_changes.py output (immutable per file)
      {id}.json                                # Initiatives with change_summary field
      _batches/                                # Crash recovery files (auto-cleaned)
      _batches_combine/                        # Crash recovery for combining pass

  clustering/<scheme>/                         # Clustering output (one subdir per scheme)
    {id}_{algo}_{model}_{params}.json          # Per-initiative clustering result
                                               # Example: 12970_agglomerative_google_embeddinggemma-300m_
                                               #   distance_threshold=0.75_linkage=average_
                                               #   max_cluster_size=20_max_depth=3_sub_cluster_scale=0.75.json

  embeddings/<model>/                          # Cached sentence embeddings (per model)
    {id}.npz                                   # NumPy compressed arrays (hash-validated)

  classification/                              # Classification output (immutable per file)
    {id}.json                                  # Initiative with nuclear stance labels
    _failed.json                               # Failed classification prompts (if any)

  cluster_summaries/<scheme>/                  # Cluster summary output (one subdir per scheme)
    {id}.json                                  # Per-initiative cluster summaries (immutable)
    _cluster_cache.json                        # Content-addressed cache (SHA-256 of feedback IDs)
    _batches_p1/                               # Phase 1 crash recovery (policy + feedback chunks)
    _batches_p2/                               # Phase 2 crash recovery (feedback combining)
    _batches_p3/                               # Phase 3 crash recovery (cluster combining by depth)

  cluster_rewrites/<format>/<scheme>/          # Rewrite output per format per scheme
    {id}.json                                  # Per-initiative rewrites (immutable per file)
    _batches/                                  # Crash recovery files

  webapp/                                      # Pre-computed webapp data (overwritten every build)
    initiative_index.json                      # Single JSON array of all initiative summaries
    global_stats.json                          # Aggregate cross-initiative statistics
    country_stats.json                         # Per-country drill-down statistics
    initiative_details/                        # Stripped copies of initiative JSONs
      {id}.json                                # No extracted_text on feedback attachments
```

### Scheme Name Encoding

Clustering scheme directory names encode the algorithm, model, and all parameters. `pipeline.sh` parses these names into CLI flags for `cluster_all_initiatives.py`. Example:

```
agglomerative_google_embeddinggemma-300m_distance_threshold=0.75_linkage=average_max_cluster_size=20_max_depth=3_sub_cluster_scale=0.75
```

Breakdown:

| Segment | Meaning |
|---|---|
| `agglomerative` | Clustering algorithm |
| `google_embeddinggemma-300m` | Sentence embedding model (slash replaced with underscore) |
| `distance_threshold=0.75` | AgglomerativeClustering distance threshold |
| `linkage=average` | AgglomerativeClustering linkage method |
| `max_cluster_size=20` | Clusters above this size are recursively sub-clustered |
| `max_depth=3` | Maximum recursion depth for sub-clustering |
| `sub_cluster_scale=0.75` | Distance threshold multiplier per recursion level |

---

## Initiative JSON Schema

**Path:** `data/scrape/initiative_details/{id}.json`

This is the central data structure. Each initiative has one JSON file that is progressively enriched by merge scripts throughout the pipeline.

### Full JSON Example

```json
{
  "id": 12970,
  "url": "https://ec.europa.eu/info/law/better-regulation/have-your-say/initiatives/12970-...",
  "short_title": "EU school fruit, vegetables and milk scheme - review",
  "summary": "This initiative aims to review the EU school fruit, vegetables and milk scheme...",
  "reference": "Ares(2021)4215064",
  "type_of_act": "Regulation",
  "type_of_act_code": "PROP_REG",
  "department": "AGRI",
  "status": "ADOPTED",
  "stage": "ADOPTION_WORKFLOW",
  "topics": ["Agriculture"],
  "policy_areas": ["Agriculture and rural development"],
  "published_date": "2021/06/29 00:00:00",
  "last_cached_at": "2026-03-15T14:23:01+00:00",

  "documents_before_feedback": [ /* Document[] - see below */ ],
  "documents_after_feedback": [ /* Document[] - see below */ ],
  "middle_feedback": [ /* Feedback[] - see below */ ],
  "before_feedback_summary": "Concatenated summaries of all documents before feedback...",
  "after_feedback_summary": "Concatenated summaries of all documents after feedback...",
  "change_summary": "The Commission's proposal was amended after public consultation to...",
  "diff": "--- before\n+++ after\n@@ -1,5 +1,7 @@\n...",
  "cluster_policy_summary": {
    "title": "EU School Fruit, Vegetables and Milk Scheme Review",
    "summary": "The Commission proposes to revise..."
  },
  "cluster_summaries": {
    "0": {
      "title": "Support for Expanding the Scheme",
      "summary": "Multiple respondents support...",
      "feedback_count": 7,
      "rewrites": {
        "reddit": {
          "title": "Stakeholders want the school food scheme expanded",
          "body": "Most feedback supports expanding the EU school food program..."
        }
      }
    },
    "1": {
      "title": "Concerns About Implementation Costs",
      "summary": "Several respondents raised concerns...",
      "feedback_count": 4
    },
    "-1:503089": {
      "title": "Oatly's Position on Plant-Based Options",
      "summary": "Oatly recommends including plant-based...",
      "feedback_count": 1
    }
  },

  "publications": [
    {
      "publication_id": 15000,
      "type": "CFE_IMPACT_ASSESS",
      "section_label": "Call for evidence",
      "reference": "Ares(2021)4215064",
      "published_date": "2021/06/29 00:00:00",
      "adoption_date": null,
      "planned_period": "",
      "feedback_end_date": "2021/07/27 00:00:00",
      "feedback_period_weeks": 4,
      "feedback_status": "CLOSED",
      "total_feedback": 74,
      "documents": [
        {
          "label": "Call for evidence - Ares(2021)4215064",
          "title": "Call for evidence for an evaluation",
          "filename": "PART-2021-04215064.pdf",
          "download_url": "https://ec.europa.eu/info/law/better-regulation/brpapi/download/...",
          "reference": "Ares(2021)4215064",
          "doc_type": "MAIN",
          "category": "CALL_FOR_EVIDENCE",
          "pages": 5,
          "size_bytes": 250000,
          "extracted_text": "# Full markdown text extracted from the PDF...",
          "extracted_text_without_ocr": "Original text before OCR replacement...",
          "summary": "This call for evidence outlines the Commission's plan to review..."
        }
      ],
      "feedback": [
        {
          "id": 503089,
          "url": "https://ec.europa.eu/info/law/better-regulation/have-your-say/initiatives/12970/F503089_en",
          "date": "2021/07/27 14:23:01",
          "feedback_text": "We support the review of the school scheme and recommend...",
          "feedback_text_original": "Original text before any API-level translation...",
          "language": "EN",
          "user_type": "COMPANY",
          "country": "SWE",
          "company_size": "LARGE",
          "organization": "Oatly AB",
          "first_name": "John",
          "surname": "Doe",
          "status": "PUBLISHED",
          "publication": "Call for evidence",
          "tr_number": "12345-67890",
          "combined_feedback_summary": "We support the review... [attachment summary]...",
          "cluster_feedback_summary": {
            "title": "Oatly's Position on Plant-Based Options",
            "summary": "Oatly recommends including plant-based..."
          },
          "nuclear_stance": "DOES NOT MENTION",
          "nuclear_stance_reasoning": "The feedback discusses school food programs...",
          "attachments": [
            {
              "id": 6276475,
              "filename": "Oatly_position.pdf",
              "document_id": "DOC-2021-12345",
              "download_url": "https://ec.europa.eu/info/law/better-regulation/api/download/...",
              "pages": 3,
              "size_bytes": 120000,
              "extracted_text": "Full text (translated to English if needed)...",
              "extracted_text_before_translation": "Original non-English text...",
              "extracted_text_without_ocr": "Original pre-OCR text...",
              "extracted_text_error": "Error message if extraction failed...",
              "summary": "Oatly's position paper argues for the inclusion of..."
            }
          ]
        }
      ]
    }
  ]
}
```

### Top-Level Fields

| Field | Type | Source | Description |
|---|---|---|---|
| `id` | `int` | API | Initiative ID (unique identifier, used as filename) |
| `url` | `string` | API | Full URL to the initiative on the "Have Your Say" portal |
| `short_title` | `string` | API | Short title of the initiative |
| `summary` | `string` | API | Commission-provided summary of the initiative |
| `reference` | `string` | API | EU reference number (e.g. `Ares(2021)4215064`) |
| `type_of_act` | `string` | API | Human-readable act type (e.g. "Regulation", "Directive") |
| `type_of_act_code` | `string` | API | Machine-readable act type code (e.g. `PROP_REG`, `DEL_REG_DRAFT`) |
| `department` | `string` | API | Responsible DG (e.g. `AGRI`, `ENER`, `ENV`) |
| `status` | `string` | API | Initiative status: `ADOPTED`, `OPEN`, `SUSPENDED`, `ABANDONED`, etc. |
| `stage` | `string` | API | Current stage: `PLANNING`, `INITIATIVE`, `OPC`, `ADOPTION_WORKFLOW`, etc. |
| `topics` | `string[]` | API | Topic classifications (e.g. `["Agriculture"]`, `["Energy", "Climate"]`) |
| `policy_areas` | `string[]` | API | Policy area classifications |
| `published_date` | `string` | API | Publication date (`YYYY/MM/DD HH:MM:SS` format) |
| `last_cached_at` | `string` | Scraper | ISO 8601 timestamp of when this file was last fetched/updated |

### Derived Top-Level Fields

These fields are added by analysis and merge scripts throughout the pipeline.

| Field | Type | Set by | Description |
|---|---|---|---|
| `documents_before_feedback` | `Document[]` | `initiative_stats.py` | Documents from publications up to and including the first one that received feedback |
| `documents_after_feedback` | `Document[]` | `initiative_stats.py` | Documents from the final post-feedback publication (empty list if none exist) |
| `middle_feedback` | `Feedback[]` | `initiative_stats.py` | All feedback from publications between the first feedback pub and the final pub |
| `before_feedback_summary` | `string` | `build_unit_summaries.py` | Concatenation (`\n\n`-joined) of all `summary` fields from `documents_before_feedback` |
| `after_feedback_summary` | `string` | `build_unit_summaries.py` | Concatenation of all `summary` fields from `documents_after_feedback` |
| `change_summary` | `string` | `merge_change_summaries.py` | LLM-generated description of substantive changes between before and after documents |
| `diff` | `string` | `merge_change_summaries.py` | Unified diff (`difflib.unified_diff`) between `before_feedback_summary` and `after_feedback_summary` |
| `cluster_policy_summary` | `object` | `merge_cluster_feedback_summaries.py` | Titled summary of the initiative's policy documents (see [Cluster Summary Entry](#cluster-summary-entry)) |
| `cluster_summaries` | `Record<string, object>` | `merge_cluster_feedback_summaries.py` | Per-cluster aggregate summaries keyed by cluster label (see [Cluster Summaries Schema](#cluster-summaries-schema)) |

### Publication Fields

Each initiative contains an ordered array of publications.

| Field | Type | Description |
|---|---|---|
| `publication_id` | `int` | Unique publication identifier |
| `type` | `string` | Publication type code (e.g. `CFE_IMPACT_ASSESS`, `OPC_LAUNCHED`, `PROP_REG`, `DEL_REG_DRAFT`) |
| `section_label` | `string` | Human-readable label (e.g. "Call for evidence", "Public consultation", "Commission adoption") |
| `reference` | `string` | EU reference number |
| `published_date` | `string` | Publication date (`YYYY/MM/DD HH:MM:SS`) |
| `adoption_date` | `string\|null` | Adoption date, if applicable |
| `planned_period` | `string` | Planned publication period |
| `feedback_end_date` | `string\|null` | When the feedback period closes |
| `feedback_period_weeks` | `int` | Duration of the feedback period in weeks |
| `feedback_status` | `string` | `OPEN` or `CLOSED` |
| `total_feedback` | `int` | Number of feedback items on this publication |
| `documents` | `Document[]` | Publication documents (see below) |
| `feedback` | `Feedback[]` | Feedback items submitted on this publication (see below) |

### Document Fields

Documents are attached to publications and represent Commission-published files.

| Field | Type | Source | Description |
|---|---|---|---|
| `label` | `string` | API | Display label for the document |
| `title` | `string` | API | Document title |
| `filename` | `string` | API | Original filename |
| `download_url` | `string` | API | Full URL to download the document |
| `reference` | `string` | API | EU reference number |
| `doc_type` | `string` | API | Document type: `MAIN`, `ANNEX`, etc. |
| `category` | `string` | API | Document category |
| `pages` | `int\|null` | Extraction | Number of pages (PDF only) |
| `size_bytes` | `int` | Extraction | File size in bytes |
| `extracted_text` | `string` | Extraction | Full text extracted from the file (markdown for PDFs) |
| `summary` | `string` | `merge_summaries.py` | LLM-generated summary of the document |
| `extracted_text_without_ocr` | `string` | `merge_ocr_results.py` | Original extracted text before OCR replacement (only present if OCR was applied) |
| `repair_method` | `string` | Repair script | Method used to repair extraction (if applicable) |
| `repair_old_error` | `string` | Repair script | Previous extraction error that was repaired |

### Feedback Fields

Feedback items are public comments submitted by respondents.

| Field | Type | Source | Description |
|---|---|---|---|
| `id` | `int` | API | Unique feedback identifier |
| `url` | `string` | API | Full URL to the feedback on the portal |
| `date` | `string` | API | Submission date (`YYYY/MM/DD HH:MM:SS`) |
| `feedback_text` | `string\|null` | API | Free-text comment. May be null if the respondent only attached documents. |
| `feedback_text_original` | `string` | API | Original feedback text before any API-level translation |
| `language` | `string` | API | ISO 639-1 language code (e.g. `EN`, `DE`, `FR`) |
| `user_type` | `string` | API | Respondent type code (see [Respondent Types](#respondent-types)) |
| `country` | `string` | API | ISO 3166-1 alpha-3 country code (e.g. `SWE`, `DEU`, `FRA`) |
| `company_size` | `string\|null` | API | Company size: `MICRO`, `SMALL`, `MEDIUM`, `LARGE`, or null |
| `organization` | `string\|null` | API | Organization name |
| `first_name` | `string\|null` | API | Respondent's first name |
| `surname` | `string\|null` | API | Respondent's surname |
| `status` | `string` | API | Publication status: `PUBLISHED`, `WITHDRAWN`, etc. |
| `publication` | `string` | API | Name of the publication this feedback is on |
| `tr_number` | `string` | API | EU Transparency Register number (if registered) |
| `combined_feedback_summary` | `string` | `build_unit_summaries.py` | `feedback_text` + all attachment `summary` fields, concatenated with `\n\n` |
| `cluster_feedback_summary` | `object` | `merge_cluster_feedback_summaries.py` | Titled summary for this feedback item from the cluster summarization pipeline (see [Cluster Summary Entry](#cluster-summary-entry)) |
| `nuclear_stance` | `string\|null` | `classify_initiative_and_feedback.py` | Nuclear energy stance: `SUPPORT`, `OPPOSE`, `NEUTRAL`, or `DOES NOT MENTION` |
| `nuclear_stance_reasoning` | `string\|null` | `classify_initiative_and_feedback.py` | LLM reasoning for the nuclear stance classification |
| `attachments` | `Attachment[]` | API + Extraction | Attached documents (see below) |

### Attachment Fields

Attachments are files uploaded by respondents alongside their feedback.

| Field | Type | Source | Description |
|---|---|---|---|
| `id` | `int` | API | Unique attachment identifier |
| `filename` | `string` | API | Original filename |
| `document_id` | `string` | API | EU document identifier |
| `download_url` | `string` | API | Full URL to download the attachment |
| `pages` | `int\|null` | Extraction | Number of pages (PDF only) |
| `size_bytes` | `int` | Extraction | File size in bytes |
| `extracted_text` | `string` | Extraction | Full text from the file. If OCR was applied, this is the OCR text. If translation was applied, this is the translated text. Always contains the "best available" version. |
| `summary` | `string` | `merge_summaries.py` | LLM-generated summary of the attachment |
| `extracted_text_without_ocr` | `string` | `merge_ocr_results.py` | Original text before OCR replacement. Only present when OCR was applied. |
| `extracted_text_before_translation` | `string` | `merge_translations.py` | Original non-English text before translation. Only present when translation was applied. |
| `extracted_text_error` | `string` | Extraction | Error message if text extraction failed |
| `repair_method` | `string` | Repair script | Method used to repair extraction (if applicable) |
| `repair_old_error` | `string` | Repair script | Previous extraction error that was repaired |

### Text Field Layering

The `extracted_text` field on attachments acts as a stack. Each enrichment step replaces `extracted_text` and preserves the previous value:

```
Original extraction  -->  extracted_text
     |
     v  (OCR applied)
OCR text             -->  extracted_text
Original extraction  -->  extracted_text_without_ocr
     |
     v  (Translation applied)
Translated text      -->  extracted_text
OCR text             -->  extracted_text_before_translation
Original extraction  -->  extracted_text_without_ocr
```

This means `extracted_text` always contains the most-processed version, while the `_without_ocr` and `_before_translation` fields provide access to earlier versions for auditing.

---

## Cluster Summaries Schema

**Location:** `cluster_summaries` field on initiative JSON (set by `merge_cluster_feedback_summaries.py`)

This field is a dictionary keyed by cluster label. Each entry contains a titled summary and the number of feedback items in that cluster.

### Structure

```json
{
  "cluster_summaries": {
    "0": {
      "title": "Support for Expanding the Scheme",
      "summary": "Multiple respondents from agricultural organizations and consumer groups...",
      "feedback_count": 7,
      "rewrites": {
        "reddit": {
          "title": "Stakeholders want the school food scheme expanded",
          "body": "Most feedback supports expanding the EU school food program. Farm groups and consumer orgs agree the budget should increase and cover more schools."
        }
      }
    },
    "1": {
      "title": "Concerns About Implementation Costs",
      "summary": "Several respondents raised concerns about the administrative burden...",
      "feedback_count": 4
    },
    "1.1": {
      "title": "Member State Administrative Burden",
      "summary": "A subset of respondents specifically highlighted...",
      "feedback_count": 2
    },
    "-1:503089": {
      "title": "Oatly's Position on Plant-Based Options",
      "summary": "Oatly recommends including plant-based alternatives...",
      "feedback_count": 1
    }
  }
}
```

### Cluster Label Types

| Label Pattern | Type | Description |
|---|---|---|
| `"0"`, `"1"`, `"2"`, ... | Regular cluster | Top-level cluster assigned by the clustering algorithm |
| `"0.1"`, `"0.2"`, `"1.1.3"` | Sub-cluster | Hierarchical sub-cluster from recursive sub-clustering. Depth encoded by dots (e.g. `"3.1.2"` = cluster 3, sub-cluster 1, sub-sub-cluster 2). |
| `"-1:503089"` | Noise cluster | Unclustered feedback item. The number after the colon is the feedback ID. Each noise item gets its own entry. |

### Cluster Summary Entry

Each entry in `cluster_summaries` (and also `cluster_policy_summary` and `cluster_feedback_summary`) follows this structure:

| Field | Type | Description |
|---|---|---|
| `title` | `string` | Short descriptive title (first line of LLM output) |
| `summary` | `string` | Full summary text (up to 10 paragraphs) |
| `feedback_count` | `int` | Number of feedback items in this cluster (only on `cluster_summaries` entries) |
| `rewrites` | `Record<string, RewriteEntry>` | Optional. Rewritten versions of the summary in different formats. Accumulated additively across format runs. |

### Rewrite Entry

When present in the `rewrites` dict:

| Field | Type | Description |
|---|---|---|
| `title` | `string` | Short punchy title in the target format (max 15 words for "reddit" format) |
| `body` | `string` | Concise rewrite (2-4 sentences for "reddit" format, optionally with bullet points) |

The `rewrites` field is only present on `cluster_summaries` entries (not on `cluster_policy_summary` or `cluster_feedback_summary`). Multiple format merges accumulate additively -- running a "reddit" rewrite then a hypothetical "twitter" rewrite would produce:

```json
"rewrites": {
  "reddit": { "title": "...", "body": "..." },
  "twitter": { "title": "...", "body": "..." }
}
```

---

## Cluster Rewrite Output Schema

**Path:** `data/cluster_rewrites/<format>/<scheme>/{id}.json`

Output of `rewrite_cluster_summaries.py`. One file per initiative per format per scheme.

```json
{
  "initiative_id": 1008,
  "format": "reddit",
  "cluster_rewrites": {
    "0": {
      "title": "Industry overwhelmingly supports the Commission's proposal",
      "body": "Nearly all respondents back the proposal. Business groups say it reduces regulatory fragmentation. Environmental groups support it but want stronger enforcement mechanisms."
    },
    "1": {
      "title": "Small businesses worry about compliance costs",
      "body": "SMEs and micro-enterprises flag the disproportionate burden of new reporting requirements. Several request a phased rollout or size-based exemptions."
    },
    "-1:1213": {
      "title": "German consumer org pushes for labeling reform",
      "body": "The Federation of German Consumer Organisations argues current labeling is misleading and proposes mandatory front-of-pack nutrition scoring."
    }
  }
}
```

### Fields

| Field | Type | Description |
|---|---|---|
| `initiative_id` | `int` | Initiative ID |
| `format` | `string` | Rewrite format name (e.g. `"reddit"`) |
| `cluster_rewrites` | `Record<string, RewriteEntry>` | Rewritten summaries keyed by cluster label. Keys match the labels in `cluster_summaries`. |

Each `RewriteEntry`:

| Field | Type | Description |
|---|---|---|
| `title` | `string` | Short title in the target format style |
| `body` | `string` | Concise rewrite in the target format style |

### Available Formats

| Format | Identity | Output Style |
|---|---|---|
| `reddit` | "Sharp policy analyst who writes like a top Reddit commenter" | Max 15-word title + 2-4 sentences + optional 3-5 bullet points. Strips formality, meta-commentary, and hedging. |

---

## Clustering Output Schema

**Path:** `data/clustering/<scheme>/{id}_{algo}_{model}_{params}.json`

Output of `cluster_all_initiatives.py`. Each file is a deep copy of the initiative JSON (from `unit_summaries/`) with clustering metadata and assignments added at the top level.

### Added Fields

```json
{
  "feedback_hash": "a1b2c3d4e5f6...",
  "cluster_model": "google/embeddinggemma-300m",
  "cluster_algorithm": "agglomerative",
  "cluster_params": {
    "distance_threshold": 0.75,
    "linkage": "average",
    "max_cluster_size": 20,
    "max_depth": 3,
    "sub_cluster_scale": 0.75
  },
  "cluster_n_clusters": 12,
  "cluster_noise_count": 3,
  "cluster_silhouette": 0.342,
  "cluster_assignments": {
    "503089": "0",
    "503090": "0",
    "503091": "1",
    "503092": "1.1",
    "503093": "1.2",
    "503094": "-1",
    "503095": "2"
  }
}
```

| Field | Type | Description |
|---|---|---|
| `feedback_hash` | `string` | SHA-256 hash of sorted feedback texts. Used for cache validation in embeddings. |
| `cluster_model` | `string` | Sentence embedding model name (e.g. `google/embeddinggemma-300m`) |
| `cluster_algorithm` | `string` | Clustering algorithm: `agglomerative` or `hdbscan` |
| `cluster_params` | `object` | Algorithm-specific parameters (see below) |
| `cluster_n_clusters` | `int` | Total number of clusters found (excluding noise) |
| `cluster_noise_count` | `int` | Number of feedback items assigned to noise (`-1`) |
| `cluster_silhouette` | `float\|null` | Silhouette score (-1 to 1, higher = better separation). Sampled at 2,000 items for large initiatives. Null if < 2 clusters. |
| `cluster_assignments` | `Record<string, string>` | Maps feedback ID (as string) to cluster label. Labels are hierarchical strings like `"0"`, `"1.2"`, `"-1"`. |

### Cluster Parameters by Algorithm

**Agglomerative:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `distance_threshold` | `float` | `0.96` | Maximum distance for merging clusters |
| `linkage` | `string` | `average` | Linkage method: `ward`, `complete`, `average`, `single` |
| `max_cluster_size` | `int` | `20` | Clusters larger than this are recursively sub-clustered |
| `max_depth` | `int` | `4` | Maximum recursion depth |
| `sub_cluster_scale` | `float` | `0.75` | Distance threshold is multiplied by this at each recursion level |

**HDBSCAN:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `min_cluster_size` | `int` | `5` | Minimum number of points to form a cluster |
| `min_samples` | `int` | `3` | Minimum samples in a neighborhood for core points |
| `max_cluster_size` | `int` | `20` | Clusters larger than this are recursively sub-clustered |
| `max_depth` | `int` | `4` | Maximum recursion depth |
| `sub_cluster_scale` | `float` | `0.75` | Same as agglomerative |

### Cluster Label Hierarchy

Labels use a dot-separated hierarchy from recursive sub-clustering:

```
"0"       -- top-level cluster 0
"0.1"     -- sub-cluster 1 of cluster 0
"0.1.2"   -- sub-sub-cluster 2 of sub-cluster 0.1
"-1"      -- noise (unclustered at top level)
```

Noise points at sub-levels are absorbed back into the parent cluster (they keep the parent's label). Only top-level noise gets the `"-1"` label.

---

## Cluster Summary Output Schema

**Path:** `data/cluster_summaries/<scheme>/{id}.json`

Output of `summarize_clusters.py`. One file per initiative with three levels of summaries.

```json
{
  "initiative_id": 12970,
  "short_title": "EU school fruit, vegetables and milk scheme - review",
  "policy_summary": {
    "title": "Review of the EU School Food Distribution Scheme",
    "summary": "The European Commission is reviewing the EU school fruit, vegetables and milk scheme established under Regulation (EU) 2016/791..."
  },
  "feedback_summaries": {
    "503089": {
      "title": "Oatly's Position on Plant-Based Options",
      "summary": "Oatly AB recommends including plant-based alternatives alongside dairy and fruit products..."
    },
    "503090": {
      "title": "Agricultural Chamber Support for Scheme Continuation",
      "summary": "The Austrian Agricultural Chamber expresses strong support..."
    }
  },
  "cluster_summaries": {
    "0": {
      "title": "Support for Expanding the Scheme",
      "summary": "Seven respondents from agricultural organizations and consumer groups express strong support for expanding the scheme...",
      "feedback_count": 7
    },
    "1": {
      "title": "Concerns About Implementation",
      "summary": "Four respondents highlight administrative and implementation challenges...",
      "feedback_count": 4
    },
    "-1:503089": {
      "title": "Oatly's Position on Plant-Based Options",
      "summary": "Oatly AB recommends including plant-based alternatives...",
      "feedback_count": 1
    }
  }
}
```

### Fields

| Field | Type | Description |
|---|---|---|
| `initiative_id` | `int` | Initiative ID |
| `short_title` | `string` | Initiative short title |
| `policy_summary` | `object\|null` | Titled summary of all publication documents (title + summary). Null if no policy documents. |
| `feedback_summaries` | `Record<string, object>` | Per-feedback-item summaries keyed by feedback ID (as string). Each has `title` and `summary`. |
| `cluster_summaries` | `Record<string, object>` | Per-cluster summaries keyed by cluster label. Each has `title`, `summary`, and `feedback_count`. |

### Three-Phase Summarization

1. **Phase 1 (Policy + Feedback items):** Each policy document section and each feedback item (text + attachments) is independently summarized. Long texts are chunked at sentence boundaries and recursively combined.

2. **Phase 2 (Feedback combining):** Multi-chunk feedback summaries are recursively combined into a single titled summary per feedback item.

3. **Phase 3 (Cluster summaries):** Bottom-up recursive combining by cluster hierarchy. At each depth level, child summaries (feedback item summaries or sub-cluster summaries) are greedily packed within a character budget and combined. Noise items (`-1`) get their individual feedback summary promoted directly as `"-1:{feedback_id}"`.

### Content-Addressed Cache

The file `_cluster_cache.json` in each scheme's cluster summary directory maps SHA-256 hashes of sorted feedback ID sets to their cluster summaries:

```json
{
  "sha256_of_sorted_feedback_ids": {
    "title": "Cached Cluster Summary Title",
    "summary": "Cached cluster summary text..."
  }
}
```

This allows clusters with unchanged membership (same set of feedback IDs) to skip LLM inference on subsequent runs.

---

## Before/After Analysis Schema

**Path:** `data/analysis/before_after/{id}.json`

Output of `initiative_stats.py -o`. Same structure as the initiative JSON with three additional top-level fields. Only initiatives with feedback are included.

### Added Fields

| Field | Type | Description |
|---|---|---|
| `documents_before_feedback` | `Document[]` | Documents from publications up to and including the first one that received feedback. These represent the Commission's position *before* public input. |
| `documents_after_feedback` | `Document[]` | Documents from the final non-`OPC_LAUNCHED` publication that has documents. Empty list when no post-feedback documents exist. Represents the Commission's position *after* public input. |
| `middle_feedback` | `Feedback[]` | All feedback from publications between the first feedback publication and the final document publication. Excludes feedback on the final publication when post-feedback docs exist; includes all feedback otherwise. |

### Timeline Logic

```
Publication 1 (Call for evidence)    -->  documents_before_feedback
  Feedback on Pub 1                  -->  middle_feedback
Publication 2 (Public consultation)
  Feedback on Pub 2                  -->  middle_feedback
Publication 3 (Commission adoption)  -->  documents_after_feedback
```

If no post-feedback publication exists, `documents_after_feedback` is an empty list.

---

## Unit Summaries Schema

**Path:** `data/analysis/unit_summaries/{id}.json`

Output of `build_unit_summaries.py`. Same structure as the summaries output with additional top-level and per-feedback fields that consolidate individual summaries into per-initiative units.

### Added Fields

| Field | Level | Type | Description |
|---|---|---|---|
| `before_feedback_summary` | Top-level | `string` | Concatenation (`\n\n`-joined) of all `summary` fields from `documents_before_feedback` |
| `after_feedback_summary` | Top-level | `string` | Concatenation of all `summary` fields from `documents_after_feedback` |
| `combined_feedback_summary` | Per feedback item in `middle_feedback[]` | `string` | The `feedback_text` + all attachment `summary` fields, concatenated with `\n\n` |

### Example

```json
{
  "id": 12970,
  "before_feedback_summary": "Summary of document 1...\n\nSummary of document 2...",
  "after_feedback_summary": "Summary of final adoption document...",
  "middle_feedback": [
    {
      "id": 503089,
      "feedback_text": "We support the review...",
      "combined_feedback_summary": "We support the review...\n\nOatly's position paper argues for the inclusion of...",
      "attachments": [
        {
          "id": 6276475,
          "summary": "Oatly's position paper argues for the inclusion of..."
        }
      ]
    }
  ]
}
```

---

## Classification Output Schema

**Path:** `data/classification/{id}.json`

Output of `classify_initiative_and_feedback.py`. Same structure as the unit summaries input with nuclear stance classification labels added.

### Added Fields

| Field | Level | Type | Description |
|---|---|---|---|
| `before_feedback_nuclear_stance` | Top-level | `string` | Nuclear stance in documents before feedback |
| `before_feedback_nuclear_stance_reasoning` | Top-level | `string` | LLM reasoning for the before-feedback classification |
| `after_feedback_nuclear_stance` | Top-level | `string` | Nuclear stance in documents after feedback |
| `after_feedback_nuclear_stance_reasoning` | Top-level | `string` | LLM reasoning for the after-feedback classification |
| `before_feedback_nuclear_stance_complex` | Top-level | `string` | Multi-step complex classification result (before) |
| `before_feedback_nuclear_stance_complex_reasoning` | Top-level | `string` | Full reasoning chain for complex classification (before) |
| `after_feedback_nuclear_stance_complex` | Top-level | `string` | Multi-step complex classification result (after) |
| `after_feedback_nuclear_stance_complex_reasoning` | Top-level | `string` | Full reasoning chain for complex classification (after) |
| `nuclear_stance` | Per feedback item | `string` | Nuclear stance of this feedback item |
| `nuclear_stance_reasoning` | Per feedback item | `string` | LLM reasoning for the feedback classification |

### Valid Labels

| Label | Description |
|---|---|
| `SUPPORT` | Explicitly supports nuclear energy inclusion in EU policy |
| `OPPOSE` | Explicitly opposes nuclear energy inclusion |
| `NEUTRAL` | Mentions nuclear energy without taking a clear position |
| `DOES NOT MENTION` | No reference to nuclear energy, nuclear plants, or SMRs |

### Complex Classification Hierarchy

The complex classifier follows a multi-step hierarchy:

1. **Step 0** -- Unit of analysis: aggregate by initiative
2. **Step 1** -- Relevance filter: nuclear + energy context
3. **Step 2** -- Commission stance: +1 inclusion, 0 neutral, -1 exclusion
4. **Step 3** -- Legitimacy logic: Technocratic, Input, Procedural-Institutional
5. **Step 4** -- Dominance rules and output parsing

When multiple classification runs are performed, results are stored as arrays instead of single values.

---

## Webapp Data Schemas

### Initiative Index (`data/webapp/initiative_index.json`)

A single JSON array of `InitiativeSummary` objects. Pre-computed by `build_webapp_index.py`. Deduplicated to remove initiatives sharing identical feedback ID sets.

```json
[
  {
    "id": 12970,
    "short_title": "EU school fruit, vegetables and milk scheme - review",
    "department": "AGRI",
    "stage": "ADOPTION_WORKFLOW",
    "status": "ADOPTED",
    "topics": ["Agriculture"],
    "policy_areas": ["Agriculture and rural development"],
    "published_date": "2021/06/29 00:00:00",
    "last_cached_at": "2026-03-15T14:23:01+00:00",
    "type_of_act": "Regulation",
    "reference": "Ares(2021)4215064",
    "total_feedback": 74,
    "country_counts": {
      "SWE": 3,
      "DEU": 12,
      "FRA": 8,
      "BEL": 5
    },
    "user_type_counts": {
      "COMPANY": 15,
      "BUSINESS_ASSOCIATION": 20,
      "EU_CITIZEN": 10,
      "NGO": 8
    },
    "feedback_timeline": [0, 2, 5, 12, 8, 15, 10, 5, 3, 2, 1, 0, 0, 4, 2, 1, 1, 2, 0, 1],
    "last_feedback_date": "2021-07-27T14:23:01+00:00",
    "has_open_feedback": false,
    "feedback_ids": [503089, 503090, 503091]
  }
]
```

| Field | Type | Description |
|---|---|---|
| `id` | `int` | Initiative ID |
| `short_title` | `string` | Short title |
| `department` | `string` | Responsible DG |
| `stage` | `string` | Current stage |
| `status` | `string` | Initiative status |
| `topics` | `string[]` | Topic classifications |
| `policy_areas` | `string[]` | Policy area classifications |
| `published_date` | `string` | Publication date |
| `last_cached_at` | `string` | ISO 8601 timestamp of last cache |
| `type_of_act` | `string` | Human-readable act type |
| `reference` | `string` | EU reference number |
| `total_feedback` | `int` | Total feedback count (sum across all publications) |
| `country_counts` | `Record<string, int>` | Feedback count by country (ISO 3166-1 alpha-3) |
| `user_type_counts` | `Record<string, int>` | Feedback count by respondent type |
| `feedback_timeline` | `int[]` | 20-bucket histogram of feedback over time |
| `last_feedback_date` | `string` | ISO 8601 date of the most recent feedback |
| `has_open_feedback` | `bool` | Whether any publication still has `feedback_status: "OPEN"` |
| `feedback_ids` | `int[]` | Sorted array of all feedback IDs (used for deduplication) |

### Global Stats (`data/webapp/global_stats.json`)

Aggregate cross-initiative statistics.

```json
{
  "total_initiatives": 2970,
  "total_feedback": 458231,
  "by_country": [["DEU", 85432], ["FRA", 52100], ["BEL", 38201]],
  "by_topic": [["Energy", 125000], ["Environment", 98000], ["Agriculture", 45000]],
  "initiatives_by_topic": [["Energy", 450], ["Environment", 380]],
  "by_user_type": [["EU_CITIZEN", 180000], ["COMPANY", 95000], ["BUSINESS_ASSOCIATION", 72000]],
  "by_department": [["ENER", 125000], ["ENV", 98000], ["AGRI", 45000]],
  "by_stage": [["ADOPTION_WORKFLOW", 1200], ["OPC", 800], ["PLANNING", 500]],
  "top_topics_by_country": {
    "DEU": [["Energy", 25000], ["Environment", 18000]],
    "FRA": [["Agriculture", 12000], ["Energy", 10000]]
  },
  "top_countries_by_topic": {
    "Energy": [["DEU", 25000], ["FRA", 10000]],
    "Environment": [["DEU", 18000], ["BEL", 8000]]
  },
  "feedback_by_month": [["2020-01", 1500], ["2020-02", 1800]],
  "feedback_by_month_by_topic": {
    "months": ["2020-01", "2020-02", "2020-03"],
    "series": {
      "Energy": [500, 600, 450],
      "Environment": [300, 400, 350]
    }
  },
  "feedback_by_month_by_country": {
    "months": ["2020-01", "2020-02", "2020-03"],
    "series": {
      "DEU": [200, 250, 180],
      "FRA": [150, 180, 120]
    }
  }
}
```

| Field | Type | Description |
|---|---|---|
| `total_initiatives` | `int` | Total number of initiatives (after deduplication) |
| `total_feedback` | `int` | Total feedback items across all initiatives |
| `by_country` | `[string, int][]` | Feedback counts by country, sorted descending |
| `by_topic` | `[string, int][]` | Feedback counts by topic, sorted descending |
| `initiatives_by_topic` | `[string, int][]` | Initiative counts by topic, sorted descending |
| `by_user_type` | `[string, int][]` | Feedback counts by respondent type, sorted descending |
| `by_department` | `[string, int][]` | Feedback counts by responsible DG, sorted descending |
| `by_stage` | `[string, int][]` | Initiative counts by stage, sorted descending |
| `top_topics_by_country` | `Record<string, [string, int][]>` | For each country: top 10 topics by feedback count |
| `top_countries_by_topic` | `Record<string, [string, int][]>` | For each topic: top 10 countries by feedback count |
| `feedback_by_month` | `[string, int][]` | Total feedback per month (YYYY-MM), sorted chronologically |
| `feedback_by_month_by_topic` | `TimeSeriesGroup` | Monthly breakdown for top 10 topics |
| `feedback_by_month_by_country` | `TimeSeriesGroup` | Monthly breakdown for top 15 countries |

#### TimeSeriesGroup Structure

```json
{
  "months": ["2020-01", "2020-02", "2020-03"],
  "series": {
    "Energy": [500, 600, 450],
    "Environment": [300, 400, 350]
  }
}
```

| Field | Type | Description |
|---|---|---|
| `months` | `string[]` | Shared month axis (YYYY-MM, sorted chronologically) |
| `series` | `Record<string, int[]>` | Named time series. Each array is aligned with `months`. |

### Country Stats (`data/webapp/country_stats.json`)

Per-country drill-down statistics. A dictionary keyed by ISO 3166-1 alpha-3 country code.

```json
{
  "DEU": {
    "total_feedback": 85432,
    "by_topic": [["Energy", 25000], ["Environment", 18000], ["Digital", 12000]],
    "by_user_type": [["EU_CITIZEN", 40000], ["COMPANY", 20000], ["BUSINESS_ASSOCIATION", 15000]],
    "top_initiatives": [
      {"id": 12970, "short_title": "EU school fruit...", "count": 150},
      {"id": 13500, "short_title": "Energy efficiency...", "count": 120}
    ],
    "recent_feedback": [
      {
        "date": "2026/03/15 10:23:01",
        "user_type": "COMPANY",
        "organization": "Siemens AG",
        "first_name": "Hans",
        "surname": "Mueller",
        "initiative_id": 13500,
        "initiative_title": "Energy efficiency directive revision",
        "feedback_text": "We welcome the Commission's proposal...",
        "url": "https://ec.europa.eu/.../F600123_en",
        "attachments": [
          {"filename": "Siemens_position.pdf", "download_url": "https://..."}
        ]
      }
    ],
    "topic_timeline": {
      "months": ["2020-01", "2020-02"],
      "series": {
        "Energy": [500, 600],
        "Environment": [300, 400]
      }
    }
  }
}
```

| Field | Type | Description |
|---|---|---|
| `total_feedback` | `int` | Total feedback items from this country |
| `by_topic` | `[string, int][]` | Top 20 topics by feedback count, sorted descending |
| `by_user_type` | `[string, int][]` | All respondent types by feedback count, sorted descending |
| `top_initiatives` | `object[]` | Top 20 initiatives by feedback count from this country |
| `top_initiatives[].id` | `int` | Initiative ID |
| `top_initiatives[].short_title` | `string` | Initiative short title |
| `top_initiatives[].count` | `int` | Number of feedback items from this country on this initiative |
| `recent_feedback` | `object[]` | 20 most recent feedback items from this country |
| `recent_feedback[].date` | `string` | Submission date |
| `recent_feedback[].user_type` | `string` | Respondent type code |
| `recent_feedback[].organization` | `string\|null` | Organization name |
| `recent_feedback[].first_name` | `string\|null` | Respondent first name |
| `recent_feedback[].surname` | `string\|null` | Respondent surname |
| `recent_feedback[].initiative_id` | `int` | Initiative ID |
| `recent_feedback[].initiative_title` | `string` | Initiative short title |
| `recent_feedback[].feedback_text` | `string\|null` | First 150 characters of feedback text |
| `recent_feedback[].url` | `string\|null` | URL to feedback on the portal |
| `recent_feedback[].attachments` | `object[]` | Attachment metadata (filename + download URL only) |
| `topic_timeline` | `TimeSeriesGroup` | Monthly feedback for top 5 topics from this country |

### Stripped Initiative Details (`data/webapp/initiative_details/{id}.json`)

Full initiative JSON with large text fields removed from feedback attachments for reduced file size. The following fields are stripped from every attachment in `publications[].feedback[].attachments[]` and `middle_feedback[].attachments[]`:

- `extracted_text`
- `extracted_text_without_ocr`
- `extracted_text_before_translation`

All other fields (summaries, cluster summaries, metadata) are preserved. These files are used by the webapp for detail pages.

---

## Respondent Types

Feedback respondents are classified by the EU portal into these types:

| Code | Display Name | Description |
|---|---|---|
| `EU_CITIZEN` | EU Citizen | Individual EU citizens |
| `COMPANY` | Company | Individual companies |
| `BUSINESS_ASSOCIATION` | Business Association | Industry and trade associations |
| `NGO` | NGO | Non-governmental organizations |
| `TRADE_UNION` | Trade Union | Labour unions |
| `ACADEMIC_RESEARCH_INSTITTUTION` | Academic/Research Institution | Academic and research institutions. Note: the double "T" in `INSTITTUTION` is the original API spelling. |
| `PUBLIC_AUTHORITY` | Public Authority | National or regional government bodies |
| `CONSUMER_ORG` | Consumer Organisation | Consumer advocacy groups |
| `ENVIRONMENTAL_ORGANISATION` | Environmental Organisation | Environmental advocacy groups |
| `NON_EU_CITIZEN` | Non-EU Citizen | Citizens from outside the EU |
| `OTHER` | Other | Other organisations not fitting above categories |

### Webapp Color Mappings

The webapp assigns consistent colors to each respondent type:

| Code | Color | Hex |
|---|---|---|
| `EU_CITIZEN` | Green | `#27ae60` |
| `COMPANY` | Blue | `#2980b9` |
| `BUSINESS_ASSOCIATION` | Dark Blue | `#1a5276` |
| `NGO` | Orange | `#e67e22` |
| `PUBLIC_AUTHORITY` | Red | `#c0392b` |
| `ACADEMIC_RESEARCH_INSTITTUTION` | Purple | `#8e44ad` |
| `TRADE_UNION` | Amber | `#8b6914` |
| `ENVIRONMENTAL_ORGANISATION` | Emerald | `#239b56` |
| `CONSUMER_ORG` | Teal | `#1abc9c` |
| `NON_EU_CITIZEN` | Teal (dark) | `#16a085` |
| `OTHER` | Gray | `#95a5a6` |

---

## Derived Field Lifecycle

Every derived field in the initiative JSON is set by a specific pipeline script and preserved across re-scrapes by the scraper's merge strategy. This table documents the full lifecycle.

### Derived Fields on Documents (`publications[].documents[]`)

| Field | Set by | Preserved when |
|---|---|---|
| `extracted_text` | `scrape_eu_initiative_details.py` (initial extraction), `merge_ocr_results.py` (OCR replacement) | `pages` and `size_bytes` unchanged |
| `extracted_text_without_ocr` | `merge_ocr_results.py` | `pages` and `size_bytes` unchanged |
| `summary` | `merge_summaries.py` (from `summarize_documents.py` output) | `pages` and `size_bytes` unchanged |

### Derived Fields on Feedback Attachments (`publications[].feedback[].attachments[]`)

| Field | Set by | Preserved when |
|---|---|---|
| `extracted_text` | `scrape_eu_initiative_details.py` (initial), `merge_ocr_results.py` (OCR), `merge_translations.py` (translation) | `pages`, `size_bytes`, and `document_id` unchanged |
| `extracted_text_without_ocr` | `merge_ocr_results.py` | `pages`, `size_bytes`, and `document_id` unchanged |
| `extracted_text_before_translation` | `merge_translations.py` | `pages`, `size_bytes`, and `document_id` unchanged |
| `summary` | `merge_summaries.py` (from `summarize_documents.py` output) | `pages`, `size_bytes`, and `document_id` unchanged |

### Derived Fields on Feedback Items (`publications[].feedback[]`)

| Field | Set by | Preserved when |
|---|---|---|
| `combined_feedback_summary` | `build_unit_summaries.py` | Not directly preserved by scraper (lives in `analysis/` output). Regenerated each run. |
| `cluster_feedback_summary` | `merge_cluster_feedback_summaries.py` | `feedback_text` unchanged |
| `nuclear_stance` | `classify_initiative_and_feedback.py` | Not directly merged into `initiative_details/`. Lives in `classification/` output. Merged into detail via `merge_cluster_feedback_summaries.py` indirectly. |
| `nuclear_stance_reasoning` | `classify_initiative_and_feedback.py` | Same as above |

### Derived Fields at Initiative Top Level

| Field | Set by | Preserved across re-scrapes? |
|---|---|---|
| `documents_before_feedback` | `initiative_stats.py` | Not in `initiative_details/`. Lives in `analysis/before_after/`. |
| `documents_after_feedback` | `initiative_stats.py` | Same as above |
| `middle_feedback` | `initiative_stats.py` | Same as above |
| `before_feedback_summary` | `build_unit_summaries.py` | Not in `initiative_details/`. Lives in `analysis/unit_summaries/`. |
| `after_feedback_summary` | `build_unit_summaries.py` | Same as above |
| `change_summary` | `merge_change_summaries.py` | Yes -- preserved as top-level derived field |
| `diff` | `merge_change_summaries.py` | Yes -- preserved as top-level derived field |
| `cluster_policy_summary` | `merge_cluster_feedback_summaries.py` | Yes -- preserved as top-level derived field |
| `cluster_summaries` | `merge_cluster_feedback_summaries.py` | Yes -- preserved as top-level derived field |

### Merge Script Summary

| Script | Target | Fields Set | Merge Type |
|---|---|---|---|
| `merge_ocr_results.py` | `initiative_details/{id}.json` | `extracted_text` (replaced), `extracted_text_without_ocr` (preserved original) | In-place mutation on documents and attachments |
| `merge_translations.py` | `initiative_details/{id}.json` | `extracted_text` (replaced), `extracted_text_before_translation` (preserved original) | In-place mutation on attachments |
| `merge_summaries.py` | `initiative_details/{id}.json` | `summary` on documents and attachments | In-place mutation, matched by `doc_id` and `(feedback_id, attachment_id)` |
| `merge_change_summaries.py` | `initiative_details/{id}.json` | `change_summary`, `diff` at top level | In-place mutation |
| `merge_cluster_feedback_summaries.py` | `initiative_details/{id}.json` | `cluster_feedback_summary` on feedback items, `cluster_policy_summary` and `cluster_summaries` at top level | In-place mutation |
| `merge_cluster_rewrites.py` | `initiative_details/{id}.json` | `cluster_summaries[label].rewrites[format]` | In-place additive mutation (multiple formats accumulate) |

### Re-Scrape Preservation Rules

When the scraper (`scrape_eu_initiative_details.py`) re-fetches a stale initiative, it applies these merge rules:

1. **Documents:** If `pages` and `size_bytes` are unchanged, all derived fields (`extracted_text`, `extracted_text_without_ocr`, `summary`) are copied from the old version.
2. **Feedback attachments:** If `pages`, `size_bytes`, and `document_id` are unchanged, all derived fields are copied.
3. **Feedback items:** If `feedback_text` is unchanged, `cluster_feedback_summary` is copied.
4. **Top-level derived fields:** `change_summary`, `diff`, `cluster_policy_summary`, and `cluster_summaries` are always preserved from the old record.
5. **Terminal initiatives:** Initiatives with status `SUSPENDED` or `ABANDONED`, and `ADOPTION_WORKFLOW` initiatives with all-closed feedback, are never re-checked regardless of `--max-age`.
6. **Corrupt files:** If a cached JSON file is corrupt (truncated write, encoding error), the scraper logs a warning and re-fetches from scratch.
