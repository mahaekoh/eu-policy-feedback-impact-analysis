# viewers/ — Standalone Data Viewers

Two standalone HTML files for interactively browsing pipeline output in a browser. No build step, no dependencies, no server required -- just open the HTML file directly.

---

## viewer.html

Interactive browser for per-initiative JSON files from `data/scrape/initiative_details/`.

### How to use

1. Open `viewer.html` in any modern browser.
2. Use the file picker to load one or more initiative JSON files (e.g. `data/scrape/initiative_details/12096.json`).

### Features

- **Tabbed navigation**: Before Feedback, After Feedback, Feedback, Publications.
- **Document display**: download links to original documents, feedback portal links for each feedback item.
- **Expandable text blocks**: summaries, extracted text, pre-translation originals (`extracted_text_before_translation`), pre-OCR originals (`extracted_text_without_ocr`).
- **User type color coding**: visual distinction by feedback submitter type.
- **Feedback filtering**: filter by user type, search text, or hide empty feedback.
- **Attachment display**: download links and extracted text for each feedback attachment.
- **Infinite scroll**: chunked rendering for initiatives with large feedback lists.

---

## feedback-viewer.html

Interactive browser for clustered feedback results from `data/clustering/<scheme>/`.

### How to use

1. Open `feedback-viewer.html` in any modern browser.
2. Use the file picker to load one or more clustering JSON files (e.g. `data/clustering/agglomerative_google_embeddinggemma-300m_.../12096.json`).

### Features

- **Cluster metadata**: displays algorithm, embedding model, parameters, and silhouette score.
- **Nested cluster tree**: expandable/collapsible hierarchy for sub-clustered feedback.
- **Distribution bars**: per-cluster country and user-type breakdowns.
- **Feedback search**: text search across all feedback items within clusters.
- **Sorting**: sort clusters by size or alphabetically.
- **Individual feedback items**: full feedback text with attachments and extracted text.
