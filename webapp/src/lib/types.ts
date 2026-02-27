export interface Attachment {
  id: number;
  filename: string;
  document_id: string;
  download_url: string;
  pages: number | null;
  size_bytes: number;
  extracted_text: string;
  summary?: string;
  extracted_text_without_ocr?: string;
  extracted_text_before_translation?: string;
  extracted_text_error?: string;
  repair_method?: string;
  repair_old_error?: string;
}

export interface Feedback {
  id: number;
  user_type: string;
  date: string;
  status: string;
  publication: string;
  url: string;
  tr_number: string;
  first_name: string | null;
  surname: string | null;
  organization: string | null;
  country: string | null;
  language: string | null;
  company_size: string | null;
  feedback_text: string | null;
  feedback_text_original?: string;
  attachments: Attachment[];
  combined_feedback_summary?: string;
  nuclear_stance?: string | null;
  nuclear_stance_reasoning?: string | null;
}

export interface Document {
  label: string;
  title: string;
  filename: string;
  download_url: string;
  reference: string;
  doc_type: string;
  category: string;
  pages: number | null;
  size_bytes: number;
  extracted_text: string;
  summary?: string;
  extracted_text_without_ocr?: string;
  extracted_text_before_translation?: string;
  repair_method?: string;
  repair_old_error?: string;
}

export interface Publication {
  publication_id: number;
  type: string;
  section_label: string;
  reference: string;
  published_date: string;
  adoption_date: string | null;
  planned_period: string;
  feedback_end_date: string | null;
  feedback_period_weeks: number;
  feedback_status: string;
  total_feedback: number;
  documents: Document[];
  feedback: Feedback[];
}

export interface Initiative {
  id: number;
  url: string;
  short_title: string;
  summary?: string | null;
  reference: string;
  type_of_act: string;
  type_of_act_code?: string;
  department: string;
  status: string;
  stage: string;
  published_date: string;
  topics: string[];
  policy_areas: string[];
  publications: Publication[];
  last_cached_at?: string;
  documents_before_feedback?: Document[];
  documents_after_feedback?: Document[];
  middle_feedback?: Feedback[];
  before_feedback_summary?: string;
  after_feedback_summary?: string;
  change_summary?: string;
}

export interface InitiativeSummary {
  id: number;
  short_title: string;
  department: string;
  stage: string;
  status: string;
  topics: string[];
  policy_areas: string[];
  published_date: string;
  last_cached_at: string;
  type_of_act: string;
  reference: string;
  total_feedback: number;
  country_counts: Record<string, number>;
  user_type_counts: Record<string, number>;
  feedback_timeline: number[];
  last_feedback_date: string;
  has_open_feedback: boolean;
  feedback_ids: number[];
}

export type SortOption =
  | "most_discussed"
  | "recently_discussed"
  | "newest";

export const USER_TYPE_COLORS: Record<string, { border: string; bg: string; text: string }> = {
  EU_CITIZEN:                        { border: "border-green-500",  bg: "bg-green-100",  text: "text-green-800" },
  COMPANY:                           { border: "border-blue-500",   bg: "bg-blue-100",   text: "text-blue-800" },
  BUSINESS_ASSOCIATION:              { border: "border-blue-800",   bg: "bg-blue-50",    text: "text-blue-900" },
  NGO:                               { border: "border-orange-500", bg: "bg-orange-100", text: "text-orange-800" },
  PUBLIC_AUTHORITY:                   { border: "border-red-500",    bg: "bg-red-100",    text: "text-red-800" },
  ACADEMIC_RESEARCH_INSTITTUTION:    { border: "border-purple-500", bg: "bg-purple-100", text: "text-purple-800" },
  TRADE_UNION:                       { border: "border-amber-700",  bg: "bg-amber-100",  text: "text-amber-900" },
  ENVIRONMENTAL_ORGANISATION:        { border: "border-emerald-500",bg: "bg-emerald-100",text: "text-emerald-800" },
  CONSUMER_ORG:                      { border: "border-pink-500",   bg: "bg-pink-100",   text: "text-pink-800" },
  NON_EU_CITIZEN:                    { border: "border-teal-500",   bg: "bg-teal-100",   text: "text-teal-800" },
  OTHER:                             { border: "border-gray-400",   bg: "bg-gray-100",   text: "text-gray-700" },
};

export function getUserTypeColor(userType: string) {
  return USER_TYPE_COLORS[userType] ?? USER_TYPE_COLORS.OTHER;
}

export function formatUserType(userType: string): string {
  return userType
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .replace("Ngo", "NGO")
    .replace("Eu ", "EU ");
}

// --- Clustering types ---

export interface ClusterData {
  cluster_model: string;
  cluster_algorithm: string;
  cluster_params: Record<string, unknown>;
  cluster_n_clusters: number;
  cluster_noise_count: number;
  cluster_silhouette: number | null;
  cluster_assignments: Record<string, string | number>;
}

export interface ClusterNode {
  label: string;
  directItems: Feedback[];
  children: ClusterNode[];
  allItems: Feedback[];
}

/** Hex colors for user type bar chart segments */
export const USER_TYPE_BAR_COLORS: Record<string, string> = {
  EU_CITIZEN: "#27ae60",
  COMPANY: "#2980b9",
  BUSINESS_ASSOCIATION: "#1a5276",
  NGO: "#e67e22",
  PUBLIC_AUTHORITY: "#c0392b",
  ACADEMIC_RESEARCH_INSTITTUTION: "#8e44ad",
  TRADE_UNION: "#8b6914",
  ENVIRONMENTAL_ORGANISATION: "#239b56",
  CONSUMER_ORG: "#1abc9c",
  NON_EU_CITIZEN: "#16a085",
  OTHER: "#95a5a6",
};

/** Palette for country bar segments */
export const COUNTRY_BAR_COLORS = [
  "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
  "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
  "#86bcb6", "#8cd17d", "#b6992d", "#499894", "#d37295",
  "#a0cbe8", "#ffbe7d", "#d4a6c8", "#fabfd2", "#d7b5a6",
];

/** Short labels for user types in bar chart segments */
export const USER_TYPE_SHORT: Record<string, string> = {
  EU_CITIZEN: "Citizen",
  COMPANY: "Co.",
  BUSINESS_ASSOCIATION: "Biz",
  NGO: "NGO",
  PUBLIC_AUTHORITY: "Gov",
  ACADEMIC_RESEARCH_INSTITTUTION: "Acad",
  TRADE_UNION: "Union",
  ENVIRONMENTAL_ORGANISATION: "Env",
  CONSUMER_ORG: "Cons",
  NON_EU_CITIZEN: "Non-EU",
  OTHER: "?",
};

/** ISO 3166-1 alpha-3 to alpha-2 mapping */
const ISO3_TO_ISO2: Record<string, string> = {
  ABW:"AW",AFG:"AF",AGO:"AO",AIA:"AI",ALA:"AX",ALB:"AL",AND:"AD",ARE:"AE",
  ARG:"AR",ARM:"AM",ASM:"AS",ATA:"AQ",ATF:"TF",ATG:"AG",AUS:"AU",AUT:"AT",
  AZE:"AZ",BDI:"BI",BEL:"BE",BEN:"BJ",BES:"BQ",BFA:"BF",BGD:"BD",BGR:"BG",
  BHR:"BH",BHS:"BS",BIH:"BA",BLM:"BL",BLR:"BY",BLZ:"BZ",BMU:"BM",BOL:"BO",
  BRA:"BR",BRB:"BB",BRN:"BN",BTN:"BT",BVT:"BV",BWA:"BW",CAF:"CF",CAN:"CA",
  CCK:"CC",CHE:"CH",CHL:"CL",CHN:"CN",CIV:"CI",CMR:"CM",COD:"CD",COG:"CG",
  COK:"CK",COL:"CO",COM:"KM",CPV:"CV",CRI:"CR",CUB:"CU",CUW:"CW",CXR:"CX",
  CYM:"KY",CYP:"CY",CZE:"CZ",DEU:"DE",DJI:"DJ",DMA:"DM",DNK:"DK",DOM:"DO",
  DZA:"DZ",ECU:"EC",EGY:"EG",ERI:"ER",ESH:"EH",ESP:"ES",EST:"EE",ETH:"ET",
  FIN:"FI",FJI:"FJ",FLK:"FK",FRA:"FR",FRO:"FO",GAB:"GA",GBR:"GB",GEO:"GE",
  GGY:"GG",GHA:"GH",GIB:"GI",GIN:"GN",GLP:"GP",GMB:"GM",GNB:"GW",GNQ:"GQ",
  GRC:"GR",GRD:"GD",GRL:"GL",GTM:"GT",GUF:"GF",GUM:"GU",GUY:"GY",HKG:"HK",
  HMD:"HM",HND:"HN",HRV:"HR",HTI:"HT",HUN:"HU",IDN:"ID",IMN:"IM",IND:"IN",
  IOT:"IO",IRL:"IE",IRN:"IR",IRQ:"IQ",ISL:"IS",ISR:"IL",ITA:"IT",JAM:"JM",
  JEY:"JE",JOR:"JO",JPN:"JP",KAZ:"KZ",KEN:"KE",KGZ:"KG",KHM:"KH",KIR:"KI",
  KNA:"KN",KOR:"KR",KWT:"KW",LAO:"LA",LBN:"LB",LBR:"LR",LBY:"LY",LCA:"LC",
  LIE:"LI",LKA:"LK",LSO:"LS",LTU:"LT",LUX:"LU",LVA:"LV",MAC:"MO",MAF:"MF",
  MAR:"MA",MCO:"MC",MDA:"MD",MDG:"MG",MDV:"MV",MEX:"MX",MKD:"MK",MLI:"ML",
  MLT:"MT",MMR:"MM",MNE:"ME",MNG:"MN",MNP:"MP",MOZ:"MZ",MRT:"MR",MSR:"MS",
  MTQ:"MQ",MUS:"MU",MWI:"MW",MYS:"MY",MYT:"YT",NAM:"NA",NCL:"NC",NER:"NE",
  NFK:"NF",NGA:"NG",NIC:"NI",NLD:"NL",NOR:"NO",NPL:"NP",NRU:"NR",NZL:"NZ",
  OMN:"OM",PAK:"PK",PAN:"PA",PCN:"PN",PER:"PE",PHL:"PH",PLW:"PW",POL:"PL",
  PRI:"PR",PRK:"KP",PRT:"PT",PRY:"PY",PSE:"PS",PYF:"PF",QAT:"QA",REU:"RE",
  ROU:"RO",RUS:"RU",RWA:"RW",SAU:"SA",SDN:"SD",SEN:"SN",SGP:"SG",SGS:"GS",
  SHN:"SH",SJM:"SJ",SLB:"SB",SLE:"SL",SLV:"SV",SMR:"SM",SOM:"SO",SPM:"PM",
  SRB:"RS",SSD:"SS",STP:"ST",SUR:"SR",SVK:"SK",SVN:"SI",SWE:"SE",SWZ:"SZ",
  SXM:"SX",SYC:"SC",SYR:"SY",TCA:"TC",TCD:"TD",TGO:"TG",THA:"TH",TJK:"TJ",
  TKM:"TM",TLS:"TL",TON:"TO",TTO:"TT",TUN:"TN",TUR:"TR",TUV:"TV",TWN:"TW",
  TZA:"TZ",UGA:"UG",UKR:"UA",UMI:"UM",UNK:"XK",URY:"UY",USA:"US",UZB:"UZ",
  VAT:"VA",VEN:"VE",VGB:"VG",VIR:"VI",VNM:"VN",VUT:"VU",WLF:"WF",WSM:"WS",
  YEM:"YE",ZAF:"ZA",ZMB:"ZM",ZWE:"ZW",
};

/** Convert country code (2-letter or 3-letter) to flag emoji */
export function countryToFlag(code: string): string {
  if (!code) return "";
  let alpha2 = code.toUpperCase();
  if (alpha2.length === 3) {
    alpha2 = ISO3_TO_ISO2[alpha2] || "";
  }
  if (alpha2.length !== 2) return code;
  const cp1 = 0x1f1e6 + (alpha2.charCodeAt(0) - 65);
  const cp2 = 0x1f1e6 + (alpha2.charCodeAt(1) - 65);
  return String.fromCodePoint(cp1, cp2);
}

/** Build cluster tree from flat assignments + feedback lookup */
export function buildClusterTree(
  assignments: Record<string, string | number>,
  feedbackLookup: Map<string, Feedback>
): ClusterNode[] {
  // Group items by their exact cluster label
  const labelItems = new Map<string, Feedback[]>();
  for (const [fbId, label] of Object.entries(assignments)) {
    const key = String(label);
    const fb = feedbackLookup.get(fbId);
    if (!fb) continue;
    if (!labelItems.has(key)) labelItems.set(key, []);
    labelItems.get(key)!.push(fb);
  }

  function buildNode(prefix: string): ClusterNode {
    const directItems = labelItems.get(prefix) || [];
    const children: ClusterNode[] = [];

    // Find immediate child prefixes (prefix.X)
    const childSet = new Set<string>();
    for (const label of labelItems.keys()) {
      if (label === prefix) continue;
      if (label.startsWith(prefix + ".")) {
        const rest = label.slice(prefix.length + 1);
        const nextSeg = rest.split(".")[0];
        childSet.add(prefix + "." + nextSeg);
      }
    }
    const sortedChildren = Array.from(childSet).sort((a, b) =>
      a.localeCompare(b, undefined, { numeric: true })
    );
    for (const childPrefix of sortedChildren) {
      children.push(buildNode(childPrefix));
    }

    // Collect all items from this node and all descendants
    const allItems = [...directItems];
    for (const child of children) {
      allItems.push(...child.allItems);
    }

    return { label: prefix, directItems, children, allItems };
  }

  // Find top-level labels (first segment before any dot)
  const topSet = new Set<string>();
  for (const label of labelItems.keys()) {
    topSet.add(label.split(".")[0]);
  }

  const clusters: ClusterNode[] = [];
  const sortedTop = Array.from(topSet).sort((a, b) =>
    a.localeCompare(b, undefined, { numeric: true })
  );
  for (const topLabel of sortedTop) {
    clusters.push(buildNode(topLabel));
  }

  // Default sort: largest first
  clusters.sort((a, b) => b.allItems.length - a.allItems.length);
  return clusters;
}

/** Compute country and user type stats for a set of feedback items */
export function computeClusterStats(items: Feedback[]) {
  const typeCounts = new Map<string, number>();
  const countryCounts = new Map<string, number>();

  for (const fb of items) {
    const ut = fb.user_type || "OTHER";
    typeCounts.set(ut, (typeCounts.get(ut) || 0) + 1);
    if (fb.country) {
      countryCounts.set(fb.country, (countryCounts.get(fb.country) || 0) + 1);
    }
  }

  return {
    sortedTypes: Array.from(typeCounts.entries()).sort((a, b) => b[1] - a[1]),
    sortedCountries: Array.from(countryCounts.entries()).sort(
      (a, b) => b[1] - a[1]
    ),
  };
}
