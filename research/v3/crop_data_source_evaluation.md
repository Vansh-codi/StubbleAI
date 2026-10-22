# V3 Crop-Data Source Evaluation

**Purpose:** Evaluate crop-data options for the six-state StubbleAI project and deployment.
**States:** Assam, Bihar, Haryana, Madhya Pradesh, Punjab, Uttar Pradesh.
**Status:** Source evaluation in progress. No crop dataset approved or integrated.
**Last reviewed:** 18 October 2026.

## 1. Decision principles

- Evaluate sources before downloading or integrating data.
- Separate annual crop statistics, spatial land cover/crop maps, and crop calendars; they are not interchangeable.
- Verify coverage for all six states, relevant years, district boundaries, spatial/temporal resolution, missingness, accuracy, provenance, access conditions, licensing and redistribution terms.
- Record retrieval date, version, source URL, processing steps, file hashes and validation results for any dataset eventually selected.
- Treat crop features as project/deployment candidates. Do not claim a predictive improvement until a controlled evaluation supports it.
- Record the reference year of every crop calendar wherever it is used. A decades-old calendar, including the DES 1998-99 calendar, must be labelled historical and must not be presented as current without explicit supporting evidence.
- Do not alter the frozen V2 baseline or its paper results.

## 2. Candidate-source comparison

### A. Government district crop statistics

**Candidate:** Directorate of Economics and Statistics, Department of Agriculture and Farmers Welfare, Government of India.

- District Area, Production & Yield query report:
  https://data.desagri.gov.in/website/apy-query-report-web
- Major-contributing-district report:
  https://data.desagri.gov.in/website/crops-report-major-contributing-district-web

**Potential use:** Annual or seasonal crop area, production and yield for descriptive dashboard context, state/district summaries, and carefully designed lagged features.

**Limitations and checks required:**
- Verify that all six states, relevant districts, crop types, seasons and years are selectable and downloadable.
- Establish whether reports provide complete district coverage or only selected/contributing districts.
- Check reporting delays, revisions, missing years, district-name/code changes and unit consistency.
- Confirm the portal's access conditions and reuse/redistribution terms before bundling data with the project.
- Annual or seasonal statistics do not directly provide weekly harvest timing.

**Initial assessment:** High-priority candidate for district-level descriptive context. Predictive use is not approved yet.

### B. NASA MODIS annual land cover

**Candidate:** MCD12Q1 Version 6.1, annual global land-cover product at 500 m.

- NASA product information:
  https://modis-land.gsfc.nasa.gov/landcover.html
- NASA product record:
  https://data.nasa.gov/dataset/modis-terraaqua-land-cover-type-yearly-l3-global-500m-sin-grid-v061-fac3a
- Product guide:
  https://landweb.modaps.eosdis.nasa.gov/data/userguide/MCD12Q1Userguide04042018.pdf

**Potential use:** Approximate district cropland fraction or land-cover context after spatial aggregation to the project's district boundaries.

**Limitations and checks required:**
- This is a land-cover classification, not a verified crop-species map or a direct harvest-date dataset.
- Check the exact classification layer and cropland class definition before deriving features.
- Assess mixed pixels, classification quality, temporal availability and changes in district boundaries.
- Confirm the applicable product-version documentation, attribution requirements and reuse terms.
- Annual land cover may be too static to represent short-term crop residue or harvest activity.

**Initial assessment:** Candidate for a spatial cropland-context feature, not a substitute for crop calendars or harvest observations.

### C. Government crop-calendar tables

**Candidate:** Government of India's *Agricultural Statistics at a Glance 2024–25*, which includes crop-calendar tables.

- Source document:
  https://nccd.gov.in/uploads/Agricultural_Statisitcs_at_a_Glance_2024_25_11da3cd37d.pdf

**Potential use:** Broad state-level sowing/harvesting windows to support interpretation, seasonal dashboard context and hypotheses for future features.

**Limitations and checks required:**
- Verify which of the six states and which crops are represented in the relevant tables.
- Determine whether dates are state-level, regional or district-specific.
- Check the reference period, table footnotes, units and whether ranges represent typical rather than observed dates.
- Do not turn broad monthly windows into false district-level daily precision.
- Verify reuse terms before republishing extracted tables.

**Initial assessment:** Candidate for coarse seasonal context. Not approved as a district-day harvest label.

### D. Crop-type maps and satellite-derived crop classifications

**Candidate class:** Published or government-supported crop-type maps, including products derived from Sentinel-2 or other satellite observations.

**Potential use:** Crop-specific area fractions or dominant crop types, if a trustworthy product covers the pilot states and relevant seasons.

**Status:** No specific product approved.

**Checks required before selection:**
- Identify a concrete dataset and its provider, version, publication and documentation.
- Verify coverage across all six states and relevant crop seasons/years.
- Establish native resolution, class definitions, validation methodology and reported accuracy.
- Check confusion between crops, fallow land, mixed pixels and changing seasonal conditions.
- Verify licensing, attribution, access restrictions and permission to redistribute derived outputs.
- Confirm that the feature could be generated consistently for the intended deployment period.

**Initial assessment:** Potentially valuable, but not ready for download or model integration until a specific source passes review.

## 3. Coverage and validation checklist

For each shortlisted dataset, record the following before approving it:

| Check | Required evidence |
|---|---|
| Six-state coverage | Explicit coverage check for Assam, Bihar, Haryana, Madhya Pradesh, Punjab and Uttar Pradesh |
| Spatial join | Matching method to current district IDs/boundaries; unmatched and ambiguous records counted |
| Temporal coverage | Years, seasons and observation/reporting dates documented |
| Resolution | Native spatial and temporal resolution distinguished from any derived resolution |
| Missingness | Missing districts, crops, years and observations quantified |
| Quality | Source accuracy/validation evidence and relevant limitations |
| Provenance | Provider, exact product/version, retrieval date and URL |
| Licensing | Reuse, redistribution, attribution and commercial/deployment conditions checked |
| Reproducibility | Processing steps, parameters, output schema and file hashes recorded |
| Model suitability | Feature availability at prediction time established to prevent leakage |

## 4. Proposed evaluation sequence

1. Inspect the Government district crop-statistics portal for the six-state crop/season/year coverage and export options.
2. Inspect the official crop-calendar tables for relevant crop and state coverage; record their actual level of granularity.
3. Evaluate whether MODIS annual cropland context adds useful spatial information at district level.
4. Identify a concrete crop-type mapping product only if crop-specific information is needed and a credible source is available.
5. Compare provenance, coverage, temporal usefulness, quality and licensing before choosing a dataset.
6. Only after approval, download a limited sample and audit it before full acquisition.
7. If a feature is approved, define a separate controlled ablation protocol with fixed splits, a baseline comparison, pooled and per-state metrics, and saved predictions.

## 5. Feature-use policy

- **Annual crop statistics:** initially dashboard/context only; predictive use requires a leakage-safe, justified experiment.
- **Annual cropland fraction:** candidate spatial context feature; validate resolution, boundary aggregation and usefulness.
- **Crop type:** candidate feature only after source-specific accuracy and coverage checks.
- **Harvest calendar/phenology:** potentially relevant to short-term fire risk, but only use the temporal and geographic precision the source actually supports.

No source should be described as integrated, validated, or improving the model until the corresponding checks and experiment are complete.

## 6. Current decision

**No crop dataset has been selected or downloaded under this evaluation.**

The next action is source coverage and licensing review. The current V3 fire/weather dataset remains a seasonal October–November 2023–2025 pilot; crop-data availability does not remove the need for broader temporal coverage and separate model validation.
## 7. Verified source findings and unresolved checks

### A. Government district crop statistics — DES

**Verified from the official interface:**
- The APY query report exposes state, district, crop, season, year range and report format.
- The major-contributing-district report has an "All District" option as well as a district-count option. Prefer an all-district export for coverage evaluation; do not assume a top-contributing-district report is complete.
- Official source pages:
  - https://data.desagri.gov.in/website/apy-query-report-web
  - https://data.desagri.gov.in/website/crops-report-major-contributing-district-web

**Reuse terms found:** The DES website policy says reproduction is allowed after obtaining permission, with source acknowledgement; the policy excludes third-party material from that permission. Do not assume DES extracts can be redistributed with a deployed application. Check the exact dataset/report terms and obtain written permission where required.
- Policy: https://desagri.gov.in/website-policies/

**Still unresolved:** Actual year/crop/season coverage and completeness across all six states; export behavior; missing district-years; units and revisions; and whether the applicable dataset has a separate licence or reuse notice.

**Required district reconciliation:** Preserve source district/state labels and any official codes. Match to the project's district identifiers using state plus verified LGD/district codes where available. Audit name aliases, spelling changes, renamed/split districts, unmatched rows and ambiguous matches. Never join on district name alone.

### B. NASA MODIS MCD12Q1 Version 6.1

**Verified from the official NASA catalog:**
- Annual global land-cover classification at 500 m, with several classification schemes and quality-related layers.
- The catalog warns that classification training sites may no longer reflect land-cover changes, particularly after 2021; use recent-year outputs cautiously.
- Product record: https://data.nasa.gov/dataset/modis-terraaqua-land-cover-type-yearly-l3-global-500m-sin-grid-v061-fac3a
- User guide: https://lpdaac.usgs.gov/documents/1409/MCD12_User_Guide_V61.pdf

**Licence evidence:** The NASA catalog lists the licence as U.S. government works. NASA Earthdata's general data-use guidance says unrestricted NASA-led mission data are generally CC0 unless a product is marked with a restriction or licence; users should acknowledge NASA as the source. Check the actual product record and applicable notices at the time of use rather than relying on a generic rule alone.
- NASA data-use guidance: https://www.earthdata.nasa.gov/engage/open-data-services-software/data-use-policy

**Suitability:** Candidate for broad land-cover/cropland context, not crop-species identification or harvest-date labels. A 500 m land-cover class is not proof of a specific crop. Before use, inspect the class legend, quality layers, available years, boundary aggregation method and local validity.

**Study-period vintage risk:** StubbleAI's current fire/weather study rows cover 2023-2025, whereas NASA warns that MODIS classification training sites may not reflect land-cover changes, particularly after 2021. This creates a potential vintage mismatch: land-cover labels applied to 2023-2025 rows may not reflect intervening urbanization or crop-pattern changes. This risk is not proof that the product is inaccurate locally; assess it against the product documentation and, where feasible, independent recent evidence before using it as a feature.

**Still unresolved:** No six-state extraction, district-level aggregation or local accuracy audit has been performed for StubbleAI. No MODIS data has been approved or integrated by this evaluation.

### C. Government crop-calendar sources

**DES calendar warning:** The official DES SDDS page states that its crop calendar is very old (1998-99) and that a renewal methodology was being finalized. Treat this version as historical, not current.
- Official page: https://desagri.gov.in/divisions-cell/special-data-dissemination-standards-sdds/

**Separate newer candidate:** *Agricultural Statistics at a Glance 2024-25* lists an appendix titled "Crop Calendar of Major Crops."
- Publication: https://desagri.gov.in/wp-content/uploads/2024/11/Agricultural-Statisitcs-at-a-Glance-2024-25_%E0%A4%95%E0%A5%83%E0%A4%B7%E0%A4%BF-%E0%A4%B8%E0%A4%BE%E0%A4%82%E0%A4%96%E0%A5%8D%E0%A4%AF%E0%A4%BF%E0%A4%95%E0%A5%80-%E0%A4%8F%E0%A4%95-%E0%A4%9D%E0%A4%B2%E0%A4%95-2024%E2%80%9325.pdf

The existence of this appendix does not by itself establish district-level dates or complete coverage of the six pilot states. Its actual table structure, reference period, crop coverage, geographic granularity and reuse conditions must be inspected before use. Do not silently substitute the newer publication for the old DES calendar without recording which source and table are being used.

### D. Current source-selection position

- **First priority to evaluate:** DES all-district APY statistics for descriptive dashboard context, subject to coverage, district-code reconciliation and reuse clearance. Annual/seasonal DES statistics are dashboard-context data only at this stage: their temporal resolution does not support within-season fire-risk timing, so they are not currently a model-feature candidate.
- **Second candidate:** MODIS MCD12Q1 for broad spatial cropland context, subject to local quality and aggregation checks.
- **Calendar candidate:** Inspect the 2024-25 publication's actual crop-calendar appendix; do not treat DES's 1998-99 calendar as current.
- **Crop-type maps:** No concrete product is approved; identify and audit a specific product before considering download.

No candidate has been downloaded or integrated under this evaluation. The next step is to inspect the candidate tables/metadata and document the six-state coverage and reuse decision before acquiring any dataset.
