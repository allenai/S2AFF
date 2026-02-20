use pyo3::prelude::*;
use rayon::prelude::*;
use regex::Regex;
use rkyv::{Archive, Serialize as RkyvSerialize};
use rkyv::option::ArchivedOption;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};

use memmap2::Mmap;
use once_cell::sync::Lazy;
use std::str;

static RE_BASIC: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[^,a-zA-Z0-9\s]+").expect("invalid RE_BASIC"));
static RE_APOSTROPHE_S: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(\w+)'s").expect("invalid RE_APOSTROPHE_S"));
static RE_WHITESPACE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\s+").expect("invalid RE_WHITESPACE"));

static STOPWORDS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    let words = [
        "it", "its", "is", "are", "a", "an", "the", "and", "as", "of", "at", "by", "for",
        "with", "into", "from", "in",
    ];
    words.into_iter().collect()
});

static INVERTED_ABBREV: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut m = HashMap::new();
    m.insert("acad", "academy");
    m.insert("amer", "american");
    m.insert("assoc", "association");
    m.insert("coll", "college");
    m.insert("co", "company");
    m.insert("corp", "corporation");
    m.insert("commun", "communication");
    m.insert("dept", "department");
    m.insert("div", "division");
    m.insert("dr", "doctor");
    m.insert("electr", "electrical");
    m.insert("eng", "engineering");
    m.insert("europ", "european");
    m.insert("exec", "executive");
    m.insert("fac", "faculty");
    m.insert("found", "foundation");
    m.insert("gov", "government");
    m.insert("inc", "incorporated");
    m.insert("info", "information");
    m.insert("inst", "institute");
    m.insert("intern", "international");
    m.insert("lab", "laboratory");
    m.insert("libr", "library");
    m.insert("nat", "national");
    m.insert("med", "medicine");
    m.insert("mech", "mechanical");
    m.insert("prof", "professor");
    m.insert("progr", "program");
    m.insert("psychol", "psychology");
    m.insert("sch", "school");
    m.insert("sci", "science");
    m.insert("soc", "society");
    m.insert("technol", "technology");
    m.insert("univ", "university");
    m.insert("tech", "technology");
    m
});

static TEXT_UNIDECODE: Lazy<Vec<String>> = Lazy::new(|| {
    let data = include_bytes!("../assets/text_unidecode.bin");
    let text = str::from_utf8(data).expect("text_unidecode.bin must be valid UTF-8");
    text.split('\0').map(|s| s.to_string()).collect()
});

const TIE_SCORE_SCALE: f64 = 1e12;

fn text_unidecode(input: &str) -> String {
    let replaces = &*TEXT_UNIDECODE;
    let mut out = String::new();
    for ch in input.chars() {
        let codepoint = ch as usize;
        if codepoint == 0 {
            out.push('\0');
            continue;
        }
        if let Some(rep) = replaces.get(codepoint - 1) {
            out.push_str(rep);
        }
    }
    out
}

fn rust_log_enabled() -> bool {
    env::var("S2AFF_RUST_LOG")
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

fn rust_log(message: &str) {
    if rust_log_enabled() {
        eprintln!("[s2aff_rust] {message}");
    }
}

fn rust_log_elapsed(label: &str, start: Instant) {
    if rust_log_enabled() {
        let elapsed = start.elapsed().as_secs_f64();
        eprintln!("[s2aff_rust] {label} ({elapsed:.2}s)");
    }
}

fn fix_text(input: &str) -> String {
    if input.is_empty() {
        return "".to_string();
    }
    let mut s = text_unidecode(input);
    s = s
        .replace("#TAB#", "")
        .replace(".", "")
        .replace(" & ", " and ")
        .replace("&", "n");
    s = RE_APOSTROPHE_S
        .replace_all(&s, |caps: &regex::Captures| {
            format!("{}s", &caps[1])
        })
        .to_string();
    s = RE_BASIC.replace_all(&s, " ").to_string();
    s = RE_WHITESPACE.replace_all(&s, " ").to_string();
    s.trim().to_string()
}

fn normalize_geoname_id(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(s) => {
            let t = s.trim();
            if t.is_empty() {
                None
            } else {
                Some(t.to_string())
            }
        }
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(i.to_string())
            } else if let Some(f) = n.as_f64() {
                if (f.fract() - 0.0).abs() < f64::EPSILON {
                    Some((f as i64).to_string())
                } else {
                    Some(f.to_string())
                }
            } else {
                Some(n.to_string())
            }
        }
        _ => Some(value.to_string()),
    }
}

#[derive(Clone, Debug)]
struct Address {
    city: Option<String>,
    state: Option<String>,
    state_code: Option<String>,
    country_geonames_id: Option<String>,
    country_code: Option<String>,
    country_name: Option<String>,
}

#[derive(Clone, Debug)]
struct Relationship {
    id: String,
    rel_type: String,
}

#[derive(Clone, Debug)]
struct RorRecord {
    id: String,
    name: String,
    aliases: Vec<String>,
    labels: Vec<String>,
    acronyms: Vec<String>,
    addresses: Vec<Address>,
    relationships: Vec<Relationship>,
    works_count: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Archive, RkyvSerialize)]
struct RorEntryLite {
    names: Vec<String>,
    acronyms: Vec<String>,
    city: Vec<String>,
    state: Vec<String>,
    country: Vec<String>,
    works_count: i64,
}

#[pyclass]
struct CandidateMatch {
    #[pyo3(get)]
    field_matches: Vec<Vec<String>>, // list per field
    #[pyo3(get)]
    field_matched_text_lens: Vec<usize>,
    #[pyo3(get)]
    field_any_text_in_query: Vec<bool>,
    #[pyo3(get)]
    matched_split_set: Vec<String>,
}

const RUST_CACHE_VERSION: u32 = 8;

#[derive(Serialize, Deserialize, Archive, RkyvSerialize)]
struct RustIndexState {
    version: u32,
    use_prob_weights: bool,
    word_multiplier: f64,
    word_multiplier_is_log: bool,
    max_intersection_denominator: bool,
    ns: Vec<usize>,
    insert_early_candidates_ind: Option<usize>,
    reinsert_cutoff_frac: f64,
    score_based_early_cutoff: f64,
    typed_keys: Vec<String>,
    id_to_key: Vec<String>,
    key_to_id: HashMap<String, usize>,
    word_index: HashMap<String, Vec<u32>>,
    word_lengths: Vec<f64>,
    ngram_index: HashMap<String, Vec<u32>>,
    ngram_lengths: Vec<f64>,
    address_index: HashMap<String, Vec<u32>>,
    ror_ids: Vec<String>,
    ror_id_to_int: HashMap<String, u32>,
    idf_lookup: HashMap<String, f64>,
    idf_lookup_min: f64,
    ror_entries_lite: HashMap<String, RorEntryLite>,
    typed_id_to_ror_id_int: Vec<u32>,
}

struct ArchivedIndex {
    #[allow(dead_code)] // keep mmap alive for archived pointers
    mmap: Arc<Mmap>,
    root: *const ArchivedRustIndexState,
}

fn tie_score_key(score: f64) -> i64 {
    let scaled = score * TIE_SCORE_SCALE;
    if scaled >= 0.0 {
        (scaled + 0.5).floor() as i64
    } else {
        (scaled - 0.5).ceil() as i64
    }
}

impl ArchivedIndex {
    fn root(&self) -> &ArchivedRustIndexState {
        unsafe { &*self.root }
    }
}

unsafe impl Send for ArchivedIndex {}
unsafe impl Sync for ArchivedIndex {}

enum IndexStore {
    Owned(RustIndexState),
    Archived(Arc<ArchivedIndex>),
}

fn archived_vec_string_to_owned(vec: &rkyv::Archived<Vec<String>>) -> Vec<String> {
    vec.iter().map(|s| s.as_str().to_string()).collect()
}

fn archived_entry_to_owned(entry: &rkyv::Archived<RorEntryLite>) -> RorEntryLite {
    RorEntryLite {
        names: archived_vec_string_to_owned(&entry.names),
        acronyms: archived_vec_string_to_owned(&entry.acronyms),
        city: archived_vec_string_to_owned(&entry.city),
        state: archived_vec_string_to_owned(&entry.state),
        country: archived_vec_string_to_owned(&entry.country),
        works_count: entry.works_count,
    }
}

fn v2_display_name(rec: &Value) -> Option<String> {
    if let Some(names) = rec.get("names").and_then(|v| v.as_array()) {
        for n in names {
            if let Some(types) = n.get("types").and_then(|v| v.as_array()) {
                let has_display = types.iter().any(|t| t.as_str() == Some("ror_display"));
                if has_display {
                    if let Some(value) = n.get("value").and_then(|v| v.as_str()) {
                        return Some(value.to_string());
                    }
                }
            }
        }
        for n in names {
            if let Some(value) = n.get("value").and_then(|v| v.as_str()) {
                return Some(value.to_string());
            }
        }
    }
    None
}

fn title_case_type(value: &str) -> String {
    let mut chars = value.chars();
    match chars.next() {
        None => "".to_string(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

fn parse_ror_record(rec: &Value) -> Option<RorRecord> {
    let is_v2 = rec.get("names").is_some();
    let id = rec.get("id")?.as_str()?.to_string();

    if is_v2 {
        let name = v2_display_name(rec)?;
        let mut aliases = Vec::new();
        let mut labels = Vec::new();
        let mut acronyms = Vec::new();
        if let Some(names) = rec.get("names").and_then(|v| v.as_array()) {
            for n in names {
                let value = match n.get("value").and_then(|v| v.as_str()) {
                    Some(v) => v.to_string(),
                    None => continue,
                };
                let types = n
                    .get("types")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|t| t.as_str())
                            .collect::<Vec<&str>>()
                    })
                    .unwrap_or_default();
                if types.iter().any(|t| *t == "alias") {
                    aliases.push(value.clone());
                }
                if types.iter().any(|t| *t == "label") {
                    labels.push(value.clone());
                }
                if types.iter().any(|t| *t == "acronym") {
                    acronyms.push(value.clone());
                }
            }
        }

        let mut addresses = Vec::new();
        if let Some(locations) = rec.get("locations").and_then(|v| v.as_array()) {
            if let Some(loc0) = locations.get(0) {
                let g = loc0.get("geonames_details").unwrap_or(&Value::Null);
                let addr = Address {
                    city: g.get("name").and_then(|v| v.as_str()).map(|s| s.to_string()),
                    state: g
                        .get("country_subdivision_name")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string()),
                    state_code: g
                        .get("country_subdivision_code")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string()),
                    country_geonames_id: None,
                    country_code: g.get("country_code").and_then(|v| v.as_str()).map(|s| s.to_string()),
                    country_name: g
                        .get("country_name")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string()),
                };
                addresses.push(addr);
            }
        }

        let mut relationships = Vec::new();
        if let Some(rels) = rec.get("relationships").and_then(|v| v.as_array()) {
            for rel in rels {
                let rid = match rel.get("id").and_then(|v| v.as_str()) {
                    Some(v) => v.to_string(),
                    None => continue,
                };
                let rtype = rel
                    .get("type")
                    .and_then(|v| v.as_str())
                    .map(title_case_type)
                    .unwrap_or_else(|| "".to_string());
                relationships.push(Relationship { id: rid, rel_type: rtype });
            }
        }

        Some(RorRecord {
            id,
            name,
            aliases,
            labels,
            acronyms,
            addresses,
            relationships,
            works_count: 0,
        })
    } else {
        let name = rec.get("name")?.as_str()?.to_string();
        let aliases = rec
            .get("aliases")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
            .unwrap_or_else(Vec::new);
        let labels = rec
            .get("labels")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.get("label").and_then(|l| l.as_str()).map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_else(Vec::new);
        let acronyms = rec
            .get("acronyms")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
            .unwrap_or_else(Vec::new);

        let mut addresses = Vec::new();
        if let Some(addrs) = rec.get("addresses").and_then(|v| v.as_array()) {
            for addr in addrs {
                let city = addr.get("city").and_then(|v| v.as_str()).map(|s| s.to_string());
                let state = addr.get("state").and_then(|v| v.as_str()).map(|s| s.to_string());
                let state_code = addr
                    .get("state_code")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let country_geonames_id = addr.get("country_geonames_id").cloned();
                let addr = Address {
                    city,
                    state,
                    state_code,
                    country_geonames_id: country_geonames_id
                        .as_ref()
                        .and_then(|v| normalize_geoname_id(v)),
                    country_code: None,
                    country_name: None,
                };
                addresses.push(addr);
            }
        }

        let mut relationships = Vec::new();
        if let Some(rels) = rec.get("relationships").and_then(|v| v.as_array()) {
            for rel in rels {
                let rid = match rel.get("id").and_then(|v| v.as_str()) {
                    Some(v) => v.to_string(),
                    None => continue,
                };
                let rtype = rel
                    .get("type")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "".to_string());
                relationships.push(Relationship { id: rid, rel_type: rtype });
            }
        }

        Some(RorRecord {
            id,
            name,
            aliases,
            labels,
            acronyms,
            addresses,
            relationships,
            works_count: 0,
        })
    }
}

fn load_ror_records_fast(path: &str) -> Result<Vec<RorRecord>, String> {
    let mut bytes = std::fs::read(path).map_err(|e| format!("failed to open ror_data: {e}"))?;
    let raw: Vec<Value> = simd_json::serde::from_slice(&mut bytes)
        .map_err(|e| format!("failed to parse ror_data: {e}"))?;
    let total = raw.len();
    let mut records: Vec<RorRecord> = Vec::with_capacity(total);
    for (idx, rec) in raw.iter().enumerate() {
        if let Some(parsed) = parse_ror_record(rec) {
            if !parsed.name.is_empty() {
                records.push(parsed);
            }
        }
        if rust_log_enabled() && (idx + 1) % 10_000 == 0 {
            rust_log(&format!("parse_ror_record progress {}/{}", idx + 1, total));
        }
    }
    Ok(records)
}

fn load_country_info(
    path: &str,
) -> Result<(HashMap<String, Vec<String>>, HashMap<String, Vec<String>>), String> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(path)
        .map_err(|e| format!("failed to read country info: {e}"))?;

    let headers = rdr
        .headers()
        .map_err(|e| format!("failed to read country info headers: {e}"))?
        .clone();

    let idx_geoname = headers
        .iter()
        .position(|h| h == "geonameid")
        .ok_or("country_info missing geonameid")?;
    let idx_country = headers
        .iter()
        .position(|h| h == "Country")
        .ok_or("country_info missing Country")?;
    let idx_iso = headers
        .iter()
        .position(|h| h == "ISO")
        .ok_or("country_info missing ISO")?;
    let idx_iso3 = headers
        .iter()
        .position(|h| h == "ISO3")
        .ok_or("country_info missing ISO3")?;
    let idx_fips = headers
        .iter()
        .position(|h| h == "fips")
        .ok_or("country_info missing fips")?;

    let mut country_codes_dict = HashMap::new();
    let mut country_by_iso2 = HashMap::new();

    for result in rdr.records() {
        let record = result.map_err(|e| format!("failed to read country info row: {e}"))?;
        let geoname = record.get(idx_geoname).unwrap_or("").trim();
        let country = record.get(idx_country).unwrap_or("").trim();
        let iso = record.get(idx_iso).unwrap_or("").trim();
        let iso3 = record.get(idx_iso3).unwrap_or("").trim();
        let fips = record.get(idx_fips).unwrap_or("").trim();

        let geoid = if geoname.is_empty() { None } else { Some(geoname.to_string()) };
        let entry = vec![
            country.to_string(),
            iso.to_string(),
            iso3.to_string(),
            fips.to_string(),
        ];
        if let Some(geoid) = geoid {
            country_codes_dict.insert(geoid, entry.clone());
        }
        if !iso.is_empty() {
            country_by_iso2.insert(iso.to_uppercase(), entry);
        }
    }

    Ok((country_codes_dict, country_by_iso2))
}

fn load_works_counts(path: &str) -> Result<HashMap<String, i64>, String> {
    let mut rdr = csv::ReaderBuilder::new()
        .from_path(path)
        .map_err(|e| format!("failed to read works_counts: {e}"))?;
    let headers = rdr
        .headers()
        .map_err(|e| format!("failed to read works_counts headers: {e}"))?
        .clone();
    let idx_ror = headers
        .iter()
        .position(|h| h == "ror")
        .ok_or("works_counts missing ror column")?;
    let idx_wc = headers
        .iter()
        .position(|h| h == "works_count")
        .ok_or("works_counts missing works_count column")?;

    let mut map = HashMap::new();
    for result in rdr.records() {
        let record = result.map_err(|e| format!("failed to read works_counts row: {e}"))?;
        let ror = record.get(idx_ror).unwrap_or("").to_string();
        if ror.is_empty() {
            continue;
        }
        let wc = record
            .get(idx_wc)
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(0);
        map.insert(ror, wc);
    }
    Ok(map)
}

fn apply_ror_edits(records: &mut [RorRecord], id_to_idx: &HashMap<String, usize>, path: &str) -> Result<(), String> {
    let file = File::open(path).map_err(|e| format!("failed to open ror_edits: {e}"))?;
    let reader = BufReader::new(file);
    for line in reader.lines() {
        let line = line.map_err(|e| format!("failed to read ror_edits line: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let val: Value = serde_json::from_str(&line)
            .map_err(|e| format!("failed to parse ror_edits json: {e}"))?;
        let ror_id = match val.get("ror_id").and_then(|v| v.as_str()) {
            Some(v) => v,
            None => continue,
        };
        let action = val.get("action").and_then(|v| v.as_str()).unwrap_or("");
        let key = val.get("key").and_then(|v| v.as_str()).unwrap_or("");
        let value = val.get("value").and_then(|v| v.as_str()).unwrap_or("");
        let idx = match id_to_idx.get(ror_id) {
            Some(i) => *i,
            None => continue,
        };

        let target_list: Option<&mut Vec<String>> = match key {
            "aliases" => Some(&mut records[idx].aliases),
            "acronyms" => Some(&mut records[idx].acronyms),
            "labels" => Some(&mut records[idx].labels),
            _ => None,
        };

        if let Some(list) = target_list {
            if action == "append" {
                if !list.iter().any(|i| i == value) {
                    list.push(value.to_string());
                }
            } else if action == "remove" {
                list.retain(|i| i != value);
            }
        }
    }
    Ok(())
}

fn strip_parens(name: &str) -> &str {
    match name.split('(').next() {
        Some(v) => v.trim(),
        None => name.trim(),
    }
}

fn build_ror_entry_lite(
    rec: &RorRecord,
    country_codes_dict: &HashMap<String, Vec<String>>,
    country_by_iso2: &HashMap<String, Vec<String>>,
) -> RorEntryLite {
    let official_name = fix_text(&rec.name).to_lowercase().replace(',', "");
    let aliases: Vec<String> = rec
        .aliases
        .iter()
        .map(|i| fix_text(i).to_lowercase().replace(',', ""))
        .collect();
    let labels: Vec<String> = rec
        .labels
        .iter()
        .map(|i| fix_text(i).to_lowercase().replace(',', ""))
        .collect();
    let acronyms: Vec<String> = rec
        .acronyms
        .iter()
        .map(|i| fix_text(i).to_lowercase().replace(',', ""))
        .collect();

    let mut names_set = HashSet::new();
    names_set.insert(official_name);
    for a in aliases {
        names_set.insert(a);
    }
    for l in labels {
        names_set.insert(l);
    }
    let names: Vec<String> = names_set.into_iter().collect();

    let mut city = Vec::new();
    let mut state = Vec::new();
    let mut country = Vec::new();

    if let Some(addr0) = rec.addresses.get(0) {
        let city_raw = addr0.city.clone().unwrap_or_default();
        let state_raw = addr0.state.clone().unwrap_or_default();
        let city_fix = fix_text(&city_raw).to_lowercase().replace(',', "");
        let state_fix = fix_text(&state_raw).to_lowercase().replace(',', "");

        let mut state_code: Vec<String> = Vec::new();
        if let Some(sc) = addr0.state_code.clone() {
            state_code = sc
                .split('-')
                .filter(|i| i.chars().all(|c| c.is_ascii_alphabetic()))
                .map(|i| fix_text(i).to_lowercase().replace(',', ""))
                .collect();
        }

        let mut country_and_codes: Option<Vec<String>> = None;
        if let Some(geoid) = addr0.country_geonames_id.clone() {
            if let Some(entry) = country_codes_dict.get(&geoid) {
                country_and_codes = Some(entry.clone());
            }
        }
        if country_and_codes.is_none() {
            if let Some(code) = addr0.country_code.clone() {
                let iso2 = code.to_uppercase();
                if let Some(entry) = country_by_iso2.get(&iso2) {
                    country_and_codes = Some(entry.clone());
                }
            }
        }

        let mut country_and_codes = country_and_codes.unwrap_or_else(|| vec!["".to_string(); 4]);

        let mut seen = HashSet::new();
        for i in country_and_codes.iter_mut() {
            *i = fix_text(i).to_lowercase().replace(',', "");
        }
        let extras_fixed: Vec<String> = if country_and_codes.iter().any(|c| c == "china") {
            vec!["pr".to_string(), "prc".to_string()]
        } else {
            Vec::new()
        };

        let mut country_elements = Vec::new();
        for i in country_and_codes.into_iter().chain(extras_fixed.into_iter()) {
            if i.is_empty() {
                continue;
            }
            if seen.insert(i.clone()) {
                country_elements.push(i);
            }
        }

        let mut state_info = Vec::new();
        state_info.push(state_fix.clone());
        state_info.extend(state_code);
        state_info.retain(|i| !seen.contains(i));
        state_info.retain(|i| i != &city_fix);

        city.push(city_fix);
        state.extend(state_info);
        country.extend(country_elements);
    }

    RorEntryLite {
        names,
        acronyms,
        city,
        state,
        country,
        works_count: rec.works_count,
    }
}

fn build_query_ngrams(q: &str, max_ngram_len: usize) -> Vec<String> {
    let q_split: Vec<&str> = q.split_whitespace().collect();
    if q_split.is_empty() {
        return Vec::new();
    }
    let longest = std::cmp::min(max_ngram_len, q_split.len());
    let mut ngrams = Vec::new();
    for i in (1..=longest).rev() {
        for window in q_split.windows(i) {
            let mut ngram = window.join(" ");
            ngram = ngram.replace('|', "\\|");
            ngrams.push(ngram);
        }
    }
    ngrams
}

fn build_regex_from_ngrams(ngrams: &[String], use_word_boundaries: bool) -> Option<Regex> {
    if ngrams.is_empty() {
        return None;
    }
    let parts: Vec<String> = ngrams
        .iter()
        .map(|ng| {
            if use_word_boundaries {
                format!("\\b{}\\b", ng)
            } else {
                ng.clone()
            }
        })
        .collect();
    let pattern = parts.join("|");
    Regex::new(&pattern).ok()
}

fn find_query_ngrams_in_text_precompiled(
    t: &str,
    regex: &Option<Regex>,
    len_filter: usize,
    remove_stopwords: bool,
) -> Vec<String> {
    let regex = match regex {
        Some(r) => r,
        None => return Vec::new(),
    };
    if t.is_empty() {
        return Vec::new();
    }
    let mut matches = Vec::new();
    for m in regex.find_iter(t) {
        let span_len = m.end().saturating_sub(m.start());
        if span_len <= len_filter {
            continue;
        }
        let text = &t[m.start()..m.end()];
        if remove_stopwords && STOPWORDS.contains(text) {
            continue;
        }
        matches.push(text.to_string());
    }
    matches
}

fn find_query_ngrams_in_text(
    q: &str,
    t: &str,
    len_filter: usize,
    remove_stopwords: bool,
    use_word_boundaries: bool,
    max_ngram_len: usize,
) -> Vec<String> {
    if q.is_empty() || t.is_empty() {
        return Vec::new();
    }
    let ngrams = build_query_ngrams(q, max_ngram_len);
    let regex = build_regex_from_ngrams(&ngrams, use_word_boundaries);
    find_query_ngrams_in_text_precompiled(t, &regex, len_filter, remove_stopwords)
}

fn matched_len_from_tokens(tokens: &HashSet<String>) -> usize {
    let mut total = 0usize;
    let mut count = 0usize;
    for t in tokens {
        total += t.len();
        count += 1;
    }
    if count > 0 {
        total += count - 1;
    }
    total
}

fn get_text_ngrams(
    text: &str,
    idf_lookup: Option<&HashMap<String, f64>>,
    idf_min: f64,
    ns: &[usize],
) -> (Vec<String>, Vec<f64>) {
    if text.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let words: Vec<&str> = text.split(' ').collect();
    let mut unigram_inv_probs = Vec::with_capacity(words.len());
    if let Some(idf) = idf_lookup {
        for w in &words {
            let weight = idf.get(*w).copied().unwrap_or(idf_min);
            unigram_inv_probs.push(weight);
        }
    } else {
        unigram_inv_probs.resize(words.len(), 1.0);
    }

    let mut ngrams = Vec::new();
    let mut weights = Vec::new();

    for (word, weight) in words.iter().zip(unigram_inv_probs.iter()) {
        let bytes = word.as_bytes();
        let len = bytes.len();
        for n in ns {
            if len < *n || *n == 0 {
                continue;
            }
            let denom = ((len as f64) - (*n as f64) + 1.0).max(1e-6);
            let w = *weight / denom;
            for i in 0..=(len - n) {
                let ng = std::str::from_utf8(&bytes[i..i + n]).unwrap_or("");
                if ng.contains(' ') {
                    continue;
                }
                ngrams.push(ng.to_string());
                weights.push(w);
            }
        }
    }
    (ngrams, weights)
}

fn ordered_unique_ids(all_ids: &[u32]) -> Vec<u32> {
    let mut seen = HashSet::new();
    let mut ordered = Vec::new();
    for id in all_ids {
        if seen.insert(*id) {
            ordered.push(*id);
        }
    }
    ordered
}

fn merge_ranked_lists_with_order(
    lists: &[Vec<(String, f64)>],
    use_log: bool,
) -> (HashMap<String, f64>, Vec<String>) {
    let mut ret: HashMap<String, f64> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    for d in lists {
        for (k, v) in d {
            let val = if use_log { (1.0 + v).ln() } else { *v };
            if let Some(existing) = ret.get_mut(k) {
                *existing += val;
            } else {
                ret.insert(k.clone(), val);
                order.push(k.clone());
            }
        }
    }
    (ret, order)
}

fn merge_ranked_arrays_with_order(
    ids_list: &[Vec<u32>],
    scores_list: &[Vec<f64>],
    use_log: bool,
) -> (Vec<u32>, Vec<f64>) {
    let mut ids_all: Vec<u32> = Vec::new();
    let mut scores_all: Vec<f64> = Vec::new();
    for (ids, scores) in ids_list.iter().zip(scores_list.iter()) {
        if ids.is_empty() {
            continue;
        }
        ids_all.extend_from_slice(ids);
        if use_log {
            scores_all.extend(scores.iter().map(|v| (1.0 + *v).ln()));
        } else {
            scores_all.extend_from_slice(scores);
        }
    }

    if ids_all.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let mut merged: HashMap<u32, (f64, usize)> = HashMap::new();
    for (idx, (id, score)) in ids_all.iter().zip(scores_all.iter()).enumerate() {
        let entry = merged.entry(*id).or_insert((*score, idx));
        if entry.1 != idx {
            entry.0 += *score;
        }
    }

    let mut merged_vec: Vec<(u32, f64, usize)> = merged
        .into_iter()
        .map(|(id, (sum, first_idx))| (id, sum, first_idx))
        .collect();
    merged_vec.sort_by_key(|item| item.2);

    let ids: Vec<u32> = merged_vec.iter().map(|i| i.0).collect();
    let scores: Vec<f64> = merged_vec.iter().map(|i| i.1).collect();

    (ids, scores)
}

// legacy jaccard helpers removed; runtime versions live on RorIndex

fn build_idf_lookup(texts: &[String]) -> (HashMap<String, f64>, f64) {
    let mut df: HashMap<String, usize> = HashMap::new();
    for text in texts {
        let mut seen = HashSet::new();
        for token in text.split_whitespace() {
            if token.len() < 2 {
                continue;
            }
            if seen.insert(token.to_string()) {
                *df.entry(token.to_string()).or_insert(0) += 1;
            }
        }
    }

    let n_samples = texts.len() as f64;
    let mut idf_lookup = HashMap::new();
    let mut idf_min = f64::INFINITY;
    for (term, doc_count) in df {
        let idf = ((1.0 + n_samples) / (1.0 + doc_count as f64)).ln() + 1.0;
        if idf < idf_min {
            idf_min = idf;
        }
        idf_lookup.insert(term, idf);
    }
    if !idf_min.is_finite() {
        idf_min = 1.0;
    }
    (idf_lookup, idf_min)
}

static ARCHIVED_CACHE: Lazy<Mutex<HashMap<String, Arc<ArchivedIndex>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

fn load_cache(path: &str) -> Result<Arc<ArchivedIndex>, String> {
    if let Some(existing) = ARCHIVED_CACHE.lock().unwrap().get(path).cloned() {
        return Ok(existing);
    }
    let file = File::open(path).map_err(|e| format!("failed to open rust cache: {e}"))?;
    let mmap = unsafe { Mmap::map(&file).map_err(|e| format!("failed to mmap rust cache: {e}"))? };
    let root = unsafe { rkyv::archived_root::<RustIndexState>(&mmap[..]) }
        as *const ArchivedRustIndexState;
    let archived = Arc::new(ArchivedIndex {
        mmap: Arc::new(mmap),
        root,
    });
    let version = archived.root().version;
    if version != RUST_CACHE_VERSION {
        return Err(format!(
            "rust cache version mismatch (found {}, expected {})",
            version, RUST_CACHE_VERSION
        ));
    }
    ARCHIVED_CACHE
        .lock()
        .unwrap()
        .insert(path.to_string(), Arc::clone(&archived));
    Ok(archived)
}

fn save_cache(path: &str, state: &RustIndexState) -> Result<(), String> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create rust cache dir: {e}"))?;
    }
    let bytes = rkyv::to_bytes::<_, 256>(state)
        .map_err(|e| format!("failed to encode rust cache: {e}"))?;
    let mut file = File::create(path).map_err(|e| format!("failed to create rust cache: {e}"))?;
    file.write_all(&bytes)
        .map_err(|e| format!("failed to write rust cache: {e}"))?;
    Ok(())
}

impl RustIndexState {
    fn into_index(self) -> RorIndex {
        let use_prob_weights = self.use_prob_weights;
        let word_multiplier = self.word_multiplier;
        let word_multiplier_is_log = self.word_multiplier_is_log;
        let max_intersection_denominator = self.max_intersection_denominator;
        let ns = self.ns.clone();
        let insert_early_candidates_ind = self.insert_early_candidates_ind;
        let reinsert_cutoff_frac = self.reinsert_cutoff_frac;
        let score_based_early_cutoff = self.score_based_early_cutoff;
        let store = IndexStore::Owned(self);
        RorIndex {
            use_prob_weights,
            word_multiplier,
            word_multiplier_is_log,
            max_intersection_denominator,
            ns,
            insert_early_candidates_ind,
            reinsert_cutoff_frac,
            score_based_early_cutoff,
            store,
        }
    }
}

#[pyclass]
pub struct RorIndex {
    use_prob_weights: bool,
    word_multiplier: f64,
    word_multiplier_is_log: bool,
    max_intersection_denominator: bool,
    ns: Vec<usize>,
    insert_early_candidates_ind: Option<usize>,
    reinsert_cutoff_frac: f64,
    score_based_early_cutoff: f64,
    store: IndexStore,
}

impl RorIndex {
    fn from_archived(archived: Arc<ArchivedIndex>) -> Self {
        let root = archived.root();
        let ns: Vec<usize> = root.ns.as_slice().iter().map(|v| *v as usize).collect();
        let insert_early_candidates_ind = match root.insert_early_candidates_ind {
            ArchivedOption::Some(v) => Some(v as usize),
            ArchivedOption::None => None,
        };
        RorIndex {
            use_prob_weights: root.use_prob_weights,
            word_multiplier: root.word_multiplier,
            word_multiplier_is_log: root.word_multiplier_is_log,
            max_intersection_denominator: root.max_intersection_denominator,
            ns,
            insert_early_candidates_ind,
            reinsert_cutoff_frac: root.reinsert_cutoff_frac,
            score_based_early_cutoff: root.score_based_early_cutoff,
            store: IndexStore::Archived(archived),
        }
    }

    fn id_to_key(&self, idx: u32) -> &str {
        let idx = idx as usize;
        match &self.store {
            IndexStore::Owned(state) => state
                .id_to_key
                .get(idx)
                .map(|s| s.as_str())
                .unwrap_or(""),
            IndexStore::Archived(archived) => archived
                .root()
                .id_to_key
                .get(idx)
                .map(|s| s.as_str())
                .unwrap_or(""),
        }
    }

    fn word_lengths(&self) -> &[f64] {
        match &self.store {
            IndexStore::Owned(state) => state.word_lengths.as_slice(),
            IndexStore::Archived(archived) => archived.root().word_lengths.as_slice(),
        }
    }

    fn ngram_lengths(&self) -> &[f64] {
        match &self.store {
            IndexStore::Owned(state) => state.ngram_lengths.as_slice(),
            IndexStore::Archived(archived) => archived.root().ngram_lengths.as_slice(),
        }
    }

    fn word_index_get(&self, token: &str) -> Option<&[u32]> {
        match &self.store {
            IndexStore::Owned(state) => state.word_index.get(token).map(|v| v.as_slice()),
            IndexStore::Archived(archived) => archived
                .root()
                .word_index
                .get(token)
                .map(|v| v.as_slice()),
        }
    }

    fn ngram_index_get(&self, token: &str) -> Option<&[u32]> {
        match &self.store {
            IndexStore::Owned(state) => state.ngram_index.get(token).map(|v| v.as_slice()),
            IndexStore::Archived(archived) => archived
                .root()
                .ngram_index
                .get(token)
                .map(|v| v.as_slice()),
        }
    }

    fn address_index_get(&self, token: &str) -> Option<&[u32]> {
        match &self.store {
            IndexStore::Owned(state) => state.address_index.get(token).map(|v| v.as_slice()),
            IndexStore::Archived(archived) => archived
                .root()
                .address_index
                .get(token)
                .map(|v| v.as_slice()),
        }
    }

    fn typed_id_to_ror_id_int(&self, idx: u32) -> u32 {
        let idx = idx as usize;
        match &self.store {
            IndexStore::Owned(state) => *state.typed_id_to_ror_id_int.get(idx).unwrap_or(&0),
            IndexStore::Archived(archived) => *archived
                .root()
                .typed_id_to_ror_id_int
                .get(idx)
                .unwrap_or(&0),
        }
    }

    fn ror_id_to_int_get(&self, ror_id: &str) -> Option<u32> {
        match &self.store {
            IndexStore::Owned(state) => state.ror_id_to_int.get(ror_id).copied(),
            IndexStore::Archived(archived) => archived.root().ror_id_to_int.get(ror_id).copied(),
        }
    }

    fn ror_id_at(&self, idx: u32) -> &str {
        let idx = idx as usize;
        match &self.store {
            IndexStore::Owned(state) => state.ror_ids.get(idx).map(|s| s.as_str()).unwrap_or(""),
            IndexStore::Archived(archived) => archived
                .root()
                .ror_ids
                .get(idx)
                .map(|s| s.as_str())
                .unwrap_or(""),
        }
    }

    fn idf_lookup_min(&self) -> f64 {
        match &self.store {
            IndexStore::Owned(state) => state.idf_lookup_min,
            IndexStore::Archived(archived) => archived.root().idf_lookup_min,
        }
    }

    fn idf_lookup_get(&self, token: &str) -> Option<f64> {
        match &self.store {
            IndexStore::Owned(state) => state.idf_lookup.get(token).copied(),
            IndexStore::Archived(archived) => archived.root().idf_lookup.get(token).copied(),
        }
    }

    fn ror_entry_lite_get(&self, ror_id: &str) -> Option<RorEntryLite> {
        match &self.store {
            IndexStore::Owned(state) => state.ror_entries_lite.get(ror_id).cloned(),
            IndexStore::Archived(archived) => archived
                .root()
                .ror_entries_lite
                .get(ror_id)
                .map(archived_entry_to_owned),
        }
    }

    fn word_multiplier_value(&self) -> f64 {
        if self.word_multiplier_is_log {
            1.0
        } else {
            self.word_multiplier
        }
    }

    fn get_text_ngrams_runtime(&self, text: &str) -> (Vec<String>, Vec<f64>) {
        if text.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let words: Vec<&str> = text.split(' ').collect();
        let mut unigram_inv_probs = Vec::with_capacity(words.len());
        if self.use_prob_weights {
            let idf_min = self.idf_lookup_min();
            for w in &words {
                let weight = self.idf_lookup_get(*w).unwrap_or(idf_min);
                unigram_inv_probs.push(weight);
            }
        } else {
            unigram_inv_probs.resize(words.len(), 1.0);
        }

        let mut ngrams = Vec::new();
        let mut weights = Vec::new();

        for (word, weight) in words.iter().zip(unigram_inv_probs.iter()) {
            let bytes = word.as_bytes();
            let len = bytes.len();
            for n in &self.ns {
                if len < *n || *n == 0 {
                    continue;
                }
                let denom = ((len as f64) - (*n as f64) + 1.0).max(1e-6);
                let w = *weight / denom;
                for i in 0..=(len - n) {
                    let ng = std::str::from_utf8(&bytes[i..i + n]).unwrap_or("");
                    if ng.contains(' ') {
                        continue;
                    }
                    ngrams.push(ng.to_string());
                    weights.push(w);
                }
            }
        }
        (ngrams, weights)
    }

    fn jaccard_ngram_nns_int(&self, candidate: &str) -> Vec<(String, f64)> {
        let (candidate_ngrams, candidate_ngram_weights) = self.get_text_ngrams_runtime(candidate);

        if candidate_ngrams.is_empty() {
            return Vec::new();
        }

        let mut unique_ngrams = Vec::new();
        let mut seen_ngrams: HashSet<String> = HashSet::new();
        let mut ngram_first_weight: HashMap<String, f64> = HashMap::new();
        for (ng, weight) in candidate_ngrams.iter().zip(candidate_ngram_weights.iter()) {
            if seen_ngrams.insert(ng.clone()) {
                unique_ngrams.push(ng.clone());
                // Python uses the first-seen weight for a given ngram
                ngram_first_weight.insert(ng.clone(), *weight);
            }
        }

        let lengths = self.ngram_lengths();
        let mut all_ids: Vec<u32> = Vec::new();
        let mut intersections = vec![0f64; lengths.len()];
        let mut weights_in_inverted = Vec::new();
        let mut candidate_ngrams_in_inverted = 0usize;

        for ng in unique_ngrams.iter() {
            if let Some(ids) = self.ngram_index_get(ng) {
                let weight = *ngram_first_weight.get(ng).unwrap_or(&0.0);
                candidate_ngrams_in_inverted += 1;
                weights_in_inverted.push(weight);
                for id in ids {
                    intersections[*id as usize] += weight;
                    all_ids.push(*id);
                }
            }
        }

        if all_ids.is_empty() {
            return Vec::new();
        }

        let (num_candidate_ngrams, num_relevant_candidate_ngrams) = if self.use_prob_weights {
            (
                candidate_ngram_weights.iter().sum::<f64>(),
                weights_in_inverted.iter().sum::<f64>(),
            )
        } else {
            (
                candidate_ngrams.len() as f64,
                candidate_ngrams_in_inverted as f64,
            )
        };

        let mut scores = vec![0f64; lengths.len()];
        for i in 0..lengths.len() {
            let denom = if self.max_intersection_denominator {
                num_relevant_candidate_ngrams + lengths[i] - intersections[i]
            } else {
                num_candidate_ngrams + lengths[i] - intersections[i]
            };
            if denom > 0.0 {
                scores[i] = intersections[i] / denom;
            }
        }

        let ordered_ids = ordered_unique_ids(&all_ids);
        ordered_ids
            .into_iter()
            .map(|i| (self.id_to_key(i).to_string(), scores[i as usize]))
            .collect()
    }

    fn jaccard_word_nns_int(&self, candidate: &str, word_multiplier: f64) -> Vec<(String, f64)> {
        let mut unigrams: Vec<String> = candidate
            .split_whitespace()
            .map(|t| t.to_string())
            .collect();
        unigrams.sort();
        unigrams.dedup();

        if unigrams.is_empty() {
            return Vec::new();
        }

        let mut unigram_inv_probs: HashMap<String, f64> = HashMap::new();
        if self.use_prob_weights {
            let idf_min = self.idf_lookup_min();
            for u in &unigrams {
                let weight = self.idf_lookup_get(u).unwrap_or(idf_min);
                unigram_inv_probs.insert(u.clone(), weight);
            }
        }

        let mut all_ids: Vec<u32> = Vec::new();
        let mut all_weights: Vec<f64> = Vec::new();
        let mut matched_unigrams: HashSet<String> = HashSet::new();

        for unigram in &unigrams {
            if let Some(ids) = self.word_index_get(unigram) {
                matched_unigrams.insert(unigram.clone());
                for id in ids {
                    all_ids.push(*id);
                    if self.use_prob_weights {
                        let weight = *unigram_inv_probs.get(unigram).unwrap_or(&self.idf_lookup_min());
                        all_weights.push(weight);
                    }
                }
            }
        }

        if all_ids.is_empty() {
            return Vec::new();
        }

        let lengths = self.word_lengths();
        let mut intersections = vec![0f64; lengths.len()];
        if self.use_prob_weights {
            for (id, w) in all_ids.iter().zip(all_weights.iter()) {
                intersections[*id as usize] += *w;
            }
        } else {
            for id in &all_ids {
                intersections[*id as usize] += 1.0;
            }
        }

        let (num_candidate_unigrams, num_relevant_candidate_unigrams) = if self.use_prob_weights {
            (
                unigram_inv_probs.values().sum::<f64>(),
                matched_unigrams
                    .iter()
                    .map(|u| *unigram_inv_probs.get(u).unwrap_or(&self.idf_lookup_min()))
                    .sum::<f64>(),
            )
        } else {
            (unigrams.len() as f64, matched_unigrams.len() as f64)
        };

        let mut scores = vec![0f64; lengths.len()];
        for i in 0..lengths.len() {
            let denom = if self.max_intersection_denominator {
                num_relevant_candidate_unigrams + lengths[i] - intersections[i]
            } else {
                num_candidate_unigrams + lengths[i] - intersections[i]
            };
            if denom > 0.0 {
                scores[i] = word_multiplier * intersections[i] / denom;
            }
        }

        let ordered_ids = ordered_unique_ids(&all_ids);
        ordered_ids
            .into_iter()
            .map(|i| (self.id_to_key(i).to_string(), scores[i as usize]))
            .collect()
    }

    fn jaccard_word_nns_int_arrays(&self, candidate: &str, word_multiplier: f64) -> (Vec<u32>, Vec<f64>) {
        let mut unigrams: Vec<String> = Vec::new();
        for token in candidate.split_whitespace() {
            if !unigrams.iter().any(|t| t == token) {
                unigrams.push(token.to_string());
            }
        }

        if unigrams.is_empty() {
            return (Vec::new(), Vec::new());
        }
        unigrams.sort();

        let mut unigram_inv_probs: HashMap<String, f64> = HashMap::new();
        if self.use_prob_weights {
            let idf_min = self.idf_lookup_min();
            for u in &unigrams {
                let weight = self.idf_lookup_get(u).unwrap_or(idf_min);
                unigram_inv_probs.insert(u.clone(), weight);
            }
        }

        let mut all_ids: Vec<u32> = Vec::new();
        let mut all_weights: Vec<f64> = Vec::new();
        let mut matched_unigrams: HashSet<String> = HashSet::new();

        for unigram in &unigrams {
            if let Some(ids) = self.word_index_get(unigram) {
                matched_unigrams.insert(unigram.clone());
                for id in ids {
                    all_ids.push(*id);
                    if self.use_prob_weights {
                        let weight = *unigram_inv_probs.get(unigram).unwrap_or(&self.idf_lookup_min());
                        all_weights.push(weight);
                    }
                }
            }
        }

        if all_ids.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let lengths = self.word_lengths();
        let mut intersections = vec![0f64; lengths.len()];
        if self.use_prob_weights {
            for (id, w) in all_ids.iter().zip(all_weights.iter()) {
                intersections[*id as usize] += *w;
            }
        } else {
            for id in &all_ids {
                intersections[*id as usize] += 1.0;
            }
        }

        let (num_candidate_unigrams, num_relevant_candidate_unigrams) = if self.use_prob_weights {
            (
                unigram_inv_probs.values().sum::<f64>(),
                matched_unigrams
                    .iter()
                    .map(|u| *unigram_inv_probs.get(u).unwrap_or(&self.idf_lookup_min()))
                    .sum::<f64>(),
            )
        } else {
            (unigrams.len() as f64, matched_unigrams.len() as f64)
        };

        let mut scores = vec![0f64; lengths.len()];
        for i in 0..lengths.len() {
            let denom = if self.max_intersection_denominator {
                num_relevant_candidate_unigrams + lengths[i] - intersections[i]
            } else {
                num_candidate_unigrams + lengths[i] - intersections[i]
            };
            if denom > 0.0 {
                scores[i] = word_multiplier * intersections[i] / denom;
            }
        }

        let ordered_ids = ordered_unique_ids(&all_ids);
        let mut ordered_scores = Vec::with_capacity(ordered_ids.len());
        for id in &ordered_ids {
            ordered_scores.push(scores[*id as usize]);
        }
        (ordered_ids, ordered_scores)
    }

    fn jaccard_ngram_nns_int_arrays(&self, candidate: &str) -> (Vec<u32>, Vec<f64>) {
        let (candidate_ngrams, candidate_ngram_weights) = self.get_text_ngrams_runtime(candidate);

        if candidate_ngrams.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let mut unique_ngrams = Vec::new();
        let mut seen_ngrams: HashSet<String> = HashSet::new();
        let mut ngram_first_weight: HashMap<String, f64> = HashMap::new();
        for (ng, weight) in candidate_ngrams.iter().zip(candidate_ngram_weights.iter()) {
            if seen_ngrams.insert(ng.clone()) {
                unique_ngrams.push(ng.clone());
                // Python uses the first-seen weight for a given ngram
                ngram_first_weight.insert(ng.clone(), *weight);
            }
        }
        unique_ngrams.sort();

        let lengths = self.ngram_lengths();
        let mut all_ids: Vec<u32> = Vec::new();
        let mut intersections = vec![0f64; lengths.len()];
        let mut weights_in_inverted = Vec::new();
        let mut candidate_ngrams_in_inverted = 0usize;

        for ng in unique_ngrams.iter() {
            if let Some(ids) = self.ngram_index_get(ng) {
                let weight = *ngram_first_weight.get(ng).unwrap_or(&0.0);
                candidate_ngrams_in_inverted += 1;
                weights_in_inverted.push(weight);
                for id in ids {
                    intersections[*id as usize] += weight;
                    all_ids.push(*id);
                }
            }
        }

        if all_ids.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let (num_candidate_ngrams, num_relevant_candidate_ngrams) = if self.use_prob_weights {
            (
                candidate_ngram_weights.iter().sum::<f64>(),
                weights_in_inverted.iter().sum::<f64>(),
            )
        } else {
            (
                candidate_ngrams.len() as f64,
                candidate_ngrams_in_inverted as f64,
            )
        };

        let mut scores = vec![0f64; lengths.len()];
        for i in 0..lengths.len() {
            let denom = if self.max_intersection_denominator {
                num_relevant_candidate_ngrams + lengths[i] - intersections[i]
            } else {
                num_candidate_ngrams + lengths[i] - intersections[i]
            };
            if denom > 0.0 {
                scores[i] = intersections[i] / denom;
            }
        }

        let ordered_ids = ordered_unique_ids(&all_ids);
        let mut ordered_scores = Vec::with_capacity(ordered_ids.len());
        for id in &ordered_ids {
            ordered_scores.push(scores[*id as usize]);
        }
        (ordered_ids, ordered_scores)
    }
    fn get_candidates_v7_inner(
        &self,
        main: &[String],
        address: Option<&str>,
        early_candidates: &[String],
    ) -> Result<(Vec<String>, Vec<f64>), String> {
        let mut ranked_before_order: Vec<u32> = Vec::new();
        let mut ranked_before_max: HashMap<u32, f64> = HashMap::new();

        for m in main.iter() {
            let mut words: Vec<String> = Vec::new();
            for w in m.to_lowercase().replace(',', "").split(' ') {
                if STOPWORDS.contains(w) {
                    continue;
                }
                let mapped = INVERTED_ABBREV.get(w).unwrap_or(&w);
                words.push(mapped.to_string());
            }
            let main_fixed = words.join(" ");
            if main_fixed.is_empty() {
                continue;
            }

            let word_multiplier = if self.word_multiplier_is_log {
                1.0
            } else {
                self.word_multiplier
            };

            let (word_ids, word_scores) =
                self.jaccard_word_nns_int_arrays(&main_fixed, word_multiplier);
            let (ngram_ids, ngram_scores) = self.jaccard_ngram_nns_int_arrays(&main_fixed);

            let (ranked_before_ids, ranked_before_scores) = merge_ranked_arrays_with_order(
                &[word_ids, ngram_ids],
                &[word_scores, ngram_scores],
                self.word_multiplier_is_log,
            );

            if ranked_before_ids.is_empty() {
                continue;
            }

            let cutoff = if self.word_multiplier_is_log {
                (1.0 + self.score_based_early_cutoff).ln()
            } else {
                self.score_based_early_cutoff
            };

            for (id_val, score) in ranked_before_ids.iter().zip(ranked_before_scores.iter()) {
                if *score <= cutoff {
                    continue;
                }
                if !ranked_before_max.contains_key(id_val) {
                    ranked_before_order.push(*id_val);
                    ranked_before_max.insert(*id_val, *score);
                } else if let Some(current) = ranked_before_max.get_mut(id_val) {
                    if *score > *current {
                        *current = *score;
                    }
                }
            }
        }

        if ranked_before_order.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let mut ranked_with_idx: Vec<(usize, f64)> = ranked_before_order
            .iter()
            .enumerate()
            .map(|(i, id)| (i, *ranked_before_max.get(id).unwrap_or(&0.0)))
            .collect();
        ranked_with_idx.sort_by(|a, b| {
            let a_key = tie_score_key(a.1);
            let b_key = tie_score_key(b.1);
            let score_cmp = b_key.cmp(&a_key);
            if score_cmp != std::cmp::Ordering::Equal {
                return score_cmp;
            }
            let a_typed = ranked_before_order[a.0];
            let b_typed = ranked_before_order[b.0];
            let a_ror = self.ror_id_at(self.typed_id_to_ror_id_int(a_typed));
            let b_ror = self.ror_id_at(self.typed_id_to_ror_id_int(b_typed));
            let ror_cmp = a_ror.cmp(b_ror);
            if ror_cmp != std::cmp::Ordering::Equal {
                return ror_cmp;
            }
            a.0.cmp(&b.0)
        });

        let mut ranked_before_sorted_ids: Vec<u32> = Vec::with_capacity(ranked_with_idx.len());
        let mut ranked_before_sorted_scores: Vec<f64> = Vec::with_capacity(ranked_with_idx.len());
        for (idx, score) in ranked_with_idx {
            ranked_before_sorted_ids.push(ranked_before_order[idx]);
            ranked_before_sorted_scores.push(score);
        }

        let mut ranked_unique: Vec<(u32, f64)> = Vec::new();
        let mut seen_rors: HashSet<u32> = HashSet::new();
        for (typed_id, score) in ranked_before_sorted_ids
            .iter()
            .zip(ranked_before_sorted_scores.iter())
        {
            let ror_id_int = self.typed_id_to_ror_id_int(*typed_id);
            if seen_rors.insert(ror_id_int) {
                ranked_unique.push((ror_id_int, *score));
            }
        }

        let mut early_candidates_tuples: Vec<(u32, f64)> = Vec::new();
        if let Some(insert_idx) = self.insert_early_candidates_ind {
            for cand in early_candidates.iter() {
                if let Some(idx) = self.ror_id_to_int_get(&cand) {
                    if !seen_rors.contains(&idx) {
                        early_candidates_tuples.push((idx, -0.1));
                    }
                }
            }
            let split = insert_idx.min(ranked_unique.len());
            let mut merged = Vec::new();
            merged.extend_from_slice(&ranked_unique[..split]);
            merged.extend(early_candidates_tuples.clone());
            merged.extend_from_slice(&ranked_unique[split..]);
            ranked_unique = merged;
        }

        if ranked_unique.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let mut ranked_after_address_filter = ranked_unique.clone();
        if let Some(addr) = address {
            if !addr.is_empty() {
                let mut address_tokens: HashSet<String> = HashSet::new();
                for w in addr.to_lowercase().replace(',', "").split(' ') {
                    if STOPWORDS.contains(w) {
                        continue;
                    }
                    address_tokens.insert(w.to_string());
                }

                let mut acceptable_rors: HashSet<u32> = HashSet::new();
                for token in address_tokens {
                    if let Some(ids) = self.address_index_get(&token) {
                        for id in ids {
                            acceptable_rors.insert(*id);
                        }
                    }
                }

                let mut filtered = Vec::new();
                for (ror_id, score) in ranked_unique.iter() {
                    if acceptable_rors.contains(ror_id) {
                        filtered.push((*ror_id, *score));
                    }
                }

                if filtered.is_empty() {
                    ranked_after_address_filter = ranked_unique;
                } else {
                    let mut removed = Vec::new();
                    for (ror_id, score) in ranked_unique.iter() {
                        if !acceptable_rors.contains(ror_id) {
                            removed.push((*ror_id, *score));
                        }
                    }
                    if !removed.is_empty() {
                        let top_score = filtered[0].1;
                        let cutoff = self.reinsert_cutoff_frac * top_score;
                        let mut removed_to_reinsert = Vec::new();
                        for (rid, score) in removed {
                            if score >= cutoff {
                                removed_to_reinsert.push((rid, -0.15));
                            }
                        }
                        let ec_len = if self.insert_early_candidates_ind.is_some() {
                            early_candidates_tuples.len()
                        } else {
                            0
                        };
                        let base = self.insert_early_candidates_ind.unwrap_or(0);
                        let insert_ind = ec_len + base + 10;
                        let split = insert_ind.min(filtered.len());
                        let mut merged = Vec::new();
                        merged.extend_from_slice(&filtered[..split]);
                        merged.extend(removed_to_reinsert);
                        merged.extend_from_slice(&filtered[split..]);
                        ranked_after_address_filter = merged;
                    } else {
                        ranked_after_address_filter = filtered;
                    }
                }
            }
        }

        let mut candidates = Vec::new();
        let mut scores = Vec::new();
        for (rid, score) in ranked_after_address_filter {
            candidates.push(self.ror_id_at(rid).to_string());
            scores.push(score);
        }
        Ok((candidates, scores))
    }
}

#[pymethods]
impl RorIndex {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(
        ror_data_path,
        ror_edits_path,
        country_info_path,
        works_counts_path,
        use_prob_weights,
        word_multiplier,
        word_multiplier_is_log,
        max_intersection_denominator,
        ns,
        insert_early_candidates_ind,
        reinsert_cutoff_frac,
        score_based_early_cutoff,
        cache_path=None
    ))]
    fn new(
        ror_data_path: String,
        ror_edits_path: String,
        country_info_path: String,
        works_counts_path: String,
        use_prob_weights: bool,
        word_multiplier: f64,
        word_multiplier_is_log: bool,
        max_intersection_denominator: bool,
        ns: Vec<usize>,
        insert_early_candidates_ind: Option<usize>,
        reinsert_cutoff_frac: f64,
        score_based_early_cutoff: f64,
        cache_path: Option<String>,
    ) -> PyResult<Self> {
        let total_start = Instant::now();
        rust_log(&format!("RorIndex::new start cache_path={cache_path:?}"));
        if let Some(path) = cache_path.as_ref() {
            if std::path::Path::new(path).exists() {
                rust_log(&format!("cache hit: loading {}", path));
                let cache_start = Instant::now();
                match load_cache(path) {
                    Ok(archived) => {
                        rust_log_elapsed("cache load", cache_start);
                        rust_log_elapsed("RorIndex::new total", total_start);
                        return Ok(RorIndex::from_archived(archived));
                    }
                    Err(err) => {
                        rust_log(&format!("cache load failed: {err}; rebuilding"));
                    }
                }
            } else {
                rust_log(&format!("cache miss: {}", path));
            }
        } else {
            rust_log("cache disabled (no path provided)");
        }
        let country_start = Instant::now();
        let (country_codes_dict, country_by_iso2) =
            load_country_info(&country_info_path).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        rust_log_elapsed("load_country_info", country_start);

        let works_start = Instant::now();
        let works_counts =
            load_works_counts(&works_counts_path).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        rust_log_elapsed("load_works_counts", works_start);

        let ror_start = Instant::now();
        if let Ok(meta) = std::fs::metadata(&ror_data_path) {
            rust_log(&format!("ror_data size={} bytes", meta.len()));
        }
        rust_log("load_ror_data start (simd-json)");
        let parsing_done = Arc::new(AtomicBool::new(false));
        let heartbeat = if rust_log_enabled() {
            let done = Arc::clone(&parsing_done);
            Some(std::thread::spawn(move || {
                let mut elapsed = 0u64;
                loop {
                    if done.load(Ordering::Relaxed) {
                        break;
                    }
                    std::thread::sleep(Duration::from_secs(5));
                    elapsed += 5;
                    if elapsed % 30 == 0 && !done.load(Ordering::Relaxed) {
                        eprintln!("[s2aff_rust] parsing ror_data... {}s elapsed", elapsed);
                    }
                }
            }))
        } else {
            None
        };
        let mut records = load_ror_records_fast(&ror_data_path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(e)
        })?;
        parsing_done.store(true, Ordering::Relaxed);
        if let Some(handle) = heartbeat {
            let _ = handle.join();
        }
        rust_log_elapsed("load_ror_data", ror_start);
        rust_log(&format!("records parsed: {}", records.len()));

        let mut id_to_idx: HashMap<String, usize> = HashMap::new();
        for (i, rec) in records.iter().enumerate() {
            id_to_idx.insert(rec.id.clone(), i);
        }

        for rec in records.iter_mut() {
            rec.works_count = works_counts.get(&rec.id).copied().unwrap_or(0);
        }

        let edits_start = Instant::now();
        apply_ror_edits(&mut records, &id_to_idx, &ror_edits_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        rust_log_elapsed("apply_ror_edits", edits_start);

        // Fix works_count for parent/child with same name (ignoring parens)
        let works_fix_start = Instant::now();
        for _ in 0..2 {
            let ids: Vec<String> = records.iter().map(|r| r.id.clone()).collect();
            for rid in ids {
                let parent_idx = match id_to_idx.get(&rid) {
                    Some(i) => *i,
                    None => continue,
                };
                let parent_name = records[parent_idx].name.clone();
                let parent_wc = records[parent_idx].works_count;
                let mut rels = records[parent_idx].relationships.clone();
                rels.sort_by_key(|r| -(records[id_to_idx[&r.id]].works_count as i64));
                for rel in rels {
                    if rel.rel_type != "Child" {
                        continue;
                    }
                    let child_idx = match id_to_idx.get(&rel.id) {
                        Some(i) => *i,
                        None => continue,
                    };
                    let child_wc = records[child_idx].works_count;
                    let child_name = records[child_idx].name.clone();
                    if strip_parens(&parent_name) == strip_parens(&child_name) {
                        if child_wc > parent_wc {
                            records[parent_idx].works_count = child_wc;
                            records[child_idx].works_count = parent_wc;
                            break;
                        }
                    }
                }
            }
        }
        rust_log_elapsed("fix_works_counts", works_fix_start);

        // Build ROR entries for stage2 features
        let ror_entries_start = Instant::now();
        let ror_entries_lite: HashMap<String, RorEntryLite> = records
            .par_iter()
            .map(|rec| {
                (
                    rec.id.clone(),
                    build_ror_entry_lite(rec, &country_codes_dict, &country_by_iso2),
                )
            })
            .collect();
        rust_log_elapsed("build_ror_entries_lite", ror_entries_start);

        // Build indices
        let build_names_start = Instant::now();
        let mut typed_keys: Vec<String> = Vec::new();
        let mut typed_names: Vec<String> = Vec::new();
        let mut texts: Vec<String> = Vec::new();

        let mut address_index: HashMap<String, HashSet<String>> = HashMap::new();

        for rec in &records {
            let ror_id = &rec.id;
            let official_name = fix_text(&rec.name).to_lowercase().replace(',', "");
            let aliases: Vec<String> = rec
                .aliases
                .iter()
                .map(|i| fix_text(i).to_lowercase().replace(',', ""))
                .collect();
            let labels: Vec<String> = rec
                .labels
                .iter()
                .map(|i| fix_text(i).to_lowercase().replace(',', ""))
                .collect();
            let acronyms: Vec<String> = rec
                .acronyms
                .iter()
                .map(|i| fix_text(i).to_lowercase().replace(',', ""))
                .collect();

            let mut all_names = Vec::new();
            all_names.push(official_name.clone());
            all_names.extend(aliases.clone());
            all_names.extend(labels.clone());
            all_names.extend(acronyms.clone());

            let types_of_names: Vec<String> = vec!["official_name".to_string()]
                .into_iter()
                .chain((0..aliases.len()).map(|i| format!("alias_{i}")))
                .chain((0..labels.len()).map(|i| format!("label_{i}")))
                .chain((0..acronyms.len()).map(|i| format!("acronym_{i}")))
                .collect();

            texts.push(all_names.join(" "));

            for (name, type_of_name) in all_names.iter().zip(types_of_names.iter()) {
                let mut cleaned = name.clone();
                let mut filtered_words: Vec<String> = Vec::new();
                for w in cleaned.split(' ') {
                    if !STOPWORDS.contains(w) {
                        filtered_words.push(w.to_string());
                    }
                }
                cleaned = filtered_words.join(" ");
                let typed_key = format!("{}__{}", ror_id, type_of_name);
                typed_keys.push(typed_key);
                typed_names.push(cleaned);
            }

            if let Some(addr0) = rec.addresses.get(0) {
                let city_raw = addr0.city.clone().unwrap_or_default();
                let city_tokens: Vec<String> = fix_text(&city_raw)
                    .to_lowercase()
                    .replace(',', "")
                    .split_whitespace()
                    .map(|s| s.to_string())
                    .collect();

                let state_raw = addr0.state.clone().unwrap_or_default();
                let state_tokens: Vec<String> = fix_text(&state_raw)
                    .to_lowercase()
                    .replace(',', "")
                    .split_whitespace()
                    .map(|s| s.to_string())
                    .collect();

                let mut state_code_tokens: Vec<String> = Vec::new();
                if let Some(sc) = addr0.state_code.clone() {
                    state_code_tokens = sc
                        .split('-')
                        .filter(|i| i.chars().all(|c| c.is_ascii_alphabetic()))
                        .map(|i| fix_text(i).to_lowercase().replace(',', ""))
                        .collect();
                }

                let mut country_and_codes: Option<Vec<String>> = None;
                if let Some(geoid) = addr0.country_geonames_id.clone() {
                    if let Some(entry) = country_codes_dict.get(&geoid) {
                        country_and_codes = Some(entry.clone());
                    }
                }
                if country_and_codes.is_none() {
                    if let Some(code) = addr0.country_code.clone() {
                        let iso2 = code.to_uppercase();
                        if let Some(entry) = country_by_iso2.get(&iso2) {
                            country_and_codes = Some(entry.clone());
                        }
                    }
                }
                let mut country_and_codes =
                    country_and_codes.unwrap_or_else(|| vec!["".to_string(); 4]);
                for i in country_and_codes.iter_mut() {
                    *i = fix_text(i).to_lowercase().replace(',', "");
                }
                if country_and_codes.iter().any(|c| c == "china") {
                    country_and_codes.push("pr".to_string());
                    country_and_codes.push("prc".to_string());
                }

                let mut fixed_elements: HashSet<String> = HashSet::new();
                for i in city_tokens
                    .into_iter()
                    .chain(state_tokens.into_iter())
                    .chain(country_and_codes.into_iter())
                    .chain(state_code_tokens.into_iter())
                {
                    if i.len() <= 1 {
                        continue;
                    }
                    if STOPWORDS.contains(i.as_str()) {
                        continue;
                    }
                    fixed_elements.insert(i);
                }

                for elem in fixed_elements {
                    address_index
                        .entry(elem)
                        .or_insert_with(HashSet::new)
                        .insert(ror_id.clone());
                }
            }
        }
        rust_log(&format!(
            "typed_keys={} typed_names={} address_tokens={}",
            typed_keys.len(),
            typed_names.len(),
            address_index.len()
        ));
        rust_log_elapsed("collect_names_and_addresses", build_names_start);

        let idf_start = Instant::now();
        let (idf_lookup, idf_lookup_min) = if use_prob_weights {
            build_idf_lookup(&texts)
        } else {
            (HashMap::new(), 1.0)
        };
        rust_log_elapsed("build_idf_lookup", idf_start);

        let word_ngram_start = Instant::now();
        let total_typed = typed_names.len();
        let progress_counter = if rust_log_enabled() {
            Some(Arc::new(AtomicUsize::new(0)))
        } else {
            None
        };

        let (mut word_index, mut ngram_index, word_lengths_pairs, ngram_lengths_pairs) = typed_names
            .par_iter()
            .enumerate()
            .fold(
                || {
                    (
                        HashMap::new(),
                        HashMap::new(),
                        Vec::<(usize, f64)>::new(),
                        Vec::<(usize, f64)>::new(),
                    )
                },
                |mut acc, (idx, name)| {
                    let (ngrams, ngram_weights) = if use_prob_weights {
                        get_text_ngrams(name, Some(&idf_lookup), idf_lookup_min, &ns)
                    } else {
                        get_text_ngrams(name, None, idf_lookup_min, &ns)
                    };

                    let mut unique_ngrams = HashSet::new();
                    let mut ngram_largest_weight: HashMap<String, f64> = HashMap::new();
                    for (ng, w) in ngrams.iter().zip(ngram_weights.iter()) {
                        if unique_ngrams.insert(ng.clone()) {
                            // Python keeps the first weight it sees for a given ngram
                            ngram_largest_weight.insert(ng.clone(), *w);
                        }
                    }

                    let ngram_len = if use_prob_weights {
                        unique_ngrams
                            .iter()
                            .map(|ng| *ngram_largest_weight.get(ng).unwrap_or(&0.0))
                            .sum::<f64>()
                    } else {
                        unique_ngrams.len() as f64
                    };
                    acc.3.push((idx, ngram_len));

                    let idx_u32 = idx as u32;
                    for ng in unique_ngrams {
                        acc.1.entry(ng).or_insert_with(Vec::new).push(idx_u32);
                    }

                    let mut name_unigrams: HashSet<String> = HashSet::new();
                    for w in name.split(' ') {
                        name_unigrams.insert(w.to_string());
                    }
                    let word_len = if use_prob_weights {
                        name_unigrams
                            .iter()
                            .map(|w| *idf_lookup.get(w).unwrap_or(&idf_lookup_min))
                            .sum::<f64>()
                    } else {
                        name_unigrams.len() as f64
                    };
                    acc.2.push((idx, word_len));
                    for w in name_unigrams {
                        acc.0.entry(w).or_insert_with(Vec::new).push(idx_u32);
                    }

                    if let Some(counter) = progress_counter.as_ref() {
                        let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
                        if count % 50_000 == 0 {
                            rust_log(&format!(
                                "build_word_ngram_indices progress {}/{}",
                                count, total_typed
                            ));
                        }
                    }
                    acc
                },
            )
            .reduce(
                || {
                    (
                        HashMap::new(),
                        HashMap::new(),
                        Vec::<(usize, f64)>::new(),
                        Vec::<(usize, f64)>::new(),
                    )
                },
                |mut acc, other| {
                    for (k, v) in other.0 {
                        acc.0.entry(k).or_insert_with(Vec::new).extend(v);
                    }
                    for (k, v) in other.1 {
                        acc.1.entry(k).or_insert_with(Vec::new).extend(v);
                    }
                    acc.2.extend(other.2);
                    acc.3.extend(other.3);
                    acc
                },
            );

        let mut word_lengths = vec![0.0f64; total_typed];
        for (idx, val) in word_lengths_pairs {
            if idx < word_lengths.len() {
                word_lengths[idx] = val;
            }
        }
        let mut ngram_lengths = vec![0.0f64; total_typed];
        for (idx, val) in ngram_lengths_pairs {
            if idx < ngram_lengths.len() {
                ngram_lengths[idx] = val;
            }
        }

        for ids in word_index.values_mut() {
            ids.sort_unstable();
        }
        for ids in ngram_index.values_mut() {
            ids.sort_unstable();
        }
        rust_log_elapsed("build_word_ngram_indices", word_ngram_start);

        let id_to_key = typed_keys.clone();
        let mut key_to_id = HashMap::new();
        for (i, k) in typed_keys.iter().enumerate() {
            key_to_id.insert(k.clone(), i);
        }

        let mut ror_ids = Vec::new();
        let mut ror_id_to_int: HashMap<String, u32> = HashMap::new();
        for rec in &records {
            let idx = ror_ids.len() as u32;
            ror_ids.push(rec.id.clone());
            ror_id_to_int.insert(rec.id.clone(), idx);
        }

        let mut typed_id_to_ror_id_int: Vec<u32> = Vec::with_capacity(typed_keys.len());
        for key in &typed_keys {
            let ror_id = key.split("__").next().unwrap_or(key);
            let idx = ror_id_to_int.get(ror_id).copied().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "missing ror id for typed key: {ror_id}"
                ))
            })?;
            typed_id_to_ror_id_int.push(idx);
        }

        let address_int_start = Instant::now();
        let mut address_index_int: HashMap<String, Vec<u32>> = HashMap::new();
        for (token, rids) in address_index {
            let mut ids: Vec<u32> = Vec::new();
            for rid in rids {
                if let Some(idx) = ror_id_to_int.get(&rid) {
                    ids.push(*idx);
                }
            }
            address_index_int.insert(token, ids);
        }
        rust_log_elapsed("build_address_index", address_int_start);

        let state = RustIndexState {
            version: RUST_CACHE_VERSION,
            use_prob_weights,
            word_multiplier,
            word_multiplier_is_log,
            max_intersection_denominator,
            ns,
            insert_early_candidates_ind,
            reinsert_cutoff_frac,
            score_based_early_cutoff,
            typed_keys,
            id_to_key,
            key_to_id,
            word_index,
            word_lengths,
            ngram_index,
            ngram_lengths,
            address_index: address_index_int,
            ror_ids,
            ror_id_to_int,
            idf_lookup,
            idf_lookup_min,
            ror_entries_lite,
            typed_id_to_ror_id_int,
        };

        if let Some(path) = cache_path.as_ref() {
            let save_start = Instant::now();
            let _ = save_cache(path, &state);
            rust_log_elapsed("save_cache", save_start);
        }

        rust_log_elapsed("RorIndex::new total", total_start);
        Ok(state.into_index())
    }

    #[pyo3(signature=(main, address=None, early_candidates=Vec::new()))]
    fn get_candidates_v4(
        &self,
        main: Vec<String>,
        address: Option<String>,
        early_candidates: Vec<String>,
    ) -> PyResult<(Vec<String>, Vec<f64>)> {
        let mut ranked_before_order: Vec<String> = Vec::new();
        let mut ranked_before_max: HashMap<String, f64> = HashMap::new();

        for m in main {
            let mut words: Vec<String> = Vec::new();
            for w in m.to_lowercase().replace(',', "").split(' ') {
                if STOPWORDS.contains(w) {
                    continue;
                }
                let mapped = INVERTED_ABBREV.get(w).unwrap_or(&w);
                words.push(mapped.to_string());
            }
            let main_fixed = words.join(" ");
            if main_fixed.is_empty() {
                continue;
            }

            let ranked_words_before =
                self.jaccard_word_nns_int(&main_fixed, self.word_multiplier_value());
            let ranked_ngrams_before = self.jaccard_ngram_nns_int(&main_fixed);

            let (ranked_before, ranked_before_local_order) = merge_ranked_lists_with_order(
                &[ranked_words_before, ranked_ngrams_before],
                self.word_multiplier_is_log,
            );

            if ranked_before.is_empty() {
                continue;
            }

            let cutoff = if self.word_multiplier_is_log {
                (1.0 + self.score_based_early_cutoff).ln()
            } else {
                self.score_based_early_cutoff
            };

            for key in ranked_before_local_order {
                if let Some(value) = ranked_before.get(&key) {
                    if *value <= cutoff {
                        continue;
                    }
                    if !ranked_before_max.contains_key(&key) {
                        ranked_before_order.push(key.clone());
                        ranked_before_max.insert(key.clone(), *value);
                    } else {
                        let current = ranked_before_max.get(&key).copied().unwrap_or(*value);
                        if *value > current {
                            ranked_before_max.insert(key.clone(), *value);
                        }
                    }
                }
            }
        }

        if ranked_before_max.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        // stable sort by score desc, tie by original order
        let mut ranked_before: Vec<(String, f64)> = ranked_before_order
            .iter()
            .map(|k| (k.clone(), *ranked_before_max.get(k).unwrap_or(&0.0)))
            .collect();
        let mut order_index: HashMap<String, usize> = HashMap::new();
        for (i, k) in ranked_before_order.iter().enumerate() {
            order_index.insert(k.clone(), i);
        }
        ranked_before.sort_by(|a, b| {
            if (b.1 - a.1).abs() > f64::EPSILON {
                b.1.partial_cmp(&a.1).unwrap()
            } else {
                let ia = order_index.get(&a.0).copied().unwrap_or(0);
                let ib = order_index.get(&b.0).copied().unwrap_or(0);
                ia.cmp(&ib)
            }
        });

        let mut ranked_unique: Vec<(String, f64)> = Vec::new();
        let mut seen_rors: HashSet<String> = HashSet::new();
        for (ror_id_with_type, score) in ranked_before {
            let ror_id = ror_id_with_type
                .split("__")
                .next()
                .unwrap_or(&ror_id_with_type)
                .to_string();
            if seen_rors.insert(ror_id.clone()) {
                ranked_unique.push((ror_id, score));
            }
        }

        let mut early_candidates_tuples: Vec<(String, f64)> = Vec::new();
        if let Some(insert_idx) = self.insert_early_candidates_ind {
            for cand in early_candidates {
                if !seen_rors.contains(&cand) {
                    early_candidates_tuples.push((cand, -0.1));
                }
            }
            let mut merged = Vec::new();
            let split = insert_idx.min(ranked_unique.len());
            merged.extend_from_slice(&ranked_unique[..split]);
            merged.extend(early_candidates_tuples.clone());
            merged.extend_from_slice(&ranked_unique[split..]);
            ranked_unique = merged;
        }

        if ranked_unique.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let mut ranked_after_address_filter = ranked_unique.clone();
        if let Some(addr) = address {
            if !addr.is_empty() {
                let mut address_tokens: HashSet<String> = HashSet::new();
                for w in addr.to_lowercase().replace(',', "").split(' ') {
                    if STOPWORDS.contains(w) {
                        continue;
                    }
                    address_tokens.insert(w.to_string());
                }

                let mut acceptable_rors: HashSet<u32> = HashSet::new();
                for token in address_tokens {
                    if let Some(ids) = self.address_index_get(&token) {
                        for id in ids {
                            acceptable_rors.insert(*id);
                        }
                    }
                }

                let mut filtered = Vec::new();
                for (ror_id, score) in ranked_unique.iter() {
                    if let Some(rid) = self.ror_id_to_int_get(ror_id) {
                        if acceptable_rors.contains(&rid) {
                            filtered.push((ror_id.clone(), *score));
                        }
                    }
                }

                if filtered.is_empty() {
                    ranked_after_address_filter = ranked_unique;
                } else {
                    let mut removed = Vec::new();
                    for (ror_id, score) in ranked_unique.iter() {
                        if let Some(rid) = self.ror_id_to_int_get(ror_id) {
                            if !acceptable_rors.contains(&rid) {
                                removed.push((ror_id.clone(), *score));
                            }
                        }
                    }
                    if !removed.is_empty() {
                        let top_score = filtered[0].1;
                        let cutoff = self.reinsert_cutoff_frac * top_score;
                        let mut removed_to_reinsert = Vec::new();
                        for (rid, score) in removed {
                            if score >= cutoff {
                                removed_to_reinsert.push((rid, -0.15));
                            }
                        }
                        let ec_len = if self.insert_early_candidates_ind.is_some() {
                            early_candidates_tuples.len()
                        } else {
                            0
                        };
                        let base = self.insert_early_candidates_ind.unwrap_or(0);
                        let insert_ind = ec_len + base + 10;
                        let split = insert_ind.min(filtered.len());
                        let mut merged = Vec::new();
                        merged.extend_from_slice(&filtered[..split]);
                        merged.extend(removed_to_reinsert);
                        merged.extend_from_slice(&filtered[split..]);
                        ranked_after_address_filter = merged;
                    } else {
                        ranked_after_address_filter = filtered;
                    }
                }
            }
        }

        let mut candidates = Vec::new();
        let mut scores = Vec::new();
        for (rid, score) in ranked_after_address_filter {
            candidates.push(rid);
            scores.push(score);
        }
        Ok((candidates, scores))
    }

    #[pyo3(signature=(main, address=None, early_candidates=Vec::new()))]
    fn get_candidates_v7(
        &self,
        main: Vec<String>,
        address: Option<String>,
        early_candidates: Vec<String>,
    ) -> PyResult<(Vec<String>, Vec<f64>)> {
        self.get_candidates_v7_inner(&main, address.as_deref(), &early_candidates)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
    }

    #[pyo3(signature=(mains, addresses=None, early_candidates_list=None))]
    fn get_candidates_v7_batch(
        &self,
        mains: Vec<Vec<String>>,
        addresses: Option<Vec<Option<String>>>,
        early_candidates_list: Option<Vec<Vec<String>>>,
    ) -> PyResult<(Vec<Vec<String>>, Vec<Vec<f64>>)> {
        let addresses = addresses.unwrap_or_default();
        let early_candidates_list = early_candidates_list.unwrap_or_default();
        let has_addresses = !addresses.is_empty();
        let has_early = !early_candidates_list.is_empty();

        if has_addresses && addresses.len() != mains.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "addresses length mismatch",
            ));
        }
        if has_early && early_candidates_list.len() != mains.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "early_candidates_list length mismatch",
            ));
        }

        let empty: &[String] = &[];
        let results: Vec<Result<(Vec<String>, Vec<f64>), String>> = if mains.len() > 1 {
            mains
                .par_iter()
                .enumerate()
                .map(|(idx, main)| {
                    let addr_opt = if has_addresses {
                        addresses[idx].as_deref()
                    } else {
                        None
                    };
                    let early = if has_early {
                        &early_candidates_list[idx]
                    } else {
                        empty
                    };
                    self.get_candidates_v7_inner(main.as_slice(), addr_opt, early)
                })
                .collect()
        } else {
            mains
                .iter()
                .enumerate()
                .map(|(idx, main)| {
                    let addr_opt = if has_addresses {
                        addresses[idx].as_deref()
                    } else {
                        None
                    };
                    let early = if has_early {
                        &early_candidates_list[idx]
                    } else {
                        empty
                    };
                    self.get_candidates_v7_inner(main.as_slice(), addr_opt, early)
                })
                .collect()
        };

        let mut candidates_list = Vec::with_capacity(results.len());
        let mut scores_list = Vec::with_capacity(results.len());
        for result in results {
            match result {
                Ok((cands, scores)) => {
                    candidates_list.push(cands);
                    scores_list.push(scores);
                }
                Err(err) => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(err));
                }
            }
        }

        Ok((candidates_list, scores_list))
    }

    fn match_query_context_batch(
        &self,
        queries: Vec<String>,
        candidates_list: Vec<Vec<String>>,
    ) -> PyResult<Vec<Vec<CandidateMatch>>> {
        if queries.len() != candidates_list.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "queries and candidates_list length mismatch",
            ));
        }

        let mut output: Vec<Vec<CandidateMatch>> = Vec::with_capacity(queries.len());
        for (query, candidates) in queries.iter().zip(candidates_list.iter()) {
            let q = query.to_string();
            let ngrams = build_query_ngrams(&q, 7);
            let regex = build_regex_from_ngrams(&ngrams, false);
            let mut query_matches: Vec<CandidateMatch> = Vec::with_capacity(candidates.len());

            for cand in candidates {
                let entry = self.ror_entry_lite_get(cand).unwrap_or_else(|| RorEntryLite {
                    names: vec![cand.clone()],
                    acronyms: Vec::new(),
                    city: Vec::new(),
                    state: Vec::new(),
                    country: Vec::new(),
                    works_count: 0,
                });

                let fields: [&[String]; 5] = [
                    &entry.names,
                    &entry.acronyms,
                    &entry.city,
                    &entry.state,
                    &entry.country,
                ];

                let mut field_matches: Vec<Vec<String>> = Vec::with_capacity(5);
                let mut field_matched_text_lens: Vec<usize> = Vec::with_capacity(5);
                let mut field_any_text_in_query: Vec<bool> = Vec::with_capacity(5);

                let mut matched_split_set: HashSet<String> = HashSet::new();

                for field in fields.iter() {
                    let mut match_text: Vec<String> = Vec::new();
                    let mut any_text_in_query = false;

                    for text in field.iter() {
                        if q.contains(text) {
                            any_text_in_query = true;
                        }
                        let forward_matches =
                            find_query_ngrams_in_text_precompiled(text, &regex, 1, true);
                        for m in forward_matches {
                            for token in m.split_whitespace() {
                                if !STOPWORDS.contains(token) {
                                    matched_split_set.insert(token.to_string());
                                }
                            }
                            match_text.push(m);
                        }
                        let reverse_matches =
                            find_query_ngrams_in_text(text, &q, 1, true, false, 7);
                        for m in reverse_matches {
                            for token in m.split_whitespace() {
                                if !STOPWORDS.contains(token) {
                                    matched_split_set.insert(token.to_string());
                                }
                            }
                            match_text.push(m);
                        }
                    }

                    if match_text.is_empty() {
                        field_matches.push(Vec::new());
                        field_matched_text_lens.push(0);
                        field_any_text_in_query.push(any_text_in_query);
                        continue;
                    }

                    let mut indices: Vec<usize> = (0..match_text.len()).collect();
                    indices.sort_by_key(|i| (Reverse(match_text[*i].len()), *i));

                    let mut match_text_set: Vec<String> = Vec::new();
                    for i in indices {
                        let t = &match_text[i];
                        if match_text_set.iter().any(|existing| existing == t) {
                            continue;
                        }
                        if match_text_set.iter().any(|existing| existing.contains(t)) {
                            continue;
                        }
                        match_text_set.push(t.clone());
                    }

                    let mut matched_text_unigrams: HashSet<String> = HashSet::new();
                    for t in &match_text_set {
                        for token in t.split_whitespace() {
                            matched_text_unigrams.insert(token.to_string());
                        }
                    }
                    matched_text_unigrams.retain(|t| !STOPWORDS.contains(t.as_str()));
                    let matched_text_len = matched_len_from_tokens(&matched_text_unigrams);

                    field_matches.push(match_text_set);
                    field_matched_text_lens.push(matched_text_len);
                    field_any_text_in_query.push(any_text_in_query);
                }

                let matched_split_vec: Vec<String> = matched_split_set.into_iter().collect();
                query_matches.push(CandidateMatch {
                    field_matches,
                    field_matched_text_lens,
                    field_any_text_in_query,
                    matched_split_set: matched_split_vec,
                });
            }

            output.push(query_matches);
        }

        Ok(output)
    }

    fn debug_index_counts(&self) -> PyResult<(usize, usize, usize, usize, usize)> {
        let (word_len, ngram_len, address_len, typed_len, ror_len) = match &self.store {
            IndexStore::Owned(state) => (
                state.word_index.len(),
                state.ngram_index.len(),
                state.address_index.len(),
                state.typed_keys.len(),
                state.ror_ids.len(),
            ),
            IndexStore::Archived(archived) => (
                archived.root().word_index.len(),
                archived.root().ngram_index.len(),
                archived.root().address_index.len(),
                archived.root().typed_keys.len(),
                archived.root().ror_ids.len(),
            ),
        };
        Ok((word_len, ngram_len, address_len, typed_len, ror_len))
    }

    fn debug_word_postings_len(&self, token: String) -> PyResult<usize> {
        let key = token.as_str();
        let len = match &self.store {
            IndexStore::Owned(state) => state.word_index.get(key).map(|v| v.len()).unwrap_or(0),
            IndexStore::Archived(archived) => archived
                .root()
                .word_index
                .get(key)
                .map(|v| v.len())
                .unwrap_or(0),
        };
        Ok(len)
    }

    fn debug_word_tokens(&self) -> PyResult<Vec<String>> {
        let tokens = match &self.store {
            IndexStore::Owned(state) => state.word_index.keys().cloned().collect(),
            IndexStore::Archived(archived) => archived
                .root()
                .word_index
                .keys()
                .map(|k| k.as_str().to_string())
                .collect(),
        };
        Ok(tokens)
    }

    fn debug_ngram_postings_len(&self, token: String) -> PyResult<usize> {
        let key = token.as_str();
        let len = match &self.store {
            IndexStore::Owned(state) => state.ngram_index.get(key).map(|v| v.len()).unwrap_or(0),
            IndexStore::Archived(archived) => archived
                .root()
                .ngram_index
                .get(key)
                .map(|v| v.len())
                .unwrap_or(0),
        };
        Ok(len)
    }

    fn debug_ngram_tokens(&self) -> PyResult<Vec<String>> {
        let tokens = match &self.store {
            IndexStore::Owned(state) => state.ngram_index.keys().cloned().collect(),
            IndexStore::Archived(archived) => archived
                .root()
                .ngram_index
                .keys()
                .map(|k| k.as_str().to_string())
                .collect(),
        };
        Ok(tokens)
    }

    fn debug_address_tokens(&self) -> PyResult<Vec<String>> {
        let tokens = match &self.store {
            IndexStore::Owned(state) => state.address_index.keys().cloned().collect(),
            IndexStore::Archived(archived) => archived
                .root()
                .address_index
                .keys()
                .map(|k| k.as_str().to_string())
                .collect(),
        };
        Ok(tokens)
    }

    fn debug_typed_id_for_key(&self, key: String) -> PyResult<Option<u32>> {
        let key_ref = key.as_str();
        let idx = match &self.store {
            IndexStore::Owned(state) => state
                .key_to_id
                .get(key_ref)
                .copied()
                .map(|v| v as u32),
            IndexStore::Archived(archived) => archived.root().key_to_id.get(key_ref).copied(),
        };
        Ok(idx)
    }

    fn debug_word_length_at(&self, idx: usize) -> PyResult<f64> {
        let len = match &self.store {
            IndexStore::Owned(state) => state.word_lengths.get(idx).copied().unwrap_or(0.0),
            IndexStore::Archived(archived) => archived
                .root()
                .word_lengths
                .get(idx)
                .copied()
                .unwrap_or(0.0),
        };
        Ok(len)
    }

    fn debug_ngram_length_at(&self, idx: usize) -> PyResult<f64> {
        let len = match &self.store {
            IndexStore::Owned(state) => state.ngram_lengths.get(idx).copied().unwrap_or(0.0),
            IndexStore::Archived(archived) => archived
                .root()
                .ngram_lengths
                .get(idx)
                .copied()
                .unwrap_or(0.0),
        };
        Ok(len)
    }

    fn debug_fix_text(&self, input: String) -> PyResult<String> {
        Ok(fix_text(&input))
    }

    fn debug_text_unidecode(&self, input: String) -> PyResult<String> {
        Ok(text_unidecode(&input))
    }

    fn debug_fix_text_steps(&self, input: String) -> PyResult<(String, String, String, String, String)> {
        let mut s = text_unidecode(&input);
        let step1 = s.clone();
        s = s
            .replace("#TAB#", "")
            .replace(".", "")
            .replace(" & ", " and ")
            .replace("&", "n");
        let step2 = s.clone();
        s = RE_APOSTROPHE_S
            .replace_all(&s, |caps: &regex::Captures| {
                format!("{}s", &caps[1])
            })
            .to_string();
        let step3 = s.clone();
        s = RE_BASIC.replace_all(&s, " ").to_string();
        let step4 = s.clone();
        s = RE_WHITESPACE.replace_all(&s, " ").to_string();
        let step5 = s.trim().to_string();
        Ok((step1, step2, step3, step4, step5))
    }

    fn debug_unidecode_len(&self) -> PyResult<usize> {
        Ok(TEXT_UNIDECODE.len())
    }

    fn debug_unidecode_codepoint(&self, codepoint: u32) -> PyResult<Option<String>> {
        if codepoint == 0 {
            return Ok(Some("\0".to_string()));
        }
        let idx = (codepoint as usize).saturating_sub(1);
        Ok(TEXT_UNIDECODE.get(idx).cloned())
    }

    fn debug_ror_entry_country(&self, ror_id: String) -> PyResult<Vec<String>> {
        Ok(self
            .ror_entry_lite_get(&ror_id)
            .map(|entry| entry.country)
            .unwrap_or_default())
    }
}

#[pymodule]
fn s2aff_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RorIndex>()?;
    m.add_class::<CandidateMatch>()?;
    Ok(())
}
