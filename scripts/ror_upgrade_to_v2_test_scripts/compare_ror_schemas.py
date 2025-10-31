"""
Temporary script to compare ROR v1 and v2 schema processing.
Loads both schema versions and compares the resulting RORIndex structures.
"""

import os
from s2aff.ror import RORIndex
from s2aff.consts import PROJECT_ROOT_PATH


def summarize_index(idx, label):
    """Generate summary statistics for a RORIndex."""
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")

    # Basic counts
    print(f"Total ROR records: {len(idx.ror_dict)}")
    print(f"GRID mappings: {len(idx.grid_to_ror)}")
    print(f"ISNI mappings: {len(idx.isni_to_ror)}")

    # Sample a few records to check structure
    sample_ids = list(idx.ror_dict.keys())[:3]

    print(f"\nSample records (first 3):")
    for ror_id in sample_ids:
        rec = idx.ror_dict[ror_id]
        print(f"\n  ID: {ror_id}")
        print(f"    Name: {rec.get('name', 'N/A')}")
        print(f"    Aliases count: {len(rec.get('aliases', []))}")
        print(f"    Acronyms count: {len(rec.get('acronyms', []))}")
        print(f"    Labels count: {len(rec.get('labels', []))}")

        if rec.get("addresses"):
            addr = rec["addresses"][0]
            print(f"    Address:")
            print(f"      City: {addr.get('city', 'N/A')}")
            print(f"      State: {addr.get('state', 'N/A')}")
            print(f"      Country geonames ID: {addr.get('country_geonames_id', 'N/A')}")
            print(f"      Country code (v2): {addr.get('country_code', 'N/A')}")

        ext_ids = rec.get("external_ids", {})
        print(f"    External IDs: {list(ext_ids.keys())}")

        print(f"    Wikipedia page: {rec.get('wikipedia_page', [])}")
        print(f"    Wikipedia URL: {rec.get('wikipedia_url', 'N/A')}")

    # Check for records with missing fields
    missing_city = sum(1 for r in idx.ror_dict.values() if r.get("addresses") and not r["addresses"][0].get("city"))
    missing_geo_id = sum(
        1 for r in idx.ror_dict.values() if r.get("addresses") and not r["addresses"][0].get("country_geonames_id")
    )

    print(f"\nData quality:")
    print(f"  Records with addresses but no city: {missing_city}")
    print(f"  Records with addresses but no country_geonames_id: {missing_geo_id}")

    # Index structures
    print(f"\nIndex structures:")
    print(f"  word_index['inverted_index'] keys: {len(idx.word_index['inverted_index'])}")
    print(f"  ngram_index['inverted_index'] keys: {len(idx.ngram_index['inverted_index'])}")
    print(f"  address_index['full_index'] keys: {len(idx.address_index['full_index'])}")
    print(f"  address_index['city_index'] keys: {len(idx.address_index['city_index'])}")
    print(f"  ror_name_direct_lookup keys: {len(idx.ror_name_direct_lookup)}")
    print(f"  ror_address_counter keys: {len(idx.ror_address_counter)}")


def compare_indices(idx_v1, idx_v2):
    """Compare two RORIndex instances."""
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")

    # Compare record counts
    common_ids = set(idx_v1.ror_dict.keys()) & set(idx_v2.ror_dict.keys())
    only_v1 = set(idx_v1.ror_dict.keys()) - set(idx_v2.ror_dict.keys())
    only_v2 = set(idx_v2.ror_dict.keys()) - set(idx_v1.ror_dict.keys())

    print(f"\nRecord overlap:")
    print(f"  Common ROR IDs: {len(common_ids)}")
    print(f"  Only in v1: {len(only_v1)}")
    print(f"  Only in v2: {len(only_v2)}")

    # Compare a common record in detail
    if common_ids:
        sample_id = list(common_ids)[0]
        rec_v1 = idx_v1.ror_dict[sample_id]
        rec_v2 = idx_v2.ror_dict[sample_id]

        print(f"\nDetailed comparison of {sample_id}:")
        print(f"  Name match: {rec_v1.get('name') == rec_v2.get('name')}")
        print(f"    v1: {rec_v1.get('name')}")
        print(f"    v2: {rec_v2.get('name')}")

        print(f"  Aliases count: v1={len(rec_v1.get('aliases', []))}, v2={len(rec_v2.get('aliases', []))}")
        print(f"  Acronyms count: v1={len(rec_v1.get('acronyms', []))}, v2={len(rec_v2.get('acronyms', []))}")
        print(f"  Labels count: v1={len(rec_v1.get('labels', []))}, v2={len(rec_v2.get('labels', []))}")

        if rec_v1.get("addresses") and rec_v2.get("addresses"):
            addr_v1 = rec_v1["addresses"][0]
            addr_v2 = rec_v2["addresses"][0]
            print(f"  City match: {addr_v1.get('city') == addr_v2.get('city')}")
            print(f"    v1: {addr_v1.get('city')}")
            print(f"    v2: {addr_v2.get('city')}")
            print(f"  State match: {addr_v1.get('state') == addr_v2.get('state')}")
            print(
                f"  Country geonames ID: v1={addr_v1.get('country_geonames_id')}, v2={addr_v2.get('country_geonames_id')}"
            )
            print(f"  Country code (v2 field): v1={addr_v1.get('country_code')}, v2={addr_v2.get('country_code')}")

        print(f"  External IDs match: {rec_v1.get('external_ids') == rec_v2.get('external_ids')}")
        if rec_v1.get("external_ids") != rec_v2.get("external_ids"):
            print(f"    v1 keys: {set(rec_v1.get('external_ids', {}).keys())}")
            print(f"    v2 keys: {set(rec_v2.get('external_ids', {}).keys())}")

        print(f"  Wikipedia page: v1={rec_v1.get('wikipedia_page')}, v2={rec_v2.get('wikipedia_page')}")

    # Compare index sizes
    print(f"\nIndex structure comparison:")
    print(
        f"  word_index['inverted_index']: v1={len(idx_v1.word_index['inverted_index'])}, v2={len(idx_v2.word_index['inverted_index'])}"
    )
    print(
        f"  ngram_index['inverted_index']: v1={len(idx_v1.ngram_index['inverted_index'])}, v2={len(idx_v2.ngram_index['inverted_index'])}"
    )
    print(
        f"  address_index['full_index']: v1={len(idx_v1.address_index['full_index'])}, v2={len(idx_v2.address_index['full_index'])}"
    )
    print(
        f"  address_index['city_index']: v1={len(idx_v1.address_index['city_index'])}, v2={len(idx_v2.address_index['city_index'])}"
    )
    print(f"  ror_name_direct_lookup: v1={len(idx_v1.ror_name_direct_lookup)}, v2={len(idx_v2.ror_name_direct_lookup)}")


if __name__ == "__main__":
    # NOTE: these are not in the repo by default
    # you'll have to go get them to run this test script
    # this script is here more for documentation purposes
    v1_file = os.path.join(PROJECT_ROOT_PATH, "data", "v1.73-2025-10-28-ror-data.json")
    v2_file = os.path.join(PROJECT_ROOT_PATH, "data", "v1.73-2025-10-28-ror-data_schema_v2.json")

    print("Loading v1 schema ROR data...")
    idx_v1 = RORIndex(ror_data_path=v1_file)

    print("Loading v2 schema ROR data...")
    idx_v2 = RORIndex(ror_data_path=v2_file)

    # Generate summaries
    summarize_index(idx_v1, "V1 SCHEMA INDEX")
    summarize_index(idx_v2, "V2 SCHEMA INDEX (after coercion)")

    # Compare
    compare_indices(idx_v1, idx_v2)

    print(f"\n{'='*60}")
    print("CONCLUSION")
    print(f"{'='*60}")
    print("If the comparison shows matching structures and similar data,")
    print("the v2-to-v1 coercion is working correctly.")
