import os
import sys
import json
import random
import re
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Set
from SPARQLWrapper import SPARQLWrapper, JSON
from collections import defaultdict

# Configuration
os.chdir(sys.path[0])  # Change working directory to script directory
SPARQL_ENDPOINT = "http://localhost:8890/sparql"
NS_PREFIX = "http://rdf.freebase.com/ns/"

# Initialize SPARQL wrapper
sparql = SPARQLWrapper(SPARQL_ENDPOINT)
sparql.setReturnFormat(JSON)


def remove_ns_prefix(id):
    if id.startswith("ns:"):
        return id[3:]
    else:
        return id

def execute_sparql(query: str, retries: int = 3) -> Optional[Dict]:
    """Execute SPARQL query with retry logic"""
    for attempt in range(retries):
        try:
            sparql.setQuery(query)
            results = sparql.query().convert()
            return results
        except Exception as e:
            if attempt == retries - 1:
                print(f"SPARQL query failed after {retries} attempts: {e}")
                return None
            print(f"Attempt {attempt + 1} failed, retrying...")
    return None


def remove_prefix(value: str) -> str:
    """Remove Freebase URI prefix"""
    if not value:
        return value
    return value.replace(NS_PREFIX, "").replace("ns:", "")


def get_entity_name(entity_id: str, cache: Dict = {}) -> str:
    """Get entity name from Freebase ID with caching"""
    if entity_id in cache:
        return cache[entity_id]
    
    entity_id_clean = remove_prefix(entity_id)
    
    # Skip if not an entity ID
    if not (entity_id_clean.startswith("m.") or entity_id_clean.startswith("g.")):
        cache[entity_id] = entity_id_clean
        return entity_id_clean
    
    query = f"""
    PREFIX ns: <http://rdf.freebase.com/ns/>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    SELECT DISTINCT ?name WHERE {{
        VALUES ?entity {{ ns:{entity_id_clean} }}
        {{
            ?entity ns:type.object.name ?name .
        }}
        UNION
        {{
            ?entity owl:sameAs ?name .
        }}
    }}
    LIMIT 1
    """
    results = execute_sparql(query)
    if results and results.get("results", {}).get("bindings"):
        name = results["results"]["bindings"][0]["name"]["value"]
        cache[entity_id] = name
        return name
    
    cache[entity_id] = entity_id_clean
    return entity_id_clean


def shorten_relation(relation: str) -> str:
    """Shorten relation to last two parts"""
    relation = remove_prefix(relation)
    parts = relation.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return relation


def extract_all_triples_from_sparql(sparql_query: str) -> List[Tuple[str, str, str]]:
    """
    从SPARQL查询语句中提取所有三元组模式，并实例化
    这些才是真正与问题相关的三元组
    """
    # 提取sparql中的三元组模式
    lines = sparql_query.split('\n')
    triples = re.findall(
        r"(\?\w+|ns:[\w.:]+)\s+(ns:[\w.:]+)\s+(\?\w+|ns:[\w.:]+)", sparql_query
    )
    triples = [list(triple) for triple in triples]

    variables = re.findall(r"(\?\w+)", sparql_query)
    variables = sorted(list(set(variables)))

    topic_ents = set()
    all_rels = set()
    for triple in triples:
        all_rels.add(triple[1])
        if triple[0].startswith('ns:'):
            topic_ents.add(triple[0])
        if triple[-1].startswith('ns:'):
            topic_ents.add(triple[0])
    topic_ents = list(topic_ents)
    all_rels = list(all_rels)

    for i, line in enumerate(lines):
        if line.startswith('SELECT DISTINCT'):
            parts = line.split()
            parts = parts[:2] + variables
            lines[i] = " ".join(parts)

    sparql_query = "\n".join(lines)

    sparql.setQuery(sparql_query)
    results = sparql.query().convert()

    try:
        all_bound_triples = []
        all_topic_node_to_path = []
        for binding in results["results"]["bindings"]:
            ans = binding["x"]["value"].replace(NS_PREFIX, "")
            bound_triples = []
            for triple in triples:
                bound_triple = triple.copy()

                if triple[0].startswith("?") and triple[0][1:] in binding:
                    bound_triple[0] = binding[triple[0][1:]]["value"].replace(NS_PREFIX, "ns:")

                if triple[-1].startswith("?") and triple[-1][1:] in binding:
                    bound_triple[-1] = binding[triple[-1][1:]]["value"].replace(
                        NS_PREFIX, "ns:"
                    )

                # remove all ns prefix:
                for i in range(3):
                    bound_triple[i] = remove_ns_prefix(bound_triple[i])

                if bound_triple[0].startswith("?") or bound_triple[-1].startswith("?"):
                    # ignore unbound triple without bound var
                    continue

                bound_triples.append(bound_triple)

            all_bound_triples.extend(bound_triples)

    except Exception as e:
        # in some samples, answer entity and topic entity are the same
        crucial_edges = []
        print(e)
        print(all_topic_node_to_path)

    return all_bound_triples

def get_related_triples(entity_id: str, max_triples: int = 100) -> List[Tuple[str, str, str]]:
    """
    Get triples related to an entity (for noise generation)
    Gets both outgoing and incoming triples
    """
    entity_id = remove_prefix(entity_id)
    triples = []
    
    # Query for outgoing relations
    query_out = f"""
    PREFIX ns: <{NS_PREFIX}>
    SELECT DISTINCT ?pred ?obj WHERE {{
        ns:{entity_id} ?pred ?obj .
        FILTER(!isLiteral(?obj) || lang(?obj) = "en" || lang(?obj) = "")
        FILTER(?pred != ns:type.object.type)
        FILTER(?pred != ns:common.topic.webpage)
        FILTER(?pred != ns:common.topic.image)
    }}
    LIMIT {max_triples}
    """
    
    results = execute_sparql(query_out)
    if results:
        for binding in results.get("results", {}).get("bindings", []):
            pred = remove_prefix(binding["pred"]["value"])
            obj = binding["obj"]["value"]
            if obj.startswith(NS_PREFIX):
                obj = remove_prefix(obj)
            triples.append([entity_id, pred, obj])
    
    # Query for incoming relations
    query_in = f"""
    PREFIX ns: <{NS_PREFIX}>
    SELECT DISTINCT ?subj ?pred WHERE {{
        ?subj ?pred ns:{entity_id} .
        FILTER(?pred != ns:type.object.type)
        FILTER(?pred != ns:common.topic.webpage)
        FILTER(?pred != ns:common.topic.image)
    }}
    LIMIT {max_triples}
    """
    
    results = execute_sparql(query_in)
    if results:
        for binding in results.get("results", {}).get("bindings", []):
            subj = remove_prefix(binding["subj"]["value"])
            pred = remove_prefix(binding["pred"]["value"])
            triples.append([subj, pred, entity_id])
    
    return triples


def format_triple_as_text(triple: Tuple[str, str, str], name_cache: Dict = {}) -> str:
    """
    Format triple as readable text
    Converts IDs to names where possible
    """
    subj, pred, obj = triple
    
    # Get names for entities (those starting with m. or g.)
    if subj.startswith("m.") or subj.startswith("g."):
        subj_name = get_entity_name(subj, name_cache)
    else:
        subj_name = subj
    
    if obj.startswith("m.") or obj.startswith("g."):
        obj_name = get_entity_name(obj, name_cache)
    else:
        obj_name = obj
    
    return f"{subj_name} {pred} {obj_name}"

def extract_topic_entities(sample: Dict, dataset: str) -> List[str]:
    """
    Extract topic entities from sample based on dataset format
    """
    entities = []
    
    if dataset == "webqsp":
        # WebQSP format
        if "Parses" in sample and sample["Parses"]:
            topic_mid = sample["Parses"][0].get("TopicEntityMid")
            if topic_mid:
                entities.append(remove_prefix(topic_mid))
    
    elif dataset == "grailqa":
        # GrailQA format
        if "graph_query" in sample and "nodes" in sample["graph_query"]:
            for node in sample["graph_query"]["nodes"]:
                if node.get("node_type") == "entity":
                    entity_id = node.get("id")
                    if entity_id:
                        entities.append(remove_prefix(entity_id))
    
    elif dataset == "cwq":
        # CWQ format - 从topic_entity中获取
        if "topic_entity" in sample:
            for entity_id in sample["topic_entity"].keys():
                entities.append(remove_prefix(entity_id))
    
    return list(set(entities))  # Remove duplicates

def extract_answers(sample: Dict, dataset: str) -> List[str]:
    """
    Extract answer entities from sample based on dataset format
    """
    answers = []
    
    if dataset == "webqsp":
        if "Parses" in sample and sample["Parses"]:
            for answer in sample["Parses"][0].get("Answers"):
                answer_name = answer.get("EntityName")
                if answer_name:
                    answers.append(answer_name)
    
    elif dataset == "grailqa":
        if "answer" in sample:
            for answer in sample["answer"]:
                if isinstance(answer, dict):
                    answer_name = answer.get("entity_name") or answer.get("answer_argument", "")
                else:
                    answer_name = str(answer)
                if answer_name:
                    answers.append(answer_name)
    
    elif dataset == "cwq":
        if "answer" in sample:
            if isinstance(sample["answer"], list):
                for ans in sample["answer"]:
                    if isinstance(ans, dict):
                        answers.append(ans.get("answer_argument", str(ans)))
                    else:
                        answers.append(str(ans))
            else:
                answers.append(str(sample["answer"]))
    
    return answers


def process_sample(sample: Dict, dataset: str, noise_multiplier: int = 8) -> Optional[Dict]:
    """
    Process a single sample to extract relevant and noise triples
    
    Args:
        sample: Sample from dataset
        dataset: Dataset name (webqsp, grailqa, cwq)
        noise_multiplier: Number of noise triples per relevant triple (default: 8)
    
    Returns:
        Processed sample with question, answers, and ctxs
    """
    # Extract question
    if dataset == "webqsp":
        question = sample.get("RawQuestion")
    elif dataset == "grailqa":
        question = sample.get("question")
    elif dataset == "cwq":
        question = sample.get("question")
    else:
        return None
    
    if not question:
        return None
    
    # Extract SPARQL query
    sparql_query = None
    if dataset == "webqsp":
        if "Parses" in sample and sample["Parses"]:
            sparql_query = sample["Parses"][0].get("Sparql")
    elif dataset == "grailqa":
        sparql_query = sample.get("sparql_query")
    elif dataset == "cwq":
        sparql_query = sample.get("sparql")
    
    if not sparql_query:
        return None
    
    # Extract answers
    answers = extract_answers(sample, dataset)
    
    # Extract topic entities
    topic_entities = extract_topic_entities(sample, dataset)

    # 从SPARQL语句中提取相关三元组
    # print(f"\nProcessing: {question[:50]}...")
    relevant_triples = extract_all_triples_from_sparql(sparql_query)
    
    # If no relevant triples found, skip this sample
    if not relevant_triples:
        print(f"Warning: No relevant triples extracted from SPARQL: {question[:50]}...")
        return None
    
    # print(f"Found {len(relevant_triples)} relevant triples from SPARQL")
    
    # Calculate how many noise triples we need: relevant_count × noise_multiplier
    gold_count = len(relevant_triples)
    noise_count = gold_count * noise_multiplier
    
    # print(f"Need {noise_count} noise triples ({gold_count} × {noise_multiplier})")
    
    # Collect noise triples from all topic entities
    all_noise_triples = []
    for entity_id in topic_entities:
        entity_triples = get_related_triples(entity_id, max_triples=200)
        all_noise_triples.extend(entity_triples)
    
    # print(f"Collected {len(all_noise_triples)} candidate noise triples")
    
    # Filter out triples that are in relevant_triples
    candidate_noise = [t for t in all_noise_triples if t not in relevant_triples]
    
    # print(f"After filtering, {len(candidate_noise)} noise candidates remain")
    
    # Randomly sample noise triples
    noise_triples = []
    if len(candidate_noise) >= noise_count:
        noise_triples = random.sample(candidate_noise, noise_count)
    else:
        noise_triples = candidate_noise
        # print(f"Warning: Only found {len(noise_triples)} noise triples, needed {noise_count}")
    
    # Format triples as ctxs with name caching
    name_cache = {}
    ctxs = []
    
    # Add relevant triples
    for triple in relevant_triples:
        text = format_triple_as_text(triple, name_cache)
        ctxs.append({
            "text": text,
            "isgold": True
        })
    
    # Add noise triples
    for triple in noise_triples:
        text = format_triple_as_text(triple, name_cache)
        ctxs.append({
            "text": text,
            "isgold": False
        })
    
    # Shuffle ctxs to mix gold and noise
    random.shuffle(ctxs)
    
    # print(f"Final: {len([c for c in ctxs if c['isgold']])} gold + {len([c for c in ctxs if not c['isgold']])} noise = {len(ctxs)} total triples")
    
    return {
        "question": question,
        "answers": answers,
        "ctxs": ctxs
    }


def process_dataset(
    input_path: str,
    output_path: str,
    dataset: str,
    noise_multiplier: int = 8,
    max_samples: Optional[int] = None
):
    """
    Process entire dataset
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file
        dataset: Dataset name (webqsp, grailqa, cwq)
        noise_multiplier: Number of noise triples per relevant triple
        max_samples: Maximum number of samples to process (None for all)
    """
    print(f"Loading dataset from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    if dataset == "webqsp":
        samples = samples["Questions"]

    if max_samples:
        samples = samples[:max_samples]
    
    print(f"Processing {len(samples)} samples from {dataset}...")
    processed_samples = []
    
    for sample in tqdm(samples):
        try:
            processed = process_sample(sample, dataset, noise_multiplier)
            if processed:
                processed_samples.append(processed)
                # print(f"✓ Successfully processed")
            else:
                print(f"✗ Skipped")
        except Exception as e:
            print(f"✗ Error processing sample: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"Successfully processed {len(processed_samples)}/{len(samples)} samples")
    print(f"{'='*60}")
    
    # Save results
    print(f"Saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_samples, f, indent=2, ensure_ascii=False)
    
    print(f"Done! Saved {len(processed_samples)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build dataset with relevant and noise triples")
    parser.add_argument(
        "--dataset",
        type=str,
        default="webqsp",
        choices=["webqsp", "grailqa", "cwq", "all"],
        help="Dataset to process"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="..",
        help="Input directory containing dataset files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="..",
        help="Output directory for processed files"
    )
    parser.add_argument(
        "--noise_multiplier",
        type=int,
        default=8,
        help="Number of noise triples per relevant triple (default: 8)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process per dataset (None for all)"
    )

    args = parser.parse_args()
    
    # Dataset paths
    dataset_paths = {
        "webqsp": ("webqsp/WebQSP.train.json", "webqsp/webqsp_with_triples.train.json"),
        "grailqa": ("grailqa/grailqa.train.json", "grailqa/grailqa_with_triples.train.json"),
        "cwq": ("cwq/cwq.train.json", "cwq/cwq_with_triples.train.json")
    }
    
    if args.dataset == "all":
        datasets_to_process = ["webqsp", "grailqa", "cwq"]
    else:
        datasets_to_process = [args.dataset]
    
    for dataset_name in datasets_to_process:
        input_file, output_file = dataset_paths[dataset_name]
        input_path = os.path.join(args.input_dir, input_file)
        output_path = os.path.join(args.output_dir, output_file)
        
        if not os.path.exists(input_path):
            print(f"Warning: Input file {input_path} not found, skipping {dataset_name}")
            continue
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name.upper()}")
        print(f"{'='*60}")
        
        process_dataset(
            input_path=input_path,
            output_path=output_path,
            dataset=dataset_name,
            noise_multiplier=args.noise_multiplier,
            max_samples=args.max_samples
        )


if __name__ == "__main__":
    main()