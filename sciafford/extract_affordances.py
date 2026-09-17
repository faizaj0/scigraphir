"""
SciAfford: extract role-typed contribution and problem affordance representations.

Document affordance representations describe contribution capabilities, mechanisms, dependencies
and limitations. problem requirement representations describe problem requirements and limitations.
Field-neutral functional descriptions support affordance-lifted graph matching.

Document schema: task, domain, contributions[{name, achieves[], overcomes[],
mechanism[], builds_on[], improves_on[]}], prior_methods[], task_limitations[],
findings[]. Query schema: task, domain, needs[], current_methods[],
task_limitations[]. These schema keys are shared with the graph builder.

Providers: OpenAI (default) or Anthropic. Extractions use a resumable JSONL cache.
From the sciafford directory:
    python extract_affordances.py --dataset sir4_cs --side doc --workers 32
    python extract_affordances.py --dataset sir4_cs --side query --workers 32
"""
import argparse, json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("SCIGRAPHIR_ROOT") or os.path.dirname(HERE)   # repo root
DATA_ROOT = os.path.join(ROOT, "retriever", "data")
CACHE = os.path.join(HERE, "cache")

# Corpus + cache resolution lives in one module so the five pipeline scripts
# cannot drift. Default dataset "tomato" reproduces every path below exactly.
sys.path.insert(0, ROOT)
from scigraphir_paths import add_dataset_arg, banner, corpus_dir, affordances_path, set_dataset  # noqa: E402

# load OPENAI_API_KEY / ANTHROPIC_API_KEY from .env.local (script isn't run with them exported)
_envf = os.path.join(HERE, ".env.local")
if os.path.exists(_envf):
    for _line in open(_envf):
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

# ---- shared domain-strip / granularity rules (reused in both prompts) ----
RULES = (
    "RULES (apply, then RE-READ and fix):\n"
    "1. DOMAIN-STRIP every function / mechanism / limitation / overcomes item: no organ, disease, "
    "material, organism, dataset, or field noun. 'segment brain MRI' -> 'partition an input into "
    "labeled regions'. task / domain / method-name may keep the domain.\n"
    "2. Reusable, not vague: not 'analyze data' / 'build a model' / 'improve accuracy'.\n"
    "3. PROVENANCE by position: a method the paper INTRODUCES (\"we propose/develop/present …\") -> a "
    "contribution (its own node). A method it USES/builds on (\"using / based on / following …\") -> "
    "that contribution's builds_on. A prior method it BEATS/REPLACES (named in the background/gap) -> "
    "that contribution's improves_on. builds_on / improves_on are method->method edges: list the OTHER "
    "methods by name. When unsure, it is builds_on, not a new contribution.\n"
    "4. A LIMITATION is a functional flaw of a METHOD or a TASK, stripped to the failure itself "
    "('not scalable', 'requires manual tuning', 'not robust to heterogeneity'), NOT the domain symptom.\n"
    "5. overcomes = which limitation(s) the contribution resolves; it should mirror a prior_method / "
    "task limitation in wording so they can be matched.\n"
    "6. Your own words about THIS text. Never copy the example verbatim."
)

SYS_DOC = (
    'Extract the relational affordance representation of a scientific paper so others can find it as an INSPIRATION for '
    "their method.\n"
    "A CONTRIBUTION is a METHOD (a node). `builds_on` and `improves_on` are edges FROM that method TO "
    "OTHER methods (method-to-method) — NOT from the paper. The paper only CONTRIBUTES the method.\n\n"
    "FIELDS:\n"
    "- task (1-2, domain OK): the problem the paper addresses.\n"
    "- domain (1, short): the field.\n"
    "- contributions (1-2): each is a method the paper INTRODUCES, with:\n"
    "    name (domain OK, what the method is CALLED),\n"
    "    achieves (1-2 stripped functions = the reusable capability it OFFERS),\n"
    "    overcomes (0-3 stripped limitations it resolves),\n"
    "    mechanism (0-2 stripped: HOW it works internally),\n"
    "    builds_on (2-4 method NAMES, class-level: OTHER methods this method is built from / relies on "
    "— a method->method dependency),\n"
    "    improves_on (0-2: each {name, has_limitation (1-2 stripped flaws)} — prior methods this method "
    "SUPERSEDES or beats; record their flaws).\n"
    "- task_limitations (0-2, stripped): inherent challenges of the task itself.\n"
    "- findings (0-2, domain OK): key results (anchor).\n"
    "- causal_findings (0-2): ONLY a result that states a REUSABLE cause->effect MECHANISM or a "
    "property->affordance another field could reuse. Each {cause (stripped), effect (stripped), "
    "about (stripped phenomenon/function)}. A domain-specific result ('X correlates with age in cohort "
    "Y') is NOT causal — leave it in findings. This is how a FINDINGS paper (not a method paper) "
    "becomes an inspiration.\n\n"
    "WORKED EXAMPLE (method paper) — 'nnU-Net: a self-configuring method for deep-learning biomedical image segmentation':\n"
    '{"task":["biomedical image segmentation"],"domain":"medical imaging / deep learning",'
    '"contributions":[{"name":"nnU-Net","achieves":["partition an input into labeled regions",'
    '"automatically configure a processing pipeline from input properties"],'
    '"overcomes":["dataset-dependence","need for manual configuration"],'
    '"mechanism":["an encoder-decoder that learns multi-scale features and reconstructs a dense labeled map"],'
    '"builds_on":["supervised deep learning","convolutional encoder-decoder network"],'
    '"improves_on":[{"name":"specialized segmentation solutions","has_limitation":["dataset-dependent","require manual tuning"]}]}],'
    '"task_limitations":["heterogeneous datasets and hardware make a single design non-trivial"],'
    '"findings":["surpasses specialized solutions on 23 public datasets"],"causal_findings":[]}\n\n'
    "WORKED EXAMPLE (findings paper) — 'The Additive-Area Heuristic: an efficient but illusory means of visual area approximation':\n"
    '{"task":["visual area estimation"],"domain":"cognitive psychology","contributions":[],'
    '"task_limitations":["area estimation is understudied vs number"],'
    '"findings":["area distortions visible in simple demonstrations"],'
    '"causal_findings":[{"cause":"estimating combined area by summing component areas",'
    '"effect":"systematic distortion of perceived area","about":"visual area estimation"}]}\n'
    "Notice: a findings paper has empty contributions; its value is the causal_finding.\n\n"
    + RULES + "\nOutput ONLY the JSON object."
)

SYS_QUERY = (
    'A researcher states a NEED (a problem, not a solution). Extract its affordance representation so it can be matched to '
    "a paper whose contribution overcomes the same limitation and achieves the same function.\n\n"
    "FIELDS:\n"
    "- task (0-1, domain OK): the named problem.\n"
    "- domain (1, short).\n"
    "- needs (1-3, stripped functions incl. implicit sub-steps): the reusable operations a useful "
    "paper would let them DO. 'measure lesion volume in a scan' -> 'partition an input into labeled "
    "regions' AND 'measure the extent of a delineated part'.\n"
    "- current_methods (0-3): each {name, has_limitation (1-2 stripped flaws)} — the approaches the "
    "query says it currently uses and why they fall short.\n"
    "- task_limitations (0-2, stripped): inherent challenges the query flags.\n\n"
    "WORKED EXAMPLE — 'How to quantify white-matter-hyperintensity volume from brain MRI, where current "
    "tools (BIANCA, DeepMedic) do not scale and visual rating is rater-dependent?':\n"
    '{"task":["quantification of white matter hyperintensity"],"domain":"neuroimaging",'
    '"needs":["partition an input into labeled regions","measure the extent of a delineated part"],'
    '"current_methods":[{"name":"BIANCA / DeepMedic segmentation","has_limitation":["limited scalability","poor reproducibility"]},'
    '{"name":"visual rating scales","has_limitation":["rater variability"]}],'
    '"task_limitations":["reproducibility across large heterogeneous cohorts"]}\n\n'
    + RULES + "\nOutput ONLY the JSON object."
)


# ONE client for the whole run, built once and shared across threads.
# This used to be `OpenAI()` INSIDE call_openai, i.e. a fresh client -- and so a
# fresh connection pool and TLS handshake -- for every one of the 24,384
# documents. The SDK's client is thread-safe and pools up to 1000 connections,
# so constructing it per call threw away every keep-alive and added a full
# handshake to each request. That overhead is per-call, so it scales with
# workers: the more parallel you go, the more of the run is spent shaking hands.
_CLIENTS: dict = {}


def _client(provider):
    if provider not in _CLIENTS:
        if provider == "openai":
            import httpx
            from openai import OpenAI
            # The SDK pools 1000 connections but keeps only 100 ALIVE. Above ~100
            # workers the surplus connections are torn down and re-handshaked
            # every call, so raising --workers starts buying TLS round-trips
            # instead of throughput. Match keepalive to the pool size.
            _CLIENTS[provider] = OpenAI(
                max_retries=0,                          # we do our own backoff
                http_client=httpx.Client(
                    limits=httpx.Limits(max_connections=1000,
                                        max_keepalive_connections=1000),
                    timeout=httpx.Timeout(90.0, connect=10.0)))
        else:
            import anthropic
            _CLIENTS[provider] = anthropic.Anthropic(max_retries=0)
    return _CLIENTS[provider]


def call_openai(sys_msg, text, model):
    client = _client("openai")
    r = client.chat.completions.create(
        model=model, temperature=0.2, max_tokens=700,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": sys_msg},
                  {"role": "user", "content": str(text)[:3000]}])
    return json.loads(r.choices[0].message.content)


def call_anthropic(sys_msg, text, model):
    client = _client("anthropic")
    r = client.messages.create(
        model=model, max_tokens=900, temperature=0.2,
        system=sys_msg + "\nReturn ONLY valid JSON, no prose.",
        messages=[{"role": "user", "content": str(text)[:3000]}])
    out = r.content[0].text.strip()
    if out.startswith("```"):
        out = out.split("```")[1].lstrip("json").strip()
    return json.loads(out)


def gen_one(text, side, provider, model):
    """Return the affordance representation, or {} if every attempt failed. Callers MUST NOT cache {}.

    Backoff is exponential with jitter, not a flat 2s. Under a rate-limit burst
    a flat sleep re-synchronises every worker onto the same retry instant, so
    all N of them collide again, exhaust the 4 attempts together, and return {}
    en masse. That is precisely the failure mode that gets worse with more
    workers, which is when you least want it.
    """
    sys_msg = SYS_DOC if side == "doc" else SYS_QUERY
    for attempt in range(5):
        try:
            d = call_openai(sys_msg, text, model) if provider == "openai" else call_anthropic(sys_msg, text, model)
            if isinstance(d, dict) and d:
                return d
        except Exception:
            pass
        time.sleep(min(2 ** attempt, 16) * (0.5 + random.random()))
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", required=True, choices=["doc", "query"])
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="only first N (for a quick test)")
    add_dataset_arg(ap)
    a = ap.parse_args()
    model = a.model or ("gpt-4o-mini" if a.provider == "openai" else "claude-haiku-4-5-20251001")
    set_dataset(a.dataset)
    print(banner())

    DATA = corpus_dir(a.split)
    if a.side == "doc":
        items = list(json.load(open(f"{DATA}/raw/documents.json")).items())
    else:
        items = [(x["id"], x["question"]) for x in json.load(open(f"{DATA}/raw/{a.split}.json"))]
    if a.limit:
        items = items[:a.limit]

    # THE cache path. This line used to build the TOMATO filename directly,
    # so a --dataset sir4_cs_smoke run appended 898 SIR-4 affordance representations into the
    # TOMATO caches and then read them back as TOMATO. affordances_path() is the
    # only thing that may decide this; it preserves the TOMATO naming quirk
    # (unsuffixed for test) and scopes every other dataset into its own dir.
    cache = affordances_path(a.side, a.split)
    done = set()
    if os.path.exists(cache):
        for line in open(cache):
            try: done.add(json.loads(line)["id"])
            except Exception: pass
    todo = [(i, t) for i, t in items if i not in done]
    print(f"[SciAfford] {a.side}: {len(items)} items, {len(done)} cached, {len(todo)} to do "
          f"(provider={a.provider}, model={model}) -> {cache}")

    with open(cache, "a") as out, ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(gen_one, t, a.side, a.provider, model): i for i, t in todo}
        failed = 0
        for f in tqdm(as_completed(futs), total=len(futs)):
            i = futs[f]
            affordance = f.result()
            # An empty affordance representation means every retry failed. Caching it would mark the
            # id "done" forever, so a rate-limit burst would silently blank those
            # documents and no rerun would ever fix them. Leave them out; the
            # next run picks them up because they are absent from `done`.
            if not affordance:
                failed += 1
                continue
            out.write(json.dumps({"id": i, "affordance": affordance}) + "\n")
            out.flush()
        if failed:
            print(f"[SciAfford] {failed} items failed all retries and were NOT cached -- "
                  f"rerun the same command to pick them up")
    print(f"[SciAfford] done -> {cache}")


if __name__ == "__main__":
    main()
