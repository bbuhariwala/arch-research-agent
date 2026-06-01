"""
Eval pipeline for Architecture Research Agent.

Measures output quality across 5 dimensions:
- Recommendation Clarity
- Context Specificity  
- Tradeoff Coverage
- Citation Quality
- Operational Realism

Uses LLM-as-a-judge pattern with deterministic checks.
Scores one dimension at a time to reduce position bias.
"""

import json
import anthropic
from dotenv import load_dotenv
from src.graph import run_research_agent
from src.logger import save_research_result
import os
from datetime import datetime

load_dotenv()

EVAL_SYSTEM_PROMPT = """You are an expert evaluator of architecture 
recommendation systems. You score outputs against a precise rubric. 

You are rigorous, consistent, and never give benefit of the doubt 
on ambiguous cases — when in doubt, score lower.

You always return valid JSON and nothing else. No preamble, 
no explanation outside the JSON."""

RUBRIC = {
    "recommendation_clarity": {
        "name": "Recommendation Clarity",
        "question": "Does the output give a clear, direct, actionable recommendation or does it hedge without resolution?",
        "scale": """
5 - Specific recommendation with clear reasoning. Names exact 
    technology AND deployment model. 
    Example: "Use Kafka via AWS MSK Serverless"
4 - Clear recommendation but missing specificity on deployment 
    model or configuration. Example: "Use Kafka" without 
    specifying managed vs self-hosted.
3 - Recommendation present but heavily qualified.
    Example: "Kafka is probably better but RabbitMQ could work"
2 - No clear recommendation. Presents both options equally 
    without resolution. Leaves decision entirely to user.
1 - Contradicts itself OR recommends something that doesn't 
    match the user's stated requirements.""",
        "disqualifier": "If recommendation contradicts user's stated hard constraints → automatic score of 1"
    },

    "context_specificity": {
        "name": "Context Specificity",
        "question": "Does the recommendation reflect the user's specific context from the clarification answers?",
        "scale": """
5 - Recommendation explicitly references user's scale, team size, 
    budget, and constraints. Would be different advice for a 
    different user profile.
4 - References most context but misses one constraint.
3 - Acknowledges context exists but gives largely generic advice. 
    Context mentioned but doesn't change the recommendation.
2 - Ignores user context almost entirely. Same answer a user 
    without context would get.
1 - Directly contradicts user's stated context. Example: recommends 
    self-managed Kafka to a team that said they have no Kafka experience.""",
        "disqualifier": "If recommendation would be identical regardless of user context → automatic score of 1"
    },

    "tradeoff_coverage": {
        "name": "Tradeoff Coverage",
        "question": "Are the tradeoffs for each option covered fairly, completely, and with appropriate depth?",
        "scale": """
5 - Both options have specific pros AND cons listed. Tradeoffs are 
    concrete not generic. Example: "Kafka requires ZooKeeper/KRaft 
    expertise" not just "Kafka is complex"
4 - Good coverage but one significant tradeoff missing for either option.
3 - Tradeoffs present but superficial or one-sided. Recommended option 
    gets more favorable treatment.
2 - Only the recommended option's tradeoffs covered. Non-recommended 
    option dismissed without analysis.
1 - No tradeoffs discussed OR tradeoffs are factually incorrect.""",
        "disqualifier": "If a major known tradeoff for the recommended option is completely absent → score capped at 2"
    },

    "citation_quality": {
        "name": "Citation Quality",
        "question": "Are specific claims, numbers, and benchmarks backed by inline citations from retrieved sources?",
        "scale": """
5 - Every specific number or benchmark has [source: url] inline. 
    All cited URLs appear in retrieved content.
4 - Most claims cited. 1-2 specific numbers uncited but no 
    hallucinated sources detected.
3 - Inconsistent citation. Some claims cited, others not. Hard to 
    distinguish sourced vs asserted claims.
2 - Sources only listed at bottom. No inline citations. Can't trace 
    which source supports which claim.
1 - No citations at all OR hallucinated sources detected.""",
        "disqualifier": "Any hallucinated citation → automatic score of 1"
    },

    "operational_realism": {
        "name": "Operational Realism",
        "question": "Does the analysis address what engineering teams actually deal with in production — cost, operational complexity, team skill requirements?",
        "scale": """
5 - Covers cost with specific estimates, operational complexity with 
    concrete examples, team skill requirements relative to user's 
    stated team profile.
4 - Covers most operational concerns but missing one significant area.
3 - Mentions operational concerns vaguely without specifics relevant 
    to user's context.
2 - Purely theoretical comparison. No production operational considerations.
1 - Operational advice is actively wrong or dangerous. Example: recommends 
    self-managed Kafka as "simple" to a team with no distributed systems 
    experience.""",
        "disqualifier": "If cost estimates are off by more than one order of magnitude → automatic score of 1"
    }
}

def score_dimension(question: str, context: str, output: str, dimension_key: str) -> dict:
    """
    Score one dimension of the output using LLM-as-a-judge
    Scores one dimension at a time to reduce position bias
    and produce more consistent, focused results.
    
    Returns dict with score, reasoning, evidence, 
    and whether disqualifier was triggered.
    """
    client = anthropic.Anthropic()
    dimension = RUBRIC[dimension_key]

    prompt = f"""Score this architecture recommendation on ONE dimension only.
    QUESTION ASKED:
    {question}

    USER CONTEXT PROVIDED:
    {context}

    AGENT OUTPUT TO EVALUATE:
    {output}

    DIMENSION TO SCORE: {dimension['name']}

    SCORING QUESTION: {dimension['question']}

    SCALE:
    {dimension['scale']}

    DISQUALIFIER: {dimension['disqualifier']}

    Return ONLY this JSON, nothing else:
    {{
        "dimension": "{dimension['name']}",
        "score": <integer 1-5>,
        "reasoning": "<one specific sentence explaining the score>",
        "disqualifier_triggered": <true or false>,
        "evidence": "<exact quote from output that most influenced your score>"
    }}"""

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=500,
        temperature=0.1,
        system=EVAL_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        raw = response.content[0].text.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        print(f"  ⚠️ Could not parse score for {dimension_key}: {e}")
        return {
            "dimension": dimension["name"],
            "score": 0,
            "reasoning": f"Parse error: {e}",
            "disqualifier_triggered": False,
            "evidence": ""
        }

def score_output(question: str, context: str, output: str) -> dict:
    """
    Score all five dimensions for one agent output.
    Returns full results including per-dimension scores
    and overall average.
    """

    print(f"\n Scoring output...")
    results = {}
    scores = []
    for dimension_key in RUBRIC.keys():
        print(f"Scoring: {RUBRIC[dimension_key]['name']}...")
        result = score_dimension(question, context, output, dimension_key)
        results[dimension_key] = result
        if result["score"] > 0:
            scores.append(result["score"])
        
        disqualifier = "DISQUALIFIER" if result["disqualifier_triggered"] else ""
        print(f"→ {result['score']}/5{disqualifier}")

    overall = sum(scores) / len(scores) if scores else 0
    
    return {
        "question": question,
        "context": context,
        "dimensions": results,
        "overall_score": round(overall, 2),
        "disqualifiers_triggered": sum(
            1 for r in results.values() 
            if r["disqualifier_triggered"]
        )
    }


def run_deterministic_checks(output: str) -> dict:
    """
    Fast structural checks that don't require an LLM call.
    These catch obvious failures immediately.
    
    Think of these as unit tests for your agent output.
    """
    checks = {}
    # Has a reccomendation section
    checks["has_recommendation_section"] = "## Recommendation" in output
    # Has inline citations
    checks["has_inline_citations"] = "[source:" in output.lower()
    # Has sources section
    checks["has_sources_section"] = "## Sources" in output

    # Minimum length — too short = incomplete
    word_count = len(output.split())
    checks["meets_minimum_length"] = word_count >= 300
    checks["word_count"] = word_count

    # Has tradeoffs section
    checks["has_tradeoffs_section"] = (
        "## Key Tradeoffs" in output or 
        "## Tradeoffs" in output or
        "tradeoff" in output.lower()
    )

    # Has operational considerations
    checks["has_operational_section"] = (
        "## Operational" in output or
        "operational" in output.lower()
    )

     # All checks passed
    structural_checks = [
        v for k, v in checks.items() 
        if k != "word_count"
    ]

    checks["all_passed"] = all(structural_checks)
    checks["passed_count"] = sum(1 for v in structural_checks if v)
    checks["total_count"] = len(structural_checks)
    
    return checks

# Fixed test cases — these never change
# Same questions, same contexts, every eval run
# This is what makes results comparable over time

TEST_CASES = [{
        "id": "tc1",
        "question": "Should I use Kafka or RabbitMQ for a high-throughput event pipeline?",
        "context": """
        - 500K messages per second, average 2KB per message
        - Need message replay for compliance
        - 2 person team, comfortable with Kubernetes, no Kafka experience
        - AWS infrastructure, budget conscious
        - Need per-key ordering and at-least-once delivery
        """
    },
    {
        "id": "tc2", 
        "question": "Should I use PostgreSQL or MongoDB for a user profile service?",
        "context": """
        - 50M users, read-heavy workload (95% reads)
        - User profiles are semi-structured, schema evolves frequently
        - 5 person team, strong SQL experience, no MongoDB experience
        - Need ACID transactions for payment-related profile updates
        - P99 latency requirement under 10ms
        """
    },
    {
        "id": "tc3",
        "question": "Should I use REST or GraphQL for a mobile app API?",
        "context": """
        - iOS and Android apps with 1M daily active users
        - Multiple client types with different data needs
        - 3 person backend team, strong REST experience
        - Bandwidth is a concern — many users on slower mobile connections
        - Need to iterate quickly on API without breaking clients
        """
    },
    {
        "id": "tc4",
        "question": "Should I use Redis or Memcached for our caching layer?",
        "context": """
        - Caching session data and computed results
        - Need pub/sub for real-time notifications
        - Data structures vary — strings, lists, sorted sets
        - 99.99% uptime requirement
        - Small team, already using Redis for rate limiting
        """
    },
    {
        "id": "tc5",
        "question": "Should I use microservices or a monolith for a new SaaS product?",
        "context": """
        - Early stage startup, 4 engineers total
        - Expecting to iterate quickly on features
        - No existing infrastructure
        - Planning to scale to enterprise customers in 18 months
        - Team has experience with both architectures
        """
    }
]

def run_evals(test_cases :list = None, save_results: bool  = True) -> dict:
    """
    Run the full eval pipeline against all test cases.
    
    For each test case:
    1. Run deterministic checks
    2. Score all 5 dimensions with LLM judge
    3. Aggregate results
    4. Save to file
    
    Returns full results including per-test and aggregate scores.
    """

    if test_cases is None:
        test_cases = TEST_CASES
    
    print(f"\n{'='*60}")
    print(f"EVAL RUN — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Test cases: {len(test_cases)}")
    print(f"\n{'='*60}")

    all_results = []

    for i, tc in enumerate(test_cases):
        print(f"\n{'-'*60}")
        print(f"Test Case {i+1}/{len(test_cases)} : {tc['id']}")
        print(f"Question: {tc['question'][:60]}...")
        print(f"{'─'*60}")
        print("\n Running agent...")
        try:
            output = run_research_agent(tc["question"],prefilled_clarifications=tc["context"])
        except Exception as e:
            print(f" Agent failed: {e}")
            print(f" Agent failed: {e}")
            all_results.append({
                "test_case_id": tc["id"],
                "question": tc["question"],
                "error": str(e),
                "overall_score": 0
            })
            continue
        
        # Run deterministic checks
        print("\n  ✓ Running deterministic checks...")
        det_checks = run_deterministic_checks(output)
        print(f"  Passed: {det_checks['passed_count']}/{det_checks['total_count']}")
        if not det_checks["all_passed"]:
            failed = [k for k, v in det_checks.items() 
                     if k not in ["word_count", "all_passed", "passed_count", "total_count"] 
                     and not v]
            print(f"  Failed checks: {failed}")
        
         # Score with LLM judge
        scores = score_output(tc["question"], tc["context"], output)

        result = {
            "test_case_id": tc["id"],
            "question": tc["question"],
            "context": tc["context"],
            "output": output,
            "deterministic_checks": det_checks,
            "llm_scores": scores,
            "overall_score": scores["overall_score"]
        }

        all_results.append(result)
        
        print(f"\n  Overall Score: {scores['overall_score']}/5.0")
        if scores["disqualifiers_triggered"] > 0:
            print(f"  ⚠️ Disqualifiers triggered: {scores['disqualifiers_triggered']}")

    # Aggregate results
    valid_scores = [r["overall_score"] for r in all_results if "error" not in r]
    aggregate = {
        "run_timestamp": datetime.now().isoformat(),
        "total_test_cases": len(test_cases),
        "successful_runs": len(valid_scores),
        "overall_average": round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else 0,
        "dimension_averages": {},
        "test_results": all_results
    }

    # Calculate per-dimension averages
    for dim_key in RUBRIC.keys():
        dim_scores = []
        for r in all_results:
            if "error" not in r and dim_key in r["llm_scores"]["dimensions"]:
                dim_scores.append(r["llm_scores"]["dimensions"][dim_key]["score"])
        if dim_scores:
            aggregate["dimension_averages"][dim_key] = round(
                sum(dim_scores) / len(dim_scores), 2
            )

     # Print summary
    print(f"\n{'='*60}")
    print("EVAL SUMMARY")
    print(f"{'='*60}")
    print(f"Overall Average: {aggregate['overall_average']}/5.0")
    print(f"\nDimension Averages:")
    for dim_key, avg in aggregate["dimension_averages"].items():
        dim_name = RUBRIC[dim_key]["name"]
        bar = "█" * int(avg) + "░" * (5 - int(avg))
        print(f"  {dim_name:<25} {bar} {avg}/5.0")
    
    weakest = min(
        aggregate["dimension_averages"].items(), 
        key=lambda x: x[1]
    )
    print(f"\nWeakest dimension: {RUBRIC[weakest[0]]['name']} ({weakest[1]}/5.0)")
    print(f"Recommended action: focus improvements here first")
    
    # Save results
    if save_results:
        os.makedirs("eval_results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"eval_results/eval_{timestamp}.json"
        with open(filepath, "w") as f:
            json.dump(aggregate, f, indent=2)
        print(f"\nResults saved to: {filepath}")
    
    return aggregate

