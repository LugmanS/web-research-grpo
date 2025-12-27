import art
from art.local.backend import LocalBackend
from art.utils import iterate_dataset
from datasets import load_dataset
from pydantic import BaseModel
from dataclasses import asdict, dataclass
from openai import AsyncOpenAI
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Literal
from tenacity import retry, stop_after_attempt
from prompt import policy_judge_base_prompt
from qdrant_client import QdrantClient
from google import genai
import os
import weave
import json
import re
import asyncio

load_dotenv()

qdrant = QdrantClient(host=os.getenv("QDRANT_HOST"), port=os.getenv("QDRANT_PORT"))

openai_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

genai_client = genai.Client(vertexai=True, project=os.getenv("GENAI_PROJECT_ID"), location=os.getenv("GENAI_LOCATION"))

# -----------------------------
# Data models
# -----------------------------

@dataclass
class Hop:
    question: str
    answer: str

@dataclass
class Scenario:
    id: str
    full_question: str
    steps: List[Hop]
    final_answer: str

    @property
    def required_tool_calls(self) -> int:
        return len(self.steps)

class ProjectTrajectory(art.Trajectory):
    final_answer: str | None = None
    
class StepScenario(BaseModel):
    step: int
    scenario: Scenario
    
@dataclass
class ToolConfig:
    name: str
    args: Dict[str, str]

    @property
    def required_keys(self) -> set:
        return set(self.args.keys())
    
    
# -----------------------------
# Tools
# -----------------------------

@dataclass
class SearchResult:
    title: str
    content: str

tool_definitions = [
    {
        "type": "function",
        "function": {
        "name": "search_documents",
        "description": "Retrieve relevant documents using a search query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A natural-language search query."}
            },
            "required": ["query"]
        }
        }
    }
]

async def search_documents(query: str) -> list[SearchResult]:
    if not query:
        raise ValueError("No query provided to perform search.")
    
    embedding_res = await openai_client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = embedding_res.data[0].embedding
    
    if not query_embedding:
        raise ValueError("No query embedding generated.")
    
    hits = qdrant.query_points(collection_name="documents", query=query_embedding, limit=8, with_payload=True)
    
    results = [SearchResult(title=point.payload.get("title"), content=point.payload.get("content")) for point in hits.points]
    return [asdict(result) for result in results]
    

tools_cfg = [
    ToolConfig(name="search_documents", args={"query": "string"}),
]

# -----------------------------
# Reward functions
# -----------------------------

@dataclass
class Step:
    content: str
    tool_name: str
    tool_args: str  # JSON string
    tool_result: str

@dataclass
class Trajectory:
    steps: List[Step]
    final_answer: str

def raw_chat_messages_to_trajectory(raw_messages: List[Dict[str, Any]]):
    steps: List[Step] = []

    tool_results: Dict[str, str] = {}
    for m in raw_messages:
        if m.get("role") == "tool":
            tcid = m.get("tool_call_id")
            if tcid:
                tool_results[tcid] = m.get("content", "") or ""

    final_answer: str = ""
    for m in raw_messages:
        role = m.get("role")

        if role == "assistant":
            tool_calls = m.get("tool_calls") or []
            assistant_content = m.get("content", "") or ""

            for tc in tool_calls:
                fn = (tc.get("function") or {})
                tool_name = fn.get("name") or ""
                tool_args = fn.get("arguments")
                tool_call_id = tc.get("id") or ""
                tool_result = tool_results.get(tool_call_id, "")

                steps.append(
                    Step(
                        content=assistant_content,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        tool_result=tool_result,
                    )
                )

            if (not tool_calls) and assistant_content.strip():
                final_answer = assistant_content.strip()

    return Trajectory(steps=steps, final_answer=final_answer)

ANGLE_BRACKET_RE = re.compile(r"[<>]")

def parse_json_object(s: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(s)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def verify_format(steps: List[Step], tools_cfg: List[ToolConfig]) -> bool:
    """
    Sanity checks:
    - step.content should not contain < or >
    - tool_name must be in tools_cfg
    - tool_args must parse to a JSON object (dict)
    - tool_args keys must exactly match required keys for that tool
    """
    tool_cfg_map = {cfg.name: cfg for cfg in tools_cfg}

    for step in steps:
        # Disallow angle brackets in reasoning/content (avoid tag injection)
        if step.content is None or ANGLE_BRACKET_RE.search(step.content):
            return False

        # Tool name must exist
        if step.tool_name not in tool_cfg_map:
            return False

        # Tool args must be JSON dict
        args_obj = parse_json_object(step.tool_args)
        if args_obj is None:
            return False

        # Keys must match the tool schema keys exactly
        required_keys = tool_cfg_map[step.tool_name].required_keys
        if set(args_obj.keys()) != required_keys:
            return False

    return True

class StepEvaluation(BaseModel):
    idx: int
    reasoning: str
    verdict: Literal["achieved", "not_achieved"]
    
class FinalAnswer(BaseModel):
    reasoning: str
    accepted: bool

class PolicyJudgeResponse(BaseModel):
    step_evaluations: List[StepEvaluation]
    final_answer: FinalAnswer
    

@retry(stop=stop_after_attempt(3))
async def judge_policy_generation(gold: Scenario, traj: Trajectory) -> PolicyJudgeResponse:
    
    judge_input = {}
    judge_input["question"] = gold.full_question
    judge_input["policy_trajectory"] = [{"idx": idx, "search_query": step["tool_args"]} for idx, step in enumerate(traj.steps)]
    judge_input["ideal_trajectory"] = [{"idx": idx, "search_query": step["question"], "intermediate_answer": step["answer"]} for idx, step in enumerate(gold.steps)]
    judge_input["gold_answer"] = gold.final_answer
    judge_input["policy_answer"] = traj.final_answer
    
    response = genai_client.models.generate_content(model=os.getenv("GENAI_MODEL_NAME"), contents=policy_judge_base_prompt + json.dumps(judge_input, indent=2))
    
    raw_content = response.candidates[0].content.parts[0].text

    try:
        return PolicyJudgeResponse.model_validate_json(raw_content)
    except Exception as e:
        print(f"Judge parse error: {e}\nRaw: {response}")
        return PolicyJudgeResponse(
            step_evaluations=[],
            final_answer=FinalAnswer(reasoning=f"Parse error: {e}\nRaw: {response}", accepted=False),
        )

def harmonic(n: int) -> float:
    return sum(1.0 / k for k in range(1, n + 1)) if n > 0 else 0.0

def tool_reward_time_discounted(
    actual_calls: int,
    required_calls: int,
    base: float = 1.0,
    over_penalty_slope: float = 0.2,
    min_reward: float = -1.0,
) -> float:
    """
    Time-discounted / harmonic tool-call reward.

    - Under-calling: reward grows as H_T / H_R (diminishing returns)
    - Over-calling: penalize each extra call linearly

    Returns in roughly [min_reward, base].
    """
    T = max(0, int(actual_calls))
    R = max(0, int(required_calls))

    if R == 0:
        # No tool calls required: discourage spam
        r = base - over_penalty_slope * T
        return max(min_reward, r)

    if T == 0:
        return max(min_reward, 0.0)

    if T <= R:
        r = base * (harmonic(T) / harmonic(R))
    else:
        r = base - over_penalty_slope * (T - R)

    return max(min_reward, r)

@dataclass
class RewardConfig:
    # format reward
    format_penalty: float = -2.0

    # tool reward params
    tool_base: float = 1.0
    no_tool_penalty: float = -1.0

    # final reward
    lambda_final: float = 1.0
    r_final_correct: float = 2.0
    r_final_wrong: float = -2.0


async def compute_reward(
    item: Scenario,
    traj: Trajectory,
    tools_cfg: List[ToolConfig],
    cfg: RewardConfig = RewardConfig(),
) -> Dict[str, float]:
    """
    Simple reward:
    1) Format sanity: +lambda_format if ok else -lambda_format
    2) Tool-call reward: time-discounted harmonic based on number of tool calls vs required
    3) Final correctness: judged externally (LLM call), +2 or -2 (scaled by lambda_final)
    """
    breakdown: Dict[str, float] = {}

    # (1) Format
    is_format_ok = verify_format(traj.steps, tools_cfg)
    breakdown["is_format_ok"] = is_format_ok
    breakdown["format_penalty"] = cfg.format_penalty
    breakdown["format_reward"] = cfg.format_penalty if not is_format_ok else 0
    
    judge_report = await judge_policy_generation(item, traj)
    
    # (2) Tool-call reward
    required_calls = item.required_tool_calls
    valid_calls = sum(1 for step in judge_report.step_evaluations if step.verdict == "achieved")
    
    breakdown["tool_calls_required"] = float(required_calls)
    breakdown["tool_calls_actual"] = float(valid_calls)

    if required_calls > 0 and len(traj.steps) == 0:
        breakdown["tool_calls_reward"] = cfg.no_tool_penalty
    elif required_calls == valid_calls:
        breakdown["tool_calls_reward"] = cfg.tool_base
    else:
        complexity_penality = cfg.tool_base / required_calls
        
        harmonic_sum = 0
        for step in judge_report.step_evaluations:
            if step.verdict == "achieved":
                harmonic_sum += 1.0 / (step.idx + 1)
                
        breakdown["tool_calls_reward"] = complexity_penality * harmonic_sum

    # (3) Final reward (LLM judge)
    is_final_correct = judge_report.final_answer.accepted
    breakdown["final_correct"] = is_final_correct
    breakdown["final_reward"] = cfg.lambda_final * (cfg.r_final_correct if is_final_correct else cfg.r_final_wrong)

    breakdown["total"] = breakdown["format_reward"] + breakdown["tool_calls_reward"] + breakdown["final_reward"]
    
    return breakdown

# -----------------------------
# Rollout
# -----------------------------

def to_message_dict(m):
    if isinstance(m, dict):
        return m
    if hasattr(m, "message"):          # Choice
        return m.message.model_dump()
    if hasattr(m, "model_dump"):       # Message
        return m.model_dump()
    raise TypeError(type(m))


@weave.op
async def rollout(model: art.Model, step_scenario: StepScenario) -> ProjectTrajectory:
    scenario = step_scenario.scenario
    
    traj = ProjectTrajectory(
        reward=0.0,
        messages_and_choices=[],
        metadata={
            "scenario_id": scenario.id,
            "step": step_scenario.step,
        },
    )
    
    traj.messages_and_choices = [
        {"role": "system", "content": "You are an autonomous research assistant equipped with tools. You solve user queries by reasoning step by step and gathering information dynamically. You operate in an iterative loop where you first think through your strategy, planning, query formulation, and analysis of previous tool outputs first. After this reasoning step, you immediately take action by calling the appropriate tool to fetch the necessary information, without providing a final answer yet. You continue this pattern of reasoning followed by tool calls until you have gathered sufficient information. Only after that should you provide the final response to the user in a simple chat style text."},
        {"role": "user", "content": scenario.full_question},
    ]
    traj.tools = tool_definitions
    
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )
    
    while True:
        response = await client.chat.completions.create(
            model=model.get_inference_name(),
            temperature=1,
            messages=traj.messages(),
            tools=traj.tools,
        )
        
        response_message = response.choices[0].message
        traj.messages_and_choices.append(response.choices[0])
         
        if not response_message.tool_calls:
            traj.final_answer = response_message.content
            messages_dict = [
                to_message_dict(c)
                for c in traj.messages_and_choices
            ]
            reward_formatted_traj = raw_chat_messages_to_trajectory(messages_dict)
            reward = await compute_reward(scenario, reward_formatted_traj, tools_cfg)
            traj.reward = reward["total"]
            break
        
        try:
            for tool_call in response_message.tool_calls:
                tool_name: str = tool_call.function.name
                if tool_name == "search_documents":
                    tool_args = json.loads(tool_call.function.arguments)
                    result = await search_documents(**tool_args)
                    traj.messages_and_choices.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": str(result),
                        }
                    )

        except Exception as e:
            print(f"Error executing tool call: {e}")
            return traj
        
    return traj

# -----------------------------
# Training
# -----------------------------

training_config = {
    "groups_per_step": 8,
    "num_epochs": 20,
    "rollouts_per_group": 4,
    "learning_rate": 1e-5,
    "max_steps": 2000,
    "validation_step_interval": 100,
    "persist_checkpoint_interval": 100,
}

async def train(model: art.Model, scenarios: List[Scenario], validation_scenarios: List[Scenario]):
    training_iterator = iterate_dataset(
        scenarios,
        groups_per_step=training_config["groups_per_step"],
        num_epochs=training_config["num_epochs"],
        initial_step=await model.get_step(),
    )

    for batch in training_iterator:
        print(
            f"Training step {batch.step}, epoch {batch.epoch}, epoch step {batch.epoch_step}"
        )
        print(f"Batch contains {len(batch.items)} scenarios")

        # Create trajectory groups for this batch (similar to train.py)
        train_groups = []
        for scenario in batch.items:
            train_groups.append(
                art.TrajectoryGroup(
                    (
                        rollout(model, StepScenario(step=batch.step, scenario=scenario))
                        for _ in range(training_config["rollouts_per_group"])
                    )
                )
            )

        # Gather all trajectory groups
        finished_train_groups = await art.gather_trajectory_groups(
            train_groups,
            pbar_desc="gather",
            max_exceptions=training_config["rollouts_per_group"] * len(batch.items),
        )

        judged_groups = []
        for group in finished_train_groups:
            judged_groups.append(group)

        if batch.step > 200 and batch.step % training_config["validation_step_interval"] == 0:
            print("Running validation at step", batch.step)
            validation_groups = []
            for scenario in validation_scenarios:
                validation_groups.append(
                    art.TrajectoryGroup([rollout(model, StepScenario(step=batch.step, scenario=scenario))])
                )

            finished_validation_groups = await art.gather_trajectory_groups(
                validation_groups,
                pbar_desc="gather",
                max_exceptions=training_config["rollouts_per_group"] * len(validation_scenarios),
            )

            await model.log(
                finished_validation_groups,
                split="val"
            )

        if batch.step % training_config["persist_checkpoint_interval"] != 0:
            await model.delete_checkpoints()
            
        await model.train(
            judged_groups,
            config=art.TrainConfig(learning_rate=training_config["learning_rate"]),
        )

        print(f"Completed training step {batch.step}")

        # Stop after max_steps for demo purposes (adjust as needed)
        if batch.step >= training_config["max_steps"]:
            break
        
# -----------------------------
# Main
# -----------------------------

async def main():
    
    BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
    DATASET_ID = "lugman-madhiai/sampled-misique"

    model = art.TrainableModel(
        name="semantic-search-agent-v1",
        project="art-grpo",
        base_model=BASE_MODEL_ID,
    )

    backend = LocalBackend()

    await model.register(backend)

    train_dataset = load_dataset(DATASET_ID, split="train")
    print(f"Train dataset contains {len(train_dataset)} total samples")
    
    validation_dataset = load_dataset(DATASET_ID, split="test")
    print(f"Validation dataset contains {len(validation_dataset)} total samples")
    
    train_scenarios = [Scenario(**example) for example in train_dataset]
    validation_scenarios = [Scenario(**example) for example in validation_dataset]

    # await rollout(model, StepScenario(step=0, scenario=train_scenarios[0]))
    
    await train(model, train_scenarios, validation_scenarios)
    
if __name__ == "__main__":
    asyncio.run(main())