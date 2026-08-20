

GRAPH_PROMPT = """
You are a video graph construction assistant.
Please analyze the given video segment and extract structured information in JSON format in English.

Your goal is to produce information useful for:
1. graph-based retrieval,
2. temporal reasoning,
3. state change tracking,
4. evidence aggregation across video segments.

Identify and describe the following:

1. entities:
- List all distinct objects, people, animals, or other significant elements present in the video segment.
- Each entity should be stable and specific.
- If the same object appears with a changed status, keep the entity name the same and put the change in "states".

2. actions:
- Describe visible actions or interactions.
- Each action must be attached to an existing entity name.
- Prefer short action phrases.
- Include interactions between entities if visible.

3. scenes:
- Describe the environment, place, or visual context.
- Keep them concise.

4. states:
- Describe object/person states that are visually relevant for reasoning.
- Examples:
  - laptop: closed / open
  - plate: in sink / on cabinet
  - person: sitting / standing / holding rope
  - door: closed / open
  - food: being cooked / finished
- Only include states clearly supported by the segment.

5. temporal cues:
- Describe whether this segment likely represents:
  - beginning of an action,
  - ongoing action,
  - end/result of an action,
  - transition between states.
- Keep this concise.

6. summary:
- Provide one short sentence summarizing the key evidence in this segment.

Special rule:
- If the video is filmed from a first-person point of view, use "me" as the subject when appropriate.

Ensure the output strictly follows the JSON format below.
Return a single JSON object, not a list or array.

{
    "entities": [{"entity name": "", "description": ""}],
    "actions": [{"entity name": "", "action description": ""}],
    "scenes": [{"location": ""}],
    "states": [{"entity name": "", "state description": ""}],
    "temporal": {"stage": "", "evidence": ""},
    "summary": ""
}

Rules:
- "entity name" in actions and states must appear in entities.
- Do not hallucinate invisible content.
- Be concise but informative.
- Do not output anything outside the JSON.
"""

REASONING_PROMPT = """Given a question of a long video and potential candidates:
Question: {query}

Candidates: {candidates}

Your task is to plan retrieval for long-video question answering.
You do NOT need to answer the question. You only need to identify what information is needed for retrieval and reasoning.

Please think about:
- important entities
- important actions
- important scenes
- important object states
- whether temporal order matters
- whether multiple segments are needed
- whether the question depends on beginning / end / entire video
- what reasoning tool is needed

Return ONLY valid JSON with the following fields:
{{
    "keywords": [],
    "states": [],
    "temporal_keywords": [],
    "candidates_necessary": "yes or no",
    "multiple": "yes or no",
    "time": "begin or end or none",
    "tool": "object counting or action counting or order or state change or none",
    "global": "yes or no"
}}

Field definitions:
- keywords: entity / action / scene words useful for retrieval. Do not copy full candidate texts unless necessary.
- states: state-related clues such as open, closed, empty, full, in sink, on table.
- temporal_keywords: cues like before, after, first, then, finally, while, at the end.
- candidates_necessary: whether options are needed to guide retrieval.
- multiple: whether multiple video segments are likely needed.
- time: whether question specifically asks about the beginning or end.
- tool: choose exactly one from [object counting, action counting, order, state change, none].
- global: whether the question is about the overall topic/content of the whole video.

Example output:
{{
    "keywords": ["laptop", "interacting", "open"],
    "states": ["closed", "open"],
    "temporal_keywords": ["before", "after"],
    "candidates_necessary": "no",
    "multiple": "yes",
    "time": "none",
    "tool": "state change",
    "global": "no"
}}
"""

PRED_PROMPT = "Respond with only the letter of the correct option.\n"

PRED_SQL_PROMPT = "Given the useful information: {sql_input}. But be aware that it may still omit crucial details so it's essential to check the video for completeness.\n"

SQL_PROMPT = """Given a question of a long video and potential candidates:
Question: {query}

Candidates: {candidates}

Break the question into several sub-questions for segment-level verification.

Goal:
- verify relevant entities / actions / scenes / states,
- support temporal reasoning,
- reduce hard negatives,
- prepare for evidence aggregation.

Rules:
1. Use short and precise sub-questions.
2. Prefer yes/no questions.
3. Use counting only when the question is explicitly about number.
4. For order/state-change questions, include sub-questions about state or event presence, not the final answer directly.
5. Do NOT mention exact timestamps.
6. Do NOT output explanations.

Return ONLY valid JSON:
{{
    "Q1": "...",
    "Q2": "..."
}}

Examples:

Example 1:
Question: Did I open the laptop?
Candidates:
A. Maybe
B. No
C. I don't know
D. Yes
Output:
{{
    "Q1": "Is there a laptop shown in the video segment?",
    "Q2": "Is someone interacting with the laptop in the video segment?",
    "Q3": "Is the laptop closed in the video segment?",
    "Q4": "Is the laptop open in the video segment?"
}}

Example 2:
Question: Where was the plate before I put bread on it?
Candidates:
A. cabinet
B. refrigerator
C. sink
D. floor
Output:
{{
    "Q1": "Is there a plate shown in the video segment?",
    "Q2": "Is bread being put on the plate in the video segment?",
    "Q3": "Is the plate in the sink in the video segment?",
    "Q4": "Is the plate in the cabinet in the video segment?",
    "Q5": "Is the plate in the refrigerator in the video segment?",
    "Q6": "Is the plate on the floor in the video segment?"
}}

Example 3:
Question: According to the video, what is the chronological order in which the following actions occur?
Candidates:
(a) Weaving in the ends.
(b) Crocheting a single crochet.
(c) Finishing the handcraft.
(d) Making a slip knot.
(e) Crocheting a chain.
Output:
{{
    "Q1": "Is there a scene showing weaving in the ends?",
    "Q2": "Is there a scene showing crocheting a single crochet?",
    "Q3": "Is there a scene showing finishing the handcraft?",
    "Q4": "Is there a scene showing making a slip knot?",
    "Q5": "Is there a scene showing crocheting a chain?"
}}
"""
SQL_ANSWER_COUNT_PROMPT = """You are a video segment analyzer.
You will be given counting-related questions about a single video segment.

Return ONLY valid JSON.
For each question:
- output a non-negative integer,
- if the answer cannot be determined from the segment, output 0.

Questions: {questions}

Example output:
{{
    "Q1": 2
}}
"""

# SQL_ANSWER_PROMPT = """You are a video segment analyzer.
# Given a list of verification questions about a single video segment, answer each question in JSON format.

# Allowed answer values:
# - "yes"
# - "no"
# - a short state value only when the question explicitly asks about state and the state is clearly visible

# However, if unsure, answer "no".

# Return ONLY valid JSON.

# Questions: {questions}

# Example:
# {{
#     "Q1": "Is there a laptop in the segment?",
#     "Q2": "Is the laptop open in the segment?"
# }}

# Output:
# {{
#     "Q1": "yes",
#     "Q2": "no"
# }}
# """

SQL_ANSWER_PROMPT = """You are a video segment analyzer.
Given a list of verification questions about a single video segment, answer each question in JSON format.

Allowed answer values:
- "yes"
- "no"
- a short state value only when the question explicitly asks about state and the state is clearly visible

Additional rules:
- For yes/no questions, each value must be exactly "yes" or "no".
- Do NOT repeat or paraphrase the question text.
- Do NOT output a sentence or explanation.
- If unsure, output "no".

Return ONLY valid JSON.

Questions: {questions}

Example:
{{
    "Q1": "Is there a laptop in the segment?",
    "Q2": "Is the laptop open in the segment?"
}}

Output:
{{
    "Q1": "yes",
    "Q2": "no"
}}
"""

ORIG_SQL_ANSWER_PROMPT = """Given a list of questions related to the video, generate corresponding answers in JSON format.
The answer must be either "yes" or "no".
Do not provide any additional explanations or responses beyond the required format.
Questions: {questions}

For Example:
Questions: {{
"Q1": "Is there ...",
"Q2": "Does the video show ..."
}}
Your output
{{
    "Q1": "yes",
    "Q2": "no"
}}

Ensure that each response adheres strictly to the specified answer types.\n
"""


AGGREGATE_PROMPT = """You are given a multiple-choice question and verification results from multiple video segments.
The segment numbers are in temporal order: smaller index means earlier in time.

Your task:
1. Aggregate evidence across segments.
2. Track temporal order when needed.
3. Track state changes when needed.
4. Summarize useful evidence without giving the final answer.

Question: {query}
Candidates: {candidates}
Information: {input}

Return one concise summary within 40 words.
Do NOT output the final option letter.
Do NOT mention uncertainty unless evidence is clearly insufficient.
"""

SUBTITLE_PROMPT = """You are given a question, answer options, and subtitles of a video.
Find subtitle segments relevant to the question.

Return ONLY a Python-style list of time indices.
If nothing is relevant, return [].

Question: {query}
Options: {candidates}
Subtitles: {subtitles}

Limit the output to at most 10 indices.
"""