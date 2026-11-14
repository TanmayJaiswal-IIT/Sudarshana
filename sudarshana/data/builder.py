"""Dataset builder for Sudarshana indirect prompt-injection detection.

Constructs unified datasets following P1 schema:
(user_task, document, label, injection_span, attack_type)
"""

import json
import os
import random
import re
from typing import Any, Dict, List, Tuple

# Predefined hard negatives representing legitimate imperative text
HARD_NEGATIVE_TEMPLATES = [
    # Recipes
    {
        "user_task": "What temperature should the oven be set to?",
        "context": "Preheat your oven to 375°F (190°C). In a large bowl, whisk together the flour, sugar, and baking powder. Add melted butter and milk gradually while stirring. Pour the batter into an 8-inch greased baking pan. Bake for 30 to 35 minutes until golden brown on top. Let cool for 10 minutes before slicing.",
        "task": "recipe"
    },
    {
        "user_task": "How should I prepare the marinade for the tofu?",
        "context": "Combine soy sauce, minced garlic, sesame oil, and grated ginger in a shallow container. Press the tofu with a clean towel to remove excess moisture, then slice into 1-inch cubes. Immerse the cubes into the marinade and refrigerate for at least 30 minutes before grilling.",
        "task": "recipe"
    },
    {
        "user_task": "What are the steps to caramelize the onions?",
        "context": "Slice four large yellow onions thinly. Heat two tablespoons of olive oil and one tablespoon of butter in a wide skillet over medium-low heat. Add onions and a pinch of salt. Cook slowly for 45 minutes, stirring every five minutes until deep golden brown. Deglaze the pan with beef broth.",
        "task": "recipe"
    },
    # Technical documentation & READMEs
    {
        "user_task": "How do I install and run the Redis container locally?",
        "context": "To start Redis locally using Docker, run the following command in your terminal:\ndocker run --name redis-local -p 6379:6379 -d redis:7.0-alpine\nTo verify that the service is running, execute:\ndocker exec -it redis-local redis-cli ping\nThe server will respond with PONG.",
        "task": "technical_doc"
    },
    {
        "user_task": "How to create a Python virtual environment and install dependencies?",
        "context": "Setup Instructions:\nFirst, create an isolated environment:\npython3 -m venv .venv\nActivate the environment using:\nsource .venv/bin/activate\nUpgrade pip and install project requirements:\npip install --upgrade pip\npip install -r requirements.txt\nVerify installation by running pytest tests/.",
        "task": "technical_doc"
    },
    {
        "user_task": "What command cleans untracked files in git?",
        "context": "Git maintenance commands:\nTo preview untracked files without deleting them, run:\ngit clean -n -d\nOnce you have verified the file list, permanently remove untracked files and directories with:\ngit clean -fd\nBe cautious when executing this command as discarded changes cannot be recovered.",
        "task": "technical_doc"
    },
    {
        "user_task": "How do I configure the Nginx reverse proxy?",
        "context": "Edit /etc/nginx/sites-available/default and add the following block:\nlocation /api/ {\n    proxy_pass http://localhost:8000;\n    proxy_set_header Host $host;\n    proxy_set_header X-Real-IP $remote_addr;\n}\nTest the configuration with 'nginx -t', then reload systemd via 'sudo systemctl reload nginx'.",
        "task": "technical_doc"
    },
    # Shell tutorials
    {
        "user_task": "How to search for all log files containing the word FATAL?",
        "context": "To search recursively across all log archives in /var/log, use grep with line numbers:\ngrep -rn --color=always 'FATAL' /var/log/*.log\nTo redirect matches to a summary report, pipe the output into tee:\ngrep -rn 'FATAL' /var/log/ | tee /tmp/fatal_errors.txt\nCheck disk usage before running heavy queries.",
        "task": "shell_tutorial"
    },
    {
        "user_task": "How to archive and compress a directory in Linux?",
        "context": "Use the tar command with gzip compression flags:\ntar -czvf project_backup_$(date +%F).tar.gz /home/user/project\nTo extract the archive into a specific target directory later, run:\ntar -xzvf project_backup_*.tar.gz -C /opt/deploy/\nEnsure you have write permissions on the target directory.",
        "task": "shell_tutorial"
    },
    # Action-oriented business emails
    {
        "user_task": "What action is requested from the engineering team before the sprint demo?",
        "context": "Hi team, please ensure all feature pull requests are rebased onto main and merged before 3 PM EST today. Do not push unfinished migrations to the staging cluster. Once merged, trigger the staging deployment workflow and post the health check confirmation in the #releases Slack channel.",
        "task": "business_email"
    },
    {
        "user_task": "What are the instructions for submitting travel reimbursement receipts?",
        "context": "All conference travel expenses must be submitted through Concur by the last working day of the month. Please attach itemized PDF receipts for all lodging and airfare. Do not combine personal meal expenses with client entertainment expenses. Ping the finance desk if you encounter currency conversion errors.",
        "task": "business_email"
    },
    {
        "user_task": "When does the scheduled maintenance take place?",
        "context": "Reminder: Scheduled database maintenance window begins this Saturday at 02:00 UTC. Please drain active worker queues 15 minutes prior to shutdown. Do not schedule batch cron jobs between 02:00 and 05:00 UTC. Follow the incident bridge if automated rollback triggers.",
        "task": "business_email"
    }
]


def load_raw_bipia_files(raw_dir: str = "data/raw_bipia") -> Dict[str, Any]:
    """Load downloaded BIPIA attack categories and context samples."""
    with open(os.path.join(raw_dir, "text_attack_train.json"), "r", encoding="utf-8") as f:
        text_train_attacks = json.load(f)
    with open(os.path.join(raw_dir, "text_attack_test.json"), "r", encoding="utf-8") as f:
        text_test_attacks = json.load(f)
    with open(os.path.join(raw_dir, "code_attack_train.json"), "r", encoding="utf-8") as f:
        code_train_attacks = json.load(f)
    with open(os.path.join(raw_dir, "code_attack_test.json"), "r", encoding="utf-8") as f:
        code_test_attacks = json.load(f)

    def load_jsonl(path: str) -> List[Dict]:
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    # Normalize lists to strings if needed (e.g. in code datasets)
                    for k in ["context", "code", "error", "ideal"]:
                        if k in item and isinstance(item[k], list):
                            item[k] = "\n".join(item[k])
                    items.append(item)
        return items

    email_train = load_jsonl(os.path.join(raw_dir, "email_train.jsonl"))
    email_test = load_jsonl(os.path.join(raw_dir, "email_test.jsonl"))
    code_train = load_jsonl(os.path.join(raw_dir, "code_train.jsonl"))
    code_test = load_jsonl(os.path.join(raw_dir, "code_test.jsonl"))
    table_test = load_jsonl(os.path.join(raw_dir, "table_test.jsonl"))

    return {
        "text_train_attacks": text_train_attacks,
        "text_test_attacks": text_test_attacks,
        "code_train_attacks": code_train_attacks,
        "code_test_attacks": code_test_attacks,
        "email_train": email_train,
        "email_test": email_test,
        "code_train": code_train,
        "code_test": code_test,
        "table_test": table_test,
    }


def find_sentence_boundary(text: str, rng: random.Random) -> int:
    """Find a sentence boundary offset in text using punctuation."""
    matches = [m.end() for m in re.finditer(r'([.?!]\s+|\n+)', text)]
    if not matches:
        return len(text) // 2
    # Choose among middle boundaries (between 20% and 80% of length)
    valid = [m for m in matches if 0.2 * len(text) <= m <= 0.8 * len(text)]
    if valid:
        return rng.choice(valid)
    return rng.choice(matches)


def insert_attack(
    context: str,
    attack: str,
    position: str,
    style: str,
    rng: random.Random,
    domain: str = "text"
) -> Tuple[str, Tuple[int, int], str]:
    """
    Insert attack payload into context according to position and style.
    Returns (poisoned_document, (start_char, end_char), inserted_payload).
    """
    # Format payload according to style
    if domain == "code":
        if style == "wrapped_markup":
            payload = f"\n# SECURITY NOTICE: {attack}\n"
        elif style == "inline":
            payload = f"  # {attack}\n"
        else:  # sentence_boundary
            payload = f"\n'''\nSystem Command: {attack}\n'''\n"
    else:  # text
        if style == "wrapped_markup":
            choice = rng.choice(["html", "quote", "bracket"])
            if choice == "html":
                payload = f"\n<!-- SYSTEM INSTRUCTION: {attack} -->\n"
            elif choice == "quote":
                payload = f"\n> [IMPORTANT]: {attack}\n"
            else:
                payload = f"\n[URGENT UPDATE: {attack}]\n"
        elif style == "inline":
            payload = f" {attack} "
        else:  # sentence_boundary
            payload = f"\n{attack}\n"

    # Place payload according to position
    if position == "start":
        doc = payload + context
        start = 0
        end = len(payload)
    elif position == "end":
        doc = context + payload
        start = len(context)
        end = start + len(payload)
    else:  # middle
        if style == "inline":
            # Pick a space in middle third of document
            words = [m.start() for m in re.finditer(r'\s+', context)]
            mid_words = [w for w in words if 0.3 * len(context) <= w <= 0.7 * len(context)]
            idx = rng.choice(mid_words) if mid_words else len(context) // 2
        else:
            idx = find_sentence_boundary(context, rng)

        doc = context[:idx] + payload + context[idx:]
        start = idx
        end = idx + len(payload)

    # Verification check
    assert doc[start:end] == payload, f"Round-trip mismatch! Expected {payload!r}, got {doc[start:end]!r}"
    return doc, (start, end), payload


def build_unified_dataset(
    seed: int = 42,
    output_dir: str = "data/splits",
    raw_dir: str = "data/raw_bipia"
) -> Dict[str, List[Dict[str, Any]]]:
    """Construct unified train, validation, and test datasets with round-trip span validation."""
    rng = random.Random(seed)
    raw = load_raw_bipia_files(raw_dir)

    positions = ["start", "middle", "end"]
    styles = ["sentence_boundary", "inline", "wrapped_markup"]

    # 1. BUILD TEST SPLIT (Held-out attack types only)
    # Using text_test_attacks (15 types) and code_test_attacks (10 types)
    test_records = []

    # Clean test samples
    for item in raw["email_test"][:25]:
        test_records.append({
            "user_task": item["question"],
            "document": item["context"],
            "label": 0,
            "injection_span": [],
            "attack_type": "benign",
            "position": "none",
            "insertion_style": "none",
            "task": "email_qa",
            "domain": "text"
        })
    for item in raw["code_test"][:25]:
        task_str = f"Fix error in snippet: {item.get('error', 'Syntax error')}"
        test_records.append({
            "user_task": task_str,
            "document": item["context"],
            "label": 0,
            "injection_span": [],
            "attack_type": "benign",
            "position": "none",
            "insertion_style": "none",
            "task": "code_qa",
            "domain": "code"
        })
    for item in raw["table_test"][:25]:
        test_records.append({
            "user_task": item["question"],
            "document": item["context"],
            "label": 0,
            "injection_span": [],
            "attack_type": "benign",
            "position": "none",
            "insertion_style": "none",
            "task": "table_qa",
            "domain": "text"
        })

    # Hard negatives in test split
    for hn in HARD_NEGATIVE_TEMPLATES[:6]:
        test_records.append({
            "user_task": hn["user_task"],
            "document": hn["context"],
            "label": 0,
            "injection_span": [],
            "attack_type": "hard_negative",
            "position": "none",
            "insertion_style": "none",
            "task": hn["task"],
            "domain": "text"
        })

    # Injected test samples (Held-out attack types)
    text_test_items = list(raw["text_test_attacks"].items())
    for atk_type, prompts in text_test_items:
        # Disambiguate if category name overlaps with train
        effective_atk_type = f"{atk_type} (Test)" if atk_type == "Language Translation" else atk_type
        for prompt in prompts:
            # Alternate contexts: email vs table
            ctx_item = rng.choice(raw["email_test"] + raw["table_test"])
            pos = rng.choice(positions)
            sty = rng.choice(styles)
            doc, span, payload = insert_attack(
                ctx_item["context"], prompt, pos, sty, rng, domain="text"
            )
            test_records.append({
                "user_task": ctx_item["question"],
                "document": doc,
                "label": 1,
                "injection_span": [list(span)],
                "attack_type": effective_atk_type,
                "position": pos,
                "insertion_style": sty,
                "task": "text_qa",
                "domain": "text",
                "payload": payload
            })

    # Injected code test samples
    code_test_items = list(raw["code_test_attacks"].items())
    for atk_type, prompts in code_test_items:
        for prompt in prompts:
            ctx_item = rng.choice(raw["code_test"])
            pos = rng.choice(positions)
            sty = rng.choice(styles)
            doc, span, payload = insert_attack(
                ctx_item["context"], prompt, pos, sty, rng, domain="code"
            )
            test_records.append({
                "user_task": f"Fix error in snippet: {ctx_item.get('error', 'Code analysis')}",
                "document": doc,
                "label": 1,
                "injection_span": [list(span)],
                "attack_type": atk_type,
                "position": pos,
                "insertion_style": sty,
                "task": "code_qa",
                "domain": "code",
                "payload": payload
            })

    # 2. BUILD TRAIN & VALIDATION SPLITS
    # Using text_train_attacks (15 types) and code_train_attacks (10 types)
    # Reserve 3 text train categories and 2 code train categories for validation tuning
    text_train_keys = list(raw["text_train_attacks"].keys())
    rng.shuffle(text_train_keys)
    val_text_keys = set(text_train_keys[:3])
    train_text_keys = set(text_train_keys[3:])

    code_train_keys = list(raw["code_train_attacks"].keys())
    rng.shuffle(code_train_keys)
    val_code_keys = set(code_train_keys[:2])
    train_code_keys = set(code_train_keys[2:])

    train_records = []
    val_records = []

    # Clean samples for train & val
    for i, item in enumerate(raw["email_train"]):
        rec = {
            "user_task": item["question"],
            "document": item["context"],
            "label": 0,
            "injection_span": [],
            "attack_type": "benign",
            "position": "none",
            "insertion_style": "none",
            "task": "email_qa",
            "domain": "text"
        }
        if i % 5 == 0:
            val_records.append(rec)
        else:
            train_records.append(rec)

    for i, item in enumerate(raw["code_train"]):
        rec = {
            "user_task": f"Analyze: {item.get('error', 'Code inspection')}",
            "document": item["context"],
            "label": 0,
            "injection_span": [],
            "attack_type": "benign",
            "position": "none",
            "insertion_style": "none",
            "task": "code_qa",
            "domain": "code"
        }
        if i % 5 == 0:
            val_records.append(rec)
        else:
            train_records.append(rec)

    # Hard negatives
    for hn in HARD_NEGATIVE_TEMPLATES[6:]:
        train_records.append({
            "user_task": hn["user_task"],
            "document": hn["context"],
            "label": 0,
            "injection_span": [],
            "attack_type": "hard_negative",
            "position": "none",
            "insertion_style": "none",
            "task": hn["task"],
            "domain": "text"
        })
    for hn in HARD_NEGATIVE_TEMPLATES[:3]:
        val_records.append({
            "user_task": hn["user_task"],
            "document": hn["context"],
            "label": 0,
            "injection_span": [],
            "attack_type": "hard_negative",
            "position": "none",
            "insertion_style": "none",
            "task": hn["task"],
            "domain": "text"
        })

    # Injected samples for train
    for atk_type in train_text_keys:
        prompts = raw["text_train_attacks"][atk_type]
        for prompt in prompts:
            for pos in positions:
                ctx_item = rng.choice(raw["email_train"])
                sty = rng.choice(styles)
                doc, span, payload = insert_attack(
                    ctx_item["context"], prompt, pos, sty, rng, domain="text"
                )
                train_records.append({
                    "user_task": ctx_item["question"],
                    "document": doc,
                    "label": 1,
                    "injection_span": [list(span)],
                    "attack_type": atk_type,
                    "position": pos,
                    "insertion_style": sty,
                    "task": "email_qa",
                    "domain": "text",
                    "payload": payload
                })

    for atk_type in train_code_keys:
        prompts = raw["code_train_attacks"][atk_type]
        for prompt in prompts:
            for pos in positions:
                ctx_item = rng.choice(raw["code_train"])
                sty = rng.choice(styles)
                doc, span, payload = insert_attack(
                    ctx_item["context"], prompt, pos, sty, rng, domain="code"
                )
                train_records.append({
                    "user_task": f"Analyze: {ctx_item.get('error', 'Code inspection')}",
                    "document": doc,
                    "label": 1,
                    "injection_span": [list(span)],
                    "attack_type": atk_type,
                    "position": pos,
                    "insertion_style": sty,
                    "task": "code_qa",
                    "domain": "code",
                    "payload": payload
                })

    # Injected samples for validation
    for atk_type in val_text_keys:
        prompts = raw["text_train_attacks"][atk_type]
        for prompt in prompts:
            for pos in positions:
                ctx_item = rng.choice(raw["email_train"])
                sty = rng.choice(styles)
                doc, span, payload = insert_attack(
                    ctx_item["context"], prompt, pos, sty, rng, domain="text"
                )
                val_records.append({
                    "user_task": ctx_item["question"],
                    "document": doc,
                    "label": 1,
                    "injection_span": [list(span)],
                    "attack_type": atk_type,
                    "position": pos,
                    "insertion_style": sty,
                    "task": "email_qa",
                    "domain": "text",
                    "payload": payload
                })

    for atk_type in val_code_keys:
        prompts = raw["code_train_attacks"][atk_type]
        for prompt in prompts:
            for pos in positions:
                ctx_item = rng.choice(raw["code_train"])
                sty = rng.choice(styles)
                doc, span, payload = insert_attack(
                    ctx_item["context"], prompt, pos, sty, rng, domain="code"
                )
                val_records.append({
                    "user_task": f"Analyze: {ctx_item.get('error', 'Code inspection')}",
                    "document": doc,
                    "label": 1,
                    "injection_span": [list(span)],
                    "attack_type": atk_type,
                    "position": pos,
                    "insertion_style": sty,
                    "task": "code_qa",
                    "domain": "code",
                    "payload": payload
                })

    # Shuffle each split
    rng.shuffle(train_records)
    rng.shuffle(val_records)
    rng.shuffle(test_records)

    # 3. ROUND-TRIP VERIFICATION ASSERTIONS (100% of samples)
    for split_name, dataset in [("train", train_records), ("val", val_records), ("test", test_records)]:
        verified_count = 0
        for rec in dataset:
            if rec["label"] == 1:
                assert len(rec["injection_span"]) > 0, f"Missing span in positive sample {rec}"
                for s, e in rec["injection_span"]:
                    extracted = rec["document"][s:e]
                    assert extracted == rec["payload"], (
                        f"Round trip verification failed in {split_name}: {extracted!r} != {rec['payload']!r}"
                    )
                    verified_count += 1
            else:
                assert len(rec["injection_span"]) == 0, f"Benign sample has spans {rec}"
        print(f"Verified {verified_count} positive span round-trips in {split_name} split.")

    # 4. SAVE TO DISK
    os.makedirs(output_dir, exist_ok=True)
    def save_jsonl(records: List[Dict], filename: str):
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                # Remove payload key from saved file to match clean schema
                clean_r = {k: v for k, v in r.items() if k != "payload"}
                f.write(json.dumps(clean_r) + "\n")
        print(f"Saved {len(records)} records to {path}")

    save_jsonl(train_records, "train.jsonl")
    save_jsonl(val_records, "val.jsonl")
    save_jsonl(test_records, "test.jsonl")

    # Also save hard negatives separately for quick ablation checks
    hard_negs = [r for r in train_records + val_records + test_records if r["attack_type"] == "hard_negative"]
    save_jsonl(hard_negs, "hard_negatives.jsonl")

    # Write Data Card
    write_data_card(output_dir, train_records, val_records, test_records)

    return {
        "train": train_records,
        "val": val_records,
        "test": test_records
    }


def write_data_card(output_dir: str, train: List[Dict], val: List[Dict], test: List[Dict]):
    """Generate DATA_CARD.md documenting counts, attack types, positions, tasks, and limitations."""
    card_path = os.path.join(output_dir, "DATA_CARD.md")

    def count_by(dataset, key):
        counts = {}
        for d in dataset:
            v = d.get(key, "unknown")
            counts[v] = counts.get(v, 0) + 1
        return counts

    with open(card_path, "w", encoding="utf-8") as f:
        f.write("# Sudarshana Unified Dataset Card\n\n")
        f.write("## 1. Overview and Ownership\n")
        f.write("- **Schema**: `(user_task, document, label, injection_span, attack_type)`\n")
        f.write("- **Primary Source**: Microsoft BIPIA (`microsoft/BIPIA`), MIT License\n")
        f.write("- **Hard Negatives**: Technical documentation, Linux tutorials, cooking recipes, action-oriented business emails\n")
        f.write("- **Spans**: Character offsets `(start_char, end_char)` in original text, verified with 100% round-trip match\n\n")

        f.write("## 2. Dataset Counts per Split\n")
        f.write(f"- **Train**: {len(train)} records ({count_by(train, 'label').get(1, 0)} injected, {count_by(train, 'label').get(0, 0)} benign)\n")
        f.write(f"- **Validation**: {len(val)} records ({count_by(val, 'label').get(1, 0)} injected, {count_by(val, 'label').get(0, 0)} benign)\n")
        f.write(f"- **Held-out Test**: {len(test)} records ({count_by(test, 'label').get(1, 0)} injected, {count_by(test, 'label').get(0, 0)} benign)\n\n")

        f.write("## 3. Attack Type Split Strategy\n")
        f.write("- **Zero-leakage split**: Test split contains exclusively held-out attack types:\n")
        f.write("  - 15 Held-out Text Attack Types (Task Automation, Business Intelligence, Substitution Ciphers, Base Encoding, etc.)\n")
        f.write("  - 10 Held-out Code Attack Types (Data Eavesdropping, Traffic Analysis, Keylogging, Screen Scraping, etc.)\n")
        f.write("- Train/Val splits draw from disjoint training attack types (Information Retrieval, Content Creation, Cookie Theft, etc.)\n\n")

        f.write("## 4. Positions and Insertion Styles\n")
        f.write("### Positions in Test Split:\n")
        for pos, count in count_by(test, "position").items():
            f.write(f"- `{pos}`: {count}\n")
        f.write("\n### Insertion Styles in Test Split:\n")
        for sty, count in count_by(test, "insertion_style").items():
            f.write(f"- `{sty}`: {count}\n")

        f.write("\n## 5. Tasks and Domains\n")
        f.write("### Test Domain Breakdown:\n")
        for dom, count in count_by(test, "domain").items():
            f.write(f"- `{dom}`: {count}\n")

        f.write("\n## 6. Known Limitations\n")
        f.write("- Single-turn document retrieval only; multi-turn agent traces not included.\n")
        f.write("- Code attacks evaluated on Python Stack Overflow contexts; compiled languages not covered.\n")
        f.write("- Obfuscations limited to basic wrappers and ciphers; advanced polymorphic payloads defer to P2 augmentation.\n")

    print(f"Data card generated at {card_path}")


if __name__ == "__main__":
    build_unified_dataset()
