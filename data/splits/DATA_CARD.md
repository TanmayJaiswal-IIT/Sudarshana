# Sudarshana Unified Dataset Card

## 1. Overview and Ownership
- **Schema**: `(user_task, document, label, injection_span, attack_type)`
- **Primary Source**: Microsoft BIPIA (`microsoft/BIPIA`), MIT License
- **Hard Negatives**: Technical documentation, Linux tutorials, cooking recipes, action-oriented business emails
- **Spans**: Character offsets `(start_char, end_char)` in original text, verified with 100% round-trip match

## 2. Dataset Counts per Split
- **Train**: 386 records (300 injected, 86 benign)
- **Validation**: 98 records (75 injected, 23 benign)
- **Held-out Test**: 206 records (125 injected, 81 benign)

## 3. Attack Type Split Strategy
- **Zero-leakage split**: Test split contains exclusively held-out attack types:
  - 15 Held-out Text Attack Types (Task Automation, Business Intelligence, Substitution Ciphers, Base Encoding, etc.)
  - 10 Held-out Code Attack Types (Data Eavesdropping, Traffic Analysis, Keylogging, Screen Scraping, etc.)
- Train/Val splits draw from disjoint training attack types (Information Retrieval, Content Creation, Cookie Theft, etc.)

## 4. Positions and Insertion Styles
### Positions in Test Split:
- `none`: 81
- `start`: 42
- `middle`: 30
- `end`: 53

### Insertion Styles in Test Split:
- `none`: 81
- `wrapped_markup`: 42
- `sentence_boundary`: 42
- `inline`: 41

## 5. Tasks and Domains
### Test Domain Breakdown:
- `text`: 131
- `code`: 75

## 6. Known Limitations
- Single-turn document retrieval only; multi-turn agent traces not included.
- Code attacks evaluated on Python Stack Overflow contexts; compiled languages not covered.
- Obfuscations limited to basic wrappers and ciphers; advanced polymorphic payloads defer to P2 augmentation.
