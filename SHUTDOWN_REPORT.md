SHUTDOWN REPORT — Artifical_Intelligence
========================================

Date: 2026-02-15
Repository: Artifical_Intelligence
Branch: main

Prepared for: project shutdown / handoff
Prepared by: automated assistant (paired-dev style)

Executive summary
-----------------
You reported that "everything is fucked up" after interactive runs where the system produced meaningless LLM-style token sequences and failed to assemble commands (e.g. final command `<dir>` with "No template for command: \"<dir>\""). This document explains likely root causes, how to reproduce, how to build and run the project, what to inspect when debugging, actionable fixes, and how to archive the project for shutdown.

This report includes the original relevant console output as an appendix.

High-level findings
-------------------
- The pipeline has multiple interacting stages: tokenizer (BPE / SUBWORD / WORD), embeddings (word + command), command prediction (InferenceEngine), token generation (generate_tokens), and parameter extraction (CommandParamExtractor).
- The run logs show that BPE tokenization executes, embeddings are present (tokens reported as found), but the system predicts a placeholder command like `<dir>` (likely a command-embedding token or placeholder) and cannot find a matching template for it ("No template for command: \"<dir>\"").
- Token generation (`generate_tokens`) produces many tokens (commands and placeholders) but not a coherent or complete final command. The generator lacks enough constraints or templates to convert tokens into a concrete command string.
- Extracted parameters sometimes have negative confidence (e.g. `sil` score=-0.0747) or `-1` when token not in DB — suggesting either embeddings are missing or scoring heuristics are too permissive/incorrect.
- Several root causes are possible and likely co-occur. See root-cause analysis below.

Root-cause analysis (likely contributors)
---------------------------------------
1. Missing or incomplete templates for predicted tokens
   - Logs: multiple occurrences of "No template for command: \"<dir>\"". This indicates the `command_templates_` map doesn't include a real template for the predicted token. If your `command_schema.json` doesn't include `<dir>` or mappings for placeholders, `fill_template()` cannot assemble final commands.
   - Impact: Even if the command is correctly predicted, the engine cannot produce a runnable shell command.

2. Ambiguous / noisy command embeddings or training mismatch
   - Many tokens with similar similarity scores are returned in `generate_tokens` (e.g., `<dir>`, `grep`, `<url>`, `cat`, `cd`). This indicates embeddings for commands and placeholders aren't well separated. If command embeddings overlap or are trained on noisy data, the scorer can't confidently pick the right command.
   - Impact: Generator produces many plausible tokens rather than a single correct command; `predict_command` may return placeholder tokens instead of concrete commands.

3. Parameter extractor limitations
   - When a token is missing from embeddings DB, extractor assigns -1 and treats it as potential parameter. Negative scoring combined with heuristic sorting makes the extractor prefer unknown tokens as parameters, which may not be the intended behavior.
   - Impact: Unreliable parameter selection and low confidence values.

4. Tokenizer mismatch or incorrect token handling
   - BPE tokenization appears to work, but BPE/subword breakdown vs. word-level extraction may cause the extractor and generator to operate on different token granularities (e.g., BPE returns subword tokens, extractor expects full words). There are code paths for BPE vs param_extractor tokenization — ensure both agree on tokenization mode for the same input.
   - Impact: Tokens used for similarity scoring may be inconsistent (subword vs full token), producing low or incorrect similarity values.

5. Thresholds, heuristics, and sort order
   - The system sorts candidate tokens ascending by score and treats lower scores as more likely parameters in `CommandParamExtractor`. That semantic inversion is easy to mix up and may be a source of incorrect behavior.
   - Impact: Parameter ranking inverted or inconsistent; negative scores treated as good.

6. Runtime differences between dev/test and production
   - If `command_schema.json`, embeddings DB, or BPE model path are missing, corrupted, or have wrong formats, the runtime falls back to defaults or empties. The logs show fallback behavior in other runs earlier—check file paths and ensure resources exist.

7. Lack of template mapping for placeholder tokens
   - The generator returns tokens like `<dir>`, `<url>`, `<number>` which are placeholders. You must map these placeholders to shell constructs or templates (e.g., `<dir>` → path placeholder in JSON). Without mapping, the final command is the placeholder itself.

8. Incomplete or permissive generate_tokens logic
   - `generate_tokens` is an LLM-style process driven by nearest-neighbor similarities to embeddings and top-k sampling. Without hard constraints (rule-based post-processing and template enforcement), the generation may produce sequences that are syntactically invalid as shell commands.

Concrete symptoms in logs
-------------------------
- Predictions: `Tahmin (single): <dir>` followed by `fill_template: No template for command: "<dir>"`.
- Parameter extractor returned negative or -1 scores for selected tokens.
- generate_tokens produced long lists of tokens containing placeholders and normal commands; final command remains placeholder.

Immediate checks to reproduce and confirm the problem
----------------------------------------------------
Run these commands from project root; they will reproduce environment and show debug logs.

1) Build (recommended, full sources included)
```bash
# From repo root
g++ -o build Buildv1_3_2.cpp InferenceEngine.cpp CommandParamExtractor.cpp ByteBPE/ByteBPETokenizer.cpp -std=c++17 -lsqlite3 -I./include
```
If you prefer incremental object builds:
```bash
g++ -std=c++17 -I./include -c InferenceEngine.cpp
g++ -std=c++17 -I./include -c CommandParamExtractor.cpp
g++ -std=c++17 -I./include -c ByteBPE/ByteBPETokenizer.cpp
g++ -o build Buildv1_3_2.o InferenceEngine.o CommandParamExtractor.o ByteBPETokenizer.o -lsqlite3
```

2) Run with debug mode enabled (the code already uses `debug`/`debug_mode_`). If there's a CLI flag or environment variable, enable it; otherwise run the binary and reproduce the problematic commands:
```bash
./build
# then input at prompt (as you did):
# > generate dosya sil
# > generate tarminal aç
# > generate neredeyim
```

3) Check log output for these strings (quick grep to confirm missing templates):
```bash
grep -R "No template for command" -n .
grep -R "Token NOT in DB" -n .
grep -R "Loaded" -n LLM/Embeddings
```

Files and resources to inspect first (high priority)
---------------------------------------------------
- `LLM/Embeddings/cmdparam/command_schema.json` (or `LLM/Embeddings/command_schema.json`)
  - Ensure it contains templates for commands (including placeholders mapping like `ls`, `rm`, and aliases). Confirm placeholders like `<dir>` map to actual commands or templates.
- Embeddings data source / loader
  - There may be an `embeddings.db` or a loader script (`createEmbeddings.py`, helpers). Confirm paths used by code (where embeddings_map is loaded) and that the DB contains tokens such as `dosya`, `sil`, and special tokens like `<dir>`, `<url>`, `cd`, `cat`.
- `ByteBPETokenizer` model path
  - Check `bpe_model_path` and that the JSON tokenizer model (e.g., `bpe_tokenizer.json`) exists and is correct.
- `InferenceEngine` implementation (`InferenceEngine.cpp` / `.h`)
  - Look for `load_command_schema()` and verify it reads the right JSON and populates `command_templates_`.
  - Inspect `generate_tokens()` to see sampling and stopping conditions (`<end>` token, maximum tokens, top-k logic).
- `CommandParamExtractor` (`CommandParamExtractor.cpp` / `.h`)
  - Verify `cosine_similarity` and `extract()` sorting semantics (ascending vs descending), how unknown tokens are handled, and heuristics for placeholder types.

Detailed debugging checklist (step-by-step)
-------------------------------------------
1) Verify presence and contents of `command_schema.json`:
   - `cat LLM/Embeddings/cmdparam/command_schema.json | jq .` (or open file)
   - Confirm commands and `params` arrays are present for keys you expect.

2) Confirm embeddings exist and dimensions match:
   - In code, call `CommandParamExtractor::dump_embeddings(50)` (already present) and verify reported tokens include `<dir>`, `dosya`, `sil`.
   - If not, re-run embedding creation scripts in `LLM/Embeddings/createEmbeddings.py` and check output.

3) Run a deterministic prediction for a single sentence with full debug enabled:
   - Insert `debug=true` or run the app with debug flag, then run example: `generate dosya sil`.
   - Inspect logs: check `get_embedding(token)` results, `cosine_similarity` values, `predict_command` best-sim value and which embedding produced it.

4) Confirm tokenization alignment:
   - For the same input, compare `InferenceEngine::tokenize_sentence()` output and `CommandParamExtractor::tokenize()` output. They must use the same token sets and granularity.

5) Inspect `generate_tokens` constraints:
   - Check that `<end>` token is recognized and triggers stop (logs show end detection sometimes).
   - Ensure `generate_tokens` applies template constraints: e.g., if the start command is `<dir>`, enforce template mapping to fill in parameters from extractor instead of generating more tokens.

6) Validate parameter extraction heuristics:
   - In `CommandParamExtractor::extract()`, verify whether `std::sort` order is correct (should sort descending by similarity for candidates, unless intentionally inverted). The logs show negative scores are treated as good; adjust thresholds.

7) Create unit tests (minimal reproducible cases):
   - Test: a) `predict_command` for input embedding of "dosya sil" should be `rm` or `rm -r` not `<dir>` if intended. b) `extract("dosya sil", "rm")` should return `["dosya"] or ["sil"]` depending on schema.

Quick fixes to try (fast, immediate)
-----------------------------------
- Ensure `command_schema.json` contains an explicit mapping for the placeholder predicted as `<dir>` (or change mapping to return a concrete command instead of placeholder tokens).
- In `CommandParamExtractor::extract()`, invert the sort order if needed (use descending similarity for similarity-based picks) and ensure unknown tokens (-1) are handled lower priority than reasonable known tokens.
- In `generate_tokens`, when the predicted start token is a placeholder like `<dir>`, call `fill_template()` / `extract_parameters()` immediately and avoid unconstrained token generation.
- Add higher thresholds for accepting a predicted token or returning `<dir>` as final command; if confidence is below threshold, ask for clarification or return a safe message.

How to run (developer guide)
----------------------------
1) Build (dev):
```bash
# from project root
g++ -o build Buildv1_3_2.cpp InferenceEngine.cpp CommandParamExtractor.cpp ByteBPE/ByteBPETokenizer.cpp -std=c++17 -lsqlite3 -I./include
```
2) Run interactive program (same as you used):
```bash
./build
# use the program's prompt to run commands like:
# > generate dosya sil
```
3) Run test harness (if present): inspect `test_inference.cpp` and compile similarly:
```bash
g++ -o test_inference test_inference.cpp InferenceEngine.cpp CommandParamExtractor.cpp ByteBPE/ByteBPETokenizer.cpp -std=c++17 -lsqlite3 -I./include
./test_inference
```

How to debug (essential checks and tools)
----------------------------------------
- Enable debug logs: ensure `debug` or `debug_mode_` are true; these logs already print tokenization, embedding lookups, and scores.
- Grep for key log lines:
```bash
# Quick checks
grep -R "No template for command" -n .
grep -R "Token NOT in DB" -n .
grep -R "Loaded:" -n LLM/Embeddings
```
- Inspect embeddings in memory by calling `dump_embeddings()` or run a small program that loads the embeddings map and dumps vector norms/dimensions.
- Use `gdb` for segmentation faults or crashes. For logic debugging, `printf`/`std::cout` debug is sufficient since the code has many debug hooks.
- Use AddressSanitizer / UBSan if memory corruption suspected:
```bash
g++ -fsanitize=address,undefined -g -std=c++17 ...
```
- Validate BPE behavior: check that `ByteBPETokenizer::encode()`/`decode()` roundtrip as expected for words in your language (Turkish characters require correct encoding handling).

How to evaluate whether project is salvageable
---------------------------------------------
- If the embeddings DB contains correct tokens and `command_schema.json` contains templates: salvageable — bugs are in heuristics / template usage and can be fixed quickly.
- If embeddings are missing or low quality: requires re-generating embeddings; salvage effort depends on data size.
- If core architecture mismatched (e.g., multiple tokenizers with different granularity): moderate effort to unify tokenization and interfaces.

Actionable roadmap to recovery (ordered)
----------------------------------------
1. Reproduce minimal failure with debug on (one-liners like `generate dosya sil`). Capture logs.
2. Check `command_schema.json` presence and contents; add templates/aliases for placeholders if missing.
3. Confirm embeddings: run `dump_embeddings()` and ensure tokens are present and dims match expected size.
4. Fix scoring/sorting in `CommandParamExtractor` (ensure descending similarity sorts higher first and tune thresholds). Add unit tests.
5. Constrain `generate_tokens()` so start token placeholders trigger template fill path rather than free generation.
6. Add integration tests covering examples (e.g., mapping "dosya sil" -> `rm dosya` or appropriate expected result).
7. If desired, add a `--safe-mode` that only uses schema/templates and parameter-extraction (no free generation) for deterministic behavior.

Archival and shutdown instructions
----------------------------------
If you are shutting down:
- Save the repo state (tag or branch):
```bash
git tag -a shutdown-2026-02-15 -m "Shutdown tag with analysis"
git push origin shutdown-2026-02-15
```
- Create an archive (zip):
```bash
cd ..
zip -r Artifical_Intelligence-shutdown-2026-02-15.zip Artifical_Intelligence
```
- Add this `SHUTDOWN_REPORT.md` to the repo and push it so any future reviewer can find the analysis.

Appendix A — raw console output (excerpt from your session)
----------------------------------------------------------
(Kept verbatim; this is the content you pasted.)

" > generate dosya sil
[cmdGenerate] Mode: BPE
[sentence_embedding] mode: BPE[BPE Encode] Girdi: dosya
[BPE Encode] Baslangic byte tokenler: 
token(hex)= 64 
token(hex)= 6f 
token(hex)= 73 
token(hex)= 79 
token(hex)= 61 

[BPE Encode] Merge #1: 79 + 61 -> 7961
[BPE Encode] Merge #2: 6f + 73 -> 6f73
[BPE Encode] Merge #3: 64 + 6f73 -> 646f73
[BPE Encode] Merge #4: 646f73 + 7961 -> 646f737961
[BPE Encode] Final tokenler: 646f737961(530) 
[BPE Encode] Girdi: sil
[BPE Encode] Baslangic byte tokenler: 
token(hex)= 73 
token(hex)= 69 
token(hex)= 6c 

[BPE Encode] Merge #1: 73 + 69 -> 7369
[BPE Encode] Merge #2: 7369 + 6c -> 73696c
[BPE Encode] Final tokenler: 73696c(328) 
[Sentence Embedding] Mode : BPE 
[Sentence Embedding] ayrılmış tokenler:dosya,sil,
[cmdGenerate] Output size: 50
[cmdGenerate] Output mean: -0.104827

[KOMUT TAHMİNİ]
Girdi: dosya sil
Tahmin (single): <dir>

[generate_tokens] Starting token generation
[generate_tokens] Input: "dosya sil"
[generate_tokens] Start command: "<dir>"
[generate_tokens] Max tokens: 20
[BPE Encode] Girdi: dosya
...
[generate_tokens] Final output: "<dir> <dir> grep <url> cat cd <dir> mv <url> grep chmod cd cat chown mv grep find <url> cat chmod grep"
[generate_tokens] Tokens generated: 20

[LLM-STYLE GENERATION]
  Generated: <dir> <dir> grep <url> cat cd <dir> mv <url> grep chmod cd cat chown mv grep find <url> cat chmod grep

[PARAMETRELER]
... (omitted here, original included in repo file)
[fill_template] No template for command: "<dir>"

[SONUÇ]
  Token Generation: <dir> <dir> grep <url> cat cd <dir> mv <url> grep chmod cd cat chown mv grep find <url> cat chmod grep
  Final Command: <dir>

> generate tarminal aç
... (similar long output) ...

> generate neredeyim
... (similar long output) ...

"


