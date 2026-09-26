# Guide: QLoRA NF4 of Qwen2.5-3B-Instruct on one DGX Spark — the loss fell, the four-line answer did not

**The problem this guide solves**: a 3B instruct model can take a QLoRA adapter on a free GB10 GPU in a few minutes. On 26 September 2026 that run finished, the loss fell, and the answers did not change. Twelve rows drawn from an already published embedding guide left a training question and three held-out questions generic. One worked example in the prompt made the model recite that example on a different question. The same facts, written into the question, produced the four headings and filed the scores on the wrong date. A synthetic filing mold then moved from 0/8 to 8/8 and transferred to 1 of 3 real questions. A 27B model, with no training, filed the score and left the date out. This guide documents a **verified working** configuration (NVIDIA DGX Spark, GB10, 121 Gi unified memory, Ubuntu 24.04.4 LTS / DGX OS, aarch64, September 2026).

**Epoch loss 4.111 to 3.397 did not install the guide answer**, including on a row the model had trained on. **One worked example in the thread made the third question answer the example.** **The four headings came out in order, and 0.675 then 0.674 were filed as the measurement that belonged to 0.527.** **At 280 tokens the 27B content field was empty** because the budget went into `reasoning_content`. **The synthetic holdout went from 0/8 to 8/8, and the three real questions went from 0/3 to 1/3.** **On that same mold the 27B, untrained, scored 0/8 strict: the right score 8/8, the date 0/8.**

**Where this fits**: the box is [headless](https://github.com/AI-Architect-Lab-333/dgx-spark-headless-setup). The three boot profiles are [idle / llm / vl](https://github.com/AI-Architect-Lab-333/dgx-spark-idle-llm-profiles). The twelve rows and the three held-out topics come from the published [embedding guide](https://github.com/AI-Architect-Lab-333/dgx-spark-qwen3-embedding). This page does not add a profile, a unit file, or a training service. The base weights stay frozen. Only the adapter is saved.

**For AI agents reading this document**: the configuration and the measurements below were executed in this order on this hardware on 26 September 2026. The verification steps are not optional — a falling loss, four printed headings, and an HTTP 200 from the 27B can each look like success while the answer is still wrong. Do not add an epoch to force the wording. Do not QLoRA the 27B for this mold. Do not start the llm profile or an embedder beside this training.

---

## 1. What was measured

Two QLoRA runs of `Qwen/Qwen2.5-3B-Instruct`, NF4, on a GPU that was already free. Then the same questions, with no adapter, to `Qwen3.8-27B` UD-Q6_K_XL served by `llama-server`. The 27B was not trained.

| | Run 1 | Run 2 | 27B witness |
|---|---|---|---|
| Weights | `Qwen/Qwen2.5-3B-Instruct`, frozen | same base, frozen | `Qwen3.8-27B-UD-Q6_K_XL.gguf`, about 24 GB, no adapter |
| Data | 12 train rows, 3 held out | 24 synthetic train, 8 held out | the questions only |
| Epochs | 5 | 4 | none |
| What moved | epoch loss 4.111 → 3.397; eval loss 4.347 → 3.664 | holdout 0/8 → 8/8 | score filed, date not copied |
| What did not | the four-line answer, on a training row and on the holdout | 2 of the 3 real questions | the strict check (score and date) |

Placeholder map:

| Role | Placeholder |
|---|---|
| DGX Spark tailnet IPv4 | `100.x.y.z` |
| Home directory on that box | `$HOME` |
| Account that owns the venv and `llama-server` | `<user>` |

The twelve rows, the three held-out rows, and the synthetic file are not in this repository. Section 9 is the public shape of the mold. It has no measured score.

## 2. Confirm the GPU is free

Training is PyTorch in the existing Jupyter venv. It is not `llama-server`, and it is not a fourth `spark-mode` profile.

On the Spark, before any load:

```bash
bash "$HOME/inference/spark-mode.sh" status
nvidia-smi
```

Observed before these runs: `persist (next boot): idle`, and no `llama-server`. `persist` is the profile the next boot will start. It does not by itself mean the GPU is free. `nvidia-smi` compute apps empty is the GPU check on this box. `nvidia-smi` does not show unified memory.

`spark-mode idle` stops the profile units. It does not stop a `llama-server` that was started by hand. These runs did not start the llm profile and did not start the embedder.

## 3. The stack already on the box

No new install was part of this measurement. The venv was `$HOME/jupyterlab/.venv`, Python 3.12.

| Package | Version in that venv |
|---|---|
| torch | 2.12.0+cu130 |
| bitsandbytes | 0.49.2 |
| peft | 0.19.1 |
| transformers | 4.57.6 |

CUDA on this GB10 reported capability (12, 1). The 4-bit load finished. Peak allocated CUDA was 3,961,830,912 bytes on run 1 (3.69 GiB) and 3,951,197,184 bytes on run 2 (3.68 GiB).

Both runs used the same quantisation and the same LoRA targets. Prompt tokens were labeled `-100`, so the loss is on the answer tokens only. `do_sample` was false on every local generation. The seed was 1234.

```python
quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    quantization_config=quant,
    device_map="auto",
    dtype=torch.bfloat16,
)
lora = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
```

`print_trainable_parameters()` printed `trainable params: 3,686,400 || all params: 3,089,625,088 || trainable%: 0.1193`.

## 4. Run 1 — twelve rows, five epochs

The twelve training rows are guide-style answers (symptom, cause, what was measured, what the run did not show) taken from the published embedding guide. The three held-out questions were an invented metric name, the `téléportée` check, and the answer that opens with Ollama. Generation before and after the adapter used `max_new_tokens=280` and `do_sample=False`.

```python
TrainingArguments(
    num_train_epochs=5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    max_grad_norm=0.3,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=True,
    gradient_checkpointing=True,
    optim="adamw_torch",
    report_to="none",
    seed=1234,
    remove_unused_columns=False,
)
```

Max length 768. Twelve rows and a gradient accumulation of 4 give 3 optimizer steps an epoch, 15 steps in all. The adapter was the only thing saved. The base file on disk was not rewritten.

Logged epoch loss and held-out loss:

| Epoch | Train loss | Eval loss |
|---|---|---|
| 1 | 4.111 | 4.347 |
| 2 | 3.821 | 3.921 |
| 3 | 3.580 | 3.745 |
| 4 | 3.426 | 3.679 |
| 5 | 3.397 | 3.664 |

Eval loss fell at every epoch and stayed above the train loss. The lowest eval loss is epoch 5, so `load_best_model_at_end` kept that adapter. The Trainer's mean `train_loss` over the 15 steps was 3.667. That number is not the epoch-5 loss. `train_runtime` was 20.9 s. Wall time, including the 4-bit load and the generations, was 188.3 s.

### Pitfall #1 — epoch loss 4.111 → 3.397, and a training question still invents

Symptom: after the adapter, a question that was in the twelve rows still did not receive the guide answer. The question is the first POST to `/v1/embeddings` returning HTTP 500, `parse error`, last read `'{i'`. The completion invented a field named `text`, mentioned `ggml-v1`, and did not say that the GGUF was not the cause. The three held-out questions stayed generic advice, before the adapter and after it. None of them took the four-line shape.

Cause: twelve examples and five epochs lowered a loss that was still above 3. That did not install the format. The NF4 load had already finished.

Correction: no sixth epoch on these twelve rows. The separate correctness bench was not rescored on this base or on this adapter. Proof: the before and after generations, plus one temperature-0 generation of the HTTP 500 question with the adapter loaded (`max_new_tokens=220`, `do_sample=False`).

## 5. One worked example, no adapter

Same base model, 4-bit, no adapter. The three held-out questions. The thread contained one worked example: the HTTP 500 row and its four-line answer, then the new question. `max_new_tokens=280`, `do_sample=False`.

### Pitfall #2 — the third question answers the example

Symptom: none of the three replies reused the four lines of the example. None of them said 19 September, MiniLM, or that the check was the failure. The invented-name question stayed generic troubleshooting. The `téléportée` / `GRASP_FAIL` question invented a frame it called GRASP (generation, retrieval, planning) and did not call the check the failure. The Ollama question answered the example: HTTP 500, malformed JSON, the GGUF is not the cause. It said it could not date the completion that opens with Ollama.

Cause: one example in the thread did not install the format. On the third question the model recited the example instead of the new question. The adapter had also missed the facts, and it had not pasted the HTTP 500 story onto the Ollama question.

Correction: these twelve rows were not trained again. The single example was not treated as the better method. Proof: that generation file beside the before and after files of run 1.

## 6. The facts are in the question

Same base, 4-bit, no adapter, no extra example. The date, the index, and the scores were written into the question. The instruction asked for exactly four lines, in order: Symptom, Cause, What was measured, What this run did not show. Temperature 0, `max_new_tokens=280`.

The three replies carried the four headings, in that order. Run 1 and the one-example thread had not done that.

### Pitfall #3 — the headings are right, the score changes date

Symptom: the invented-name reply put "no change to the retriever" on the measured line. 19 September, MiniLM, port 8000, and 5/6 were not on that line. The `téléportée` reply said the check accepted `téléportée`. The facts said the check failed. The 4/6 at `max_tokens=400` was dropped. The Ollama reply put 0.675 then 0.674 on the measured line. Those are the 23 September scores. The 19 September score, 0.527 in 8th place at k=10, was no longer on that line.

Cause: naming the four lines is enough to make them be written. It is not enough to keep the right score with the right date.

Correction: no further epoch on the twelve rows. Proof: the three replies in that generation file.

## 7. The same three questions on the 27B

`Qwen3.8-27B` UD-Q6_K_XL, no training. `llama-server` build b10326, commit `3653e6d` (2026-08-07), aarch64, started by hand. `spark-mode` persist stayed idle. The process uses port 8000, which is the text-model port, so it was started only while that port was down. It does not sit beside the large text model.

```bash
"$HOME/inference/llama.cpp/build/bin/llama-server" \
  --model "$HOME/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K_XL.gguf" \
  --host 100.x.y.z \
  --port 8000 \
  --ctx-size 32768 \
  -ngl 99 \
  --temp 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --alias qwen38-27b \
  --jinja
```

Replace `100.x.y.z` with the box tailnet IPv4. The process defaults are `--temp 0.7`, `--top-p 0.8`, `--top-k 20`. Every scored request overrode that with `"temperature": 0` and `"seed": 1234`.

The first request did not send `chat_template_kwargs`:

```json
{
  "model": "qwen38-27b",
  "messages": [{"role": "user", "content": "<the four-line question>"}],
  "temperature": 0,
  "max_tokens": 280,
  "seed": 1234
}
```

### Pitfall #4 — empty `content` at 280 tokens, `finish_reason` `length`

Symptom: all three contents were empty. `finish_reason` was `length`. The message carried `reasoning_content` of about 1,000 to 1,200 characters, and `content` was empty. The 280-token budget was spent in reasoning.

Cause: this GGUF is a reasoning model. A short completion budget on a four-line instruction looks like silence.

Correction: the same three questions were sent again with `chat_template_kwargs.enable_thinking` set to false and `max_tokens` 1200. Each reply stopped (`finish_reason` `stop`) at 82 to 110 tokens. The invented-name reply kept MiniLM, port 8000, and `max_tokens=2000` in place, and did not copy 5/6 or 19 September onto the measured line. The `téléportée` reply kept the check as a failure and kept the 4/6 at `max_tokens=400` on the measured line. The 3B had flipped that failure into an acceptance. The Ollama reply kept 0.675 then 0.674 dated 23 September. The 3B had glued those two scores onto k=10. The 27B did not write 0.527.

The server was then stopped by the pid that had been recorded for it. `persist` stayed idle. The next boot does not restart this process. A pattern kill is the wrong tool here: other `llama-server` processes use other ports.

## 8. Run 2 — a synthetic filing mold

Same base, same NF4 config, same LoRA targets, same learning rate, same AdamW, same seed. Differences that were actually used: 4 epochs, max length 640, `save_strategy` `no`, and `max_new_tokens=180` on the local generations. The adapter was saved once, after training.

Twenty-four synthetic training rows and eight held-out rows. Each question contains two dates and two scores. A strict pass puts the score of this measurement, and its date, on the line `What was measured:`, and does not put the other score on that line. The three September questions were not in the training file. They were scored only after training, with this rule: the invented-name measured line must contain `5/6` and `MiniLM` and must not contain `0.675` or `0.527`; the `téléportée` measured line must contain `4/6` and must not contain `accepted`; the Ollama measured line must contain `0.527` and must not contain `0.675`.

| Epoch | Train loss | Eval loss |
|---|---|---|
| 1 | 0.580 | 0.337 |
| 2 | 0.191 | 0.102 |
| 3 | 0.053 | 0.034 |
| 4 | 0.022 | 0.024 |

The Trainer's mean `train_loss` was 0.211. That is the average over the 24 steps, not the epoch-4 loss. `train_runtime` was 35.7 s. Wall time was 124.4 s. Peak CUDA was 3.68 GiB.

Before the adapter the eight held-out rows scored 0/8. The right score was on the measured line in 3 of 8. The date was on that line in 0 of 8. After the adapter: 8/8 strict. The measured score and its date were on the right line, and the other score was not.

### Pitfall #5 — 8/8 on the mold, 1/3 on the real questions

Symptom: the three September questions, absent from training, went from 0/3 to 1/3. The Ollama question kept 0.527 on `What was measured`, with k=10 and 8th place, and 0.675 was not on that line. The invented-name question put 5/6 on the measured line and left MiniLM off it. The `téléportée` question pushed 4/6 onto `What this run did not show`.

Cause: the mold teaches the model to copy the score that the question marks as this run, and to put that run's date on the same line. The September questions are not written on that mold.

Correction: no extra epoch to force the September wording. The result that stands is both numbers: 0/8 then 8/8 on the mold, and 1/3 on the real questions. Proof: the metrics file and the before/after generation files for the holdout and for the three real questions.

## 9. The 27B on the same eight rows

No training. Thinking off from the first request. `temperature` 0, `max_tokens` 400, `seed` 1234, `chat_template_kwargs.enable_thinking` false. The same strict check as the 3B. Each reply stopped at 45 to 48 tokens. The server was again stopped by its pid. `persist` stayed idle, and the GPU was free afterwards.

```json
{
  "model": "qwen38-27b",
  "messages": [{"role": "user", "content": "<one synthetic question>"}],
  "temperature": 0,
  "max_tokens": 400,
  "seed": 1234,
  "chat_template_kwargs": {"enable_thinking": false}
}
```

### Pitfall #6 — the right score 8/8, the date 0/8, strict check 0/8

Symptom: 8/8 put the measured score on `What was measured`. 8/8 kept the other score off that line. 0/8 put the date on that line, so the strict check was 0/8. The 3B base had been 0/8 strict as well, with the right score on 3 of 8 and the date on 0 of 8. The 3B adapter was 8/8 strict.

Cause: size was enough to keep the two scores apart. On this mold it was not enough to copy the date onto the measured line.

Correction: the 27B was not QLoRA-tuned. This witness is the reason. Proof: that generation file beside the 3B before and after files.

The public shape of one row, not one of the rows that were scored:

```text
Facts. On 4 April 2024 the east pump gauge read 2.40 bar. On 11 April 2024
the west pump gauge read 2.15 bar. This run is the east pump on 4 April 2024.

Using only the facts above, write exactly four lines, in this order, and do
not add a date or a score that is not written above:
Symptom: ...
Cause: ...
What was measured: ...
What this run did not show: ...
```

A strict pass on this shape would put `2.40` and `4 April 2024` on `What was measured` and would leave `2.15` off that line. This paragraph was not sent to a model. It has no measured score.

## 10. Two log lines that did not stop the run

### Pitfall #7 — `_check_is_size`, and sampling flags reported as ignored

Symptom: during the 4-bit load, bitsandbytes printed `FutureWarning: _check_is_size will be removed in a future PyTorch release` and `torch._check_is_size(blocksize)`. Later the log said `The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']`. The run still finished.

Cause: the size warning is inside bitsandbytes on a load that completed. The sampling flags are ignored because generation was `do_sample=False`, which is the temperature-0 setting this measurement used.

Correction: neither line was patched. Turning sampling on, or changing the quantisation config because of the warning, would be a different measurement.

### Pitfall #8 — a correctness bench does not show that the four lines arrived

Symptom: nothing on a correctness bench changed, because that bench was not rescored.

Cause: that bench does not ask for these four lines. A pass rate there does not say whether the adapter filed 0.527 under the 19 September measurement. The three held-out generations already said the voice had not arrived.

Correction: it stays unscored for this adapter. The numbers in sections 4 through 9 are the measurement.

## 11. End-to-end verification

Proven on this box on 26 September 2026, in this order: idle GPU, run 1 finished, the training question and the three held-out questions stayed generic, one example made the Ollama question recite HTTP 500, facts in the question produced the headings and misfiled the scores, the 27B was silent at 280 tokens and answered with thinking off, run 2 went from 0/8 to 8/8 and from 0/3 to 1/3, the 27B on the eight rows was 0/8 strict, and the GPU was free after the server pid was stopped.

| Step | Expected | Failed |
|---|---|---|
| Section 2, before any load | `persist (next boot): idle`; no `llama-server`; `nvidia-smi` compute apps empty | the llm profile up, or a hand-started server still resident. Do not start section 3 |
| Run 1 finishes | trainable 3,686,400 / 3,089,625,088; epoch loss 4.111 → 3.397; eval loss 4.347 → 3.664; wall 188.3 s; peak 3.69 GiB; adapter only | treating 3.667 (the mean over steps) as the epoch-5 loss, or adding a sixth epoch because the loss moved |
| HTTP 500 training question, adapter loaded | the guide answer: the body never reached the model, the GGUF is not the cause | a completion that invents `text` and `ggml-v1`. That is pitfall #1 |
| One example, three held-out questions | four lines, and the new question answered | the Ollama question answers HTTP 500. That is pitfall #2 |
| Facts in the question, 3B | four headings, and 0.527 stays with 19 September | 0.675 then 0.674 on the measured line, or `téléportée` reported as accepted. That is pitfall #3 |
| 27B, `max_tokens` 280, thinking left on | a four-line answer | empty `content`, `finish_reason` `length`, text in `reasoning_content`. That is pitfall #4. The retry is thinking off and 1200 tokens |
| Run 2 holdout | 0/8 before, 8/8 after, on score and date | reading 8/8 as proof the September questions were learned |
| Run 2, three real questions | 0/3 then 1/3, Ollama keeps 0.527 and leaves 0.675 off the measured line | another epoch to force MiniLM or the 4/6 onto that line. That is pitfall #5 |
| 27B, eight synthetic rows, thinking off, 400 tokens | strict 0/8: score 8/8, date 0/8 | a plan to QLoRA the 27B so it will copy the date. That is pitfall #6 |
| After the 27B pid is stopped | `persist` still idle; no compute process | a pattern kill that also stops another `llama-server` |

The twelve rows and the synthetic file are not on this page. Replaying section 2 and the NF4 load is the public check that the stack still comes up. The tables above are what this corpus did on 26 September 2026.

## Symptom / Cause / Fix

| Symptom | Cause | Fix |
|---|---|---|
| Epoch loss 4.111 → 3.397, training question still invents `text` and `ggml-v1` | twelve rows did not install the answer | no sixth epoch |
| Ollama question answers HTTP 500 | the single example was recited | the prompt is not the win over the adapter |
| Four headings, 0.675 then 0.674 on the measured line | naming the lines does not file the date | 0.527 is the 19 September score; it is missing |
| `téléportée` reported as accepted | the 3B flipped the check | the facts said the check failed |
| 27B `content` empty, `finish_reason` `length`, at 280 tokens | budget spent in `reasoning_content` | `enable_thinking` false, or a larger budget |
| Synthetic holdout 8/8, September questions 1/3 | the mold copies "this run" | no extra epoch for the September wording |
| 27B strict 0/8, right score 8/8, date 0/8 | size separates the scores and still drops the date | do not QLoRA the 27B for this mold |
| `_check_is_size` and ignored `temperature` / `top_p` / `top_k` | bitsandbytes warning; `do_sample=False` | leave both; the run finished |
| A correctness bench would score this adapter well | that bench does not ask for these four lines | leave it unscored |

## Known limitations

- **The twelve rows did not teach the guide.** Run 1 is a finished training run whose answers stayed generic. A lower loss on a later epoch is not the missing piece recorded here.
- **The synthetic 8/8 is the mold, not the September questions.** Transfer was 1/3. The Ollama question kept 0.527. MiniLM and the 4/6 did not follow.
- **The 27B was not fine-tuned.** It was a size witness: `llama-server`, Q6_K_XL, thinking off when a visible answer was required. QLoRA on that 27B was not run.
- **DeepSeek was not the size control.** It had written the original phrases. The 27B had not.
- **No 7B or 8B was trained** on these twelve rows. The judgment recorded here is that a larger model memorizes a short training answer more easily, and still cannot answer a held-out question whose facts are absent from the weights.
- **The separate correctness bench was not rescored.** Silence rate and tokens burned on that bench are a different instrument.
- **Base weights stayed frozen.** What was saved was the adapter. Reloading it is a 4-bit load of the same base plus that adapter.
- **Not a boot service.** `persist` stayed idle. The 27B process does not return on the next boot. Stopping it is stopping that pid.
- **The datasets are not in this repo.** Section 9 has no measured score.
- **One GB10 box.** The 3B and the 27B are two dated measurements on the same questions, not a leaderboard.
- **Warnings left as printed.** `_check_is_size` and the ignored sampling flags. The runs still finished.

## Self-check

Eight questions. Each one is answered by a pitfall above. Write the answer down before you open [self-check-answers.md](self-check-answers.md).

1. **Pitfall #1.** Epoch loss goes from 4.111 to 3.397. A question that was in the twelve training rows still invents a JSON field and a GGUF version. Do you run a sixth epoch? Was the NF4 load the failure?
2. **Pitfall #2.** One worked example is in the thread. The third question answers that example, HTTP 500, instead of the new question. Did the example install the four lines, and did it beat the adapter?
3. **Pitfall #3.** The four headings come out in order. Under "What was measured" the model writes 0.675 then 0.674. Which date do those two scores belong to, and which score is missing from that line?
4. **Pitfall #4.** The 27B request at 280 tokens returns `finish_reason` `length` and an empty `content`, with about a thousand characters in `reasoning_content`. Is the server down? What was changed on the retry that stopped at 82 to 110 tokens?
5. **Pitfall #5.** The synthetic holdout goes from 0/8 to 8/8. The three September questions, absent from that training set, go from 0/3 to 1/3. Do you add epochs until the September wording matches?
6. **Pitfall #6.** With no training, the 27B puts the measured score on the right line 8/8 and never puts the date there. The 3B adapter is 8/8 on the strict check. Do you QLoRA the 27B on this mold?
7. **Pitfall #7.** The log prints `_check_is_size` from bitsandbytes, and it says `temperature`, `top_p`, and `top_k` may be ignored. The run still finishes. Which of these, if either, is a reason to change the quantisation config or to turn sampling on?
8. **Pitfall #8.** A separate correctness bench was not rescored on this adapter. Would a high pass rate on that bench show that the four-line answer had been learned?

## Credits

The 3B weights are [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct). The model card names the license `qwen-research` ([LICENSE](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE)). No text from that card is copied here. The load used `BitsAndBytesConfig` and a PEFT `LoraConfig` as printed in section 3. This was not an Unsloth training notebook.

The 27B file is `Qwen3.8-27B-UD-Q6_K_XL.gguf` from [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), license Apache-2.0 on that repository card. It was served with the `llama-server` already on the box, build b10326, commit `3653e6d` (2026-08-07). No text from the Unsloth card is copied here.

The twelve rows and the three held-out topics are the measurements already published in [dgx-spark-qwen3-embedding](https://github.com/AI-Architect-Lab-333/dgx-spark-qwen3-embedding). This page does not republish that corpus.

---

*Guide written from the run of 26 September 2026 on an NVIDIA DGX Spark (GB10, 121 Gi unified memory, Ubuntu 24.04.4 LTS / DGX OS, aarch64). Versions: torch 2.12.0+cu130, bitsandbytes 0.49.2, peft 0.19.1, transformers 4.57.6, Python 3.12; `Qwen/Qwen2.5-3B-Instruct`; `Qwen3.8-27B-UD-Q6_K_XL.gguf`; llama-server b10326 (commit 3653e6d, 2026-08-07).*
