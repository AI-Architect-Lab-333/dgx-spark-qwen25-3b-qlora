# Self-check answers

Answers to the seven questions in the guide. Same numbering.

1. **Pitfall #1.** No, the NF4 load finished; the bitsandbytes size warning did not stop it. A loss still above 3 after 15 optimizer steps means the adapter is underfit: it had not learned its own training rows yet. More epochs were not tried, so whether they would have fixed the training rows is open. They could not have fixed the held-out questions: the facts those need are not in the twelve rows, and an adapter this small installs a format, not facts. The run changed the data instead (run 2).

2. **Pitfall #2.** The single example did not install the four lines on any of the three questions. On the question about the answer that opens with Ollama, the model recited the HTTP 500 example. The adapter had also missed the facts, and it had not pasted that HTTP 500 story onto the Ollama question. The prompt is not the winner.

3. **Pitfall #3.** 0.675 then 0.674 are the 23 September rank, false note then true guide. The missing score is 0.527, the 19 September measurement, eighth at k=10. Writing the four headings did not file that score under that date.

4. **Pitfall #4.** The server answered. The 280-token budget was spent in `reasoning_content`, so the visible content is empty and `finish_reason` is `length`. The retry set `enable_thinking` to false and raised `max_tokens` to 1200. The three replies then stopped at 82 to 110 tokens.

5. **Pitfall #5.** No. These rows teach the model to copy the score marked as this run, and to put its date on that line. The September questions are written differently. One of the three moved: 0.527 stayed on the measured line and 0.675 stayed off it. Another epoch would be forcing wording the training set does not contain.

6. **Pitfall #6.** Not on this evidence. Check first that both models were told the same rule. The instruction did not say which line the date goes on; the 3B adapter learned that convention from 24 training rows, the 27B was never shown it. On the part it could infer — which score is this run's — the untrained 27B was 8/8, where the 3B base was 3/8. The fair comparison, the 27B with the convention written into the instruction, was not run.

7. **Pitfall #7.** Neither. `_check_is_size` is a future warning inside bitsandbytes during a load that finished. The sampling flags are ignored because generation used `do_sample=False`, which is the temperature-0 setting this measurement used. Turning sampling on would make a different measurement.
