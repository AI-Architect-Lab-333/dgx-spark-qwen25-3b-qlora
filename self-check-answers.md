# Self-check answers

Answers to the eight questions in the guide. Same numbering.

1. **Pitfall #1.** No sixth epoch. Twelve rows and five epochs lowered a loss that was still above 3, and they did not install the guide answer, including on a row the model had seen. The NF4 load had finished. The bitsandbytes size warning did not stop it. The failure is the data and the task.

2. **Pitfall #2.** The single example did not install the four lines on any of the three questions. On the question about the answer that opens with Ollama, the model recited the HTTP 500 example. The adapter had also missed the facts, and it had not pasted that HTTP 500 story onto the Ollama question. The prompt is not the winner.

3. **Pitfall #3.** 0.675 then 0.674 are the 23 September rank, false note then true guide. The missing score is 0.527, the 19 September measurement, eighth at k=10. Writing the four headings did not file that score under that date.

4. **Pitfall #4.** The server answered. The 280-token budget was spent in `reasoning_content`, so the visible content is empty and `finish_reason` is `length`. The retry set `enable_thinking` to false and raised `max_tokens` to 1200. The three replies then stopped at 82 to 110 tokens.

5. **Pitfall #5.** No. These rows teach the model to copy the score marked as this run, and to put its date on that line. The September questions are written differently. One of the three moved: 0.527 stayed on the measured line and 0.675 stayed off it. Another epoch would be forcing wording the training set does not contain.

6. **Pitfall #6.** No. The 27B kept the two scores apart and left the date off the measured line, so the pass count was 0/8. The 3B base had the right score on 3 of 8 and the date on 0 of 8. The 3B adapter is what added the date. No QLoRA adapter was trained on the 27B.

7. **Pitfall #7.** Neither. `_check_is_size` is a future warning inside bitsandbytes during a load that finished. The sampling flags are ignored because generation used `do_sample=False`, which is the temperature-0 setting this measurement used. Turning sampling on would make a different measurement.

8. **Pitfall #8.** No. That bench does not ask for these four lines. This run did not rescore it. The generations on the three held-out questions are the measurement that says whether the answer arrived.
