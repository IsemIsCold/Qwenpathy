# Questions to the organisers

Sent on **2026-09-25** in the official tech-track WeChat group. Fill in the answers as they arrive.

| # | Question | Answer | Date |
|---|---|---|---|
| 1 | What are the scoring weights per field? How is `response_text` judged? How much does latency count? | **Partly answered by the official rules page:** 70 automatic = 25 emotion accuracy + 20 reply quality (BLEU-4 / ROUGE-L / BERTScore vs gold) + 15 profile (average) + 10 latency; plus 30 expert review (integrity, code quality, platform records). **Still open:** is the profile scored by F1 or exact match? How are BLEU/ROUGE/BERTScore combined? | 2026-10-04 (rules page) |
| 2 | `memory_refs`: will the test set include a memory bank? What do the training `mem_` IDs refer to? | | |
| 3 | How do we submit: save a private image, or upload a Docker image? Is the evaluation hardware 2×4090? | | |
| 4 | Can we use external public data, or other open models (e.g. Qwen3)? Is there a model size limit? | **Urgent:** we plan to use Qwen3-8B. | |
| 5 | Is 20卡时 (kǎshí, card-hours) counted per GPU? Does 无卡开机 (start without GPU) use card-hours? | | |

## Follow-up questions to post (Oct 5)

- 请问对基座模型有限制吗？比如参数量上限，或者能否使用 Qwen3 等开源模型？(*Qǐngwèn duì jīzuò móxíng yǒu xiànzhì ma? Bǐrú cānshùliàng shàngxiàn, huòzhě néngfǒu shǐyòng Qwen3 děng kāiyuán móxíng?*) "Are there restrictions on the base model, e.g. a maximum size, or can we use open models like Qwen3?"
- 请问用户画像是按 F1 还是完全匹配计分？(*Qǐngwèn yònghù huàxiàng shì àn F1 háishì wánquán pǐpèi jìfēn?*) "Is the user profile scored by F1 or by exact match?"
- 请问测试集会提供记忆库吗？memory_refs 应该引用什么？(*Qǐngwèn cèshìjí huì tígōng jìyìkù ma? memory_refs yīnggāi yǐnyòng shénme?*) "Will the test set include a memory bank? What should memory_refs refer to?"
- 请问回复质量的 BLEU-4 是按字还是按词（jieba）计算？是语料级还是逐句平均？BERTScore 用的是哪个模型？(*Qǐngwèn huífù zhìliàng de BLEU-4 shì àn zì háishì àn cí (jieba) jìsuàn? Shì yǔliào jí háishì zhú jù píngjūn? BERTScore yòng de shì nǎge móxíng?*) "Is reply BLEU-4 computed per character or per word (jieba)? Over the corpus or averaged per sentence? Which model does BERTScore use?" (On the same predictions, corpus BLEU = 0.08 and sentence-averaged = 0.19, so this matters.)
