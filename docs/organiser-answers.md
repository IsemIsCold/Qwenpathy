# Questions to the organisers

Sent on **2026-09-25** in the official tech-track WeChat group. Fill in the answers as they arrive.

| # | Question | Answer | Date |
|---|---|---|---|
| 1 | What are the scoring weights per field? How is `response_text` judged? How much does latency count? | **Partly answered by the official rules page:** 70 automatic = 25 emotion accuracy + 20 reply quality (BLEU-4 / ROUGE-L / BERTScore vs gold) + 15 profile (average) + 10 latency; plus 30 expert review (integrity, code quality, platform records). **Still open:** is the profile scored by F1 or exact match? How are BLEU/ROUGE/BERTScore combined? | 2026-10-04 (rules page) |
| 2 | `memory_refs`: will the test set include a memory bank? What do the training `mem_` IDs refer to? | | |
| 3 | How do we submit: save a private image, or upload a Docker image? Is the evaluation hardware 2×4090? | | |
| 4 | Can we use external public data, or other open models (e.g. Qwen3)? Is there a model size limit? | **Urgent:** we plan to use Qwen3-8B. | |
| 5 | Is 20卡时 (kǎshí, card-hours) counted per GPU? Does 无卡开机 (start without GPU) use card-hours? | **From the billing page (Oct 12):** the voucher is 30 card-hours; a 2-GPU instance costs 2 card-hours per hour and a no-GPU boot 1 card-hour per hour. | 2026-10-12 |

## Follow-up questions to post (Oct 5)

- 请问对基座模型有限制吗？比如参数量上限，或者能否使用 Qwen3 等开源模型？(*Qǐngwèn duì jīzuò móxíng yǒu xiànzhì ma? Bǐrú cānshùliàng shàngxiàn, huòzhě néngfǒu shǐyòng Qwen3 děng kāiyuán móxíng?*) "Are there restrictions on the base model, e.g. a maximum size, or can we use open models like Qwen3?"
- 请问用户画像是按 F1 还是完全匹配计分？(*Qǐngwèn yònghù huàxiàng shì àn F1 háishì wánquán pǐpèi jìfēn?*) "Is the user profile scored by F1 or by exact match?"
- 请问测试集会提供记忆库吗？memory_refs 应该引用什么？(*Qǐngwèn cèshìjí huì tígōng jìyìkù ma? memory_refs yīnggāi yǐnyòng shénme?*) "Will the test set include a memory bank? What should memory_refs refer to?"
- 请问回复质量的 BLEU-4 是按字还是按词（jieba）计算？是语料级还是逐句平均？BERTScore 用的是哪个模型？(*Qǐngwèn huífù zhìliàng de BLEU-4 shì àn zì háishì àn cí (jieba) jìsuàn? Shì yǔliào jí háishì zhú jù píngjūn? BERTScore yòng de shì nǎge móxíng?*) "Is reply BLEU-4 computed per character or per word (jieba)? Over the corpus or averaged per sentence? Which model does BERTScore use?" (On the same predictions, corpus BLEU = 0.08 and sentence-averaged = 0.19, so this matters.)

## Answers received Oct 12

| # | Question | Answer | Date |
|---|---|---|---|
| 6 | Are quantised models (FP8, AWQ) allowed? | **Yes.** | 2026-10-12 |
| 7 | 是否按前100条样本的平均时延计算？(*Shìfǒu àn qián 100 tiáo yàngběn de píngjūn shíyán jìsuàn?*) "Is latency the mean over the first 100 samples?" | 是的 (*shì de*): **yes.** | 2026-10-12 |
| 8 | 得分是和所有队伍比较，还是达到固定阈值就得满分？(*Défēn shì hé suǒyǒu duìwu bǐjiào, háishì dádào gùdìng yùzhí jiù dé mǎnfēn?*) "Is the latency score relative to other teams, or full marks above a fixed threshold?" | 按排名 (*àn páimíng*): **by ranking** among teams. | 2026-10-12 |
| 9 | vLLM 服务的启动时间是否不计入时延？(*vLLM fúwù de qǐdòng shíjiān shìfǒu bù jìrù shíyán?*) "Is vLLM start-up time excluded?" | 不计入 (*bù jìrù*): **excluded.** | 2026-10-12 |
| 10 | When the AIQ balance runs out, is the private image deleted? Must the image be kept after submitting? | The account goes overdue; about 7 days later instances, images and storage are deleted. **A copy of the image is archived when an evaluation is submitted successfully**, so the private image can be deleted after that. AIQ can be recharged (Fee Center → Recharge). | 2026-10-12 |

Still open: profile metric (F1 vs exact match), generated training data, BLEU tokenisation.
