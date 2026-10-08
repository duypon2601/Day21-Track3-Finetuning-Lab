# Lab 21 — Evaluation Report

**Họ tên**: (cần điền)  **MSSV**: (cần điền)  **Ngày**: 2026-10-08
**Tier**: `LAPTOP`  **Base model**: `Qwen/Qwen3.5-2B`  **Phần cứng thực tế**: Apple M5, 24 GB bộ nhớ hợp nhất, backend MPS (không có CUDA), fp16

> Mọi con số dưới đây lấy từ file trong `results/`; tên file ghi ngay cạnh con số.
> Stack: torch 2.14.1 · transformers 5.19.0 · trl 1.14.2 · peft 0.21.2.

**Tóm tắt một đoạn.** Bản fine-tune `correct` đạt target **1.000** so với **0.595** của
base đã prompt tử tế, nhưng điểm regression rơi từ **0.600** xuống **0.067**: cổng hồi
quy **FAILED**, và theo đúng cổng đó tôi **không** deploy adapter này. Nguyên nhân đọc
được ngay trên đầu ra thô: model trả JSON triage cho cả câu hỏi kiến thức phổ thông. Một
thí nghiệm tiếp theo trộn 11 mẫu hỏi-đáp phổ thông (4,7%) vào tập train giữ nguyên target
1.000 và đưa regression về 0.680 — cổng PASSED cho adapter thứ hai đó. Run `qlora` **chưa
chạy được** trên máy này (xem mục 4.3).

---

## 1. Setup — lựa chọn và lý do

| | |
|---|---|
| Base model | `Qwen/Qwen3.5-2B` (mặc định của tier `LAPTOP`) |
| Dataset | corpus mặc định: 250 ticket CSKH tiếng Việt → JSON triage 4 trường; không sửa (`make verify`: eval sets unmodified) |
| Train / val | 225 / 25 (seed 42) |
| Eval | 50 ticket target + 15 câu regression, dùng đủ, không đặt `EVAL_LIMIT` |
| `max_length` | 1024 (của tier) — p95 đo được là **98**, max **101** *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epoch → **58** optimizer step (batch 1 × grad_accum 8 = batch hiệu dụng 8) |
| LoRA `correct` | text-linear (12 loại module), r=16, α=32, LR 1e-4, 16 819 200 tham số huấn luyện |

**Vì sao model này.** Máy tôi là Mac M-series, không có GPU NVIDIA. `HARDWARE-GUIDE.md`
xếp macOS vào tier `CPU` (chỉ NB1), nhưng 24 GB bộ nhớ hợp nhất đủ để train LoRA trên
MPS với model 2B ở fp16 (driver Metal giữ khoảng 6,9 GB sau khi train xong). Tôi chọn 2B
thay vì 4B của tier T4 để còn biên bộ nhớ an toàn. Cùng một base model cho mốc lẫn bản
fine-tune (`results/baselines_frozen.json` và `results/runs.csv` đều ghi
`Qwen/Qwen3.5-2B`).

**Vì sao giữ dataset mặc định.** Đây là lượt chạy đầu; tôi muốn quen khung đo trước khi
đổi corpus. Hệ quả cần nói rõ: corpus này sinh từ template, nên nhãn là một quy ước cố
định của bộ sinh dữ liệu (ví dụ "Hoàn lại" → `doi_tra`, câu kết "Cho tôi hỏi" →
`trung_tinh`). Điều đó ảnh hưởng trực tiếp tới cách đọc con số 1.000 ở mục 5.

**`max_length` lệch khỏi p95.** NB1 gợi ý 256, tier đặt 1024. Tôi giữ 1024 vì nó không
có tác dụng gì ở đây: chuỗi dài nhất là 101 token nên không mẫu nào bị cắt ở cả hai giá
trị, và batch = 1 nên không có padding nào bị tính thừa. Nếu tăng batch lên ≥ 2 thì nên
hạ về 256.

**Template có giữ khối `<think>` không?** **Có** — `results/template_check.json`:
`open_tag_present: true`, `body_present: true`, verdict "reasoning preserved". Kèm một
chi tiết quan trọng hơn cho corpus này: câu trả lời huấn luyện là JSON trần, và template
Qwen3.5 tự đóng một khối `<think>\n\n</think>` rỗng **bên trong generation prompt**, tức
là ở phía bị mask. Model vì thế học "sau khối think rỗng là JSON" và
`valid_trace_rate = 0.0` (`results/verdict.json`). Con số đó không chứng minh
reasoning-trace collapse — tôi không train trên trace nào và eval chạy với
`enable_thinking=False` — nó chỉ cho biết không có trace nào để đo.

**Hai thay đổi mã tôi phải làm để chạy được trên MPS** (đều trong
`src/labkit/generate.py`, không đụng tới phép đo):

1. `load_base()` đặt `HF_DEACTIVATE_ASYNC_LOAD=1` khi thiết bị là MPS. transformers 5.19
   nạp trọng số bằng 4 luồng; trên MPS lần đầu nó segfault (exit 139), lần sau treo ở
   "Loading weights: 0%" với CPU ~400%. Nạp tuần tự mất khoảng 7 giây.
2. `peak_vram_gb()` trả về `torch.mps.driver_allocated_memory()` trên MPS. MPS không có
   bộ đếm đỉnh, nên cột `peak_vram_gb` trong `runs.csv` là **bộ nhớ driver đang giữ ở
   cuối run**, tức cận dưới của đỉnh thật, không phải đỉnh.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | **0.3936** (37 / 94 token) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

*(results/mask_proof.json)* Đoạn được tính loss, giải mã ngược từ nhãn:

```
{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đoạn bị mask (không tính loss) kết thúc bằng
`<|im_start|>assistant\n<think>\n\n</think>\n\n` — system prompt, ticket và cả khối think
rỗng đều nằm ngoài loss; `<|im_end|>` nằm trong loss nên model học được chỗ dừng. NB3 xác
nhận lại trên toàn tập train trước khi train (`assert 0 < sup < tot`).

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

`results/baselines_frozen.json` được ghi trước khi NB3 chạy; (c) lấy từ
`results/verdict.json`.

| Run | target | regression | format | latency (ms/mẫu) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.600 | 0.000 | 2807.3 |
| (b) base + optimized prompt | 0.595 | 0.600 | 1.000 | 991.0 |
| (c) LoRA fine-tune `correct` | **1.000** | **0.067** | 1.000 | 867.0 |

**(b) có thật sự mạnh hơn (a) không?** Có: 0.000 → 0.595, format 0 → 1. Với prompt ngây
thơ model không trả JSON nào (format 0.000) và chậm gần gấp ba, phù hợp với việc nó viết
văn xuôi dài thay vì một object ngắn — tôi không lưu đầu ra thô của (a) nên không trích
được ví dụ.

**Tôi không sửa `OPTIMIZED_PROMPT`** (`make verify`: SHA `719e74d3b6232053` khớp bản
gốc). Nhưng phải nói thẳng (b) yếu ở đâu, vì đó là thứ fine-tune "thắng". Tính lại theo
từng trường từ `results/qualitative_detail.json`: product 0.94, intent 0.58, urgency
0.54, **sentiment 0.32**. (b) trả `tich_cuc` cho 49/50 ticket trong khi nhãn chia đều
17/17/16 — few-shot duy nhất trong prompt không đủ để nó hiểu quy ước sentiment của
corpus. Một prompt (b) có thêm vài ví dụ phủ ba lớp sentiment gần như chắc chắn sẽ cao
hơn 0.595; tôi không thử, vì mốc đã đóng băng trước khi thấy kết quả train.

---

## 4. Giải phẫu cấu hình sai (NB4, chấm ở NB5 §4)

Cả ba run đã chạy dùng cùng 58 step, cùng seed, cùng mask, cùng dữ liệu. Mỗi run đổi
đúng một biến so với `correct`: `attn_only` đổi **vị trí gắn adapter** (rank được nâng
để giữ nguyên ngân sách tham số), `wrong_lr` đổi **learning rate**, `qlora` đổi **độ
chính xác của base**.

| Run | vị trí | r | trainable | LR | train loss TB (NB4) | **target (NB5 §4)** | format | s | bộ nhớ GB* |
|---|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 16 819 200 | 1e-4 | 0.4500 | **1.000** | 1.000 | 295.9 | 6.93 |
| `attn_only` | q,v | 322 *(matched)* | 16 816 128 | 1e-4 | 0.4414 | **0.975** | 1.000 | 244.3 | 6.61 |
| `wrong_lr` | text-linear | 16 | 16 819 200 | 1e-5 | 1.1670 | **0.435** | 0.995 | 290.4 | 6.92 |
| `qlora` | text-linear | 16 | — | 1e-4 | *chưa chạy* | *chưa chạy* | — | — | — |

*(results/runs.csv, results/autopsy.json)* \*bộ nhớ driver MPS cuối run, xem mục 1.
`final_loss` trong `runs.csv` là loss **trung bình cả run**, không phải loss ở step cuối;
loss log cuối cùng là 0.0016 (`correct`), 0.0052 (`attn_only`), 0.4296 (`wrong_lr`).
`attn_only` lệch ngân sách 0,02% so với `correct` (ngưỡng 5%).

**4.1 — Vị trí so với rank.** Trên target `attn_only` đạt 0.975, `correct` đạt 1.000:
chênh 0.025, tức 5 trường trong 200 trường được chấm, với một seed duy nhất. Tôi coi đây
là **hoà trong sai số**, nghiêng nhẹ về `correct`, chứ không phải bằng chứng rằng vị trí
là đòn bẩy trên bài này. Thứ tự theo train loss thì **ngược lại**: `attn_only` có loss
trung bình thấp hơn (0.4414 so với 0.4500) nhưng điểm target thấp hơn — đúng tình huống
NB4 cảnh báo, dù biên độ nhỏ. Điều rút ra được là: với cùng ~16,8 M tham số, dồn hết vào
q,v ở rank 322 không mua thêm gì so với trải ra 12 loại module ở rank 16. Quét rank có
kiểm soát (phụ lục B4) nói cùng một ý từ phía kia: tăng ngân sách gấp 4 (r=64) vẫn là
1.000, giảm một nửa (r=8) là 0.975. Bài toán này quá hẹp để bộc lộ sự khác biệt mà deck
§11.2 mô tả; muốn thấy cần một tác vụ khó hơn hoặc nhiều seed.

**4.2 — `wrong_lr`.** Đường loss không phẳng mà **giảm chậm**: 2.595 → 0.430 sau 58
step, trong khi `correct` xuống dưới 0.06 từ khoảng step 25 và kết thúc ở 0.0016;
`mean_token_accuracy` dừng ở 0.93 so với 1.00. Nếu chỉ nhìn đường loss này mà không biết
LR, tôi sẽ kết luận "model đang học tốt, chỉ cần thêm epoch" hoặc "cần rank lớn hơn" —
cả hai đều sai nút vặn. Trên tác vụ nó đạt 0.435, **thấp hơn cả baseline (b) 0.595** dù
format đã gần hoàn hảo (0.995): adapter học được hình dạng JSON trước, còn quy ước nhãn
thì chưa. Đây là khác biệt lớn nhất trong cả lab: đổi một con số LR làm mất 0.565 điểm
target, trong khi đổi vị trí hay rank chỉ xê dịch 0.025.

**4.3 — `qlora`: chưa đo được.** Run này cần `bitsandbytes`, chỉ có cho Linux/CUDA;
trên macOS `import bitsandbytes` báo `ModuleNotFoundError`, và `requirements.txt` cũng
loại nó khỏi macOS bằng platform marker. Vì vậy `runs.csv` không có dòng `qlora`,
`autopsy.json` chỉ có ba run, và `make verify` báo WARN ở mục "NB4 contrast runs". Tôi
**không có số đo nào** để ủng hộ hay bác bỏ khuyến nghị "không dùng QLoRA cho dòng model
này", và không trả lời câu hỏi tiết kiệm VRAM bằng suy đoán. Để hoàn tất: chạy
`ONLY=qlora make nb4` rồi `make nb5` trên Colab T4 — nhưng khi đó cả bốn run phải chạy
lại trên cùng máy và cùng base model thì mới so được với nhau.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **FAILED**
`target Δ = +0.405` · `regression Δ = −0.533` · `valid_trace_rate = 0.00`
*(results/verdict.json)*

Lý do máy ghi: "general capability regressed by 0.533 (tolerance 0.020)". Target thì
vượt (b) rất xa, nên đây không phải ca "prompt engineering đã thắng" mà là ca thứ hai
trong thứ tự chẩn đoán của NB5: quên thảm hoạ.

Đầu ra thô cho thấy cơ chế cụ thể hơn chữ "quên" (`results/qualitative_detail.json`).
Trong 15 câu regression, bản fine-tune trả **JSON triage cho 13 câu**: hỏi thủ đô Việt
Nam thì nhận `{"intent": "hoi_thong_tin", ..., "product": "thành phố", ...}`. Model không
mất kiến thức; nó đã học rằng *mọi* lượt assistant đều là một object 4 khoá. Điều đáng
chú ý là các câu regression được hỏi **không có system prompt**, trong khi mọi mẫu train
đều có system "Phân loại ticket sau." — tức adapter không gắn hành vi vào system prompt
như tôi tưởng, mà ghi đè luôn phân phối đầu ra mặc định. Lý do hợp lý: 225/225 mẫu train
có cùng một dạng đáp án, không có mẫu nào cho model thấy trường hợp "không phải ticket
thì trả lời bình thường", và loss xuống 0.0016 nghĩa là nó đã khớp gần như tuyệt đối với
phân phối đó. Hai câu thoát được (dịch Anh–Việt, tên cũ của TP.HCM) là hai câu có dạng
khác hẳn một ticket.

**Kiểm tra chẩn đoán bằng thí nghiệm** (`scripts/replay_experiment.py`,
`results/replay_experiment.json`). Nếu nguyên nhân là thiếu mẫu "không phải ticket" thì
thêm vài mẫu như vậy phải sửa được. Tôi train thêm adapter `correct_replay`: y hệt
`correct` (cùng 58 step, LR, rank, seed) cộng 11 cặp hỏi-đáp phổ thông không có system
prompt (`data/replay_general.jsonl`, 4,66% tập train, script assert không câu nào trùng
với tập regression).

| Adapter | target | regression | format | cổng so với (b) |
|---|---|---|---|---|
| `correct` (được chấm) | 1.000 | 0.067 | 1.000 | FAILED |
| `correct_replay` | 1.000 | 0.680 | 1.000 | PASSED (Δtarget +0.405, Δregression +0.080) |

Chẩn đoán đứng vững: 11 mẫu đủ để lấy lại hành vi trả lời thường mà không mất điểm
target. Ba giới hạn phải ghi kèm. Một, phán quyết chính thức của lab vẫn là FAILED cho
`correct`; `correct_replay` được train **sau khi** tôi đã thấy kết quả, nên nó là kiểm
chứng giả thuyết chứ không phải kết quả tiền đăng ký. Hai, 0.680 > 0.600 **không** có
nghĩa adapter giỏi kiến thức hơn base: nó trả lời ngắn hơn nên khớp keyword tốt hơn (base
viết "1.000 mét" và bị chấm 0 cho keyword "1000"; adapter viết "1000 mét"). Ba, vẫn còn
1/15 câu (câu tục ngữ "Có công mài sắt") bị trả JSON, và tập regression chỉ có 15 câu.

---

## 6. Định tính — có cả ca THUA

Vì (c) đạt 1.000 trên cả 50 ticket nên không có ca thua nào trong nhóm target; các ca
thua đều nằm ở nhóm regression, và tôi đưa chúng vào bảng thay vì chỉ liệt kê ca thắng.
Nguồn: `results/qualitative_detail.json` (script sinh lại bằng greedy decode và assert
điểm trung bình khớp với `baselines_frozen.json` / `verdict.json`).

| # | Đầu vào (rút gọn) | Đúng | (b) base + prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | target #32: "…máy xay sinh tố mã đơn OD906403. Hoàn lại. Mong shop phản hồi. Shop hỗ trợ tốt." | doi_tra / trung_binh / máy xay sinh tố / tich_cuc | hoan_tien / thap / **OD906403** / tieu_cuc → 0.00 | đúng cả 4 → 1.00 | ✅ FT thắng. (b) lấy mã đơn làm product |
| 2 | target #30: "…đèn bàn LED mã đơn OD936122. Hoàn lại. Sớm nhé. Lần cuối mua ở đây." | doi_tra / trung_binh / đèn bàn LED / tieu_cuc | hoan_tien / thap / đèn bàn LED / tich_cuc → 0.25 | đúng cả 4 → 1.00 | ✅ FT thắng. "Hoàn lại" = `doi_tra` là quy ước của corpus |
| 3 | target #21: "…máy xay sinh tố mã đơn OD593253. Bảo hành bao lâu. Tôi cần trước ngày mai. Quá tệ." | hoi_thong_tin / cao / máy xay sinh tố / tieu_cuc | hoi_thong_tin / cao / **OD593253** / **tich_cuc** → 0.50 | đúng cả 4 → 1.00 | ✅ FT thắng. (b) gọi "Quá tệ" là tích cực |
| 4 | regression #0: "Thủ đô của Việt Nam là thành phố nào?" | Hà Nội | "Thủ đô của Việt Nam hiện nay là thành phố **Hà Nội**…" → 1.00 | `{"intent": "hoi_thong_tin", "urgency": "thap", "product": "thành phố", "sentiment": "trung_tinh"}` → 0.00 | ❌ **FT thua** |
| 5 | regression #8: "Ai là tác giả của Truyện Kiều?" | Nguyễn Du | "…được viết bởi nhà thơ **Nguyễn Du**…" → 1.00 | `{"intent": "hoi_thong_tin", …, "product": "Truyện Kiều", "location": "Việt Nam", "search_qu…` → 0.00 | ❌ **FT thua**, còn bịa thêm khoá |
| 6 | regression #3: "Viết một câu chúc mừng sinh nhật bằng tiếng Việt." | có "sinh nhật" | "Chúc mừng sinh nhật bạn! Hy vọng…" → 1.00 | `{"intent": "sinh_nhanh", "urgency": "trung_binh", "product": "trung_tinh", "sentiment": "tich_cuc"}` → 0.00 | ❌ **FT thua**, bịa cả giá trị enum |
| 7 | regression #13: "Thành phố Hồ Chí Minh trước đây có tên là gì?" | Sài Gòn | "…có tên là **Quảng Đông**…" → 0.00 | "…có tên là **Quận Thành phố Hồ Chí Minh**…" → 0.00 | hoà — cả hai cùng sai; base 2B vốn đã bịa ở đây |

**Mẫu chung ở các ca FT thua.** Mọi ca thua là câu hỏi ngắn, một câu, không có system
prompt — bề ngoài giống một ticket. Model ép chúng vào schema, và khi không có giá trị
hợp lệ thì bịa: enum mới (`"hoi"`, `"sinh_nhanh"`), khoá mới (`"location"`,
`"search_query"`). Ca #5 và #6 cho thấy một rủi ro mà điểm format = 1.000 trên tập target
che mất: ngoài phân phối, adapter không giữ được ngay cả schema của chính nó.

**Mẫu chung ở các ca FT thắng.** Đều là chỗ nhãn theo quy ước của bộ sinh dữ liệu chứ
không theo nghĩa tự nhiên: sentiment quyết định bởi câu kết, "Hoàn lại" là `doi_tra`,
product là danh từ đứng trước mã đơn. (b) đọc ticket theo nghĩa thường nên sai có hệ
thống.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Tôi không deploy adapter `correct`. Lý do không phải vì nó phân loại kém —
nó đúng 200/200 trường trên tập target — mà vì cái giá của điểm số đó là một model trả
lời "thủ đô Việt Nam" bằng một object JSON. Trong một hệ CSKH thật, ticket không phải
lúc nào cũng đi qua đúng một đường với đúng một system prompt; một adapter biến mọi đầu
vào thành triage sẽ hỏng âm thầm ở chỗ đầu tiên nó bị dùng lệch mục đích, và ca #5, #6 ở
mục 6 cho thấy khi đó nó còn phá cả schema. Cổng hồi quy bắt được đúng điều này; nếu tôi
chỉ báo cáo target và format, bảng kết quả sẽ là một chiến thắng tuyệt đối.

Con số 1.000 cũng cần đọc dè dặt vì một lý do thứ hai: corpus sinh từ template, tập eval
sinh từ cùng template, nên 1.000 đo việc adapter học thuộc quy ước của bộ sinh dữ liệu
chứ chưa đo năng lực phân loại ticket do người thật viết. Phần lớn khoảng cách 0.405 so
với (b) đến từ sentiment (0.32) — thứ mà vài ví dụ few-shot trong prompt nhiều khả năng
đã thu hẹp đáng kể. Nên câu hỏi "có cần fine-tune không" trên bài này vẫn chưa được trả
lời dứt khoát; cái tôi trả lời được là "adapter này, như đã train, thì chưa dùng được".

Về đòn bẩy, xếp theo biên độ đo được trên target: **learning rate** (0.565 điểm giữa
1e-4 và 1e-5) ≫ **vị trí adapter** (0.025, trong sai số) ≈ **rank** (0.025 giữa r=8 và
r=16, 0 giữa r=16 và r=64). Nhưng đòn bẩy quyết định phán quyết lại không nằm trong ba
nút đó: nó là **thành phần dữ liệu**. Thêm 11 mẫu — 4,7% tập train, không đổi một siêu
tham số nào — đưa regression từ 0.067 lên 0.680 và lật cổng từ FAILED sang PASSED. Nếu
phải đưa một bản lên production, đó sẽ là `correct_replay`, sau khi chấm lại trên một tập
regression lớn hơn 15 câu, nhiều seed, và ticket thật.

**Ba điều tôi học được.**

1. **Loss 0.0016 và target 1.000 là dấu hiệu cần nghi ngờ, không phải để ăn mừng.**
   Adapter khớp hoàn hảo với 225 mẫu cùng một dạng đáp án thì cũng đồng thời học rằng
   không còn dạng đáp án nào khác. Tôi chỉ thấy điều đó khi in đầu ra của 15 câu
   regression ra đọc; con số 0.067 nói có vấn đề, còn dòng
   `{"intent": "hoi_thong_tin", "product": "thành phố"}` mới nói vấn đề là gì.
2. **System prompt không "khoanh vùng" được hành vi đã fine-tune.** Tôi đã mặc định rằng
   vì mọi mẫu train đều có system "Phân loại ticket sau." nên adapter chỉ kích hoạt khi
   có câu đó. Sai: câu hỏi không có system prompt vẫn bị trả JSON. Muốn model phân biệt
   thì phải cho nó thấy ví dụ của phía bên kia trong dữ liệu.
3. **Đường loss của `wrong_lr` trông khoẻ mạnh.** Nó giảm đều từ 2.6 xuống 0.43, không
   phẳng, không nan. Không có gì trên đồ thị báo rằng LR sai 10 lần; chỉ điểm target
   0.435 — thấp hơn cả base có prompt tốt — mới cho thấy. Kèm theo đó, `attn_only` có
   train loss thấp hơn `correct` nhưng target thấp hơn: xếp hạng bằng loss cho ra thứ tự
   sai ở cả hai chỗ.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** (1) chạy run `qlora` cùng ba run kia trên Colab T4
để lấp ô còn trống ở mục 4; (2) lặp `correct` và `attn_only` với 3 seed để biết 0.025 là
tín hiệu hay nhiễu; (3) viết một prompt (b) mạnh hơn với ví dụ phủ đủ ba lớp sentiment —
đo trước khi train lại — để biết khoảng cách 0.405 còn lại bao nhiêu; (4) quét tỉ lệ
replay 1% / 2% / 5% để tìm mức nhỏ nhất còn giữ được regression.

---

## Phụ lục — thưởng đã làm

- [x] **B1 NB6 merge + hot-swap.** `results/merge_check.json`: trước merge 1.0000, sau
  merge 1.0000, Δ = 0.0000 (ngưỡng 0.01, n = 50). Hot-swap hai adapter `correct` và
  `attn_only` trên cùng một base đang nạp, cả hai trả cùng JSON đúng cho ticket đầu tiên.
  Merge cho overhead suy luận bằng 0 nhưng mất khả năng tháo adapter ra: với adapter
  `correct` điều đó nghĩa là hành vi "trả JSON cho mọi thứ" bị đúc luôn vào một bản
  3,5 GB, thay vì nằm trong một file 64 MB có thể bỏ đi.
- [ ] B2 dataset miền riêng — không làm.
- [ ] B3 reasoning-trace collapse — không làm. Trên corpus này `response-only` cho mask
  giống hệt `assistant-only` (đáp án không chứa trace), nên chạy hai lần sẽ ra hai kết
  quả như nhau và không chứng minh được gì.
- [x] **B4 quét rank có kiểm soát** (`scripts/rank_sweep.py`, `results/rank_sweep.json`).
  Cố định text-linear, LR 1e-4, 58 step, seed 42; chỉ đổi r (α = 2r).

  | r | trainable | train loss TB | target | format |
  |---|---|---|---|---|
  | 8 | 8 409 600 | 0.5194 | 0.975 | 1.000 |
  | 16 (`correct`) | 16 819 200 | 0.4500 | 1.000 | 1.000 |
  | 64 | 67 276 800 | 0.4081 | 1.000 | 1.000 |

  Rank không phải đòn bẩy ở đây: gấp 8 lần tham số (r=8 → r=64) đổi 0.025 điểm target,
  và từ r=16 trở lên là 0. Train loss thì giảm đều theo rank (0.519 → 0.450 → 0.408), nên
  ai xếp hạng bằng loss sẽ kết luận r=64 tốt hơn r=16 trong khi trên tác vụ chúng bằng
  nhau. 225 mẫu sinh từ template không chứa đủ thông tin để dùng hết r=16, chưa nói r=64.
  Xếp hạng ba nút theo biên độ target: LR (0.565) ≫ vị trí (0.025) ≈ rank (0.025). Một
  seed, n = 50 — hai mức 0.025 đó không phân biệt được với nhiễu.
- [ ] B5 HuggingFace Hub — không làm.

**Ngoài rubric:** thí nghiệm replay ở mục 5 (`scripts/replay_experiment.py`,
`data/replay_general.jsonl`, `results/replay_experiment.json`) và script lưu đầu ra thô
(`scripts/dump_qualitative.py`, `results/qualitative_detail.json`).
