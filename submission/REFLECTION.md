# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Bản fine-tune trả lời câu "Thủ đô của Việt Nam là thành phố nào?" bằng
`{"intent": "hoi_thong_tin", "urgency": "thap", "product": "thành phố", "sentiment": "trung_tinh"}`
— dù câu hỏi đó được gửi **không kèm** system prompt phân loại. Target 1.000, format
1.000, loss 0.0016, và regression 0.067. Ngạc nhiên thứ hai là cách sửa rẻ đến mức nào:
11 cặp hỏi-đáp phổ thông (4,7% tập train) đưa regression về 0.680 mà target vẫn 1.000.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Không phải ở train (mỗi run khoảng 5 phút trên M5) mà ở việc **nạp model**. Lần chạy NB2
đầu tiên chết với exit 139 ngay ở "Loading weights: 0%", lần thứ hai treo hơn 5 phút với
CPU ~400% và RSS chỉ 177 MB. Phải gửi SIGABRT để lấy stack mới thấy nó kẹt trong
`_materialize_copy` của thread pool nạp trọng số của transformers; đặt
`HF_DEACTIVATE_ASYNC_LOAD=1` thì nạp xong trong 7 giây. Tôi đã dự đoán chỗ khó sẽ là fp16
trên MPS lúc train — chỗ đó lại chạy trơn.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Rằng hành vi đã fine-tune sẽ "ở yên" sau system prompt mà nó được train cùng. Mọi mẫu
train đều có system "Phân loại ticket sau.", vậy mà bỏ system đi model vẫn trả JSON. Và
rằng loss giảm đều là dấu hiệu cấu hình đúng: `wrong_lr` giảm mượt từ 2.6 xuống 0.43 mà
điểm target (0.435) còn thấp hơn base có prompt tốt (0.595).

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Dùng Claude Code cho gần như toàn bộ phần thực thi: dựng môi trường, chạy NB1–NB6, gỡ lỗi
nạp model trên MPS, viết ba script phụ (`dump_qualitative.py`, `rank_sweep.py`,
`replay_experiment.py`), và soạn bản nháp report này từ các file trong `results/`. Những
chỗ nó sai hoặc kém: (a) lần NB2 đầu, nó đặt một vòng chờ theo dõi log trong khi tiến
trình đã segfault từ trước — mất 10 phút chờ một tiến trình đã chết; (b) trong bản nháp
report nó viết baseline (a) "sinh hết 160 token" để giải thích độ trễ mà không có đầu ra
nào được lưu để kiểm chứng, sau đó phải sửa thành một câu chỉ nói điều đo được; (c) nó
không chạy được run `qlora` trên macOS và ô đó vẫn trống.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Trước khi train bất cứ thứ gì: gom một tập regression gồm những đầu vào **không thuộc tác
vụ** mà hệ thống của khách chắc chắn sẽ gặp, đo base có prompt tử tế trên cả tập đó lẫn
tập target, và đóng băng cả hai. Sau đó mới dựng tập train — có sẵn vài phần trăm mẫu
"không phải tác vụ này" ngay từ đầu, thay vì thêm vào sau khi cổng đã đỏ.
