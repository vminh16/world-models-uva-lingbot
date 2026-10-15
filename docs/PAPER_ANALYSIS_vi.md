# Phân tích hai paper: UVA và LingBot-VA

Tài liệu cho người đọc repo muốn hiểu sâu hai paper. Nó giữ những phần phân tích không đủ chỗ trong technical report
(bản rút gọn: [`report/PART2_vi.md`](../report/PART2_vi.md)) và đối chiếu từng claim của paper với kết quả tái lập trong
repo này.

- **UVA:** Shuang Li, Yihuai Gao, Dorsa Sadigh, Shuran Song. *Unified Video Action Model*. RSS 2025.
- **LingBot-VA:** Lin Li, Qihang Zhang, Yiming Luo, et al. *Causal World Modeling for Robot Control*. RSS 2026.

Số liệu của paper lấy từ bảng và phần văn bản của paper. Số liệu "đo được" là của repo này (mã EXP-xx, xem
[`EXPERIMENTS.md`](EXPERIMENTS.md)).

---

## 1. Bối cảnh: hai nhánh trước đó và hạn chế của chúng

Câu hỏi chung của cả hai paper: **làm sao kết hợp khả năng dự đoán động lực học thị giác của model sinh video với việc suy
ra action, mà vẫn đủ chính xác và đủ nhanh để điều khiển robot.**

**Nhánh 1: dự đoán action trực tiếp** (policy phản xạ: Diffusion Policy, OpenVLA, π0).
- Ánh xạ thẳng từ ảnh và câu lệnh sang action, không mô hình hoá động lực học của môi trường.
- Dễ học thuộc quỹ đạo, hoặc overfit vào lịch sử action, nên tổng quát kém khi cảnh thay đổi hay có nhiễu thị giác.
- Chỉ nhìn một hoặc vài frame gần nhất (hoặc một action chunk), nên thiếu trí nhớ cho task dài và cho môi trường quan sát
  một phần.

**Nhánh 2: sinh video trước, suy ra action sau** (UniPi, Dreamitate và các model video hai chiều).
- Sinh frame video độ phân giải cao, rồi trích action từ video đã sinh; hoặc dùng attention hai chiều trên cả chuỗi.
- Attention hai chiều không tôn trọng thứ tự nhân quả của thời gian vật lý, nên video dễ thiếu nhất quán.
- Chậm: sinh pixel tốn kém, không đạt tần số điều khiển cần cho robot.
- Lỗi của video lan sang action.

UVA giải quyết bài toán bằng cách **tách** phần sinh video khỏi phần điều khiển. LingBot-VA giải quyết bằng cách làm cho
phần sinh video **nhân quả và đủ rẻ** để chạy ngay trong vòng điều khiển.

---

## 2. UVA — Unified Video Action Model

### 2.1 Ý tưởng chính

1. **Biểu diễn ẩn chung cho video và action.** Lịch sử ảnh, lịch sử action và token của frame tương lai (bị che ngẫu nhiên
   theo kiểu Masked Autoencoder) được ghép theo kênh và theo thời gian, rồi đưa qua một transformer (khởi tạo từ MAR,
   khoảng 0.5B). Biểu diễn ẩn này là cầu nối giữa miền ảnh và miền điều khiển.
2. **Hai đầu diffusion tách rời.** Thay vì khử nhiễu trên cả backbone lớn, UVA gắn hai đầu diffusion nhỏ, cùng đọc biểu
   diễn ẩn: một đầu ra video, một đầu ra action. Khi train, tối ưu đồng thời cả hai loss buộc biểu diễn phải chứa động lực
   học của cảnh. Khi điều khiển, **đầu video bị bỏ qua hoàn toàn**; chỉ đầu action chạy. Trong code, đây là chế độ
   `policy_model` của `predict_action`.
3. **Huấn luyện với mask để có nhiều chức năng.** Bằng cách che chọn lọc video hoặc action (quá khứ hay tương lai) bằng
   một mask token học được, cùng một model đóng được 5 vai trò mà không cần fine-tune lại: policy, sinh video, forward
   dynamics, inverse dynamics, và policy kết hợp planner.

### 2.2 Đóng góp cho robot learning

- **Tách lợi ích khỏi chi phí của video.** Đây là đóng góp quan trọng nhất. Video làm biểu diễn tốt hơn lúc train mà
  không làm chậm lúc chạy. UVA nhanh ngang Diffusion Policy (0.23 s so với 0.36–0.50 s trong mô phỏng), trong khi các
  phương pháp sinh video trước như UniPi mất 24 s.
- **Giám sát bằng video là một cách chống overfit.** UVA giữ hiệu năng khi lịch sử dài ra (tới 16 bước), trong khi DP-C
  suy giảm mạnh. Bỏ phần video (UVA-action) làm PushT giảm 0.98 → 0.45 và PushT-M 0.88 → 0.46.
- **Một model, nhiều vai trò.** Forward dynamics có thể dẫn đường cho một policy khác. Inverse dynamics có thể gán nhãn
  action cho video. Đây là nền cho việc dùng dữ liệu không đầy đủ.

### 2.3 Kết quả chính trong paper

| Thí nghiệm | Kết quả |
|---|---|
| Mô phỏng (Table I) | PushT 0.98, PushT-M 0.88 (+20% so với baseline tốt nhất), LIBERO-10 0.90 (hơn π0 0.85, π0-FAST 0.60, OpenVLA 0.54) |
| Ablation UVA-action | PushT 0.45, ToolHang 0.62, PushT-M 0.46, **LIBERO-10 0.86** |
| Nhiễu thị giác trên PushT (Table III) | Bình thường 0.98; đổi màu nền 0.35, thêm vật nền 0.31, đổi màu đích 0.64 (DP-C: 0.91 → 0.12 / 0.21 / 0.17) |
| Robot thật UMI (Table II) | Đơn task: Cup 0.85 (DP-UMI 0.95). Đa task, ngoài phân phối: Cup 0.65 (0.50), Towel 0.70 (0.70), Mouse 0.80 (0.40). 95 ms so với 70 ms |
| Sinh video (FVD, Table IV) | LIBERO-10 51.10, CupArrange 29.72 (UniPi: 56.55, 71.37) |
| Forward dynamics dẫn đường cho DP-C (Table V) | 38% → 60% (simulator thật làm planner: 75%) |
| Inverse dynamics trên UMI (Table VI) | 0.75 cm, 1.11° (UniPi 1.92 cm, 2.21°; SLAM 0.41 cm, 0.30°) |

### 2.4 Hạn chế tác giả tự nêu

- Chưa dùng được lượng lớn **video không có action**, nên trên robot thật đôi khi chỉ ngang DP-UMI.
- Attention trong transformer chiếm khoảng một nửa thời gian suy luận (trên robot thật), khiến UVA chậm hơn DP-UMI một
  chút. Tác giả đề xuất dùng Flash Attention.
- Lịch sử dài làm UVA kém linh hoạt hơn DP-UMI ở các pha phục hồi lỗi tức thời trong task đơn.
- Thất bại khi màu nền trùng màu vật.
- Hướng mở rộng tác giả đề xuất: thêm đầu diffusion cho âm thanh và lực.

---

## 3. LingBot-VA — Causal World Modeling for Robot Control

### 3.1 Ý tưởng chính

1. **Mixture-of-Transformers tự hồi quy, hai luồng bất đối xứng.** Token video và token action được đan xen thành **một
   chuỗi nhân quả**, dưới mặt nạ attention nhân quả nghiêm ngặt. Hai luồng có cùng độ sâu (30 lớp) nhưng khác độ rộng:
   - luồng video rộng 3072, khởi tạo từ Wan2.2-5B, dự đoán trạng thái hình ảnh ẩn kế tiếp bằng conditional flow
     matching;
   - luồng action rộng 768 (khoảng 350M tham số thêm), suy ra action bằng inverse dynamics từ chuyển trạng thái vừa dự
     đoán.

   Tổng cộng 5.3B tham số.
2. **Lịch sử trong KV cache.** Key-value của mọi token quan sát và action trong quá khứ được giữ lại. Khi chạy thật, token
   của **ảnh quan sát được** (không phải ảnh tự dự đoán) được đưa vào cache. Cách này vừa cho trí nhớ dài, vừa bỏ được
   phần tính lại ở mỗi bước.
3. **Noisy latent augmentation.** Khi train, luồng action được học đọc từ latent video còn nhiễu. Nhờ vậy, lúc chạy có thể
   dừng khử nhiễu video sớm, vì điều khiển cần cấu trúc ngữ nghĩa chứ không cần chi tiết từng pixel.
4. **Tiền huấn luyện chung video và action.** Khởi tạo từ Wan2.2-5B (đã tiền huấn luyện trên video internet), rồi
   tiền huấn luyện trên khoảng 16k giờ dữ liệu thao tác robot từ 7 nguồn, với một không gian action chung cho mọi loại
   robot (30 chiều cho robot hai tay).

### 3.2 Đóng góp cho robot learning

- **World model video dạng nhân quả, chạy liên tục.** Model vừa tưởng tượng vừa được neo vào thực tế ở mỗi bước, nên sửa
  lỗi được trong vòng kín. So với Motus (cùng backbone Wan và cùng MoT), khác biệt chính là tính nhân quả.
- **Trí nhớ dài cho task quan sát một phần.** Trong Wipe Plate, robot phải lau đúng 6 lần. Trong Search Box, khi hộp phải
  rỗng, robot phải nhớ để mở hộp trái thay vì mở lại hộp phải. Trên cả hai task, LingBot-VA vượt π0.5 rõ rệt.
- **Hiệu quả dữ liệu khi thao tác thật.** Trên 6 task thật của robot hai tay (dài hạn: làm bữa sáng, mở hộp hàng; chính
  xác: cắm ống, nhặt vít; vật biến dạng: gấp áo, gấp quần), với 50 demo mỗi task và 500 step fine-tune, model vượt π0.5
  khoảng 20% về success.
- **Đưa model 5B vào vòng điều khiển:** khoảng 0.5 s mỗi bước trên một RTX 5880 Ada, với 5 bước khử nhiễu video và 10
  bước khử nhiễu action.

### 3.3 Kết quả chính trong paper

| Thí nghiệm | Kết quả |
|---|---|
| RoboTwin 2.0, 50 task | Easy 92.0%, Hard 91.1% (Motus 88.7 / 87.0, π0.5 82.7 / 76.8, π0 65.9 / 58.4). Task dài nhất (horizon 3, Hard): 94.2%, hơn Motus 10 điểm |
| LIBERO (Spatial, Object, Goal, Long) | Trung bình 98.5% |
| Tiền huấn luyện so với khởi tạo thẳng từ Wan2.2-5B | 92.10 / 91.12% so với thấp hơn rõ rệt |
| Ablation: MoT so với chia sẻ trọng số | 92.93% so với 92.15% |
| Ablation: **bỏ dự đoán video tường minh** (chỉ dùng video làm tín hiệu huấn luyện) | **48.31%** |
| Ablation: attention hai chiều thay cho nhân quả | 81.46% |
| Ablation: 25 → 15 bước khử nhiễu video | 92.91% (không đổi); khi chạy thật dùng 5 bước |

### 3.4 Hạn chế tác giả tự nêu

1. Policy có nhớ lịch sử có thể overfit vào nhịp động lực học trong dữ liệu, và kém đi khi bị can thiệp mạnh ngoài phân
   phối, ví dụ người đảo ngược tiến độ của task.
2. Dù đã giảm số bước, backbone video vẫn chậm hơn các VLA không dự đoán video.
3. Tiền huấn luyện cần rất nhiều compute, khó tái lập khi thiếu tài nguyên.
4. Hướng mở rộng tác giả đề xuất: nén video hiệu quả hơn, và thêm cảm biến xúc giác, lực, âm thanh.

---

## 4. So sánh chi tiết

| Tiêu chí | UVA | LingBot-VA |
|---|---|---|
| Quan hệ video–action | Song song: biểu diễn ẩn chung, rồi tách thành hai đầu diffusion độc lập | Nối tiếp nhân quả: dự đoán video trước, rồi inverse dynamics suy ra action |
| Backbone | Một transformer chung (khoảng 0.5B) và hai đầu diffusion MLP nhẹ | MoT hai luồng bất đối xứng suốt các lớp: video 3072 chiều (Wan2.2-5B), action 768 chiều |
| Attention và tính nhân quả | Masked autoencoder (kiểu MAR) trên cửa sổ lịch sử và tương lai | Mặt nạ nhân quả nghiêm ngặt trên chuỗi đan xen theo thời gian |
| Cách tăng tốc | Bỏ hẳn đầu video khi điều khiển | KV cache, và dừng khử nhiễu video sớm nhờ noisy latent augmentation |
| Trí nhớ | Cửa sổ trượt cố định (PushT: 4 frame trong 1.6 s), mã hoá lại toàn bộ ở mỗi bước | Lịch sử dài trong KV cache (RoboTwin: cửa sổ 72 latent frame, tức 36 đoạn) |
| Dữ liệu và tiền huấn luyện | Train trên dữ liệu của từng benchmark; chưa dùng video không nhãn | Wan2.2-5B, rồi khoảng 16k giờ dữ liệu robot, rồi fine-tune |
| Lệch tần số video–action | Nhóm action giữa hai frame thành token, lặp lại cho khớp số token ảnh của frame, rồi qua một lớp FC | Giảm tần số video (RoboTwin: 50 Hz → 12.5 Hz), giữ action ở 50 Hz trong chuỗi đan xen |
| Tốc độ theo paper | 0.23 s (mô phỏng, 100 bước); 95 ms (thật, 16 bước) | ≈ 0.5 s mỗi bước (RTX 5880 Ada 48 GB, 5/10 bước) |
| Đo trên 1× L4 24 GB (repo này) | 0.14–0.48 s, 1.9 GB (EXP-07) | 3.8–17 s mỗi đoạn 32 action, 18 GB, có offload (EXP-11) |

---

## 5. Đánh giá phản biện

### 5.1 World model ngầm và world model tường minh lúc chạy

**UVA là world model ngầm lúc chạy.** Khi điều khiển, UVA không sinh quan sát tương lai. Học sinh video chỉ là loss phụ trợ
khi train, giúp định hình biểu diễn ẩn. Muốn dùng UVA như world model tường minh để lập kế hoạch (thí nghiệm forward
dynamics với 100 quỹ đạo lấy mẫu mỗi bước), phải chạy đầu video, và chi phí tăng mạnh.

**LingBot-VA là world model tường minh ngay trong vòng điều khiển.** Mỗi bước, luồng video thực sự dự đoán trạng thái thị
giác tương lai, rồi luồng action suy ra action từ trạng thái "tưởng tượng" đó. Noisy latent augmentation là một giải pháp
kỹ thuật thực tế để giữ tính nhân quả tường minh mà vẫn trong giới hạn độ trễ.

**Hai paper đi tới hai câu trả lời khác nhau.** Ablation "w/o video prediction" của LingBot-VA, tức chỉ dùng video làm
tín hiệu huấn luyện (gần với cách UVA chạy), làm success giảm từ 92.9% xuống 48.3%. Ngược lại, UVA đạt kết quả cao mà
không dự đoán video lúc chạy. Đây không phải so sánh trực tiếp: mỗi ablation chỉ nói về kiến trúc của chính paper đó. Có
ba yếu tố có thể giải thích sự khác biệt:
- **Task và dữ liệu:** RoboTwin là thao tác hai tay; PushT và LIBERO là một tay. Nhưng độ dài task một mình không giải
  thích được: LIBERO-10 gồm các task dài, vậy mà trong UVA lợi ích của video ở đó chỉ là 0.04, so với 0.42 ở PushT-M.
- **Kiến trúc:** luồng action của LingBot-VA rất nhỏ và được thiết kế để dựa vào dự đoán của luồng video. Bỏ dự đoán đi
  thì luồng này yếu. Đầu action của UVA đọc thẳng biểu diễn ẩn của cả transformer.
- **Điểm khởi đầu:** trong thí nghiệm EXP-03, khi fine-tune từ checkpoint đã học video, tắt loss video chỉ làm điểm giảm
  0.063. Có vẻ phần lợi ích của video đã nằm sẵn trong biểu diễn.

### 5.2 Bộ nhớ, độ trễ và khả năng mở rộng

UVA không có cache nhân quả: cửa sổ token được xử lý lại ở mỗi lần gọi. Theo cách hiểu của chúng tôi, đây là một lý do
khiến attention chiếm khoảng một nửa thời gian suy luận như tác giả báo.

KV cache của LingBot-VA bỏ được phần tính lại, nhưng **không** làm chi phí cố định. Đo trên L4 (EXP-11), với trí nhớ đầy
đủ, thời gian mỗi đoạn vẫn tăng tuyến tính từ 16 s lên 24 s, vì attention phải đọc toàn bộ cache. Thời gian chỉ đi ngang
khi cache đầy (cửa sổ 72). Cắt trí nhớ xuống 4 đoạn thì thời gian cố định, nhưng video vỡ sau khoảng 15 đoạn. Như vậy,
"giải quyết bài toán độ trễ chuỗi dài" chỉ đúng một phần.

Về phần cứng, model 5.3B cộng KV cache dùng khoảng 18 GB VRAM khi offload bộ mã hoá văn bản T5 (11 GB) và VAE sang CPU.
Không offload, chúng tôi ước tính vượt 24 GB, nhưng chưa chạy thử. Riêng việc mã hoá câu lệnh trên CPU mất khoảng 4.5
phút mỗi episode. Mức 0.5 s mỗi bước của paper được đo trên GPU 48 GB; trên L4 với cùng số bước, chúng tôi đo được 3.8 s
mỗi đoạn (cấu hình chunk khác, nên không so trực tiếp được). Trong 17 s mỗi đoạn ở config mặc định, khoảng 6 s là sinh
video và 10 s là sinh action.

### 5.3 Số bước khử nhiễu

| | Paper | Repo (config mặc định) | Đo được |
|---|---|---|---|
| UVA | 100 bước (mô phỏng), 16 bước (thật) | 100 | 5 bước: mean score PushT 0.995 so với 0.957 ở 100 bước (khoảng tin cậy chứa 0), nhanh hơn 3.5× |
| LingBot-VA | 25 → 15 bước không mất gì; chạy thật dùng 5 | 25 video / 50 action | 10/20: action lệch 0.002 (mốc đổi seed 0.016), nhanh hơn 2.4×; 5/10: lệch 0.015, nhanh hơn 4.5× |

Kết quả của chúng tôi phù hợp với claim về noisy latent của LingBot-VA (đo độ lệch action ở chế độ vòng hở, không đo
success). UVA dùng ít bước hơn nhiều so với paper mà không giảm điểm PushT (một seed, task một đích; chưa kiểm tra trên
task khó hơn). Khi so tốc độ giữa các phương pháp, cần so ở cùng chất lượng, không phải ở config mặc định.

### 5.4 Những gì paper chưa kiểm tra

- **UVA:** độ đa dạng thật của đề PushT-M. Code đánh giá gốc bốc đích từ bộ sinh ngẫu nhiên toàn cục; các simulator chạy
  song song được fork với cùng trạng thái, nên mọi episode trong một đợt có chung một đích (EXP-01, ISSUE-005). Với cấu
  hình mặc định (56 simulator một đợt), con số 0.88 nhiều khả năng được đo trên một đích. Paper còn báo checkpoint tốt
  nhất được chọn theo chính kết quả đánh giá. Trên 50 đích khác nhau, checkpoint đã công bố đạt 0.813.
- **UVA:** tên metric không thống nhất. Bảng ghi "success rate", phần mô tả PushT-M ghi "average reward". Con số khớp với
  mean score. Trên cùng model PushT, mean score 0.97 nhưng success thật chỉ 0.73.
- **LingBot-VA:** model tự tưởng tượng được bao xa khi không có quan sát thật. Trong lần chạy vòng hở 40 đoạn (EXP-11; một ảnh,
  một câu lệnh, một seed), cảnh trôi sau khoảng 10 đoạn, ngay cả khi nhớ đầy đủ.
- **LingBot-VA:** mức độ model nghe theo câu lệnh. Đổi câu lệnh thì action đổi, nhưng video vẫn làm task cũ (một lần thử).

---

## 6. Đối chiếu claim của paper với tái lập

| Claim | Paper | Tái lập trong repo | Kết luận |
|---|---|---|---|
| UVA đạt PushT 0.98 | 0.98 | 0.974 ± 0.016 | Xác nhận |
| UVA đạt LIBERO-10 0.90 | 0.90 | 0.865 [0.811, 0.906] | Xác nhận (trong khoảng tin cậy) |
| UVA đạt PushT-M 0.88 | 0.88 | 0.787 (code gốc, 5 đích mỗi lần chạy); 0.813 (50 đích) | Thấp hơn; một phần do lỗi giao thức đánh giá |
| Video giúp policy | UVA-action −0.42 trên PushT-M | −0.063 khi fine-tune 25 demo từ checkpoint đã học video | Xác nhận về chiều, độ lớn phụ thuộc bối cảnh |
| UVA học được đích mới từ ít demo (không phải claim của paper; thí nghiệm của repo) | – | 25–50 demo: khối tới gần đích (4–5 px) nhưng chỉ 2–4/30 episode đúng cả vị trí lẫn góc; gần đích hay kẹt ở góc xoay ~120° (r ≈ 0.7) | Học được vị trí thô, chưa đặt chính xác |
| UVA suy luận nhanh | 0.23 s | 0.48 s ở 100 bước, 0.14 s ở 5 bước (L4, chậm hơn GPU của paper) | Cùng bậc; giảm số bước còn nhanh hơn |
| LingBot-VA ít bước khử nhiễu không mất chất lượng | 25 → 15 bước: 92.91% | 10/20 bước gần như trùng bản gốc | Phù hợp (độ lệch action vòng hở, không đo success) |
| LingBot-VA có trí nhớ dài nhờ KV cache | Wipe Plate, Search Box | Chi phí tăng tuyến tính; vòng hở trôi sau khoảng 10 đoạn (một lần chạy) | Chưa kiểm chứng (cần vòng kín); bản thân model trôi khi chạy vòng hở |
| LingBot-VA chạy được trong thực tế | ≈ 0.5 s mỗi bước (48 GB) | 3.8–17 s mỗi đoạn, 18 GB, cần offload (24 GB) | Chạy được, nhưng xa thời gian thực trên GPU 24 GB |

---

## 7. Hướng nghiên cứu kết hợp hai ý tưởng

Hai paper bù trừ nhau. UVA mạnh ở tính linh hoạt (masking) và ở khả năng bỏ hẳn video khi cần phản xạ nhanh. LingBot-VA
mạnh ở kiến trúc bất đối xứng, trí nhớ dài và giải mã từ trạng thái còn nhiễu. Dưới đây là ba hướng kết hợp, kèm ghi chú
mức độ mới và cách kiểm chứng.

### Hướng 1: world model thích nghi theo độ bất định ("System 1 / System 2")

**Nhận xét.** Không phải lúc nào robot cũng cần tưởng tượng tường minh khung hình tương lai như LingBot-VA, và cũng không
phải lúc nào cũng bỏ qua được như UVA. Khi di chuyển trong không gian trống, phản xạ nhanh là đủ (System 1). Khi sắp tiếp
xúc, thao tác chính xác (cắm ống, gấp áo) hoặc gặp trạng thái ngoài phân phối, robot cần mô phỏng tương lai để lập kế
hoạch (System 2).

**Cách kết hợp.** Dùng một kiến trúc MoT nhân quả với cổng định tuyến động:
- Khi độ bất định thấp (entropy của phân phối action thấp, hoặc dự đoán bước trước khớp với ảnh thật): bỏ qua luồng
  video như chế độ policy của UVA, giải mã action thẳng từ KV cache.
- Khi độ bất định vượt ngưỡng (va chạm, che khuất, chuyển pha vật lý): bật luồng video, chỉnh số bước khử nhiễu theo noisy
  latent, và có thể lấy mẫu nhiều nhánh tương lai như chế độ forward dynamics planner của UVA để chọn quỹ đạo tốt nhất.

**Ghi chú.** Đây là hướng được thực nghiệm của repo hỗ trợ trực tiếp nhất. Hai paper bất đồng về việc có cần video tường
minh không (mục 5.1). Trong LingBot-VA, phần sinh video chiếm khoảng 1/3 thời gian mỗi đoạn (mục 5.2); khoảng cách lớn hơn
giữa hai model còn do chênh lệch kích thước, nên không quy hết cho video được. Cách kiểm chứng: đo success
và thời gian trung bình khi bật video ở 0%, 100% và theo ngưỡng.

### Hướng 2: huấn luyện lai nhân quả và mask

**Nhận xét.** LingBot-VA bị cố định theo một chiều suy luận nhân quả (từ quan sát và action ra quan sát kế tiếp, rồi ra
action). UVA nhờ mask ngẫu nhiên đổi linh hoạt được giữa 5 vai trò trên cùng bộ trọng số, nhưng không có KV cache và chưa
dùng được video không nhãn ở quy mô lớn.

**Cách kết hợp.** Đưa masking của UVA vào bên trong từng chunk thời gian của kiến trúc MoT có KV cache:
- **Giữa các chunk:** giữ mặt nạ nhân quả để dùng được KV cache cho trí nhớ dài.
- **Bên trong mỗi chunk:** dùng mask ngẫu nhiên hai chiều của UVA. Với video không có action, che toàn bộ luồng action.
  Với dữ liệu robot, đổi ngẫu nhiên giữa bài toán forward dynamics và inverse dynamics.

**Ghi chú.** Việc dùng video không có action chính là hạn chế mà UVA tự nêu. LingBot-VA đã dùng một phần video internet
qua Wan2.2. Phần mới ở đây là cho LingBot-VA thêm vai trò simulator thuận để đánh giá hoặc lập kế hoạch cho policy, mà
không phải đổi kiến trúc.

### Hướng 3: world model đa phương thức bất đối xứng (thị giác – lực/xúc giác – action)

**Nhận xét.** Các task thật của LingBot-VA (cắm ống, nhặt vít, rọc hộp) đều giàu tiếp xúc, trong khi thông tin thị giác
dễ bị chính tay kẹp che khuất.

**Cách kết hợp.** Áp dụng nguyên lý MoT bất đối xứng với ba luồng:
- luồng video lớn, tần số thấp (khoảng 10–12.5 Hz);
- luồng lực/xúc giác nhỏ, tần số cao;
- luồng action nhỏ, tần số cao (50 Hz trở lên).

Mỗi phương thức có đầu giải mã riêng như UVA. Nhờ KV cache và việc tách giải mã, robot cập nhật tiếp xúc và chỉnh action ở
tần số cao, trong khi vẫn được dẫn hướng bởi dự báo thị giác ở tần số thấp hơn.

**Ghi chú.** Cả hai paper đều nêu đây là hướng tương lai: UVA nhắc tới âm thanh và lực, LingBot-VA nhắc tới xúc giác, lực
và âm thanh. Điểm riêng của đề xuất này là cấp phát dung lượng và tần số khác nhau cho từng luồng.
