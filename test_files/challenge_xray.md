# Image Classification Challenge
## Mô tả
Xây dựng model phân loại ảnh X-ray phổi thành 3 lớp: **Normal**, **Pneumonia**, **COVID-19**.

## Dữ liệu
- **Input:** Thư mục `./dataset/public/` chứa ảnh .png (224x224) và file `labels.csv`
- **Output:** File `./working/submission.csv` với 2 cột: `image_id, prediction`
- Prediction là 1 trong 3 nhãn: `normal`, `pneumonia`, `covid`

## Yêu cầu bắt buộc
1. Phải sử dụng **Deep Learning** (CNN, ViT, hoặc tương đương). KHÔNG dùng ML truyền thống.
2. Phải **train model từ dữ liệu** trong `./dataset/public/`. KHÔNG hardcode kết quả.
3. KHÔNG sử dụng pseudo-labeling trên test set.
4. File code phải đọc dữ liệu từ `./dataset/public/` và xuất kết quả ra `./working/submission.csv`.
5. KHÔNG gọi API bên ngoài (OpenAI, HuggingFace Inference, v.v.)

## Đánh giá
- Macro F1-Score trên tập test ẩn.
- Baseline: 0.60
