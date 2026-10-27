# E11 – bản nộp ACIIDS 2027 (Springer LNCS/LNAI), cập nhật 2026-10-26

## Biên dịch
1. Tải `llncs.cls` và `splncs04.bst` chính thức (mẫu Springer LNCS trên Overleaf/Springer) vào cùng thư mục với `main.tex`. Gói này KHÔNG kèm hai file đó.
2. `pdflatex main && bibtex main && pdflatex main && pdflatex main`
3. `PREVIEW_layout_only.pdf` dựng bằng lớp giả lập (15 trang, sát giới hạn). Phải biên dịch lại bằng llncs.cls thật; nếu vượt 15 trang, cắt Phụ lục C/D hoặc bảng IBM1.

## Phải điền/kiểm trước khi nộp (EasyChair, hạn 31/10/2026)
- `main.tex`: tên tác giả, đơn vị, ORCID; kiểm tra CFP xem có cần ẩn danh không.
- `\repourl` trong `main.tex` đang là `TODO-ADD-REPOSITORY-URL-BEFORE-SUBMISSION`: tạo kho từ thư mục `repro_kit/` rồi điền URL/DOI.
- Chỉ nộp MỘT nơi (ACIIDS cấm nộp song song).
- Tuyên bố công cụ AI/đạo đức theo yêu cầu Springer/EasyChair.
- Tham khảo chưa xác minh được: `dinh2026registers` (đã khớp tên bài, tạp chí, tập 12:1, trang 39–71, DOI; chưa thấy danh sách tác giả trên trang gốc), `nguyen2026sentenceaware` (trang Underline chỉ ghi Tan Sang Nguyen, AAAI 2026), `nguyen2026parallelretrieval` (chưa rõ nơi xuất bản; bản thảo camera-ready 2026-10-20). Đã khớp trên trang gốc: Ni 2022, Fomicheva 2020, Nguyen 2025 (BARTBahnar), Vo 2024 (AAAI).

## Cập nhật 27/10 (rà soát cuối)
- Đã bỏ email cá nhân khỏi ô tác giả (hiện `author@example.org`): PHẢI thay bằng email thật của tác giả nộp bài.
- Đã sửa: "β=5 thấp hơn nền" (thực tế 741 > 739), k=5 là tiền định, tham chiếu phụ lục, đoạn trùng về validation, ghi chú nội bộ trong danh mục tham khảo, dấu và viết hoa tên riêng trong .bib.
- Chưa tôi kiểm chứng được: số IBM1 (405, 294, "9/32", "353") vì chưa nhận gói IBM1; con số Sinkhorn 770/781 (gốc không tái lập được, bài đã nói rõ).
- Cần xác nhận: nơi xuất bản của bản camera-ready AACL (hiện ghi "Camera-ready manuscript, 20 October 2026"), danh sách tác giả Dinh et al. và Sentence-aware.
- Cần điền: tác giả/đơn vị/ORCID, `\repourl`, tuyên bố dùng công cụ AI theo Springer/EasyChair.
