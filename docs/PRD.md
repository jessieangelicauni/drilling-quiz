# PRD — KO-DRILL: Personalized Korean Drill Quiz

**Status:** Living document — acuan keputusan desain & implementasi
**Terakhir diperbarui:** 2026-10-18 — paper (`paper/KO-DRILL_IEEE.docx`) Section V sudah disinkronkan dengan §4–§5 dokumen ini: ablasi besar dibuang, evaluasi diorganisir ulang jadi satu validasi terarah per novelty (V-B/C/D), human study diturunkan jadi pilot eksplorasi sekunder (V-E), dan taksonomi KAGAS sudah dipakai konsisten di Section IV-C + Fig. 3, bukan lagi taksonomi ad hoc.

**Audit konsistensi (2026-10-18, lanjutan):** dicek ulang PRD vs paper secara eksplisit. Hasil: (1) Critic loop di §3.2 **sudah disinkronkan** ke paper — Fig. 3 dan Section IV-C sekarang punya kotak CRITIC (LLM-as-judge) dengan jalur revise-bukan-resample. (2) Tracker default (§2.1) — ditemukan Novelty 3 (bukan cuma Novelty 1) juga dibangun di atas formalisme urn Urnings; keputusan awal "Parallel Elo default" **dibalik**: Urnings sekarang tracker utama di PRD maupun paper, konsisten satu arah. Lihat §2.1 dan §8.

**Audit konsistensi (2026-10-19):** dicek independen dari paper — apakah Urnings tetap pilihan tepat kalau dievaluasi murni dari kebutuhan implementasi, bukan dari otoritas paper. (1) Dijalankan simulasi tambahan `src/simulate_drift.py`: Glicko-2-style tracker (RD native, kontinu) unggul telak di simulasi ability **statis** (RMSE 0.11 vs Urnings 0.43), tapi begitu ability dibuat **naik seiring waktu** (simulasi pelajar yang sungguh belajar, +1.5 logit / ~400 respons) — regime yang sebenarnya relevan untuk aplikasi drilling — keunggulan itu terbalik: RD Glicko mengecil lalu "terkunci", RMSE akhir memburuk ke 1.05 (hampir menyamai Elo polos 1.21), sementara Urnings tetap stabil (0.86) karena urn-nya dibatasi (`n_p=32`) sehingga selalu punya ruang bergerak kalau data berubah arah. Glicko **ditolak** atas dasar ini — lihat §2.1 baris "Ditolak". (2) Dicek juga precedent industri (Duolingo, Math Garden, Duolingo English Test, platform game berbasis Glicko) — tidak ada satupun yang menghadapi regime identik kita (item baru + kalibrasi simultan); lihat `docs/literature-review-15.md` gap G3. (3) Fitur produk baru (daily quiz, leaderboard EXP, mastery per-topik, requeue-until-mastered) ditambahkan ke §2.1 (kriteria mastery) dan §2.4 baru (lapisan gamifikasi, sengaja dipisah total dari tracker psikometrik).

**Audit implementasi nyata (2026-10-19, lanjutan):** mulai implementasi kode produksi (bukan simulasi) — `src/urnings_tracker.py`, `src/vocab_scheduler.py` (wrapper `fsrs`), `src/grammar_scheduler.py`, `src/streak.py`, `src/daily_drill.py`, semua lolos smoke test. Sekaligus diuji komponen yang sebelumnya **cuma klaim literatur, belum pernah dicoba sendiri** (KLUE-RoBERTa, EXAONE, KAGAS validator) — lihat GPU sandbox tersedia (RTX 4070 Ti SUPER 16GB), jadi dijalankan sungguhan: (1) KLUE-RoBERTa-base terkonfirmasi jalan untuk tokenisasi/forward-pass teks Korea. (2) **EXAONE-3.5 diganti ke EXAONE-4.0-1.2B** — 3.5 butuh `trust_remote_code` yang ternyata rusak di semua versi `transformers` yang dicoba; 4.0-1.2B didukung native, tetap punya hybrid reasoning mode (lihat §2.3). (3) Generasi nyata EXAONE-4.0 untuk soal partikel 에/에서 menunjukkan distractor off-target (pakai `와`, bukan partikel lokatif) dan penjelasan tata bahasa yang salah secara faktual — bukti konkret bahwa validator KAGAS (§3.2) memang wajib, sekarang diimplementasikan di `src/distractor_validator.py` (berbasis `kiwipiepy`), terbukti menangkap kasus itu otomatis.

**Audit pipeline generasi nyata (2026-10-19, lanjutan lagi):** dibangun `src/exaone_provider.py` — `ContentProvider` sungguhan (bukan mock) untuk item grammar, lewat proses debugging yang substansial, dicatat supaya tidak terulang:
1. Permintaan JSON terstruktur zero-shot/one-shot menghasilkan JSON yang **valid secara sintaks tapi salah secara isi** di semua percobaan — model menulis meta-komentar tentang soal ("Apa isi blank-nya?"), bukan soal sungguhan.
2. `float16` menyebabkan output **rusak secara numerik** (karakter Cyrillic nyasar) — diganti ke `bfloat16`, rusaknya hilang.
3. Permintaan terbuka ("tulis satu kalimat memakai X") di non-reasoning mode gagal **hampir instan** (model echo satu token lalu berhenti) — awalnya terlihat seperti "non-reasoning mode butuh reasoning mode untuk tugas non-trivial", tapi ternyata bukan itu masalahnya.
4. Perbaikan sebenarnya: prompt **sangat terstruktur** (template eksplisit + satu contoh kerja + satu slot isian) membuat non-reasoning mode bekerja konsisten (5/5 di 5 topik beda). Model 1.2B ini butuh jauh lebih banyak scaffolding daripada contoh di model card-nya sendiri.

**Desain akhir yang mengikuti temuan ini:** LLM HANYA diminta satu hal yang terbukti reliable — satu kalimat natural memakai partikel target. LLM tidak pernah diminta membuat distractor atau menandai blank sendiri. Tiap skill = himpunan partikel yang bisa tertukar (closed set, ditentukan linguistik/KAGAS, bukan dikarang LLM); `kiwipiepy` mencari posisi partikel secara deterministik; setiap opsi (termasuk jawaban benar) divalidasi `distractor_validator` sebelum disimpan. Diuji end-to-end lewat `daily_drill.py` dengan provider sungguhan (bukan mock) — berhasil.

**Skill kedua + bug baru ditemukan (2026-10-19, lanjutan):** ditambah `particle_obj` (을/를 vs 에 — menandai objek langsung dengan partikel lokatif, kesalahan umum pembelajar). Proses menambah skill ini menyingkap bug yang tag-checking saja tidak bisa tangkap: EXAONE-4.0-1.2B menghasilkan `"라면를 먹어요"` — salah, seharusnya `"라면을"` karena 라면 berakhiran konsonan (batchim). `distractor_validator` yang hanya mengecek tag (JKO) **meloloskan** ini karena `를` memang partikel JKO yang sah, cuma alomorf yang salah untuk kata itu. Ditambahkan `has_batchim()`/`expected_object_particle()` (dekomposisi Unicode Hangul, deterministik, bukan panggilan model) ke `distractor_validator.py` untuk menangkap kelas error ini secara spesifik — diuji ulang, 4/5 generasi lolos (1 ditolak otomatis oleh guard batchim, bukan gagal diam-diam). Juga ditemukan: percobaan generalisasi awal nyaris merusak `particle_loc` yang sudah jalan — cek "distractor tidak boleh valid sebagai tag target" itu BENAR untuk `particle_obj` (beda tag, 에≠JKO) tapi SALAH untuk `particle_loc` (kedua opsi memang sama-sama JKB secara sah; itulah yang membuatnya confusable, tagger tidak menilai ketepatan kontekstual). Pelajaran: perbaikan untuk satu skill tidak otomatis general ke skill lain, perlu regresi ulang tiap kali.

**Audit "apakah semua metode tepat" (2026-10-19, lanjutan lagi) — investigasi + perbaikan/eliminasi per metode:**

1. **Konjugasi verba (sebelumnya "keterbatasan residual, belum ditangani") — DIPERBAIKI, bukan dibiarkan.** `kiwipiepy`'s analyzer ternyata terlalu permisif untuk dipakai sebagai pengecek gramatikalitas (didesain tahan terhadap teks informal/noise, jadi dia tetap berhasil "mem-parse" `마시요` yang salah, seolah valid — round-trip analyze→join tidak menangkap ini). Perbaikan sebenarnya: `exaone_provider.py` dirombak total — **LLM sekarang HANYA diminta lemma verba** (tugas pemilihan kosakata, tetap genuinely open-ended — di sinilah nilai "infinite item pool" sebenarnya berada), **bukan menulis kalimat lengkap**. Partikel benar ditentukan kode (fixed untuk `particle_loc`, aturan batchim untuk `particle_obj`), dan konjugasi dibangun deterministik lewat `kiwi.join()` (terbukti benar untuk kasus tak-beraturan: 마시다→마셔요, 보다→봐요, 하다→해요). Hasil: `particle_loc` 4/5, `particle_obj` 5/5, **nol kesalahan konjugasi atau partikel** — LLM tidak lagi punya kesempatan salah di dua hal itu karena tidak lagi diminta melakukannya.

**Residual pemilihan verba — DIPERBAIKI juga (2026-10-19, audit lanjutan ketiga).** Ditemukan verba kadang aneh secara semantik (라면→들다 "angkat", 커피→차다 "tendang"). Dicoba dulu teknik yang baru terbukti berhasil di kasus Qwen3 (diversifikasi contoh few-shot) — **tidak transfer**, malah membuat EXAONE-4.0-1.2B lebih buruk (5/6 kasus echo kata template "입력" bukan jawab; 1.2B ternyata sudah di batas kompleksitas prompt yang bisa ditanganinya). Uji model lebih besar (32B, satu-satunya ukuran lain yang ada) dipertimbangkan lalu ditolak — download besar + risiko OOM untuk masalah yang sifatnya kosmetik, tidak proporsional. Perbaikan yang dipakai: `slots` sudah berupa tuple kecil terkurasi dari awal, jadi memasangkan tiap slot dengan satu verba terkurasi (`verb_lookup`) sama validnya dan menghilangkan risiko ini total untuk semua slot yang ada sekarang. LLM tetap jadi fallback untuk slot baru yang belum ada di tabel (diuji eksplisit, jalur fallback terkonfirmasi jalan) — jalur generatif tidak dibuang, cuma tidak dipercaya untuk sesuatu yang kurasi manual sama murahnya dan jauh lebih aman.

2. **KLUE-RoBERTa (sebelumnya "❌ buntu, tidak ada data") — DIGANTI sementara, bukan dibiarkan kosong.** Karena benar-benar tidak ada dataset kalibrasi kesulitan Korea (bukan soal waktu/usaha), regression head yang butuh training tidak bisa diimplementasikan sekarang. Dibuat `src/difficulty_heuristic.py` — estimator deterministik (jumlah morfem + keberagaman tag gramatikal "lanjutan" seperti EP/ETM/EC/NNB dari `kiwipiepy`, tanpa training) dengan **interface output sama** (`mu_hat, sigma_hat2`) seperti yang akan dihasilkan encoder terlatih, jadi bisa langsung dipakai `seed_item_from_encoder()` sekarang dan ditukar modelnya nanti tanpa mengubah kode lain. `sigma_hat2` sengaja dibuat tetap tinggi (1.5) — estimator ini harus selalu terlihat tidak terlalu yakin, supaya prior yang dihasilkan lemah dan cepat ditimpa data respons asli. Diuji: urutan kesulitan monoton benar (kalimat sederhana < honorifik < kompleks-pasif). Eksplisit lebih lemah dari klaim Li et al. 2025 (encoder terlatih) — bukan diklaim setara, cuma pengganti sementara yang bisa diuji sekarang.

3. **Qwen3 bahasa Indonesia (sebelumnya "⚠️ English-only, bisa dicoba lagi") — DIELIMINASI setelah dicoba ulang, bukan dibiarkan "mungkin bisa."** Dicoba perbaikan dengan one-shot contoh Bahasa Indonesia: instruksi bahasa langsung terikuti 4/4 (dari sebelumnya 0/3). TAPI muncul bug baru yang lebih berbahaya: **3/4 percobaan membocorkan topik dari contoh** (menjelaskan kalimat makan-di-restoran seolah soal belajar-di-sekolah, karena contohnya soal belajar) — bahasa Indonesia lancar, tapi faktanya salah, lebih sulit terdeteksi dibanding sebelumnya (full bahasa Korea jelas-jelas salah). Keputusan: **Bahasa Indonesia dieliminasi secara definitif**, bukan ditunda — `qwen_feedback.py` sekarang English-only berdasarkan bukti, bukan batasan sementara.

4. **`MASTERED_RECHECK_DAYS=30`** — tidak diubah. Arah metodenya (recheck periodik) sudah benar dan perlu (lihat kasus Glicko), tapi angka 30 tetap tebakan masuk akal, bukan hasil tuning dari data pembelajar nyata — tidak ada cara memvalidasi ini tanpa data pemakaian sungguhan, jadi dibiarkan sebagai konstanta eksplisit yang bisa ditune nanti, bukan diklaim sudah benar.

**Ringkasan status akhir:** tidak ada lagi metode berstatus "buntu tanpa solusi" atau "berisiko tanpa mitigasi" di sistem ini — tiga dari empat sudah diperbaiki/diganti dengan metode yang benar-benar tepat dan teruji; satu (`RECHECK_DAYS`) tetap konstanta eksplisit yang jujur diakui sebagai tebakan, bukan ditutup-tutupi sebagai "sudah benar".

**Audit "yakin 100%?" (2026-10-19, lanjutan) — dua residual terakhir diselesaikan sejauh bisa:**

1. **Konjugasi di contoh ilustrasi Qwen (`들어요`→`듣어요`) — DIPERBAIKI.** Ditemukan: `kiwipiepy` menganggap `stem+요` (tanpa vokal penghubung, pola `마시요`/`보요`) sebagai parse yang valid (bukan error) — makanya round-trip analyze→join biasa tidak menangkapnya. Tapi untuk kasus `듣어요` (stem tak-beraturan + ending penuh tanpa perubahan bunyi), `kiwipiepy` menandainya `VV-I` dan `join()` otomatis mengoreksi ke `들어요`. Dibangun `correct_korean_conjugation()` di `distractor_validator.py`: paksa EF/EC "요" telanjang setelah stem verba jadi "어요" sebelum `join()` — menutup KEDUA kelas error (diverifikasi idempoten: tidak merusak `가요`/`해요`/`만나요` yang sudah benar, karena `kiwipiepy` selalu merepresentasikan bentuk benar sebagai stem+"어요" secara internal, tidak pernah stem+"요" telanjang). Dipasang sebagai post-processing otomatis di `qwen_feedback.py` (kedua jalur, kedua bahasa). Diverifikasi ulang end-to-end: `들어요` sekarang benar di output nyata.

2. **`MASTERED_RECHECK_DAYS=30` — TIDAK bisa disolve penuh, tapi diperbaiki sejauh mungkin secara jujur.** Dicoba menurunkan angka ini dari matematika FSRS yang sudah tervalidasi (bukan tebak sendiri) — **gagal transfer**: interval FSRS tumbuh eksponensial karena mengasumsikan review yang terjarak lama (itu mekanisme intinya), sementara "mastery" Urnings terbentuk dari banyak respons dalam waktu singkat — mensimulasikan ini sebagai "review FSRS berturut-turut" menghasilkan interval absurd (puluhan ribu hari). Root cause sama dengan KLUE-RoBERTa: **tidak ada data retensi skill Korea nyata**, dan tidak ada kode yang bisa mengganti data yang tidak ada. Yang BISA diperbaiki: `grammar_scheduler.py` sekarang punya `RecheckOutcomeLog` — mencatat setiap kejadian recheck (berapa hari sejak terakhir dipraktikkan, apakah mastery terkonfirmasi atau ternyata sudah regresi), supaya begitu ada data pemakaian nyata, angka 30 ini bisa di-tune dari bukti, bukan ditebak lagi. Constant 30 tetap ada sebagai default eksplisit, bukan diklaim benar.

**Status akhir yang jujur:** dari tabel audit terakhir, 9 dari 11 item sekarang berstatus ✅ dengan bukti, dan 1 item (`RECHECK_DAYS`) punya jalur menuju validasi nyata (bukan cuma label) walau belum tervalidasi hari ini — itu adalah batas yang sebenarnya dari apa yang bisa diselesaikan tanpa data pengguna sungguhan, bukan kekurangan usaha.
**Terkait:** `paper/KO-DRILL_IEEE.docx` (paper akademik, bahasa Inggris), repo ini (implementasi)

Dokumen ini merangkum **semua keputusan arsitektur beserta alasannya** yang sudah
dibahas dan disepakati. Tujuannya satu: supaya 3 bulan dari sekarang, saat ada
pertanyaan "kenapa dulu kita pilih X dan bukan Y", jawabannya ada di sini —
bukan harus digali ulang dari histori chat.

Prinsip kerja proyek ini: **ini riset, bukan produk produksi** — untuk
paper dan tiga novelty-nya (§3). Untuk implementasi aplikasi (§9
seterusnya), tujuannya **production-grade**: metode yang dipakai harus
yang paling tepat secara teknis, bukan sekadar "cukup jalan". Dua hal ini
hidup berdampingan, bukan kontradiksi — paper butuh novelty yang defensible,
aplikasi butuh metode yang benar.

**Keputusan di dokumen ini tidak permanen secara dogmatis.** Semua boleh
diubah nanti — TAPI hanya dengan alasan yang kuat (bukti baru, pengujian
nyata, bukan sekadar preferensi atau kemalasan mengecek ulang), bukan
ditimpa begitu saja. Pola yang sudah terjadi berkali-kali di proyek ini:
Urnings-vs-Elo dibalik dari keputusan awal setelah simulasi; Glicko sempat
menang lalu ditolak setelah uji drift; "Indonesia tidak bisa" untuk Qwen3
dibalik setelah model lebih besar diuji. Semua pembalikan itu VALID karena
disertai bukti konkret, bukan tebakan. Standar yang sama berlaku untuk
siapa pun (termasuk sesi/cloud session berikutnya) yang mau mengubah
keputusan di dokumen ini: tunjukkan pengujian/bukti, jangan cuma asumsi.
Kalau sebuah komponen tidak menjawab salah satu dari 3 novelty inti (§3)
ATAU tidak menjawab kebutuhan aplikasi (§9.4), itu kandidat kuat untuk
**tidak** dikerjakan sekarang.

---

## 1. Masalah & Kenapa Proyek Ini Ada

**Masalah:** Latihan drill terarah (grammar + kosakata Korea) efektif untuk
pembelajar, tapi menulis ribuan soal berkualitas, per level TOPIK, dengan
distractor yang masuk akal — itu kerja tim linguis bertahun-tahun. Kita tidak
punya tim itu. LLM bisa generate soal sesuai permintaan, tapi soal yang baru
digenerate **tidak punya data kalibrasi tingkat kesulitan** — berbeda dari
bank soal statis yang sudah pernah diuji ke ribuan orang.

**Kenapa kita tidak meniru Duolingo/ALEKS/Math Garden saja** (lihat riset di
§5 untuk detail dan sitasi):

- Di semua sistem besar itu, **kontennya dikarang dan di-QA manusia** dalam
  skala organisasi yang kita tidak punya, *sebelum* masuk ke mesin adaptif.
  Birdbrain (Duolingo) memilih dari soal yang sudah ada — tidak mengarang
  soal baru. ALEKS dan Math Garden dibangun manusia selama bertahun-tahun.
- Mereka tidak pernah perlu memecahkan masalah "soal baru tanpa kalibrasi,
  disajikan detik itu juga" — karena soal mereka memang tidak pernah benar-
  benar baru saat disajikan.
- Skala data mereka 5–6 orde besar lebih besar dari kita (Birdbrain: puluhan
  miliar prediksi/hari dari 50 juta+ pengguna; kita: 50–180 pelajar). Model
  berat (LSTM, DKT, dsb.) di skala kita = overfitting, bukan personalisasi.

**Kesimpulan posisi proyek:** Kita **ikuti** industri persis di bagian yang
skalanya cocok (penjadwalan, pelacakan kemampuan — pakai metode ringan/
klasik, sama seperti ALEKS-era-awal dan Math Garden). Kita **berbeda** hanya
di bagian yang terpaksa beda karena keterbatasan sumber daya: generasi dan
kalibrasi soal baru. Itulah satu-satunya tempat novelty riset kita berada.

---

## 2. Dekomposisi Sistem & Keputusan per Komponen

Sistem ini BUKAN satu model AI besar. Dipecah 3 bagian independen, karena
masing-masing punya karakteristik data yang beda dan butuh solusi beda.

### 2.1 Learner Modeling (seberapa paham siswa pada skill X)

| | |
|---|---|
| **Keputusan** | Model psikometrik eksplisit — **Urnings** sebagai tracker utama sistem (bukan Parallel Elo). Elo dan Parallel Elo dipertahankan sebagai baseline pembanding di evaluasi (§5), bukan sebagai default produksi. |
| **Alasan** | (a) Data kita kecil (puluhan-ratusan pelajar) — model neural (DKT/SAKT/AKT) butuh ratusan ribu interaksi untuk tidak overfit, dan tetap unggul tipis dibanding IRT/Elo bahkan di dataset besar menurut literatur pembanding langsung. (b) Butuh uncertainty yang bisa dipakai modul lain (§3.3) — Elo polos tidak punya ini secara native. (c) Harus interpretable untuk diaudit guru/reviewer. |
| **Kenapa bukan Elo polos** | Elo gagal spesifik ketika **item dipilih adaptif DAN dikalibrasi dinamis secara bersamaan** — persis kondisi yang dipaksa oleh item hasil-generate-LLM. Disimulasikan sendiri: 120.000 respons, SD rating Elo melebar ke 2.69 (SD asli 1.04) dan terus naik; Urnings stabil di 1.11. Kode: `src/simulate_trackers.py`. |
| **Kenapa Urnings, bukan Parallel Elo, yang jadi default** | **Diperbarui 2026-10-19 (lihat §9.18) — Novelty 1 (seed urn dari encoder) sudah dihapus dari paper, jadi alasan di bawah ini DIKOREKSI supaya tidak merujuk ke sesuatu yang sudah tidak ada.** Awalnya kami pertimbangkan Parallel Elo sebagai default (lebih ringan diimplementasikan: tanpa urn ball-exchange + Metropolis-Hastings). Alasan yang TETAP BERLAKU: (1) Novelty "escalation" (sekarang Novelty 2 di paper) dibangun literal di atas lebar urn (`w_k > τ`) — Parallel Elo tidak punya konsep ini. (2) Mekanisme "Select" (§9.15, seleksi adaptif per-siswa — bukan lagi diklaim novelty, tapi tetap fitur aplikasi nyata) juga butuh `item_urn` Urnings untuk menghitung kesulitan item yang dipakai mencocokkan ke kemampuan siswa. (3) Eksperimen §9.13 bahkan menambah bukti BARU: Urnings terbukti aman dipakai dengan prior yang mungkin salah (robust-by-construction lewat `sigma_hat2`), sementara Elo terbukti bisa "meledak" kalau prior-nya salah — jadi walau klaim "speedup" asli (dulu Novelty 1) tidak terbukti, properti *robustness* ini justru alasan baru yang independen untuk tetap Urnings. Pindah ke Parallel Elo tetap berarti kehilangan (1) dan (2) sekaligus properti robustness di (3), demi manfaat kompleksitas implementasi yang untuk kode riset (bukan produk produksi dengan SLA) tidak terlalu mahal — implementasi Urnings sudah ada dan jalan di `src/simulate_trackers.py` (`run_urnings`), ~30 baris. |
| **Posisi Parallel Elo** | Tidak dibuang — didokumentasikan sebagai alternatif "ringan" untuk deployment produksi nanti kalau proyek keluar dari tahap riset (lisensi/SLA/tim engineering kecil jadi pertimbangan saat itu), dan tetap jadi baseline pembanding di evaluasi (§5) supaya kontribusi Urnings terukur, bukan diasumsikan. |
| **Ditolak** | Deep Knowledge Tracing (DKT/SAKT/AKT) — data terlalu kecil, tidak interpretable, tidak perlu untuk novelty riset kita. Cognitive diagnosis model (NeuralCD) — butuh matriks item-skill yang di-supply expert; taksonomi kita (§3.2) justru membuat mapping ini otomatis dari proses generate, jadi CDM jadi tidak perlu. **Glicko-2** — dipertimbangkan serius (lihat `src/simulate_drift.py`): punya rating deviation (RD) native, kontinu, tanpa masalah diskritisasi urn. Menang telak di simulasi ability statis, tapi begitu ability dibuat naik seiring waktu (regime sungguhan untuk drilling app), RD-nya mengecil lalu "terkunci" — makin lama makin tertinggal dari kemampuan asli yang membaik (RMSE akhir 1.05, hampir seburuk Elo polos). Konsekuensi produk konkret: kalau dipakai untuk status mastery (lihat baris **Mastery criterion** di bawah), siswa bisa macet di status "mastered" walau sudah mulai salah lagi — bertentangan langsung dengan fitur *"requeue wrong answer until mastered"* yang mengasumsikan status itu harus bisa dicabut. Tidak ada precedent literatur yang memakai Glicko di regime item-baru-tanpa-kalibrasi ini (lihat `docs/literature-review-15.md` gap G3) — keputusan ini murni dari simulasi sendiri, bukan dari otoritas sumber luar. |
| **Multi-skill** | Satu learner punya **banyak** ability — satu per skill (partikel, honorifik, konjugasi, dst — lihat KAGAS §3.2), bukan satu angka tunggal. Urnings diperluas jadi vektor skill, bukan skalar. |
| **Mastery criterion** | Satu dimensi skill dinyatakan **mastered** kalau `k/n ≥ ambang` (contoh: 0.8) **DAN** lebar urn `w_k < τ` — persis primitive yang sama dengan uncertainty-gated escalation (Novelty 3, §3.3), dipakai ulang untuk kegunaan ketiga. Status ini **harus bisa dicabut**: kalau urn bergerak turun lagi setelah jawaban salah berulang, status mastered hilang otomatis — konsisten dengan fitur *"requeue wrong answer until mastered"*. Ini salah satu alasan Glicko ditolak (lihat baris Ditolak) — urn yang terbatas selalu punya ruang gerak, RD Glicko yang mengecil terus tidak. |
| **Kompetensi per topik (fitur produk)** | `k/n` per skill **langsung** jadi angka "kompetensi %" yang ditampilkan ke admin/guru (fitur *"Track per-user competency in each topic"*) — tidak perlu formula konversi tambahan, beda dari rating Elo/Glicko yang perlu dinormalisasi dulu sebelum bisa ditampilkan sebagai persentase. |

### 2.2 Scheduling (soal apa berikutnya, kapan diulang)

| | |
|---|---|
| **Keputusan** | FSRS (Free Spaced Repetition Scheduler) untuk kosakata/hafalan. Tidak butuh LLM atau deep learning sama sekali. |
| **Alasan** | Sudah matang, open-source, teruji di skala jutaan review (Anki). FSRS-6 dilatih dari ~700 juta review, 17–21 parameter per-kartu — tapi keunggulannya atas SM-2 baru terlihat setelah user terkumpul **ribuan** review, jadi tidak ada untungnya "membangun sendiri" dari nol. |
| **Ditolak** | Model scheduling custom/neural — tidak ada alasan reinvent sesuatu yang sudah solved dengan baik oleh komunitas open-source. |
| **Requeue wrong answer** | Fitur *"requeue wrong answer until mastered"* dipegang FSRS (due-state per item, bukan ability tracker) — item yang dijawab salah ditandai due lagi sampai benar N kali berturut-turut. **Sengaja dipisah** dari Urnings (§2.1): ini state sederhana per (user, item), bukan ability kontinu per skill. Kedua sinyal ini independen tapi bisa saling melengkapi di UI (contoh: badge "mastered" topik dari §2.1, badge "cleared" per item dari sini). |

### 2.3 Item Generation & Feedback (dari mana soal & penjelasan berasal)

Ini **satu-satunya bagian** yang butuh LLM/deep learning secara substansial.

| Peran | Model | Alasan |
|---|---|---|
| Generate soal Korea | **EXAONE-4.0-1.2B** (LG AI Research) — bukan 3.5 | Korean-first, tokenizer lebih efisien untuk Hangul, mode hybrid reasoning/non-reasoning cocok untuk kebijakan eskalasi (§3.3). **Diuji langsung 2026-10-19:** EXAONE-3.5-2.4B-Instruct butuh `trust_remote_code=True`, dan kode custom-nya ternyata rusak di semua versi `transformers` yang dicoba (4.56, 4.57, 5.0, 5.20 — gagal di titik berbeda-beda). EXAONE-4.0-1.2B didukung **native** di `transformers` (arsitektur `exaone4`, tidak perlu remote code), lebih kecil, dan tetap punya hybrid reasoning mode yang sama — upgrade murni, bukan kompromi. Catatan lisensi non-komersial tetap berlaku, fallback Qwen3 tetap relevan kalau deployment komersial. |
| Feedback penjelasan (**ID & EN**) | **Qwen3-4B** (bukan 1.7B) | Menjelaskan gramatika Korea *dalam bahasa lain* itu tugas lintas-bahasa, bukan tugas satu-bahasa — Qwen3 dilatih di 119 bahasa dengan kekuatan terdokumentasi di bahasa CJK, vocab besar. Gemma jadi alternatif kalau butuh model lebih kecil/ada kebutuhan vision. **Diuji langsung 2026-10-19** (`src/qwen_feedback.py`), termasuk satu kesimpulan yang **dibalik**: (1) Qwen3-1.7B diminta jawab Indonesia → jawab Korea, instruksi diabaikan total. (2) Diperbaiki dengan one-shot contoh Indonesia → bahasa terikuti 4/4, TAPI 3/4 **membocorkan topik contoh** ke jawaban (menjelaskan kalimat makan seolah soal belajar) — lebih berbahaya dari gagal total karena terlihat meyakinkan. (3) Dicoba generate-English-lalu-translate → gagal berbeda dan lebih parah: terjemahan **membalik klaim benar/salah**, partikel ikut diterjemahkan jadi "dari", separuh isi hilang. (4) Sempat dicatat "Indonesia dieliminasi, English-only final" — **keputusan ini dibalik setelah ditantang**: variabel yang belum diuji adalah ukuran model, bukan kelayakan tugasnya. **Qwen3-4B** dengan teknik sama (one-shot, contoh didiversifikasi 2 topik berbeda biar tidak overfit ke satu topik) → 4/4 akurat, tidak ada kebocoran topik, di jalur cepat MAUPUN jalur detail (reasoning), ~8GB VRAM (masih longgar dari 16GB). (5) Terpisah dari itu: reasoning mode di `max_new_tokens=500` **kehabisan budget di tengah `<think>`, nol output** — bukan kualitas lebih buruk, tidak ada jawaban sama sekali; `1200` baru selesai reliable. (6) Residual baru ditemukan sambil cek kualitas: contoh ilustrasi yang ditulis Qwen sendiri di jalur EN-detail mengandung salah konjugasi (`들어요`→`듣어요`) — beda pipeline dari fix konjugasi EXAONE, belum diperbaiki. (7) Simplifikasi dari rencana §3.3 asli tetap berlaku: implementasi memakai Qwen3 untuk **kedua** jalur, bukan EXAONE-non-reasoning untuk jalur cepat seperti tertulis di §3.3. **Pelajaran umum:** kesimpulan "tidak bisa" dari satu ukuran model itu terikat ke ukuran model itu, bukan ke kelayakan kemampuannya — coba ukuran lebih besar sebelum menyatakan sesuatu tidak mungkin. |
| Prediksi kesulitan soal baru | **KLUE-RoBERTa** (encoder kecil, fine-tuned) | **Bukan** model besar — riset (Li et al. 2025, EPM) membuktikan BERT/RoBERTa fine-tuned justru mengalahkan LLM jauh lebih besar untuk tugas regresi kesulitan spesifik ini. Encoder murah dijalankan real-time per soal. Harus keluarkan **mean + variance**, bukan titik estimasi tunggal — variance ini jadi bobot prior di urn Urnings (lihat Novelty 1, §3.1). |

**Prinsip pembagian kerja:** model neural/LLM *menggenerate dan merasakan*
(teks, bahasa); model statistik eksplisit *mengingat dan memutuskan*
(kemampuan siswa). LLM besar tidak menggantikan model psikometrik — LLM
melayani model psikometrik (lewat prior kesulitan).

### 2.4 Gamification Layer (EXP & Leaderboard)

| | |
|---|---|
| **Keputusan** | EXP dan leaderboard (fitur *"View quiz leaderboard: EXP rank within topic"*, *"Track daily streak"*) adalah **counter aditif sederhana** (+poin per jawaban benar, bonus streak) — **sama sekali terpisah** dari rating Urnings di §2.1. |
| **Kenapa dipisah, bukan reuse rating tracker** | (a) Rating psikometrik (urn `k/n`) bisa dan memang **harus** bisa turun kalau performa menurun — itu sifat yang benar untuk model ability. Tapi EXP yang terlihat turun di leaderboard adalah UX yang buruk (pola yang dihindari semua sistem gamifikasi besar — Duolingo XP tidak pernah berkurang). (b) Karena dipisah, rating tracker internal tidak pernah ditampilkan langsung ke user — menghapus kebutuhan rating itu "mudah dipahami user" sebagai kriteria pemilihan algoritma. Keputusan tracker di §2.1 jadi murni keputusan backend. |
| **Implikasi** | User tidak pernah melihat angka urn/rating mentah. Yang ditampilkan: EXP (akumulatif, §2.4), status mastery per topik (dari §2.1, ditampilkan sebagai badge/persen, bukan angka rating), dan streak harian (counter hari berturut-turut, independen dari keduanya). |

---

## 3. Tiga Novelty Riset — INI YANG HARUS DIPERTAHANKAN

Semua keputusan di atas cuma infrastruktur. Novelty riset ada di 3 hal ini
saja. Kalau harus memangkas scope, pangkas apa saja **selain** tiga ini.

### 3.1 Novelty 1 — Generate–Calibrate–Select Loop — **DIHAPUS dari paper 2026-10-19, lihat §9.18**

**Status: tidak lagi diklaim sebagai novelty di paper.** Bagian ini
dibiarkan apa adanya di bawah sebagai **catatan historis** (kenapa
desain ini pernah diputuskan, apa yang terjadi) — bukan desain yang
masih berlaku. Ringkasan singkat (detail lengkap di §9.18): klaim inti
("prior dari encoder mempercepat konvergensi") diuji langsung (§9.13)
dan **tidak terbukti** untuk Urnings di semua regime yang diuji; bagian
"Calibrate" dihapus dari paper sepenuhnya, bagian "Select" (seleksi
adaptif per-siswa) **tetap ada sebagai fitur aplikasi** (§9.15) tapi
TIDAK lagi diklaim sebagai novelty riset — `difficulty_heuristic.py`,
`seed_item_from_encoder()`, dan seleksi adaptif di `BankContentProvider`
semuanya tetap berjalan di kode, cuma framing "novelty"-nya yang
dicabut.

**PENTING — mismatch penomoran:** paper sekarang cuma punya **2
novelty** (apa yang di PRD ini disebut Novelty 2 dan Novelty 3, lihat
§3.2/§3.3, sekarang jadi "Novelty 1" dan "Novelty 2" di paper). PRD ini
**SENGAJA TIDAK direnumber** mengikuti paper — penomoran §3.1/§3.2/§3.3
di PRD dipertahankan apa adanya untuk kontinuitas riwayat (semua audit
note di atas sudah menulis "Novelty 1/2/3" sesuai skema lama, me-renumber
berarti mengedit ulang riwayat). Kalau butuh tahu nomor novelty yang
BENAR di paper saat ini, selalu rujuk ke paper (`src/content.py`)
langsung, bukan ke nomor di PRD ini.

**Deskripsi desain asli (historis, tidak lagi berlaku):**

**Masalah yang dijawab:** Soal baru dari LLM tidak punya data kalibrasi, jadi
tidak bisa dipakai adaptif tanpa menunggu puluhan respons dulu.

**Solusi:** Encoder KLUE-RoBERTa memprediksi (μ̂, σ̂²) dari teks soal →
langsung dikonversi jadi komposisi & ukuran urn awal di Urnings:
prediksi kesulitan jadi titik rating awal, prediksi variance menentukan
"bobot" prior itu (makin tidak yakin encoder, makin cepat prior itu
ditimpa oleh data respons asli). Soal baru langsung bisa dipakai adaptif
sejak detik pertama, bukan setelah fase pra-tes.

**Bukti (dikumpulkan 2026-10-19, lihat §9.13 untuk detail lengkap):**
klaim awal ("encoder prior konvergen lebih cepat ke rating sebenarnya
dibanding prior seragam") **TIDAK terbukti untuk Urnings** di 2 regime
simulasi yang diuji (pool tetap maupun pool tumbuh) — semua level
kualitas prior (korelasi 0.47–0.96 terhadap beta sebenarnya) berakhir
dalam rentang noise yang sama dengan baseline netral. Perbandingan
dengan Elo/Parallel Elo (regime sama, prior di-inject langsung tanpa
transformasi Urnings) justru menunjukkan arah SEBALIKNYA dari yang
diharapkan: Elo MEMANG menunjukkan speedup jelas untuk prior berkualitas
tinggi, tapi RMSE akhirnya MELEDAK (3–6x lebih buruk dari netral) kalau
prior-nya ternyata kurang akurat — sementara Urnings tidak meledak sama
sekali di kondisi manapun, karena `sigma_hat2` secara eksplisit
membatasi seberapa besar pengaruh prior.

**Klaim yang didukung bukti bukan "konvergensi lebih cepat", tapi
"robust terhadap prior yang mungkin salah"** — mengingat
`difficulty_heuristic.py` sendiri eksplisit tidak terkalibrasi, sifat
robust ini justru lebih relevan/berharga untuk kondisi nyata sistem
ini daripada klaim speedup yang mengasumsikan prior sudah akurat.
Bagian V-B paper (`src/content.py`) sudah ditambah catatan yang sama.

### 3.2 Novelty 2 — Misconception-Conditioned Diagnostic Distractors

**Masalah yang dijawab:** Jawaban salah di pilihan ganda biasanya cuma
"masuk akal", tidak informatif — sistem cuma tahu siswa salah, bukan salah
kenapa.

**Solusi:** Taksonomi kesalahan Korea **KAGAS** (14 tipe error, dari ACL 2023,
Korean Automatic Grammatical error Annotation System — bukan taksonomi ad hoc
buatan sendiri) dipakai sebagai label. LLM diminta generate distractor yang
*merepresentasikan* satu tipe error spesifik dari KAGAS, ditandai label itu.
Memilih distractor itu = bukti terhadap skill spesifik itu saja → update
hanya dimensi itu di vektor ability siswa (§2.1).

**Validator wajib** (bukan opsional): morphological analyser Korea konfirmasi
distractor memang merepresentasikan error yang dimaksud DAN bukan jawaban
yang kebetulan juga benar (ini kegagalan paling umum distractor hasil LLM
untuk bahasa Korea — banyak substitusi partikel yang "salah" di satu konteks
ternyata benar di konteks lain).

**Bukti nyata, bukan cuma antisipasi (2026-10-19):** diminta membuat soal
partikel 에 vs 에서, EXAONE-4.0-1.2B menghasilkan satu distractor yang
ternyata memakai `와` (partikel penghubung "dan", tag `JC`) — bukan partikel
lokatif sama sekali, menguji poin tata bahasa yang tidak relevan, walau
diminta eksplisit "minimal 1 distractor merepresentasikan kebingungan
에/에서". Model juga salah menyebut `에` sebagai "목적격 조사" (partikel
akusatif) di penjelasannya sendiri — padahal itu partikel lokatif/datif.
Kedua kesalahan ini **tidak akan tertangkap** kalau sistem percaya ke
self-report LLM. Validator (`src/distractor_validator.py`, berbasis
`kiwipiepy`) terbukti menangkap kasus ini secara otomatis — lihat smoke
test di file itu, direduksi dari kegagalan nyata di atas, bukan skenario
rekaan.

**Generator → Validator → Critic:** bukan cuma accept/reject (verifier gate),
tambah lapisan critic (LLM-as-judge) yang bisa minta revisi iteratif,
bukan langsung buang soal yang gagal validasi.

**Bukti dibutuhkan (minimal):** agreement rater manusia vs label KAGAS
otomatis pada subset soal; cek tingkat soal dengan >1 jawaban benar
(failure mode paling berbahaya).

### 3.3 Novelty 3 — Uncertainty-Gated Escalation

**Masalah yang dijawab:** Feedback LLM yang detail itu mahal secara komputasi
DAN hanya efektif untuk sebagian siswa/situasi (literatur: Lin et al. 2026,
signifikan di 2 dari 7 lesson saja). Generate feedback detail untuk semua
orang = boros dan tidak proporsional manfaatnya.

**Solusi:** Tiap dimensi skill siswa punya lebar urn (= ketidakpastian
pengukuran). Default: jalur cepat (EXAONE non-reasoning, feedback singkat
di-cache). Eskalasi ke jalur mahal (EXAONE reasoning mode + feedback Qwen3
kontrastif menyebut misconception spesifik) HANYA kalau ketidakpastian
tinggi ATAU misconception yang sama berulang.

**Bukti dibutuhkan (minimal):** korelasi antara trigger eskalasi dan
perbaikan pengukuran/pembelajaran nyata; perbandingan biaya compute dengan
skenario "selalu eskalasi".

**Reuse di luar novelty ini:** primitive yang sama (`w_k`, lebar urn) juga
dipakai untuk kriteria mastery per topik (§2.1) — satu mekanisme, dua
kegunaan: `w_k > τ` memicu eskalasi feedback, `w_k < τ` (dikombinasi `k/n`
tinggi) menyatakan topik mastered. Ini salah satu alasan konkret Urnings
dipertahankan dibanding Glicko-2 — lihat baris "Ditolak" di §2.1.

---

## 4. Yang SENGAJA Tidak Dikerjakan (Out of Scope)

Karena ini riset dengan sumber daya terbatas, bukan produk produksi — daftar
ini sama pentingnya dengan daftar yang dikerjakan.

- ❌ **Deep Knowledge Tracing (DKT/SAKT/AKT)** sebagai model utama — bukan
  pertanyaan riset kita, dan datanya memang tidak cukup untuk itu bekerja
  baik.
- ❌ **Grid ablasi besar** (selection regime × kalibrasi item × tracker × urn
  size × 20 replikasi/sel) — ini berlebihan untuk klaim yang perlu
  dibuktikan. Cukup satu perbandingan terarah per novelty (§3), **bukan**
  faktorial penuh.
- ❌ **RCT 3-arm 180 pelajar sebagai syarat wajib** — ini skala yang cocok
  untuk *jika* proyek berkembang jadi evaluasi efektivitas pembelajaran
  penuh, tapi bukan syarat untuk membuktikan 3 novelty di atas valid secara
  teknis. Evaluasi teknis (offline) harus bisa berdiri sendiri tanpa
  menunggu infrastruktur RCT selesai.
- ❌ **Model scheduler/Birdbrain-style custom** — FSRS sudah cukup, sudah
  dibahas §2.2.
- ❌ **Fine-tuning ulang EXAONE/Qwen3** — pakai as-is dengan prompting +
  constrained generation dulu; fine-tuning cuma dipertimbangkan kalau ada
  bukti kuat prompting tidak cukup.
- ❌ **Mengejar arsitektur "paling canggih"** — bukan kriteria sukses riset
  ini. Kriteria sukses: integrasi KT ↔ LLM ↔ validator Korea-spesifik
  terbukti bekerja, dan 3 novelty di atas terbukti valid.

---

## 5. Rencana Validasi Minimal (Pengganti Ablasi Besar)

Per novelty, satu bukti terarah — bukan faktorial penuh:

| Novelty | Yang dibuktikan | Metode minimal |
|---|---|---|
| 1 — Generate-Calibrate-Select | Prior encoder mempercepat konvergensi rating | Simulasi: urn dengan prior encoder vs urn tanpa prior (uninformative), ukur jumlah respons sampai presisi target tercapai |
| 2 — Misconception distractors | Distractor benar-benar mendiagnosis, bukan cuma "masuk akal" | Agreement instruktur vs label KAGAS otomatis (subset soal); % soal dengan >1 jawaban benar sebelum vs sesudah verifier |
| 3 — Uncertainty-gated escalation | Compute terpakai tepat sasaran | Korelasi lebar-urn dengan perbaikan estimasi; perbandingan biaya vs "selalu eskalasi" |
| Tracker (infrastruktur, bukan novelty) | Urnings terbukti paling stabil secara simulasi; Elo dan Parallel Elo jadi baseline pembanding, bukan kandidat default | Simulasi sudah ada & selesai (`src/simulate_trackers.py`) — tetap dipertahankan sebagai bukti pendukung, bukan diperluas |

**Yang TIDAK perlu:** replikasi 20x per sel, factorial penuh atas urn
size/selection regime/tracker/kalibrasi. Satu simulasi terkontrol sudah
cukup untuk klaim yang kita buat.

---

## 6. Posisi Terhadap Sistem Industri (Referensi Cepat)

| Sistem | Pendekatan | Kenapa tidak kita tiru/kenapa kita tiru sebagian |
|---|---|---|
| Duolingo (app) | HLR → Birdbrain (LSTM) | Butuh skala data 5-6 orde lebih besar; kita tiru filosofi scheduling-nya (FSRS), bukan model beratnya |
| Duolingo English Test | IRT klasik + CAT | Soal sudah dikalibrasi sebelum deploy — prekondisi yang tidak kita punya |
| ALEKS | Knowledge Space Theory | Paradigma berbeda total, dibangun manusia bertahun-tahun; bukti bahwa model ringan bukan kompromi |
| Math Garden | Elo (Klinkenberg 2011) | Preseden langsung untuk learner model kita, tapi mereka sengaja selang-seling item acak untuk hindari feedback-loop Elo — kita tidak bisa karena soal selalu baru |
| Anki | FSRS | Ditiru langsung untuk scheduling, tanpa modifikasi |

---

## 7. Pustaka Kunci per Keputusan (Quick Reference)

- **Elo gagal di seleksi-adaptif+kalibrasi-dinamis:** Bolsinova, Gergely, Brinkhuis (2026), *BJMSP* 79(1), DOI 10.1111/bmsp.12395
- **Urnings algorithm:** Bolsinova et al. (2022), *BJMSP* 75(3), DOI 10.1111/bmsp.12276
- **Encoder kecil > LLM besar untuk prediksi kesulitan:** Li et al. (2025), *Educational and Psychological Measurement* 85(6), DOI 10.1177/00131644251344973
- **KAGAS (14 tipe error Korea):** ACL 2023, "Towards Standardizing Korean Grammatical Error Correction"
- **HLR (dasar FSRS/Duolingo):** Settles & Meeder (2016), ACL, DOI 10.18653/v1/P16-1174
- **FSRS:** github.com/open-spaced-repetition
- **Math Garden / Elo pendidikan:** Klinkenberg, Straatemeier, van der Maas (2011), *Computers & Education* 57(2), DOI 10.1016/j.compedu.2011.02.003
- **ALEKS / Knowledge Space Theory:** Cosyn, Uzun, Doble, Matayoshi (2021), *J. Mathematical Psychology* 101, DOI 10.1016/j.jmp.2021.102512
- **Feedback LLM tidak selalu efektif:** Lin et al. (2026), *J. Computer Assisted Learning*, DOI 10.1002/jcal.70343

Daftar lengkap 41 referensi: `paper/KO-DRILL_IEEE.docx` → References.

---

## 8. Pertanyaan Terbuka (Belum Diputuskan)

- [x] ~~Parallel Elo vs Urnings~~ — **DIPUTUSKAN FINAL (menggantikan keputusan awal):** Urnings adalah tracker utama sistem, konsisten dengan paper (Section III-B, Fig. 1, Fig. 2, persamaan 3–5). Parallel Elo **bukan** default produksi — didemosikan jadi baseline pembanding di evaluasi (§5) dan alternatif ringan yang didokumentasikan untuk deployment produksi nanti. Alasan pembalikan dari keputusan awal: ditemukan bahwa bukan cuma Novelty 1 (persamaan 3–4), tapi juga Novelty 3 (persamaan 5, `w_k` = lebar urn) sama-sama dibangun di atas formalisme urn Urnings secara spesifik — pindah ke Parallel Elo berarti menurunkan ulang matematika dua novelty inti sekaligus, ongkos yang untuk kode riset (bukan produk produksi dengan SLA ketat) tidak sepadan dengan manfaat kompleksitas implementasi yang dihindari. PRD §2.1 dan paper sekarang konsisten satu arah.
- [ ] KAGAS 14 tipe — apakah dipakai utuh, atau disaring ke subset yang
      relevan untuk level TOPIK I–IV target kita?
- [ ] Skala validasi manusia — pilot kecil dulu (berapa orang?) sebelum
      mempertimbangkan RCT skala penuh?
- [ ] EXAONE lisensi non-komersial — perlu diputuskan sejak awal apakah
      proyek ini riset murni (OK) atau ada rencana komersialisasi (perlu
      rencana migrasi ke Qwen3 generator sejak sekarang).

---

## 9. Status Implementasi & Scope Produksi (2026-10-19)

**Tujuan proyek dikonfirmasi ulang:** production-grade, bukan sekadar
prototipe riset — metode yang dipakai harus yang paling tepat secara
teknis, bukan yang "cukup jalan". Implementasi dilanjutkan di cloud
session berikutnya berdasarkan dokumen ini.

### 9.1 Audit metode final — tidak ada yang berubah sejak §2–§3

Tidak ada metode yang statusnya terbalik sejak audit-audit sebelumnya di
dokumen ini. Ringkasan akhir, jujur soal level keyakinan (bukan klaim
100% di mana pun kecuali satu baris pertama):

| Metode | Status | Level keyakinan |
|---|---|---|
| Aturan batchim Hangul (`distractor_validator.has_batchim`) | ✅ | Mendekati 100% — fakta encoding Unicode, bukan model |
| Urnings (tracker ability) | ✅ | Tinggi — 2 simulasi independen (statis + drift) + alasan produk; bukan data pelajar sungguhan |
| Partikel & konjugasi EXAONE (deterministik) | ✅ | Sangat tinggi secara struktural (dijamin benar by construction), diuji di sampel kecil (5-10 kasus) |
| `correct_korean_conjugation()` | ✅ (sempit) | Menangkap satu pola error spesifik yang sudah ditemukan, bukan pengecek gramatikal umum |
| FSRS | ✅ | Tinggi untuk domain spaced-repetition umum; belum diuji dengan data pengguna aplikasi ini |
| kiwipiepy | ✅ | Tinggi, tapi terbukti punya gap (itulah alasan validator di atas dibangun) |
| Qwen3-4B (feedback, EN+ID, 2 jalur) | ✅ | Diuji 4/4 per kasus di beberapa konfigurasi — sampel kecil |
| `difficulty_heuristic.py` | ✅ sebagai proxy | Diklaim lemah secara eksplisit — urutan benar, akurasi absolut tidak diklaim |
| `MASTERED_RECHECK_DAYS=30` | ⚠️ | Satu-satunya yang genuinely menunggu data pemakaian nyata — sudah dicoba diturunkan dari matematika FSRS, gagal secara prinsip (bukan teknis), instrumentasi (`RecheckOutcomeLog`) sudah terpasang untuk tuning nanti |

Confidence di atas baris batchim HANYA bisa naik lewat data skala
produksi nyata, bukan lewat audit kode lebih lanjut — titik ini sudah
mencapai batas yang bisa dibuktikan lewat investigasi dan pengujian kode.

### 9.2 Modul aplikasi — lengkap untuk satu siklus penuh

Ditemukan dan ditutup satu gap nyata: `daily_drill.py` menyusun soal
tapi **tidak pernah ada logika memproses jawaban** — separuh siklus
aplikasi. Dibangun `src/answer_processing.py`:
- `submit_grammar_answer()` — update urn skill pelajar (Urnings), cek
  eskalasi (`should_escalate`), panggil `qwen_feedback` jalur
  cepat/detail sesuai hasil cek, catat ke `practice_log`.
- `submit_vocab_answer()` — update kartu FSRS.
- `complete_drill_session()` — update streak, dipanggil sekali per hari
  (bukan per soal).

Keputusan desain: **urn item tidak dipersist.** Item dari
`exaone_provider` dibuat unik per kejadian (tidak dipakai ulang antar
pelajar atau antar sesi), jadi tidak ada eksposur berulang untuk urn
item mengakumulasi bukti — beda dari urn SKILL pelajar yang memang
terakumulasi lintas semua item skill itu. Urn item netral/sekali-pakai
dipakai hanya supaya matematika `update()` yang simetris punya dua sisi
untuk dioperasikan; hanya hasil sisi pelajar yang disimpan.

Smoke test `answer_processing.py` langsung menangkap bug nyata saat
pertama ditulis: feedback salah menyebut label opsi (`"A"`) bukan teks
partikel sungguhan (`"에"`) — ditemukan dan diperbaiki lewat pengujian,
bukan diasumsikan benar dari desain.

### 9.3 Infra produksi — keputusan dicatat, implementasi DITUNDA TANPA BATAS WAKTU

Keputusan infra berikut **sudah diambil** dan tidak perlu didebat ulang
kapan pun infra dikerjakan, tapi **fokus sekarang dan di cloud session
berikutnya tetap murni application layer** — infra bukan "next", infra
ditunda sampai diminta eksplisit:

| Keputusan | Pilihan |
|---|---|
| Database | PostgreSQL |
| Serving model EXAONE/Qwen3 | Dedicated inference server (vLLM — dikonfirmasi mendukung `exaone4`/`qwen3` native di registry-nya) |
| Target deployment | Satu VM cloud dengan GPU, skala 50-180 pengguna |

### 9.4 Application layer — yang sudah ada vs yang masih kosong

**Sudah ada dan teruji** (sisi pelajar, end-to-end): tracker (`urnings_tracker.py`),
penjadwalan vocab+grammar (`vocab_scheduler.py`, `grammar_scheduler.py`),
perakitan drill harian (`daily_drill.py`), proses jawaban (`answer_processing.py`),
generasi soal (`exaone_provider.py`, `vocab_provider.py`, `content_provider.py`),
feedback (`qwen_feedback.py`), validator (`distractor_validator.py`), proxy
kesulitan (`difficulty_heuristic.py`), gamifikasi dasar (`streak.py`).

**Masih kosong — sisi admin/LMS**: *Maintain topic/subject*, *Maintain
curriculum per subject* (upload kurikulum), *Set batch publish time*,
*Set number of questions*, *Set question type*, *Auto-tagging topic*
(soal baru ditandai topiknya otomatis). Semua ini logika aplikasi murni
(CRUD + workflow), belum ada satu baris kode pun — ini kandidat kerja
berikutnya yang paling jelas, BUKAN infra.
(*Review topic & Question Bank* — approve/edit/drop soal dari bank —
**sudah tidak kosong lagi**, lihat §9.7.)

**Juga belum ada:** logika EXP/leaderboard (§2.4 sudah mendesain
pemisahannya dari tracker, tapi akumulasi EXP dan ranking per-topik belum
diimplementasikan sebagai modul), dan skill grammar KAGAS lain di luar
`particle_loc`/`particle_obj`/`time_range` (kode `PART`) dan
`honorific`/`tense_past` (kode `END`) — lihat pemetaan resmi dan kode
yang masih kosong (`VERB`, `ADJ`, `NOUN`, `MOD`, `CONJ`) di §9.8.

### 9.5 Skill ketiga: `honorific` — kategori gramatikal berbeda struktur, bukan novelty baru

Ditambahkan skill `honorific` (`-(으)세요` vs bentuk polos) ke
`exaone_provider.py`. **Bukan novelty ke-4** — tetap instance dari
Novelty 2, cuma bukti tambahan bahwa pendekatan Generator→Validator→
koreksi-deterministik general ke kategori gramatikal yang **secara
struktural beda** (infleksi morfologis honorifik, bukan substitusi
partikel seperti dua skill sebelumnya) — ini poin evaluasi yang sah
ditambahkan di Section V paper, bukan novelty terpisah.

**Dua temuan nyata selama membangunnya:**

1. **`kiwi.join([(stem,'VV'),('시','EP'),('어요','EF')])` menghasilkan
   `셔요` (kontraksi arkais/formal), bukan `세요` yang jadi standar modern
   di materi TOPIK.** `kiwipiepy` sebenarnya menyimpan `세요`/`으세요`
   sebagai morfem EF tunggal yang sudah terfusi, bukan EP+EF terurai —
   begitu dipakai langsung (`honorific_ending()`, aturan batchim yang
   sama dengan `expected_object_particle`), hasilnya benar
   (`읽으세요`, `가세요`, `가르치세요`, dst). Pelajaran: alat yang "benar
   secara gramatikal" (kiwi.join menghasilkan bentuk yang valid) belum
   tentu "benar secara konvensi pembelajaran" — perlu dicek eksplisit
   terhadap standar yang diajarkan, bukan cuma terhadap gramatikalitas.

2. **`validate_cloze_option` gagal total (0/5) untuk skill ini** — fungsi
   itu mengecek apakah SATU morfem (token tunggal) sama dengan teks opsi;
   opsi di skill ini adalah kata kerja terkonjugasi utuh (multi-morfem:
   stem+EF), jadi tidak pernah ada satu token pun yang form-nya persis
   sama dengan string penuh opsi. Dibangun `validate_sentence_final_tag()`
   di `distractor_validator.py` — cek tag morfem terakhir kalimat
   (sebelum SF), bukan cocok-string literal. Setelah fix: 5/5 berhasil,
   semua jawaban benar menyebut bentuk honorifik (`가르치세요`,
   `연구하세요`, `쉬세요`), tidak ada yang tertukar dengan bentuk polos.

### 9.6 Skill keempat & kelima: `time_range`, `tense_past`

- **`time_range`** (`부터` vs `까지`, keduanya tag `JX`). Ditemukan dulu:
  `"[waktu]___"` telanjang itu ambigu (bisa jadi awal ATAU akhir rentang)
  tanpa konteks — jebakan yang sama dengan 이/가 vs 은/는 yang sudah
  ditolak sebelumnya. Diperbaiki dengan **template berpasangan**
  (`"...까지"` sudah fixed di kalimat), memaksa `부터` jadi satu-satunya
  jawaban benar secara struktural. 5/5 berhasil.
- **`tense_past`** (lampau `-았/었-` vs sekarang, dijangkar kata `어제`).
  Struktur sama dengan `honorific` (blank mencakup kata kerja utuh) —
  dipakai ulang lewat `_build_whole_word_item()` (bagian tail yang
  memang identik digeneralisasi; bagian penentuan bentuk per skill
  tetap dipisah, `_build_verb_ending_item` vs `_build_tense_item`,
  supaya tidak mengulang insiden generalisasi-salah sebelumnya).
  Ditemukan `kiwi.join()` otomatis menangani harmoni vokal `았/었` dan
  bentuk tak-beraturan (`가다→갔어요`, `하다→했어요`) walau cuma diberi
  `었` seragam — tidak perlu logika pemilihan alomorf manual seperti di
  `을`/`를`. 5/5 berhasil.

Total sekarang: **5 skill grammar** (`particle_loc`, `particle_obj`,
`honorific`, `time_range`, `tense_past`), semua lolos regresi bersama.

### 9.7 Question bank — memisahkan GENERATE dari SERVE

**Masalah yang diangkat:** soal selalu "mulai dari 0" karena
`exaone_provider.py` dipanggil langsung tiap kali sistem butuh soal
(live, per-request) — setiap permintaan siswa = satu panggilan GPU,
dengan risiko gagal/lambat, dan tidak ada bank soal yang tersimpan.

**Solusi:** `src/question_bank.py` — `QuestionBank` (batch-generate
sejumlah soal via `exaone_provider`, simpan sebagai `PENDING`;
`approve()`/`reject()`/`edit()` untuk workflow admin — persis fitur
*"Review topic & Question Bank"* yang sebelumnya kosong di §9.4) dan
`BankContentProvider` (implementasi `ContentProvider` yang serve HANYA
dari entry `APPROVED` — tidak ada panggilan LLM sama sekali di jalur
serving). Diuji end-to-end: `assemble_daily_drill()` jalan sempurna
tanpa satu pun panggilan GPU saat perakitan drill, karena soal sudah
di-generate+approve sebelumnya secara batch.

Penyimpanan masih in-memory (dict + dataclass, pola yang sama seperti
`urnings_tracker.py`/`streak.py`) — bisa ditukar ke database sungguhan
nanti tanpa mengubah interface, konsisten dengan keputusan infra
ditunda (§9.3).

### 9.8 Pemetaan resmi ke 14 kategori KAGAS — SELESAI, dan mengoreksi asumsi di §9.4/paper

Diselesaikan dengan membaca isi Table 3 dari paper aslinya (Yoon et al.,
ACL 2023, arXiv 2210.14389) langsung — bukan dari web search sekunder,
dan bukan cuma judul/abstrak. 14 kode resmi KAGAS beserta deskripsi
satu-baris dari paper:

| Kode | Deskripsi (dari Table 3) |
|---|---|
| INS | Kata disisipkan |
| DEL | Kata dihapus |
| WS | Spasi antar kata berubah |
| WO | Urutan kata berubah |
| SPELL | Kesalahan ejaan |
| PUNCT | Kesalahan tanda baca |
| SHORT | Edit yang tidak mengubah struktur morfem (kontraksi, mis. `언어이었어요`→`언어였어요`) |
| VERB | Kesalahan pada kata kerja |
| ADJ | Kesalahan pada kata sifat |
| NOUN | Kesalahan pada kata benda |
| PART | Kesalahan pada partikel |
| END | Kesalahan pada akhiran |
| MOD | Kesalahan pada modifier |
| CONJ | Kesalahan pada konjugasi |

**Koreksi penting terhadap §9.4 dan caption Fig. 3 di paper:** 14 kode
ini adalah skema anotasi-edit untuk korpus GEC — 7 kode operasi-edit
(INS/DEL/WS/WO/SPELL/PUNCT/SHORT) + 7 kode kelas-POS
(VERB/ADJ/NOUN/PART/END/MOD/CONJ) — **BUKAN** taksonomi miskonsepsi
granular seperti "topic/nominative swap", "object-particle omission",
"honorific agreement", "irregular-stem over-regularization" yang
dipakai sebagai label m₁–m₄ di caption Fig. 3 paper. Istilah-istilah
itu bukan nama kategori resmi KAGAS; dikarang sendiri lalu ditempel
seolah berasal dari situ. **Ini perlu dikoreksi di paper** (Section
IV-C / Fig. 3 caption) — ditandai sebagai utang akademik, terpisah
dari kerja aplikasi di bawah.

Dari 14 kode, hanya **7 kode kelas-POS yang relevan sebagai skill
drill-quiz** — 7 kode operasi-edit (INS/DEL/WS/WO/SPELL/PUNCT/SHORT)
adalah mekanik anotasi korpus, bukan grammar point yang bisa diajarkan
lewat satu soal cloze/MC.

**Pemetaan 5 skill yang sudah ada** (confidence tinggi untuk kode+
deskripsi resmi di atas; confidence sedang untuk pemetaan skill→kode
berikut — judgment call dari deskripsi satu-baris, bukan dicocokkan ke
contoh resmi satu-per-satu):
- `particle_loc`, `particle_obj`, `time_range` → **PART** (ketiganya
  kesalahan partikel/posisi — `time_range` sudah ditandai tag `JX` di
  §9.6, konsisten partikel)
- `honorific`, `tense_past` → **END**, bukan CONJ seperti asumsi awal —
  keduanya soal pemilihan *akhiran* kalimat (honorifik vs polos; lampau
  vs sekarang), bukan pemilihan kata kerja/konjugasi itu sendiri

**Status sebenarnya: 5 skill menutupi 2 dari 7 kode kelas-POS yang
relevan (PART, END)** — bukan "2 dari 14" seperti klaim lama di §9.4
(baris itu sudah diperbaiki juga). Lima kode relevan yang masih kosong:
**VERB, ADJ, NOUN, MOD, CONJ** — kerangka ini yang dipakai untuk
memutuskan skill berikutnya, bukan sekadar "kesalahan umum yang
dikenal" tanpa struktur.

### 9.9 Skill keenam: `irregular_conj` — kategori CONJ resmi, bukan asumsi lagi

Ditambahkan skill ke-6, **kategori resmi pertama yang ditarget dari
kerangka terstruktur §9.8** (bukan "kesalahan umum yang dikenal" tanpa
pemetaan, seperti 5 skill sebelumnya). Target: **CONJ** — verba
tak-beraturan `르` (자르다/모르다/부르다), contoh resmi KAGAS Table 3:
`잘르러→자르러`.

**Tidak perlu LLM sama sekali untuk skill ini** — verba tak-beraturannya
sendiri sudah jadi closed set yang dikurasi (3 verba), tidak ada
"pemilihan kosakata" yang perlu digenerate.

**Temuan kunci:** `kiwi.join()` SELALU menghasilkan kontraksi yang
BENAR, apa pun varian ending yang diberikan (`자르+아요→잘라요`,
`부르+아요→불러요` — otomatis menormalkan harmoni vokal juga) — artinya
`kiwi.join()` tidak bisa dipakai membangun bentuk SALAH (distractor).
Dibangun `harmony_positive()` baru di `distractor_validator.py` —
aturan harmoni vokal berbasis dekomposisi Unicode (vokal `ㅏ`/`ㅗ` →
`-아요`, selain itu → `-어요`), dipakai untuk menyambung literal stem
tak-terkontraksi + ending yang benar harmoninya tapi SALAH karena tidak
menerapkan kontraksi tak-beraturan (`자르아요`, bukan `잘라요`) — pola
error pembelajar yang nyata dan umum.

Hasil: 5/5 berhasil (`불러요`/`부르어요`, `몰라요`/`모르어요`,
`잘라요`/`자르어요`). Total sekarang **6 skill grammar**, menutupi
**3 dari 7 kode kelas-POS relevan** (PART, END, CONJ) — sisa: **VERB,
ADJ, NOUN, MOD**.

### 9.10 Skill ketujuh: `adj_relativizer` — kategori ADJ, lalu KEPUTUSAN: berhenti mengejar cakupan 7 kode penuh

Ditambahkan skill ke-7, target: **ADJ** — akhiran relativiser
`-(으)ㄴ` pada adjektiva (형용사), mis. `맵다→매운` (bukan `*맵는`),
dikontraskan dengan kesalahan umum pembelajar: memakai akhiran
bergaya-verba `-는` pada stem adjektiva.

**Pertama kali blank-nya BUKAN di akhir kalimat** — semua skill
sebelumnya (`honorific`, `tense_past`) punya blank di akhir kalimat
(`validate_sentence_final_tag` cukup). Di sini blank ada di TENGAH
kalimat, mengubah kata benda yang mengikutinya (`"___ 음식을
좋아해요"`) — butuh validator baru, `validate_prenominal_tag()` di
`distractor_validator.py` (cek tag morfem tepat sebelum kata benda
tertentu, bukan tag morfem akhir kalimat).

**Bug nyata ditemukan sebelum kode final** (diuji empiris langsung di
kiwipiepy terinstal, bukan diasumsikan dari dokumentasi): akhiran
`-(으)ㄴ` pada stem tanpa batchim direpresentasikan kiwipiepy sebagai
**karakter jongseong gabung** (U+11AB, tercetak `ᆫ`), BUKAN huruf
`ㄴ` mandiri (U+3134) yang terlihat identik secara visual tapi
codepoint berbeda. Percobaan pertama pakai `ㄴ` mandiri menghasilkan
`kiwi.join()` yang diam-diam GAGAL — bukan error, tapi sekadar
menempel dua karakter jadi string rusak (`"크ㄴ"`, bukan `"큰"`).
Baru ketahuan setelah membandingkan langsung dengan morfem yang
diekstrak dari `kiwi.analyze()` pada bentuk yang sudah diketahui
benar. Setelah pakai codepoint yang benar: `kiwi.join()` menangani
SEMUA tak-beraturan dengan benar (ㅂ-irregular `맵다→매운`,
ㄹ-irregular `길다→긴`, stem-`으` `크다→큰`/`예쁘다→예쁜`) — juga diuji
dan dikonfirmasi: tag `VA` vs `VA-I` (bentuk tak-beraturan) TIDAK
mempengaruhi hasil `kiwi.join()` sama sekali, jadi tidak perlu deteksi
tak-beraturan eksplisit (simplifikasi yang sama seperti tag `VV`
seragam untuk verba di skill lain).

Tidak ada panggilan LLM — 6 pasangan (adjektiva, kata benda yang
cocok secara semantik) dikurasi langsung, sama seperti `irregular_conj`.
Hasil: 5/5 berhasil.

**Keputusan setelah skill ini: BERHENTI mengejar "7 dari 7 kode
KAGAS lengkap".** Dibahas eksplisit dengan user (bukan diputuskan
sepihak) — kesimpulannya:
1. Klaim paper (Novelty 2) cuma butuh taksonomi yang dipakai memang
   kode KAGAS asli (sudah benar sejak §9.8), BUKAN klaim cakupan
   lengkap semua kode — jadi tidak ada keharusan akademik untuk
   menyelesaikan VERB/NOUN/MOD.
2. `adj_relativizer` menambah pola struktural baru yang genuinely
   belum ada (blank di tengah kalimat, bukan di akhir) — nilai nyata
   untuk klaim generalisasi paper. VERB/NOUN/MOD TIDAK akan menambah
   pola struktural baru — ketiganya soal pemilihan kata/leksikal
   (semantik), bukan soal bentuk/morfologi seperti 7 skill yang sudah
   ada. Validator deterministik yang terbukti (cek tag, aturan
   batchim, `kiwi.join()`) tidak otomatis nyambung ke "apakah pilihan
   kata ini masuk akal secara makna" — itu butuh baik kurasi
   pasangan-kata closed-set (nilai tambah kecil, pola sama seperti
   `irregular_conj`) ATAU critic-loop LLM-as-judge yang sudah
   DIDESAIN di §3.2 tapi BELUM diimplementasikan sama sekali
   (investasi jauh lebih besar).
3. Alternatif "pakai yang sudah pretrained/korpus existing" (model
   GEC terlatih, atau korpus KAGAS asli Kor-Learner/Kor-Native/
   Kor-Lang8) dipertimbangkan dan ditolak untuk tahap ini: model GEC
   menyelesaikan tugas TERBALIK (memperbaiki, bukan membangun
   distractor terkontrol); korpus KAGAS asli berlisensi riset-saja
   (NIKL, bukan untuk pemakaian produksi) dan errornya "ditemukan apa
   adanya" dari esai bebas, bukan pasangan pilihan-ganda bersih yang
   dibutuhkan tracking per-skill Urnings.

**Status akhir cakupan KAGAS (TIDAK akan ditambah lagi tanpa alasan
baru):** 7 skill menutupi **4 dari 7 kode kelas-POS relevan** (PART,
END, CONJ, ADJ). VERB/NOUN/MOD dicatat eksplisit sebagai **diputuskan
tidak dikerjakan sekarang, bukan dilupakan** — sama gaya keputusan
seperti infra ditunda di §9.3. Fokus pindah ke gap aplikasi yang
masih nol: admin/LMS CRUD atau EXP/leaderboard (§9.4).

### 9.11 Skill kedelapan (di luar kerangka KAGAS): `subj_topic` — meninjau ulang keputusan lama dengan bukti baru

Setelah §9.10 memutuskan berhenti mengejar cakupan KAGAS, user
eksplisit bertanya: dari 7 skill yang ada, adakah yang harus dihapus
karena tidak penting, atau skill penting lain yang belum dikerjakan?
Jawabannya: tidak ada yang perlu dihapus (ketujuhnya materi TOPIK I-II
yang nyata dan umum), tapi ditemukan satu kandidat kuat yang BUKAN
ide baru — **이/가 (subjek, JKS) vs 은/는 (topik, JX)**, kemungkinan
kesalahan pembelajar Korea paling terkenal, sudah pernah **ditolak
eksplisit** di §9.6 (catatan `time_range`) dengan alasan "ambigu tanpa
konteks". Alasan penolakan itu sendiri yang sudah usang: `time_range`
KEMUDIAN berhasil menyelesaikan masalah ambiguitas yang identik lewat
template berpasangan — jadi penolakan lama ini layak ditinjau ulang
dengan bukti baru itu, sesuai prinsip kerja proyek ini sendiri (lihat
bagian awal dokumen).

**Pengujian morfologi: bersih total, tidak ada bug ditemukan** (baru
pertama kali dalam seri 8 skill ini) — termasuk kontraksi tak-beraturan
ganti-kata-ganti (`저+가→제가`, `나+가→내가`) yang ditangani sempurna
oleh `kiwi.join()`. Juga ditemukan: `이/가` (tag `JKS`) dan `은/는`
(tag `JX`) punya tag kiwipiepy yang BERBEDA — beda dari
`particle_loc`/`honorific` yang dua opsinya satu tag sama. Bahkan
alomorf salah (`책가`, seharusnya `책이`) langsung di-parse ulang
kiwipiepy jadi kata lain sama sekali (`가` dibaca sebagai kata kerja
"pergi") — sinyal validasi yang sangat kuat.

**Tapi masalah inti yang membuatnya ditolak dari awal TETAP ada,
cuma sekarang ditangani secara jujur, bukan dihindari:** tidak seperti
7 skill lain, `이/가` vs `은/는` **keduanya selalu gramatikal benar**
di hampir semua kalimat — perbedaannya murni pragmatik (topik vs
fokus/informasi-baru), bukan benar-salah tata bahasa. Dipilih
mekanisme pemaksa: pola tanya-jawab `누가` ("siapa") — konvensi
textbook standar bahwa jawaban atas pertanyaan `누가` memakai fokus-
subjek `이/가` (informasi baru), bukan topik `은/는`. **Ini KONVENSI
PEDAGOGIS, bukan hukum sintaksis keras** seperti 7 skill lain (yang
semuanya dijamin benar/salah secara struktural atau by construction)
— ditandai eksplisit dengan level keyakinan lebih rendah, didiskusikan
dan disetujui user sebelum dikerjakan, bukan diklaim setara dengan
skill lain.

Dibangun `validate_postnominal_tag()` di `distractor_validator.py` —
kebalikan `validate_prenominal_tag`: cek tag morfem tepat SETELAH kata
tertentu (di sini: lemma kata ganti `저`/`나`), bukan sebelum. Diperlukan
karena opsi di sini adalah satu unit kata-ganti+partikel yang
terkontraksi (`제가`), sama seperti `adj_relativizer` butuh validator
baru untuk posisi blank yang berbeda.

Tidak ada panggilan LLM — 4 pasangan (kata ganti, predikat) dikurasi
langsung. `correct_form` dan `distractor_form` SAMA-SAMA dibangun lewat
`kiwi.join()` (bukan concatenation string manual) — ini sengaja,
untuk menghindari jebakan menempel stem terkontraksi `제` (cuma valid
sebelum `가`) ke `는`, yang akan menghasilkan `*제는` yang salah.

Hasil: 5/5 berhasil. Total sekarang **8 skill grammar**. Ini skill
terakhir yang ditambahkan dengan alasan kuat (nilai pedagogis tinggi +
biaya rendah + peninjauan ulang keputusan lama) — bukan untuk mengejar
checklist apa pun.

### 9.12 Audit terhadap klaim novelty riset sendiri — Novelty 1 ternyata TIDAK tersambung ke pipeline nyata, sekarang diperbaiki

User eksplisit meminta: sebelum lanjut ke gap aplikasi (admin/LMS,
EXP/leaderboard), audit dulu apakah tiga novelty riset (§3) benar-benar
berfungsi di kode, bukan cuma diklaim di paper. Dicek satu-satu
langsung terhadap kode yang jalan (bukan dokumentasi):

- **Novelty 3 (Uncertainty-Gated Escalation): benar-benar
  terintegrasi.** `should_escalate()` dipanggil di
  `answer_processing.py`, menentukan jalur cepat/detail
  `qwen_feedback.py`. Sesuai klaim.
- **Novelty 1 (Generate–Calibrate–Select Loop): komponennya ADA
  (`estimate_difficulty()`, `seed_item_from_encoder()`, masing-masing
  lolos smoke test sendiri) TAPI TIDAK DIPANGGIL dari pipeline manapun
  di luar smoke test-nya sendiri.** `answer_processing.py` sebelumnya
  punya `_NEUTRAL_ITEM_URN = UrnState(k=16, n=32)` tetap untuk SEMUA
  item apa pun skill/kesulitannya — secara fungsional sistem yang jalan
  berperilaku seolah Novelty 1 tidak ada sama sekali.
- **Novelty 2 (critic loop): bagian "Critic (LLM-as-judge)" di Fig.
  3/Section IV-C paper murni narasi, tidak ada kodenya.** Validator yang
  ada baru mengerjakan 1 dari 3 pengecekan yang didesain (realisasi
  morfologi); 2 lainnya ("distractor bukan kebetulan juga benar",
  "round-trip solver") tidak diimplementasikan sebagai pengecekan
  umum — untuk 8 skill yang ada, risiko itu dihindari lewat konstruksi
  manual (distractor sengaja dibuat salah), bukan lewat mekanisme yang
  diklaim paper. **Belum dikerjakan di sesi ini** — dicatat sebagai
  tugas terbuka berikutnya, bukan diabaikan.

**Novelty 1 diperbaiki sekarang, lengkap dengan temuan tambahan:**
saat menelusuri ini, ditemukan alasan desain lama di §9.2 ("urn item
tidak perlu dipersist, karena item sekali-pakai per kejadian") sudah
**usang** untuk item dari `question_bank.py` — bank itu sendiri baru
ada di §9.7, SETELAH §9.2 ditulis, dan justru dirancang supaya item
APPROVED dipakai ulang ke banyak siswa. Pola yang sama lagi: keputusan
lama, alasannya sudah tidak berlaku karena perubahan lain yang datang
belakangan.

Diperbaiki dua lapis sekaligus (user eksplisit minta keduanya
sekaligus, bukan bertahap):
1. `answer_processing.py`: `submit_grammar_answer()` sekarang menerima
   `item_urn` opsional. Kalau diberikan (item dari bank) dipakai
   langsung dan urn yang sudah diperbarui dikembalikan lewat
   `AnswerResult.updated_item_urn` untuk disimpan pemanggil. Kalau
   tidak diberikan (item sekali-pakai di luar bank), di-seed otomatis
   dari `estimate_difficulty(item.stem)` untuk kejadian itu saja.
   Modul ini tetap TIDAK menyimpan apa pun sendiri (prinsip modul ini
   dari awal) — persistensi tetap tanggung jawab pemanggil.
2. `question_bank.py`: `BankEntry` sekarang punya field `item_urn`,
   di-seed otomatis saat entry dibuat. `edit()` me-reset `item_urn`
   dari nol (bukan mempertahankan evidence lama) — keputusan eksplisit:
   evidence yang terakumulasi itu mengkalibrasi TEKS LAMA yang sudah
   tidak ada lagi, membawanya begitu saja berarti salah atribusi.

**Bug matematis nyaris masuk ke smoke test — ditangkap sebelum commit,
bukan sesudah:** asumsi awal "jawaban benar di item yang di-seed LEBIH
SULIT harus menaikkan urn pelajar LEBIH BANYAK" (intuisi "mengalahkan
soal sulit = lebih berharga") ternyata **salah arah** untuk algoritma
ball-exchange yang dipakai di sini — dicek silang terhadap
`simulate_trackers.run_urnings` (implementasi referensi yang sudah
divalidasi lewat RMSE/konvergensi di §2.1/§5, bukan diasumsikan ulang)
dan dikonfirmasi lewat simulasi Monte Carlo (3000 percobaan): jawaban
benar pada item yang di-seed LEBIH MUDAH (k/n item lebih tinggi)
justru menaikkan urn pelajar LEBIH BANYAK secara rata-rata dibanding
item yang di-seed lebih sulit — berlawanan dengan intuisi naif, tapi
itu memang perilaku asli algoritma tervalidasi ini per-langkah (bukan
bug; kebenarannya dibuktikan lewat hasil RMSE jangka panjang, bukan
intuisi satu langkah). Assertion di smoke test ditulis sesuai arah
yang terverifikasi ini, bukan intuisi awal — dan dites lewat rata-rata
banyak percobaan (bukan satu draw, yang terbukti terlalu berisik
karena `y` dan penerimaan MH sama-sama stokastik).

**Yang BELUM dikerjakan, dicatat eksplisit (bukan tugas kecil, perlu
sesi sendiri):** eksperimen pembuktian penuh PRD §3.1 ("encoder prior
konvergen lebih cepat ke rating sebenarnya dibanding prior seragam")
pada skala yang berarti, dan Novelty 2's critic loop. Keduanya kandidat
kerja berikutnya yang lebih bernilai riset daripada admin/LMS CRUD/
EXP/leaderboard — tapi itu pun sudah soal prioritas berikutnya, bukan
pekerjaan sesi ini.

### 9.13 Eksperimen §3.1 dikerjakan — hasil TIDAK sesuai klaim di bawah Urnings, tapi menjelaskan KENAPA, dan memperkuat keputusan Urnings dari sudut baru

Dikerjakan tepat setelah §9.12: bukti minimal yang diminta §3.1
("encoder prior konvergen lebih cepat ke rating sebenarnya dibanding
prior seragam"). Tiga simulasi berurutan, masing-masing dibangun di
atas temuan simulasi sebelumnya — bukan satu eksperimen tunggal yang
langsung berhasil, dicatat apa adanya termasuk jalan buntu dan bug
yang ditemukan sendiri di tengah jalan.

**Simulasi 1 — `src/simulate_item_prior.py`, pool tetap (pola
`simulate_trackers.py`: 400 item, 120.000 respons, ~300 respons/item).**
Hasil pertama (1 seed) terlihat prior encoder justru memperburuk RMSE —
ternyata ada *confound* nyata di desain sendiri: `sigma_hat2` yang
tetap (mengikuti perilaku asli `difficulty_heuristic.py`, yang memang
sengaja tidak pernah mengubah keyakinannya) membuat urn item ber-prior
selalu berukuran n=26, BEDA dari baseline netral yang n=32 — jadi yang
awalnya kelihatan "prior encoder tidak membantu" sebagian cuma efek
"urn lebih kecil = lebih tidak stabil", bukan soal posisi prior itu
sendiri. Diperbaiki dengan kondisi "position only" (prior encoder,
tapi n dipaksa 32 sama seperti baseline) + diulang 8 kali (bukan 1
seed) karena satu draw YANG SUDAH TERBUKTI terlalu berisik (lihat
§9.12). Hasil setelah diperbaiki: **tidak ada keunggulan jelas di
RMSE@2000 maupun RMSE@final untuk kualitas sinyal manapun (korelasi
0.96/0.78/0.47 terhadap beta sebenarnya) — semua kondisi overlap dalam
rentang noise satu sama lain, baseline netral sedikit lebih baik di
RMSE@final.**

**Simulasi 2 — `src/simulate_item_coldstart.py`, pool TUMBUH
(bukan tetap).** Dugaan: regime pool-tetap tidak mewakili aplikasi
nyata — item di sini terus-menerus baru digenerate (`exaone_provider.py`
+ `question_bank.py`), rata-rata respons per item jauh lebih sedikit
daripada 300. Dibangun regime baru: pool mulai dari 20 item, +1 item
baru tiap 20 respons, berakhir di ~6020 item (~20 respons/item
rata-rata) — lebih dekat ke kondisi nyata. Hasil: **pola TIDAK
monoton dan membingungkan** — sinyal LEMAH (noise_std=2.0, nyaris
tidak informatif) justru RMSE@final-nya PALING BAIK (0.51), mengalahkan
baik netral (0.61) maupun sinyal KUAT (0.63). Dicurigai (belum
dikonfirmasi penuh) ini efek saturasi: noise besar mendorong lebih
banyak item ke k0 ekstrem (0 atau 32) lewat transformasi
sigmoid+pembulatan+clipping, yang berinteraksi secara tidak terduga
dengan `pick_adaptive`.

**Simulasi 3 — `src/simulate_item_coldstart_elo.py`, regime sama,
tracker Elo/Parallel Elo.** User eksplisit mengangkat pertanyaan yang
tepat: apakah pola membingungkan di Urnings itu soal mekanisme
ball-exchange Urnings secara spesifik (lihat temuan §9.12: arah
pergerakan per-langkah Urnings memang berlawanan intuisi), bukan soal
"apakah prior-injection berguna" secara umum — diuji langsung,
bukan diasumsikan. Injeksi prior ke Elo jauh lebih langsung (bh0 =
beta prediksi, angka kontinu, tanpa transformasi sigmoid/pembulatan
Urnings). Hasil **jelas dan monoton, beda total dari Urnings:**

| Tracker | Kondisi | RMSE@2000 | RMSE@final |
|---|---|---|---|
| Elo | netral | 0.7747 | **0.6029** |
| Elo | kuat (noise=0.3) | **0.7369** | 0.6248 |
| Elo | sedang (noise=0.8) | 0.7447 | 0.8489 |
| Elo | lemah (noise=2.0) | 0.7524 | **2.0871** |
| Parallel Elo | netral | 0.8503 | **0.3125** |
| Parallel Elo | kuat | **0.8203** | 0.3903 |
| Parallel Elo | sedang | 0.8309 | 0.5936 |
| Parallel Elo | lemah | 0.8282 | **1.7693** |

Prior yang BAGUS memang membantu konvergensi awal di Elo, jelas dan
monoton (sesuai intuisi/klaim §3.1) — tapi prior yang SALAH (bahkan
cuma "lemah", bukan ekstrem) membuat RMSE akhir MELEDAK jauh lebih
buruk daripada mulai netral sama sekali (1.77–2.09 vs 0.31–0.60).
**Penjelasan:** Elo tidak punya mekanisme "seberapa percaya diri" —
tiap item di-update dengan langkah tetap (`K=0.3`) terlepas dari
apakah tebakan awalnya meyakinkan atau kasar; item yang salah
kalibrasi ikut merusak estimasi SETIAP siswa yang menjawabnya sampai
cukup banyak data mengoreksinya. Urnings punya `sigma_hat2` yang
secara eksplisit mengecilkan urn (bobot prior) kalau tidak yakin —
itulah yang membuat perilakunya "datar"/tidak dramatis di Simulasi
1 & 2: bukan gagal memanfaatkan prior bagus, tapi SENGAJA membatasi
seberapa besar pengaruh prior (bagus ATAUPUN jelek).

**Kesimpulan jujur (bukan klaim sukses, bukan juga klaim gagal
total):** bukti minimal §3.1 ("prior encoder konvergen lebih cepat")
**TIDAK terbukti untuk Urnings** pada regime yang diuji — ini dicatat
apa adanya, tidak ditutupi. TAPI eksperimen Elo menjelaskan KENAPA,
dan sekaligus memberi alasan baru dan independen untuk tetap pakai
Urnings (terlepas dari alasan Novelty 1/3 yang sudah ada di §2.1/§8):
karena `difficulty_heuristic.py` sendiri eksplisit "tidak terkalibrasi
ke skala kesulitan nyata manapun", meng-inject prior yang BELUM
TENTU akurat ini lewat mekanisme Urnings jauh lebih aman (risiko
kecil kalau prior ternyata salah) dibanding lewat mekanisme Elo
(risiko BESAR, terbukti meledak di simulasi). Urnings dipertahankan
BUKAN karena manfaat besar yang terbukti, tapi karena sifat
"aman-terhadap-prior-yang-salah" yang baru terbukti konkret lewat
perbandingan ini — klaim §3.1 di PRD/paper perlu direvisi melunak
jadi ini (robustness, bukan speedup konvergensi tervalidasi), bukan
dibiarkan mengklaim manfaat yang belum terbukti.

**Belum dikerjakan (dicatat, bukan diabaikan):** pola non-monoton di
Simulasi 2 (dugaan efek saturasi k0 ekstrem) belum dikonfirmasi akar
penyebabnya secara pasti.

### 9.14 Critic loop Novelty 2 — scope kecil (verb lemma), bukan generasi bebas penuh

Dikerjakan setelah diskusi eksplisit soal scope: bukan revisit generasi
distractor bebas (yang terbukti tidak reliable di audit awal, lihat
catatan "Round 1" di `exaone_provider.py`), tapi critic loop ditempel
ke SATU-SATUNYA panggilan LLM yang masih ada di seluruh pipeline 8
skill — `_pick_verb_lemma()`. Generator → Validator → Critic (§3.2)
diuji di titik paling kecil dan paling aman dulu, bukan di titik
paling ambisius.

**Metodologi: dicari kegagalan NYATA dulu, bukan skenario karangan.**
Dicoba 20 slot di luar semua tabel `verb_lookup` terkurasi (memaksa
jalur LLM) — 12/20 gagal (output bukan bentuk dasar verba berakhiran
"다"). Diperiksa teks mentahnya satu-satu, bukan diasumsikan.

**Ditemukan: 2 dari 12 kegagalan itu BUKAN kegagalan model, tapi bug
di kode ekstraksi** — `text.split()[0]` cuma cek token PERTAMA. Untuk
slot "박물관", model sebenarnya sudah menjawab benar ("책을 읽다")
tapi kode cuma melihat "책을" (bukan "읽다") dan membuangnya. Diperbaiki
dulu TANPA LLM sama sekali (`_extract_verb_lemma()` sekarang scan semua
token) — 2 kegagalan pulih gratis, prinsip "jangan panggil LLM kalau
bug kode biasa yang menyebabkan" dipegang.

**10 kegagalan sisanya genuinely dari model** — tiga pola: (1) output
bahasa Inggris ("travel" untuk 우주선/공항/기차역), (2) kata benda,
bukan kata kerja (해변→"해변", 교도소→"감옥", dst), (3) bentuk gerund
`-기`, bukan bentuk dasar `-다` (카지노→"게임하기").

**Dibuktikan dulu kenapa blind retry TIDAK bisa membantu di sini
(bukan diasumsikan):** model pakai greedy decoding (`do_sample=False`)
— prompt yang identik dijamin menghasilkan output identik. Diuji
langsung: slot yang sama dipanggil 3x tanpa critic → gagal identik
3/3 kali. Ini yang membuat critic loop punya alasan kuat untuk ADA:
satu-satunya cara memperbaiki kegagalan pada slot yang SAMA adalah
mengubah konteks percakapannya, bukan coba lagi dengan harapan beda.

**Critic loop:** kalau `_extract_verb_lemma` gagal, model diperlihatkan
jawaban salahnya sendiri + diminta revisi dalam percakapan yang sama
(bukan prompt baru dari nol) — satu giliran tambahan, dibatasi (tidak
retry berulang tanpa batas). Hasil terhadap 10 kegagalan nyata:
**4/10 pulih (40%)** — 우주선→이동하다, 호텔→숙박하다, 기차역→이동하다,
영화관→보다. 6/10 tetap gagal bahkan setelah critic (해변, 교도소,
스타디움, 공항, 카지노, 수족관) — dicatat jujur, bukan diklaim 100%
berhasil.

**Instrumentasi produksi:** `ExaoneGrammarProvider.critic_attempts`/
`.critic_recoveries` mencatat seberapa sering critic loop benar-benar
dipakai dan seberapa sering berhasil di pemakaian nyata nanti, bukan
cuma dari eksperimen satu kali ini.

**Yang BELUM dikerjakan (scope sengaja dibatasi, dicatat eksplisit):**
generasi distractor bebas dikondisikan-miskonsepsi (visi penuh Fig. 3)
masih belum disentuh — keputusan eksplisit untuk tidak revisit itu
sekarang, karena riwayat proyek ini sudah menunjukkan pendekatan itu
tidak reliable dan lebih berisiko daripada critic loop berskala kecil
ini. Pengecekan verifier gate (b) "distractor bukan kebetulan juga
benar" dan (c) "round-trip solver" (§9.12) juga masih belum
diimplementasikan sebagai pengecekan umum — kedua ini masih open
item terpisah.

### 9.15 Novelty 1's "Select" — ternyata juga belum tersambung, ditemukan dari deskripsi fitur user sendiri

User menjelaskan perilaku yang diinginkan tanpa menyebut "Novelty 1"
sama sekali: kalau satu siswa anggap soal gampang, soal berikutnya
harus menyesuaikan jadi lebih susah untuk DIA; siswa lain yang anggap
sulit harus dapat yang lebih mudah — yaitu seleksi soal adaptif
per-individu. Dicek langsung ke kode (bukan diasumsikan ada/tidak
ada): **memang belum ada sama sekali.** `ContentProvider.next_new_grammar_item`
cuma terima `skill_id`, tidak ada info kemampuan siswa; `BankContentProvider`
(§9.7) cuma `self.rng.choice(candidates)` — pilih acak dari item yang
APPROVED untuk skill itu, tidak peduli kesulitan item maupun kemampuan
siswa sama sekali.

Ini persis bagian **"Select"** dari nama novelty-nya sendiri
("Generate–**Calibrate**–**Select** Loop") — §9.12 memperbaiki
"Calibrate" (urn item di-seed dari estimasi kesulitan), tapi estimasi
itu TIDAK PERNAH dipakai untuk MEMILIH soal mana yang disajikan.
Ditemukan murni dari user menjelaskan fitur yang diinginkan, bukan
dari audit sistematis — sama persis pola `subj_topic`/Novelty 1 lain:
gap nyata yang baru ketahuan setelah ditanya dengan cara yang
berbeda.

**Diperbaiki:** protokol `ContentProvider.next_new_grammar_item`
(`daily_drill.py`) sekarang terima `learner_urn` (urn kemampuan siswa
untuk skill itu) dan `exclude` (item yang sudah dipakai di drill yang
sama, supaya tidak dobel). `BankContentProvider` sekarang pakai ini:
urutkan kandidat berdasarkan `|kesulitan_item - kemampuan_siswa|`
(prinsip yang sama seperti `pick_adaptive` di `simulate_trackers.py`,
target ~50% peluang benar), lalu pilih acak dari 3 kandidat terdekat
(bukan selalu yang PERSIS terdekat — satu `item_urn` estimasi tunggal
itu masih noisy, apalagi baru di-seed). `ExaoneGrammarProvider` dan
`CombinedContentProvider` menerima parameter baru ini untuk
kompatibilitas protokol tapi tidak memakainya — generasi langsung
(bukan dari bank) menghasilkan SATU item baru per panggilan, tidak
ada kandidat untuk dipilih-pilih; seleksi adaptif cuma relevan untuk
jalur bank (item sudah di-batch + approve duluan).

**Diuji:** dibuat urn item buatan tangan merentang dari sangat mudah
sampai sangat sulit, dikonfirmasi siswa lemah (urn rendah) konsisten
dapat himpunan item yang lebih mudah, siswa kuat dapat yang lebih
sulit, dan keduanya TIDAK dapat himpunan yang sama. `exclude` dikonfirmasi
mencegah pengulangan item dalam satu sesi perakitan drill.

**Masih belum (dicatat, bukan diabaikan):** seleksi adaptif baru
berjalan untuk jalur bank (`BankContentProvider`) — jalur live-generation
(`ExaoneGrammarProvider`) tidak punya konsep "pilih dari kandidat" sama
sekali karena cuma generate satu item per panggilan.

### 9.16 Verifier gate (b)/(c) — dibangun, tapi (b) terbukti tidak reliable, (c) berguna setelah 2 kali perbaikan

Dibangun `src/solver_check.py` — implementasi (b) "distractor tidak
kebetulan juga benar" dan (c) "cek round-trip lewat solver" dari §3.2,
pakai Qwen3-4B (sudah ada di `qwen_feedback.py`) dalam peran BARU: bukan
menjelaskan jawaban, tapi menyelesaikan soal secara BUTA (tidak tahu
kunci jawaban).

**Percobaan pertama gagal, dan alasannya penting:** satu prompt diminta
"sebutkan SEMUA opsi yang bisa diterima" (menguji (b) dan (c) sekaligus).
Hasil pada 8 skill nyata: 4/24 bersih — termasuk solver MENOLAK jawaban
yang gramatikal benar dan MENERIMA opsi yang tidak masuk akal (mis.
menerima `도`/`만` padahal kalimatnya butuh penanda lokasi). Dicoba
perbaikan dengan contoh kerja (teknik yang dulu menyelamatkan reliability
generasi EXAONE sendiri) — malah LEBIH BURUK, model mulai echo
meta-komentar bukan jawab. **Tugas "pilih semua yang benar" (multi-select)
ternyata terlalu berat untuk mode non-reasoning model ini** — pelajaran
"model kecil butuh tugas sempit" yang sama seperti EXAONE-1.2B, kali ini
ditemukan lagi untuk model berbeda (Qwen3-4B) di tugas berbeda.

**Diperbaiki dengan memecah jadi 2 tugas sempit terpisah:**
- (c): pilih SATU jawaban terbaik (format pilihan-ganda standar + satu
  contoh kerja).
- (b): untuk tiap opsi non-kunci, tanya terpisah "apakah kalimat ini
  juga bisa diterima? ya/tidak".

**Ditemukan lagi, kedua kalinya di investigasi ini: budget token terlalu
kecil bikin hasil terlihat gagal padahal cuma terpotong.** Model ini
sering echo ULANG seluruh pertanyaan dulu sebelum menjawab (mode
non-reasoning) — token budget 5-10 motong jawaban SEBELUM sampai ke
jawaban sebenarnya, bukan berarti modelnya "tidak menjawab". Skrip
diagnostik cepat pertama JUGA sempat salah baca hasil (cuma cek karakter
PERTAMA dari teks, yang selalu "정" dari "정답:", bukan huruf jawaban
sungguhan) — setelah dibenerkan, kasus yang kelihatan "gagal total" di
percobaan pertama ternyata sebagian besar BENAR.

**Pengecekan (b) — dites ketat, TERBUKTI tidak reliable, bukan
diasumsikan.** Diuji 6 kasus diketahui (3 benar, 3 salah, termasuk
bentuk tak-terkontraksi yang jelas tidak gramatikal) — model SELALU
condong jawab "ya, bisa diterima", bahkan untuk kalimat yang jelas
salah. Dicoba 2 framing prompt berbeda DAN mode reasoning (budget 800
token) — tetap sama, model selalu bilang "맞음" (benar) untuk 4/4 kasus
termasuk yang jelas salah. Ini BUKAN masalah konfigurasi yang bisa
diperbaiki lagi (sudah divariasikan: format prompt, budget token, kata-
kata prompt, mode reasoning) — dicatat sebagai keterbatasan nyata yang
sudah diuji, bukan ditutup-tutupi. **Hasil (b) ditampilkan di output
tapi diberi label eksplisit "tidak reliable, FYI saja" — TIDAK dipakai
sebagai verdict.**

**Pengecekan (c) — setelah diperbaiki, berguna.** Dijalankan terhadap
24 item nyata (3 per skill × 8 skill): **18/24 (75%) kunci jawaban
diterima solver.** Pola kegagalan jelas dan bisa dijelaskan, bukan
acak:
- `irregular_conj`: 2/3 meleset, solver konsisten pilih bentuk TIDAK
  terkontraksi (salah) — ditemukan berulang di 3 ronde pengujian
  berbeda sepanjang investigasi ini. Bukan bukti item kita salah
  (bentuk benar sudah diverifikasi `kiwi.join()`) — bukti Qwen3-4B
  punya blind spot spesifik untuk konjugasi tak-beraturan 르.
- `subj_topic`: 2/3 meleset, solver konsisten condong ke bentuk topik
  (나는/저는) — ini justru **MENGKONFIRMASI** langsung apa yang sudah
  diakui jujur di §9.11: skill ini berbasis konvensi pedagogis, bukan
  aturan sintaksis keras. Solver independen pun tidak yakin — selaras
  dengan confidence tier yang sudah ditetapkan, bukan temuan baru yang
  mengejutkan.
- `particle_loc`, `time_range`: masing-masing 1/3 meleset — belum
  cukup pola untuk disimpulkan, dicatat sebagai temuan tunggal.

**Kesimpulan jujur:** asumsi "benar terjamin karena konstruksi" untuk
8 skill yang ada **tetap berdiri** — yang diukur eksperimen ini bukan
"apakah item kita benar" (itu sudah dijamin kode deterministik), tapi
"apakah solver independen setuju", sinyal yang LEBIH LEMAH karena
kompetensi solver sendiri bervariasi. Kegagalan yang ditemukan
sebagian besar bisa dijelaskan sebagai keterbatasan solver
(`irregular_conj`) atau soft-ambiguity yang memang sudah diungkap
sebelumnya (`subj_topic`), bukan cacat baru di konstruksi deterministik
kita.

### 9.17 Pola non-monoton di §9.13 (simulasi pool-tumbuh) — akar penyebab dikonfirmasi

§9.13 menemukan pola aneh di simulasi pool-tumbuh: prior LEMAH
(noise_std=2.0, nyaris tidak informatif) justru RMSE akhirnya PALING
BAIK, mengalahkan baik prior netral maupun prior KUAT — dicatat waktu
itu sebagai "dugaan belum dikonfirmasi: efek saturasi k0 ekstrem".
Dugaan itu sekarang diuji langsung, dua tahap.

**Tahap 1 — apakah prior lemah memang mendorong lebih banyak item ke
k0 ekstrem (0 atau 32)?** Dihasilkan 2000 item per level sinyal, dihitung
proporsi yang mentok di batas:

| Sinyal | noise_std | % item mentok (k=0 atau 32) |
|---|---|---|
| Kuat | 0.3 | 0.0% |
| Sedang | 0.8 | 0.7% |
| Lemah | 2.0 | **11.3%** |

Terkonfirmasi — sinyal lemah memang mendorong jauh lebih banyak item
ke posisi ekstrem (transformasi sigmoid+pembulatan+clipping dari
`mu_hat` yang sangat ber-noise sering mentok di ujung skala).

**Tahap 2 — apakah item yang mentok itu memang jarang dipilih
`pick_adaptive`?** Dijalankan simulasi 120.000 respons penuh dengan
pelacakan tambahan (berapa kali tiap item benar-benar dipilih untuk
disajikan):

| Kondisi | Item mentok dipilih rata-rata | Item normal dipilih rata-rata |
|---|---|---|
| Kuat (n=5 item mentok) | 0.00x | 19.95x |
| Lemah (n=739 item mentok) | **0.00x** | 22.72x |

**Terkonfirmasi penuh, bukan dugaan lagi:** item yang k0-nya mentok
di ujung **TIDAK PERNAH dipilih sama sekali** sepanjang 120.000
respons — `pick_adaptive` memang didesain memilih item yang kesulitannya
cocok dengan kemampuan siswa (target ~50% peluang benar), dan item
dengan k0 ekstrem nyaris tidak pernah "cocok" dengan siapa pun.

**Penjelasan lengkap pola non-monoton §9.13:** di kondisi sinyal LEMAH,
~11-12% item pool efektif **tidak pernah terpakai sama sekali** —
bukan "banyak item dengan tebakan acak yang saling meng-cancel", tapi
sebagian item benar-benar HILANG dari peredaran. Akibatnya, 120.000
respons yang sama terkonsentrasi ke item yang TERSISA (lebih sedikit)
— tiap item yang tersisa dapat rata-rata LEBIH BANYAK exposure
(22.72x vs 19.95x di kondisi kuat). Lebih banyak respons per item =
urn item itu makin presisi mengkalibrasi dirinya sendiri terlepas dari
titik awalnya — inilah yang membuat RMSE akhir kondisi lemah terlihat
"lebih baik": itu efek SAMPING dari pool yang efektif mengecil, BUKAN
manfaat nyata dari prior yang tidak informatif.

**Implikasi untuk sistem nyata (bukan cuma simulasi):** kalau
`difficulty_heuristic.py` pernah menghasilkan tebakan yang sangat
meleset untuk satu soal tertentu, akibatnya BUKAN cuma "soal itu
dikalibrasi agak salah" — akibatnya bisa soal itu **tidak pernah
disajikan ke siapa pun sama sekali** (dianggap "tidak cocok" buat semua
siswa). Ini pertimbangan monitoring praktis baru yang sebelumnya tidak
disadari: perlu cara mendeteksi item "mati" (tidak pernah disajikan)
di bank soal, bukan cuma memantau soal yang kalibrasinya meleset.

Status: pola non-monoton §9.13 **sudah dijelaskan penuh**, bukan lagi
open item.

### 9.18 Novelty 1 (Generate–Calibrate–Select) DIHAPUS dari paper — keputusan besar, bukan sepihak

Setelah diskusi langsung soal apakah ketiga novelty "sudah kuat dan
memberi improvement" (jawaban jujur: belum ada satupun yang tervalidasi
penuh, lihat ringkasan di percakapan sesi ini), dan setelah diminta
pendapat sebagai "senior LLM engineer" soal relevansi novelty terhadap
tujuan produk (personalized drill quiz) — user eksplisit minta Novelty
1 **dihapus sepenuhnya dari paper**, bukan sekadar dilunakkan lagi
(§3.1 sudah dilunakkan sebelumnya di sesi yang sama, tapi user
memutuskan itu belum cukup).

**Alasan keputusan (rekomendasi yang diberikan, disetujui user):**
1. Klaim inti Novelty 1 ("prior dari encoder mempercepat konvergensi")
   sudah diuji langsung (§9.13) dan **terbukti tidak berlaku** untuk
   Urnings — bukan cuma "belum tervalidasi", tapi "diuji dan gagal".
2. Bagian yang SEBENARNYA melayani tujuan "personalized drill quiz"
   bukan "Calibrate" (prediksi kesulitan dari encoder), tapi "Select"
   (pencocokan adaptif per-siswa, §9.15) — dan itu TIDAK BUTUH klaim
   "encoder mempercepat konvergensi" untuk tetap berfungsi. Urnings
   sudah mengkalibrasi item dari respons nyata (robust, terbukti
   §2.1/§9.13); Select memakai itu langsung.
3. Pola yang berulang sepanjang sesi ini: setiap kali LLM diminta
   membuat PENILAIAN terbuka (seberapa sulit kalimat ini, apakah
   distractor ini juga benar, jawaban mana yang benar) — gagal atau
   butuh dipersempit drastis (lihat §9.14, §9.16). Klaim prediksi-
   kesulitan Novelty 1 itu persis tugas penilaian semacam itu —
   konsisten, bukan kebetulan terisolasi, bahwa itu juga tidak
   berhasil.

**Yang DIHAPUS dari paper (`src/content.py`, `src/make_figures.py`,
`src/build_docx.py`), bukan cuma dicoret kata-katanya:**
- Section IV-B "Novelty 1: The Generate–Calibrate–Select Loop" —
  seluruh subsection, termasuk persamaan (2)(3)(4) dan deskripsi
  re-distilasi encoder.
- Section V-B "Validating Novelty 1" — seluruh subsection (termasuk
  paragraf temuan simulasi yang baru ditambahkan sesi ini sendiri).
- Fig. 2 lama (diagram loop Generate-Calibrate-Select) — dihapus total,
  bukan cuma caption-nya; `make_figures.py`'s `fig2()` (GCS loop)
  dihapus, fungsi-fungsi sisanya di-rename (fig3→fig2, fig4→fig3,
  fig5→fig4, fig6→fig5) dan file PNG lama dibersihkan.
- Gap "G1. Generation and calibration are studied separately" di
  Research Gaps — dihapus, gap lain direnumber (G2→G1, G3→G2, G4→G3).
- Klaim "tiga novelty"/"four gaps" di abstrak dan daftar kontribusi →
  "dua novelty"/"three gaps".
- `ANCHORS` di `build_docx.py` (pemetaan teks→figure index) diperbaiki
  supaya tidak menunjuk ke paragraf yang sudah dihapus — ini detail
  teknis yang kalau terlewat akan membuat build GAGAL DIAM-DIAM (figure
  salah tempat tanpa error), jadi dicek eksplisit sebelum build ulang.
- Seluruh lettering subsection Section IV dan V di-geser (B/C/D/E →
  sesuai jumlah subsection yang tersisa) dan SEMUA cross-reference
  "Section IV-X"/"Fig. N" terkait diperbaiki satu-satu, diverifikasi
  lewat render PDF ulang (bukan cuma grep) untuk pastikan tidak ada
  yang bolong.

**Yang DIBIARKAN ada secara eksplisit (keputusan sadar, bukan lupa):**
- Kode aplikasi (`difficulty_heuristic.py`, `seed_item_from_encoder()`,
  `item_urn` di `BankEntry`, seleksi adaptif di `BankContentProvider`)
  — **semua tetap jalan**, karena "Select" memang berguna untuk produk
  terlepas dari apakah "Calibrate" itu novelty riset atau bukan.
- Deskripsi arsitektur di Fig. 1 dan abstrak (bagian "a small encoder
  predicts difficulty") — tetap disebut sebagai KOMPONEN SISTEM, bukan
  sebagai novelty.
- Referensi [13] (Gao et al., adaptive item bank for Big Five
  personality) di daftar References paper — jadi **tidak lagi disitasi
  di teks manapun** (dulu cuma dipakai di paragraf Novelty 1 yang
  dihapus) tapi SENGAJA TIDAK dihapus dari daftar, karena menghapusnya
  butuh renumber SEMUA sitasi [14] dst sepanjang paper (~40 referensi),
  risiko salah-ketik lebih besar daripada manfaatnya untuk draf yang
  belum disubmit. **Dicatat jujur sebagai technical debt kecil**, bukan
  ditutupi: satu referensi di bibliografi paper saat ini tidak disitasi
  di teks manapun.
- Penomoran "Novelty 1/2/3" di PRD ini sendiri (§3.1/§3.2/§3.3, §8)
  **SENGAJA TIDAK direnumber** mengikuti paper — lihat catatan di awal
  §3.1 soal mismatch penomoran ini.

**Diverifikasi, bukan diasumsikan beres:** paper di-render ulang jadi
PDF (LibreOffice) dan dicek halaman demi halaman (bukan cuma grep teks
sumber) — abstrak, Related Work/Research Gaps, Design Rationale,
Proposed Method (Fig. 1 & 2 baru), Experimental Design, Discussion,
References. Semua cross-reference, nomor persamaan, nomor figure, dan
lettering subsection konsisten di versi akhir.

---

### 9.19 Validasi Novelty 2 (PRD)/Novelty 1 (paper) — batch script otomatis dibuat, rating instruktur ditunda

Konteks: user eksplisit minta ditekankan bahwa metode kami harus
dinyatakan lebih bagus daripada metode di paper literature review.
Ditinjau ulang apa yang betul-betul bisa dibuktikan untuk novelty
distractor (§3.2 PRD / "Novelty 1" di paper saat ini) tanpa mengganti
novelty-nya — kesimpulannya novelty-nya tetap tepat (sudah menyasar gap
G2 secara spesifik terhadap [12]), yang belum ada cuma eksekusi
validasinya.

**Keputusan user:** untuk draf ini, tunda rating instruktur manusia (butuh
orang sungguhan, tidak bisa disimulasikan tanpa jadi circular evidence) —
laporkan dulu cuma metrik yang bisa dihitung otomatis, catat rating
instruktur sebagai future work eksplisit.

**Kendala ditemukan:** session cloud ini tidak punya GPU dan `torch`
belum terinstal — `ExaoneGrammarProvider.__init__` (exaone_provider.py)
mem-require `.to("cuda")` tanpa fallback, jadi generasi EXAONE nyata sama
sekali tidak bisa dijalankan di sini. Sama dengan `SolverCheckProvider`
(solver_check.py), yang juga load Qwen3-4B ke CUDA.

**Yang dibuat:** `src/validate_novelty1.py` — skrip batch yang
mengagregasi tiga mekanisme yang SUDAH ADA dan sudah masing-masing
diverifikasi di smoke test file aslinya (bukan mekanisme baru), dijalankan
di N lebih besar (default 20/skill = 160 total, bukan 5/skill di smoke
test) dan dilaporkan sebagai satu tabel + file JSON:
1. Generation pass rate per skill (gate morfologis `distractor_validator`).
2. Critic-loop recovery rate (`provider.critic_attempts`/`critic_recoveries`,
   sudah ada sejak §9.14).
3. Solver round-trip key-acceptance rate — check (c) dari
   `solver_check.py`, yang sudah dikonfirmasi reliable di §9.16.

**Sengaja TIDAK dilaporkan:** check (b) solver_check.py ("distractor ini
juga benar tidak?") — §9.16 sudah menemukan yes-bias terkonfirmasi di
Qwen3-4B untuk check ini, jadi kalau dimasukkan di sini nilainya akan
false-positive-aman untuk setiap item. Mengikuti keputusan yang sama
dengan solver_check.py sendiri: dilaporkan di sana cuma untuk referensi,
tidak dipakai sebagai verdict.

**Peringatan metodologis dicatat eksplisit di docstring skrip:** jangan
pernah bilang angka dari skrip ini "mengalahkan" 53% teacher-rated-quality
di [12] (Bitew et al.) — itu skor dari rating manusia, sementara angka
skrip ini pass-rate otomatis; dua construct berbeda, membandingkan
langsung adalah category error yang akan ditangkap reviewer. Klaim yang
valid dari data skrip ini: kegagalan distractor tertangkap otomatis saat
generasi (bukan cuma lewat review manusia setelah fakta, seperti pada
[12]) — klaim soal proses deteksi, bukan klaim skor-vs-skor.

**Belum dijalankan sungguhan** — skrip ini ditulis dan dicek sintaksnya
(`py_compile`) di session tanpa GPU; user masih perlu menjalankannya di
mesin GPU sendiri (sama seperti catatan "Content generation findings" di
README) dan membawa hasilnya balik sebelum angka manapun ditulis ke
Section V-B paper (`src/content.py`). Field label misconception/KAGAS
per item (untuk metrik agreement vs rating instruktur nanti) SENGAJA
belum ditambahkan ke `GeneratedItem` di langkah ini — baru relevan kalau
fase rating instruktur manusia benar-benar dijalankan, dan pemetaan
skill→kode KAGAS resmi butuh diverifikasi dulu terhadap taksonomi KAGAS
asli (cuma `irregular_conj`→CONJ dan `adj_relativizer`→ADJ yang sudah
eksplisit dikonfirmasi di kode saat ini, §9.9/§9.10), bukan ditebak.

---

*Dokumen ini hidup — update setiap ada keputusan baru atau keputusan lama
yang direvisi. Jangan biarkan basi.*
