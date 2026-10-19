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
| **Kenapa Urnings, bukan Parallel Elo, yang jadi default** | Awalnya kami pertimbangkan Parallel Elo sebagai default (lebih ringan diimplementasikan: tanpa urn ball-exchange + Metropolis-Hastings). Tapi ternyata **dua dari tiga novelty inti** sudah dibangun di atas formalisme urn Urnings secara spesifik, bukan cuma satu: Novelty 1 (seed urn dari varians encoder, persamaan 3–4) **dan** Novelty 3 (uncertainty-gated escalation, persamaan 5, literal `w_k > τ` = lebar urn). Parallel Elo tidak punya konsep "lebar" — untuk setara butuh proxy baru yang belum diturunkan di mana pun. Pindah ke Parallel Elo berarti menurunkan ulang matematika dua novelty sekaligus, demi manfaat kompleksitas implementasi yang untuk kode riset (bukan produk produksi dengan SLA) tidak terlalu mahal — implementasi Urnings sudah ada dan jalan di `src/simulate_trackers.py` (`run_urnings`), ~30 baris. Argumen "data kecil, pathology Elo belum separah itu" juga pedang bermata dua: kalau benar, *downside* Urnings (kompleksitas ekstra) juga rendah risiko; *upside*-nya (standard error native dipakai 2 novelty) tetap berharga terlepas dari skala. Konsistensi dengan bukti simulasi paper sendiri (Fig. 5, §5 dokumen ini) juga jadi pertimbangan — sistem yang dipakai harus sama dengan yang dibuktikan terbaik, kalau tidak reviewer akan mempertanyakannya. |
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

### 3.1 Novelty 1 — Generate–Calibrate–Select Loop

**Masalah yang dijawab:** Soal baru dari LLM tidak punya data kalibrasi, jadi
tidak bisa dipakai adaptif tanpa menunggu puluhan respons dulu.

**Solusi:** Encoder KLUE-RoBERTa memprediksi (μ̂, σ̂²) dari teks soal →
langsung dikonversi jadi komposisi & ukuran urn awal di Urnings:
prediksi kesulitan jadi titik rating awal, prediksi variance menentukan
"bobot" prior itu (makin tidak yakin encoder, makin cepat prior itu
ditimpa oleh data respons asli). Soal baru langsung bisa dipakai adaptif
sejak detik pertama, bukan setelah fase pra-tes.

**Bukti dibutuhkan (minimal):** encoder prior konvergen lebih cepat ke rating
sebenarnya dibanding prior seragam/tanpa informasi. Satu perbandingan, bukan
grid ablasi besar.

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

**Masih kosong — sisi admin/LMS, belum tersentuh sama sekali** (lihat
daftar fitur awal proyek): *Maintain topic/subject*, *Maintain curriculum
per subject* (upload kurikulum), *Review topic & Question Bank*
(approve/edit/drop soal dari bank), *Set batch publish time*, *Set number
of questions*, *Set question type*, *Auto-tagging topic* (soal baru
ditandai topiknya otomatis). Semua ini logika aplikasi murni (CRUD +
workflow), belum ada satu baris kode pun — ini kandidat kerja berikutnya
yang paling jelas, BUKAN infra.

**Juga belum ada:** logika EXP/leaderboard (§2.4 sudah mendesain
pemisahannya dari tracker, tapi akumulasi EXP dan ranking per-topik belum
diimplementasikan sebagai modul), dan skill grammar KAGAS lain di luar
`particle_loc`/`particle_obj` (baru 2 dari 14 tipe).

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

---

*Dokumen ini hidup — update setiap ada keputusan baru atau keputusan lama
yang direvisi. Jangan biarkan basi.*
