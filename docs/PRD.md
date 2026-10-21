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
| **Mastery criterion** | **Diperbaiki 2026-10-20, lihat §9.28 — baris ini dulu salah.** Satu dimensi skill dinyatakan **mastered** kalau `k/n ≥ ambang` (0.8) SAJA. Dulu ditulis "DAN lebar urn `w_k < τ`", diklaim dua sinyal independen memakai ulang primitive escalation (Novelty 3, §3.3) untuk kegunaan ketiga — **terbukti mustahil secara struktural, bukan cuma salah kalibrasi**: `width(k,n)` monoton turun untuk k≥n/2 di SEMUA n, jadi kondisi width di atas urn yang sama selalu bisa direduksi ke suatu ambang-k lain, tidak pernah benar-benar independen. Dihapus, bukan dikalibrasi ulang (lihat §9.28 untuk pembuktian lengkap). Status tetap **harus bisa dicabut**: kalau k/n turun lagi setelah jawaban salah berulang, status mastered hilang otomatis — konsisten dengan fitur *"requeue wrong answer until mastered"*, dan ini TIDAK berubah dari penghapusan kondisi width (revocation selalu didorong oleh k/n, bukan width, lihat §9.28). Alasan Glicko ditolak (baris Ditolak) tetap berlaku tanpa berubah. |
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

### 9.20 Run pertama `validate_novelty1.py` (N=20/skill, user punya GPU) — dua akar masalah berbeda ditemukan, cuma satu yang di-fix

User menjalankan skrip §9.19 di mesin GPU sendiri, hasil dibawa balik
sebagai `1_validation_results.json`. Overall solver key-accept rate
73.1% (117/160) — tapi dipecah per skill, dua skill jatuh jauh di bawah
yang lain: `irregular_conj` 1/20 (5%) dan `subj_topic` 7/20 (35%). Dicek
isi soal aslinya (bukan cuma angka agregat), ketemu dua akar masalah
yang BERBEDA SAMA SEKALI:

- **`subj_topic` — bug format, bukan keterbatasan model.** Distribusi
  pilihan solver: **pilih "B" di 20/20 trial, tanpa kecuali**, independen
  dari isi soal maupun dari label jawaban benarnya. Ditelusuri ke
  `_build_subj_topic_item` (exaone_provider.py): template stem-nya
  `f"A: 누가 {predicate}? B: ___ {predicate}"` — label giliran dialog
  "A:"/"B:" bertabrakan dengan label pilihan MCQ "A:"/"B:" yang
  dirender tepat sesudahnya oleh `solver_check.py`. Prompt yang dilihat
  solver literally punya dua "A:" dan dua "B:" dengan arti berbeda.
  **Di-fix di sumbernya** (bukan cuma di prompt solver): label dialog
  diganti jadi "질문:"/"답:" (non-Latin, tidak bertabrakan dengan opsi
  A/B/C/D manapun) — karena stem yang sama juga akan ditampilkan ke
  siswa asli, jadi ini cacat konten, bukan cuma artefak evaluasi.
- **`irregular_conj` — keterbatasan genuine judge model, BUKAN di-fix.**
  Pilihan solver **mengikuti isi kalimat** (bukan posisi label):
  konsisten pilih bentuk tak-terkontraksi yang salah (자르어요, bukan
  잘라요) 19/20 kali, persis mengonfirmasi temuan sampel-kecil yang
  sudah tercatat di docstring `solver_check.py`. Sengaja TIDAK
  diperbaiki dengan menambah contoh few-shot ke prompt solver yang
  mengajarkan pola 르-irregular — itu akan membuat solver "benar" karena
  disuapi jawaban, bukan karena soalnya memang tidak ambigu, yaitu
  circular validity yang merusak gunanya check ini sebagai bukti
  independen. Dilaporkan jujur sebagai limitasi judge, sama persis
  filosofi check (b) yang yes-bias di `solver_check.py` sendiri.
- **`time_range` (65%) dipecah per kalimat** juga mengungkap satu
  template spesifik ("아침___저녁까지 운동해요") gagal 0/5 independen dari
  posisi label, sementara 4 template lain 80-100% — kemungkinan solver
  bingung karena "아침"/"저녁" bukan angka waktu seperti template lain.
  Belum diputuskan perlu di-fix atau cukup dilaporkan sebagai limitasi
  sempit — lihat apakah berulang di run selanjutnya setelah fix
  `subj_topic` di atas.

**Belum diverifikasi ulang** — fix `subj_topic` di atas butuh run ulang
`validate_novelty1.py` di mesin GPU user untuk konfirmasi bias "selalu
pilih B" benar-benar hilang, bukan diasumsikan beres. `1_validation_results.json`
yang sudah ada di repo sekarang merepresentasikan state SEBELUM fix ini
— nilai historis (before/after), bukan dihapus, tapi tidak lagi
menggambarkan kode saat ini untuk `subj_topic`.

---

### 9.21 Semua skill dipaksa 4 opsi (A-D) — user eksplisit minta, 6 skill sebelumnya cuma 2

User bertanya kenapa cuma 2 opsi lalu eksplisit minta semua soal pilihan
ganda punya 4 opsi. Dicek: `particle_loc`/`particle_obj` sudah 4 opsi
dari awal (mekanisme filler `_FILLER_PARTICLES` sudah ada); 6 skill
lain (honorific, time_range, tense_past, irregular_conj,
adj_relativizer, subj_topic) cuma 2 — bukan keterbatasan teknis, cuma
belum dikerjakan.

**Risiko utama yang dijaga:** menambah distractor sembarangan bisa
menciptakan jawaban ganda-benar — persis failure mode yang
`distractor_validator.py` dibangun untuk cegah (§3.2). Jadi SETIAP
filler tambahan diverifikasi dulu, standalone, pakai `kiwipiepy` CPU
langsung (tanpa GPU) terhadap distractor_validator yang sudah ada,
sebelum masuk kode — bukan ditebak.

**Yang dibangun** (`exaone_provider.py`):
- Helper umum `_pick_fillers()`: ambil 2 kandidat dari daftar filler
  skill, masing-masing cuma dipakai kalau validator skill itu
  **menolaknya** (fails = definitely wrong, sama persis prinsip
  `_FILLER_PARTICLES` yang particle_loc/obj sudah pakai dari awal) —
  kalau kurang dari 2 kandidat lolos, item itu di-skip (return None),
  bukan diterbitkan dengan <4 opsi.
- Helper `_assemble_options()`: susun 4 opsi + shuffle label, dipakai
  semua skill (hapus duplikasi `labels = ["A","B"]`/`shuffle` yang
  tadinya ada di 5 tempat berbeda).
- Filler per skill, masing-masing diverifikasi via `distractor_validator`
  sebelum dipakai (lihat komentar di kode untuk detail per kasus):
  `_EF_FILLER_WORDS` (학교/사과/저기/안녕 — kata benda biasa, gagal tes EF
  di posisi akhir kalimat) untuk honorific/tense_past/irregular_conj;
  `_ETM_FILLER_WORDS` (학교/사과/저기) untuk adj_relativizer;
  `_JKS_FILLER_PARTICLES` (도/만/의) untuk subj_topic;
  `_TIME_RANGE_FILLER_PARTICLES` (에/에게/로/한테) untuk time_range.
- **Catatan jujur, bukan ditutupi:** filler ini SENGAJA bukan distractor
  misconception yang sophisticated seperti distractor utama tiap skill
  (yang 2 opsi lama) — filler cuma "aman secara morfologis", setara
  dengan filler yang particle_loc/obj sudah pakai dari awal, bukan
  diagnostik. Direkam sebagai limitasi untuk dibenahi kalau fase review
  instruktur manusia (§9.19) benar-benar jalan.
- Smoke test di `__main__` diupdate: `len(item.options) == 4` (bukan
  toleransi 2/3/4 lagi), plus assersi baru per skill yang cek filler-nya
  memang berasal dari konstanta yang benar, bukan cuma cek jumlahnya.

**Diverifikasi tanpa GPU** (bukan diasumsikan beres): karena constructor
`ExaoneGrammarProvider.__init__` tetap wajib `.to("cuda")`, verifikasi
dilakukan dengan men-stub `torch`/`transformers` di `sys.modules`
(modul palsu, tidak pernah benar-benar load model) lalu menjalankan
`__main__`-nya secara nyata via `runpy` — bukan baca kode dan percaya.
Hasilnya: 5/5 sukses di SEMUA 8 skill, semua assersi lulus termasuk
assersi filler baru. Jalur fallback LLM (slot tak terkurasi) tidak bisa
diuji dengan cara ini — itu memang butuh model asli, di luar apa yang
bisa diverifikasi tanpa GPU, batasnya sama seperti sebelumnya.

**Belum dijalankan di GPU asli** — stub di atas membuktikan logikanya
benar untuk SEMUA slot terkurasi saat ini, tapi belum membuktikan jalur
LLM fallback (untuk slot baru di luar lookup table) juga tetap
menghasilkan 4 opsi dengan benar. User perlu jalankan ulang
`validate_novelty1.py` di mesin GPU untuk konfirmasi end-to-end,
termasuk cek apakah `subj_topic` yang sekarang 4 opsi + label
dialog baru (§9.20) benar-benar menghilangkan bias solver.

---

### 9.22 3 skill baru: `verb_choice`, `noun_counter`, `mod_adverb` — menutup 3 kode KAGAS yang sengaja ditunda di §9.10

User eksplisit minta implementasi VERB/NOUN/MOD, kode kelas-POS KAGAS
yang di §9.10 sengaja ditunda (bukan dilupakan) karena dianggap butuh
penilaian semantik, bukan cek morfologi seperti 8 skill yang sudah ada.
**Bukan novelty baru** — sama seperti `irregular_conj`/`adj_relativizer`
sebelumnya, ini instance tambahan dari Novelty 1 (misconception
distractor), bukan klaim riset ke-2/ke-3.

**Sebelum menulis kode**, tiga desain diverifikasi dulu satu-per-satu
langsung di `kiwipiepy` (CPU, tanpa GPU) — bukan ditebak:

- **`verb_choice`** (KAGAS VERB) — kesalahan kolokasi kata kerja ("숙제를
  봐요" bukan "숙제를 해요"). **Genuinely leksikal**: dicek langsung,
  kedua opsi (verba benar vs verba salah untuk objek ini) sama-sama tag
  EF, kiwipiepy tidak bisa membedakan — sama persis situasi
  honorific/irregular_conj. Pakai ulang `_build_whole_word_item` tanpa
  perubahan, cuma pasangan (objek, verba-benar, verba-salah) yang
  dikurasi: 숙제/하다/보다, 밥/먹다/마시다, 영화/보다/먹다, 음악/듣다/보다,
  사진/찍다/먹다.
- **`noun_counter`** (KAGAS NOUN) — kesalahan kata bantu bilangan (단위명사:
  명 untuk orang, 개 untuk benda, 권 untuk buku, 분 untuk orang-dihormati).
  Juga leksikal — kedua opsi sama-sama tag NNB, diverifikasi langsung.
  Blank di tengah kalimat (sebelum "있어요", bukan di akhir atau sebelum
  kata benda) — tidak ada validator existing yang cocok, jadi ditambah
  `validate_before_tag()` baru di `distractor_validator.py` (generalisasi
  `validate_sentence_final_tag`'s pendekatan "cek tag di posisi struktural"
  ke posisi mid-sentence manapun, ditemukan lewat `end_marker` bukan
  posisi akhir kalimat).
- **`mod_adverb`** (KAGAS MOD) — kesalahan adverb vs kata sifat atributif
  (조용히/조용하게 vs *조용한* dipakai salah sebelum kata kerja). **Temuan
  tak terduga saat verifikasi**: ternyata ini BUKAN leksikal seperti dua
  di atas — dicek langsung, akhiran adverb (-게, tag EC) dan akhiran
  atributif (-은/ᆫ, tag ETM, dipakai ulang dari `adj_relativizer_ending`
  yang sudah ada) **punya tag kiwipiepy yang BERBEDA** di posisi yang
  sama. Jadi skill ini lebih dekat ke 8 skill morfologis lama daripada ke
  `verb_choice`/`noun_counter` — ditemukan dari verifikasi langsung,
  bukan diasumsikan dari awal.

**Bug ditemukan dan diperbaiki sebelum push** (dari verifikasi langsung,
bukan asumsi): assersi smoke test awal untuk `mod_adverb` menebak
distractor-nya akan berakhir literal karakter "은"/"ᆫ" — salah, karena
batchim ETM melebur jadi satu karakter gabung dengan suku kata
sebelumnya (하+ᆫ→한), persis kuirk representasi yang sama yang sudah
dicatat di komentar `adj_relativizer` (§9.10). Diganti jadi cek
keanggotaan ke 5 bentuk distractor yang sudah diketahui, bukan tebak
pola akhiran.

**Diverifikasi tanpa GPU** (pola sama seperti §9.21): stub `torch`/
`transformers`, jalankan `__main__` sungguhan via `runpy`. Hasil: 5/5
sukses di SEMUA 11 skill (8 lama + 3 baru), termasuk assersi konten
baru per skill. Belum dijalankan di GPU asli — `validate_novelty1.py`
sudah diupdate mencakup 3 skill baru ini, user perlu jalankan ulang
untuk dapat angka solver-check yang nyata untuk ketiganya.

---

### 9.23 Audit literature review (3 gap G1/G2/G3) + investigasi critic-loop 0/0 + subj_topic "regresi" — user eksplisit minta dicek ulang, bukan diasumsikan benar

User bertanya langsung: apakah novelty terlalu lemah dibanding literature
review, dan minta 3 gap (G1/G2/G3, `src/content.py` Section II.F) dicek
ulang kebenarannya, bukan dipercaya begitu saja. Hasil, per bagian:

**G1/G2/G3 sendiri (sebagai gap riset) tetap valid** — bukan diagnosis yang
salah. Yang ditemukan salah adalah DUA SITASI pendukungnya:

1. **[29] salah paper, dikutip 4x (baris 315/528/547/648 sebelum fix).**
   Klaim G3 ("Lin et al., 7 lesson, 885 learner, signifikan di 2/7,
   effect size 0.28/0.33") sebenarnya berasal dari **Thomas, Borchers,
   Bhushan, Gatz, Gupta, & Koedinger, "LLM-Generated Feedback Supports
   Learning if Learners Choose to Use It," EC-TEL 2025 / LNCS vol. 16063**
   (diverifikasi via web search langsung ke abstract-nya: angka 885
   learner, 2/7 lesson, effect size 0.28/0.33 match persis). Paper yang
   SEBELUMNYA ada di slot [29] — **J. Lin, S. Y. Zhao, A. Gurung, X.
   Meng, K. R. Koedinger, "Retrieval-Augmented Lesson Generation...for
   Scenario-Based Tutor Training," JCAL 2026** — adalah paper BERBEDA
   (soal generate materi training untuk tutor, bukan soal efektivitas
   feedback ke learner), dan tidak dikutip di tempat lain manapun di
   `content.py` di luar slot [29] ini. **Diperbaiki**: isi referensi [29]
   diganti ke Thomas et al. (posisi list tidak diubah, jadi nomor [29]
   tidak perlu renumber di tempat lain — `REFERENCES` di-enumerate
   otomatis oleh `build_docx.py`, aman).
2. **[15] salah paper untuk klaim "Parallel Elo" (baris ~400).** Kalimat
   "Parallel Elo, proposed in the same paper [15]" salah: [15] (Bolsinova,
   Gergely, Brinkhuis, "Keeping Elo Alive," 2026) membahas remedy
   BERBEDA (K-factor adaptif berbasis tren), bukan dua rating chain
   paralel. Mekanisme "dua chain paralel" yang benar-benar disebut
   "Parallel Elo" di paper itu berasal dari **[16] (Bolsinova, Brinkhuis,
   Hofman, Maris, "Tracking a Multitude of Abilities as They Develop,"
   2022)** — paper yang SAMA yang sudah benar dikutip untuk Urnings satu
   kalimat setelahnya. Diverifikasi via web search langsung ke abstract
   [16] (deskripsi "two parallel chains of ratings...removes dependence
   of item selection on current errors" match persis istilah "Parallel
   Elo"). **Diperbaiki**: `[15]` di kalimat itu diganti jadi `[16]`.
3. **[15]/[18] untuk klaim G2 lainnya ("Elo dipakai di adaptive platform",
   "karakterisasi presisi soal kegagalan Elo") sudah DICEK dan BENAR** —
   [15] (Keeping Elo Alive) memang spesifik soal mengkarakterisasi
   kegagalan pengukuran Elo; [18] (Vesin et al., "Adaptive assessment...
   On the use of Elo-rating") memang spesifik soal Elo dipakai di
   platform adaptif nyata. Tidak diubah.
4. **G1's [10] (Bitew et al., angka "53% teacher-rated-quality") juga
   DICEK dan BENAR** — diverifikasi via web search langsung ke abstract
   paper itu, angka 53% match persis "roughly half" yang dikutip di
   `content.py` baris 131. Tidak diubah.

Kedua fix di atas sudah di-rebuild ke `paper/KO-DRILL_IEEE.docx` lewat
`src/build_docx.py` dan dikonfirmasi langsung di PDF hasil render
(`pdftotext`, bukan cuma percaya build sukses) — baris "[16]" dan "[29]"
yang benar muncul di teks maupun di daftar referensi.

**Critic-loop `critic_attempts=0` di SEMUA 11 skill (`1_validation_results.json`
terbaru) — diinvestigasi sampai akar kode, BUKAN cuma dicatat sebagai
anomali.** Ditelusuri ke `exaone_provider.py`: `_pick_verb_lemma()` (satu-
satunya pemanggil critic loop) hanya dipanggil dari `next_new_grammar_item`'s
cabang generik (baris ~808-809), dan HANYA kalau `spec["verb_lookup"].get(slot)`
mengembalikan falsy. Dicek `SKILL_SPECS` langsung: `particle_loc` (7/7 slot
ter-curated), `particle_obj` (6/6), `honorific` (5/5), `tense_past` (5/5) —
`verb_lookup` masing-masing menutupi 100% dari `slots`-nya sendiri. 7 skill
lain (`time_range`, `irregular_conj`, `adj_relativizer`, `subj_topic`,
`verb_choice`, `noun_counter`, `mod_adverb`) sama sekali tidak pernah
memanggil LLM (komentar kode sendiri bilang "No LLM call"). **Kesimpulan:
dengan konfigurasi SKILL_SPECS saat ini, jalur yang memicu critic loop
TIDAK BISA dijangkau sama sekali** — bukan bug skrip validasi, tapi
konsekuensi nyata dari pola yang sudah berulang di proyek ini (particle_loc/
obj/honorific/tense_past/irregular_conj/verb_choice/noun_counter semua
pindah dari "tanya LLM" ke "kurasi penuh" begitu terbukti closed-set
leksikal, §9.9/§9.10/§9.22). **Implikasi untuk paper: angka critic-recovery
40% yang dikutip (dari §9.14, eksperimen terisolasi 20-slot yang MEMAKSA
jalur uncurated) belum pernah direproduksi di skala `validate_novelty1.py`
yang sama dengan metrik lain, dan TIDAK AKAN bisa selama tidak ada slot
yang sungguh-sungguh uncurated di produksi.** Dua pilihan, belum diputuskan
sepihak, perlu arahan user: (a) lunakkan klaim "Generator→Validator→Critic"
di paper jadi eksplisit "mekanisme fallback untuk vocabulary belum
terkurasi, divalidasi terisolasi (§9.14), belum teruji end-to-end pada
konfigurasi produksi saat ini" — jujur, tidak overclaim; atau (b) sengaja
sisakan/tambah satu slot genuinely uncurated supaya critic loop benar-benar
ter-exercise di `validate_novelty1.py` berikutnya, dengan verifikasi
linguistik langsung (pola yang sama seperti §9.21/§9.22), bukan ditebak.
Ditambahkan juga pengecekan eksplisit di `print_report()`
(`validate_novelty1.py`) supaya skrip SENDIRI mencetak peringatan ini kalau
`critic_attempts` total 0 — bukan diam-diam tampil "-" tanpa penjelasan.

**`subj_topic` "turun" dari 35% (§9.20, 7/20) ke 10% (run 11-skill terbaru,
2/20) — diinvestigasi, BUKAN regresi baru, dua run tidak apple-to-apple.**
Fix label dialog (질문/답, §9.20) dikonfirmasi SUDAH ada di kode
(`_build_subj_topic_item`, baris ~643) — bukan hilang/ter-revert. Yang
berubah antara dua run adalah §9.21 (paksa 4 opsi) terjadi SETELAH angka
35% itu dicatat: 35% berasal dari versi subj_topic 2-opsi (A/B), sementara
10% berasal dari versi 4-opsi (A-D) yang sekarang berjalan. Ini mengubah
baseline tebak-acak solver dari 50% (2 opsi) jadi 25% (4 opsi) — faktor 2x
saja sudah menjelaskan sebagian besar selisihnya, belum butuh teori baru.
Apakah 10% (2/20) signifikan di BAWAH baseline acak 25% (indikasi solver
punya anti-preferensi sungguhan ke bentuk topic-marked, bukan cuma gagal
menebak) — dicek: pada n=20, p=0.25, hasil 2 kejadian cuma ~1.5 SD di
bawah mean (5), TIDAK cukup kuat untuk membedakan dari noise murni secara
statistik. **Jadi: konsisten dengan (bukan bukti baru independen untuk)
limitasi yang sudah diketahui jujur di §9.11/§9.16** ("pedagogical
convention, bukan hard rule" — solver yang tidak diajar konvensi
buku-teks itu wajar condong ke bentuk topic-marked yang lebih umum dalam
percakapan natural). **Tidak ada fix kode yang dibutuhkan** — angka 35%
di §9.20 sudah stale (dari konfigurasi 2-opsi yang sudah tidak ada),
jangan dikutip lagi dibanding angka 11-skill/4-opsi saat ini. `irregular_conj`
juga tetap rendah (1/20 → 2/20) across the same transition, konsisten
dengan temuan §9.20 sendiri bahwa pilihan solver di skill ini mengikuti
ISI kalimat, bukan posisi/jumlah opsi — perubahan dari 2 ke 4 opsi memang
diperkirakan tidak banyak mengubah angka ini, dan memang tidak banyak
berubah.

**Bottom line untuk ketiga poin yang user minta:** (1) 3 gap literatur
sendiri valid, tapi 2 sitasi pendukungnya salah paper — sudah diperbaiki
dan diverifikasi di PDF hasil render. (2) belum dijawab di sesi ini —
lihat percakapan soal framing "personalized" di luar PRD. (3) critic-loop
0/0 adalah temuan nyata yang butuh keputusan desain (lunakkan klaim vs.
tambah slot uncurated), subj_topic/irregular_conj BUKAN regresi, cuma
perbandingan across konfigurasi berbeda yang perlu didokumentasikan
supaya tidak disalahartikan di masa depan.

---

### 9.24 Novelty 2 — bug tau TIDAK TERCAPAI diperbaiki, lalu ditemukan masalah lebih dalam: width TIDAK mengukur error sebenarnya (n=32)

Lanjutan investigasi §9.23. User minta tegas: investigasi, jangan cuma
dicatat. Dua temuan, satu bug (sudah diperbaiki), satu temuan konseptual
yang BELUM diputuskan arah perbaikannya (perlu keputusan user).

**Bug #1 (diperbaiki): `should_escalate()` default `tau=0.2`, tapi
`width()` maksimum untuk urn n=32 (satu-satunya ukuran urn yang dipakai di
SELURUH kodebase) cuma ~0.1715 — secara matematis TIDAK PERNAH bisa
melebihi 0.2.** Dicek `answer_processing.py`: `should_escalate(new_learner_urn)`
dipanggil tanpa override tau, jadi **eskalasi berbasis ketidakpastian
mati total di produksi, selalu False, untuk learner mana pun, pada k
berapa pun.** Lolos karena smoke test di `urnings_tracker.py` vakum:
`assert should_escalate(l2) or not should_escalate(l2)` tidak pernah bisa
gagal. **Diperbaiki**: `tau` default diganti ke 0.15 (benar-benar
tercapai, dicek langsung k=0..32), smoke test diganti jadi assersi nyata
(urn paling tidak pasti k=16 harus escalate, urn ekstrem k=0/32 tidak
boleh).

**Temuan #2 (belum diputuskan arahnya — murni investigasi, bukan bug
sederhana): `width()` TIDAK mengukur jarak estimasi ke kemampuan asli
(`|theta_hat - theta_true|`) — ia mengukur seberapa SENTRAL posisi k saat
ini dalam rentang urn, independen dari benar-tidaknya estimasi itu.**
Dibuktikan lewat eksperimen terkontrol yang menghilangkan semua perancu
populasi/item (`validate_novelty2.py::controlled_position_diagnostic`):
SATU learner per nilai theta_true tetap, melawan pool item yang
diketahui PERSIS (tanpa noise estimasi item), dijalankan sampai
ekuilibrium (8000 respons, diukur di 1000 respons terakhir).

Hasil (n=32, ukuran urn produksi): learner ekstrem (theta=-2.5) berakhir
di width **tersempit** (0.092, "tampak paling yakin") tapi error
**terbesar** (0.57-0.66) dari semua posisi yang diuji; learner di tengah
(theta=0) justru width **terlebar** (0.16-0.17, "tampak paling ragu")
dengan error yang biasa-biasa saja. Korelasi width-vs-error lintas grid
theta: **-0.755 (n=32), -0.849 (n=64), -0.040 (n=128, noisy)** — robust
negatif di ukuran urn yang dipakai produksi, baru mendekati nol/hilang di
n=128. Konsisten dengan temuan lama §9.13 (lantai RMSE n=32 ≈0.45) —
error di sini memang didominasi noise diskretisasi n=32, bukan sinyal
yang width bisa ikuti.

**Kenapa ini terjadi (mekanisme, bukan cuma angka):** `width()` adalah SD
posterior Beta(k+.5, n-k+.5) pada n **TETAP** — maksimal saat k=n/2 (rekor
terbaru learner persis 50/50 melawan pool item yang dihadapi), BUKAN
fungsi dari banyaknya bukti terkumpul seperti SE klasik yang mengecil
dengan n bertambah (di Urnings, n urn tidak pernah bertambah, cuma k yang
bergeser). Jadi width mengukur "seberapa seri performa belakangan
melawan pool", bukan "seberapa jauh estimasi dari kebenaran" — dua hal
yang kebetulan searah hanya di awal pelacakan (belum ada bukti = lebar
DAN salah sekaligus), tapi berpisah begitu tracker mencapai ekuilibrium.

**Implikasi untuk klaim paper (Section V-C) — PERLU KEPUTUSAN USER, belum
saya putuskan sepihak:**
- **Opsi (a) — reframe klaim**: jangan klaim width mengukur error absolut
  (sudah terbukti salah/terfalsifikasi untuk n=32). Klaim yang BISA
  dipertahankan: "escalate saat performa belakangan learner ini melawan
  pool item yang dihadapi genuinely ambigu (dekat 50/50)" — defensible
  secara pedagogis sendiri (konsisten dengan separuh trigger lain,
  `consecutive(misconception)>=2`, yang juga soal ambiguitas/persistensi,
  bukan soal jarak ke kebenaran absolut). Section V-C perlu ditulis ulang
  supaya tidak menjanjikan bukti yang sudah terbukti tidak ada.
- **Opsi (b) — ganti/tambah sinyal**: width saja tidak cukup untuk klaim
  "menyasar learner yang terukur buruk"; perlu sinyal tambahan yang
  genuinely melacak kualitas estimasi independen dari posisi (misalnya
  SE berbasis jumlah-kunjungan eksplisit, atau volatilitas tren k
  terbaru — ide yang mirip K-factor adaptif di [15]/"Keeping Elo Alive").

**Yang TETAP valid tanpa perlu keputusan di atas**: fraksi eskalasi
realistis (setelah fix tau) — 75.8-82.2% interaksi terpicu pada riwayat
praktik 120rb respons/300 learner, hemat compute 17.8-24.2% vs
always-escalate. Ini tidak bergantung pada interpretasi width yang benar,
cuma pada seberapa sering `w_k > tau` terpenuhi.

**File:** `src/urnings_tracker.py` (bug fix + smoke test nyata, bukan
vakum), `src/validate_novelty2.py` (skrip baru, sudah termasuk
`controlled_position_diagnostic` sebagai angka yang bisa dipercaya, dan
`simulate()` sebagai pelengkap untuk fraksi eskalasi — korelasi pooled
di `simulate()` ditandai eksplisit "NOT_TRUSTED" di output supaya tidak
dikutip tanpa sadar terperangkap perancu populasi).

---

### 9.25 Kalibrasi formal width (ECE-style) — mengkuantifikasi temuan §9.24 dengan metrik standar literatur, bukan cuma korelasi mentah

User minta metrik evaluasi yang tepat secara literatur untuk mengukur
temuan §9.24 secara formal, bukan cuma korelasi Pearson ad-hoc. Dipilih
metodologi *regression calibration* dari Kuleshov, Fenner & Ermon (ICML
2018, "Accurate Uncertainties for Deep Learning Using Calibrated
Regression") — versi kontinu dari Expected Calibration Error (ECE) milik
Guo et al. (ICML 2017) yang biasa dipakai untuk classifier. Keduanya
dicek judul/tahun/venue-nya dulu (web search), belum ditambahkan ke
bibliografi paper karena belum ada teks paper yang mengutipnya — tunggu
keputusan arah (lihat §9.24, opsi a/b) sebelum ditambahkan supaya tidak
jadi sitasi yatim seperti [13]/Gao et al. yang sudah ada.

**Cara kerja** (`src/validate_novelty2.py::beta_quantile_table` +
`calibration_curve`, tanpa scipy — quantile Beta(k+.5,n-k+.5) diestimasi
via Monte Carlo 200rb sampel per k, konsisten dengan gaya numpy-only
skrip simulate_* lain di repo ini): untuk level nominal X% (10%..90%),
ambil interval kredibel urn pada skala pi, transformasi ke skala logit
(theta), cek apakah theta_true benar-benar jatuh di dalamnya. Kalibrasi
baik = observed coverage ≈ nominal.

**Hasil (n=32, ukuran urn produksi):**

| nominal | observed (terkontrol) | observed (populasi realistis) |
|---|---|---|
| 50% | 33.8% | 42.9% |
| 80% | 63.6% | 71.0% |
| 90% | 75.7% | 82.8% |

**Calibration error (rata-rata \|nominal-observed\| lintas 9 level):
11.3% (setup terkontrol), 6.2% (populasi realistis).** Arahnya KONSISTEN
di semua level dan kedua setup: observed selalu LEBIH RENDAH dari
nominal — artinya **interval kredibel urn secara sistematis terlalu
sempit (overconfident)**, bukan cuma "tidak berkorelasi" seperti temuan
korelasi di §9.24. Interval yang diklaim 90% cuma benar-benar menangkap
kebenaran 76-83% dari waktu; yang diklaim 50% cuma 34-43%.

**Ini memperkuat (bukan menggantikan) temuan §9.24 dengan angka yang
lebih presisi dan lebih mudah dikutip di paper**: bukan cuma "width tidak
melacak error" tapi "width secara sistematis meremehkan ketidakpastian
sebesar ~6-16 poin persentase tergantung level confidence." Siap dikutip
langsung begitu arah paper (§9.24 opsi a vs b) diputuskan.

**File:** `src/validate_novelty2.py` (fungsi `beta_quantile_table`,
`calibration_curve`, terintegrasi ke `__main__` dan `simulate()`/
`controlled_position_diagnostic()` yang sekarang juga mengembalikan
`k_theta_pairs` mentah untuk dikalibrasi).

---

### 9.26 Novelty 2 Section IV-C/V-C ditulis ulang (opsi a, reframe) + item discrimination index diimplementasikan — satu temuan negatif dikonfirmasi final, satu temuan POSITIF

User setuju opsi (a) dari §9.24/§9.25: reframe klaim Novelty 2, jangan
rombak sinyal. Dikerjakan:

**Paper (`src/content.py`), Section IV-C**: kalimat "a direct, natively
available measure of how uncertain the system is about that learner"
diganti jadi "a signal of how evenly matched the learner's recent record
has been" + kalimat eksplisit "we deliberately do not claim w_k is a
calibrated estimate of absolute distance from true ability." Eq. 2's
narasi juga diubah dari "genuinely uncertain about the learner" jadi
"performance ... genuinely mixed ... not about an unobservable
true-ability gap."

**Section V-C ditulis ulang total**: bukan lagi menjanjikan perbandingan
yang sudah terbukti tidak valid (width vs subsequent reduction), tapi
melaporkan APA YANG SUNGGUH DITEMUKAN: korelasi -0.76/-0.85 (n=32/64),
kalibrasi ECE-style 6-11 poin persentase overconfident (§9.25), dan
kesimpulan jujur "w_k is systematically overconfident... not merely
uncorrelated." Fraksi eskalasi (76-82%) dan hemat compute (18-24%)
dipertahankan sebagai klaim yang TETAP valid. 2 sitasi baru ditambahkan
ke bibliografi (posisi [42]/[43], append di akhir list jadi aman tanpa
renumber): Guo et al. 2017 (ICML, "On Calibration of Modern Neural
Networks") dan Kuleshov et al. 2018 (ICML, "Accurate Uncertainties for
Deep Learning Using Calibrated Regression") — keduanya dicek judul/
tahun/venue/halaman via web search sebelum ditambahkan, bukan ditebak.
Di-rebuild ke docx+pdf, dikonfirmasi `pdftotext`.

**Item discrimination index (`src/validate_item_discrimination.py`,
baru)** — metrik #2 dari analisis literatur sesi sebelumnya. Point-
biserial correlation (= Pearson correlation langsung untuk true Rasch
generating model, tidak perlu formula baru) antara respons biner ke satu
item dan true theta populasi, field-test style (semua learner dapat item
yang sama, standar validasi item klasik sebelum dipakai adaptif).

**Pertanyaan spesifik-KO-DRILL yang dijawab, bukan cuma sanity check
generik**: Section IV-D (paper) sudah lama menyatakan target 85% sukses
(Eq. 3, demi pedagogi) "deliberately sacrifices measurement efficiency"
tapi menyebutnya "a natural comparison for future work," belum pernah
diukur. Sekarang diukur: diskriminasi di β=0 (target 50%, CAT klasik)
= 0.408 ("good"); di β=+1.73 (setara target 85% untuk learner rata-rata)
= 0.335 ("good" juga, cuma turun 17.9% relatif) — **TEMUAN POSITIF**:
trade-off itu nyata tapi kecil, item yang diserve policy 85% tetap
psikometrically sehat (masih di atas ambang 0.30 yang lazim dipakai),
bukan kompromi kualitas yang serius. Ditulis ke Section IV-D, angka
"future work"-nya diganti jadi hasil ukur nyata.

**Catatan metodologis, bukan ditutupi**: ini simulasi field-test
(item TETAP untuk seluruh populasi), bukan simulasi deployment adaptif
sungguhan (tiap learner dapat item BEDA yang dicocokkan ke
kemampuannya sendiri) — beda pertanyaan, didisclosure eksplisit di
docstring skrip, jangan dibaca sebagai "discrimination item yang benar-
benar diserve ke user".

**Pola yang baik untuk diingat**: sesi investigasi novelty ini
menghasilkan dua macam hasil — satu negatif-tapi-ditemukan-dan-
dilaporkan-jujur (width/kalibrasi, §9.24-9.25), satu positif (item
discrimination, §9.26). Jangan biarkan nada investigasi terus-menerus
negatif membuat user mengira semuanya bermasalah — laporkan temuan baik
sama jujurnya dengan temuan buruk.

---

### 9.27 Field `kagas_code` ditambahkan ke `GeneratedItem` + proxy rating KAGAS pertama (bukan pengganti studi instruktur manusia)

User minta dua hal: (1) tambah field kode KAGAS ke tiap skill/item — prasyarat
yang sejak §9.19 sengaja belum dikerjakan; (2) pilih model yang tepat untuk
proxy rating otomatis yang jujur soal keterbatasannya.

**Field ditambahkan** (`src/exaone_provider.py`): `SKILL_SPECS` tiap skill
sekarang punya `"kagas_code"` sesuai pemetaan resmi §9.8/§9.9/§9.10/§9.22 —
particle_loc/particle_obj/time_range→PART, honorific/tense_past→END,
irregular_conj→CONJ, adj_relativizer→ADJ, verb_choice→VERB, noun_counter→NOUN,
mod_adverb→MOD, subj_topic→**None eksplisit** (bukan ditebak kode apa pun —
§9.11 sudah bilang ini di luar kerangka KAGAS). `GeneratedItem` dataclass
dapat 2 field baru: `kagas_code` (diisi dari spec di `_assemble_options`,
titik konstruksi tunggal, jadi otomatis berlaku ke semua 11 skill) dan
`primary_distractor_label` (label mana di antara 4 opsi yang distractor
UTAMA — bukan salah satu dari 2 filler dari `_pick_fillers` yang sama sekali
tidak dimaksudkan membawa misconception apa pun). Tanpa field kedua ini,
rater (manusia atau proxy) tidak akan tahu opsi mana yang sebenarnya mau
dinilai. Diverifikasi via `__main__` smoke test asli (GPU), bukan stub —
5/5 semua skill, `kagas_code` dan `primary_distractor_label` muncul benar
di output.

**Pemilihan model untuk proxy**: satu-satunya model yang ter-cache di
mesin ini (EXAONE-4.0-1.2B, EXAONE-3.5-2.4B, Qwen3-4B, Qwen3-1.7B) semuanya
dari 2 keluarga yang SUDAH dipakai di generator/solver — pakai salah
satunya lagi berarti bias keluarga-model yang sama persis dengan masalah
yang mau dihindari. Memuat model ketiga cuma untuk tugas klasifikasi
ringan ini dinilai tidak proporsional (biaya download+GPU besar untuk
sinyal tambahan yang sejak awal cuma dimaksudkan sebagai bukti lemah,
bukan pengganti rater manusia). **Dipilih: Claude (saya sendiri) sebagai
rater independen** — arsitektur/perusahaan/data latih sama sekali berbeda
dari EXAONE (LG AI) maupun Qwen (Alibaba), nol biaya infrastruktur
tambahan, kompetensi bahasa Korea+linguistik cukup untuk tugas klasifikasi
7-kategori ini.

**Prosedur, dijaga buta sungguhan**: generate 30 item nyata (3/skill ×
10 skill yang punya kode KAGAS resmi, subj_topic dikecualikan karena
`kagas_code=None`), simpan ke file blind yang MENGHAPUS `true_kagas_code`
DAN `skill_id` (skill_id 1:1 dengan kode jadi akan membocorkan jawaban),
baca cuma file blind itu, putuskan kode KAGAS untuk tiap distractor utama
tanpa melihat kunci, simpan keputusan ke file terpisah, BARU dibandingkan
dengan kunci asli.

**Hasil: 30/30 (100%) agreement.** Dilaporkan jujur dengan satu caveat
penting, bukan dioversell: 100% ini sebagian besar karena kategori yang
diuji (PART/END/CONJ/ADJ/VERB/NOUN/MOD) secara STRUKTURAL mudah dibedakan
dari bentuk permukaannya saja (particle vs akhiran kata kerja vs batang
tak-teratur vs akhiran relativizer vs kelas kata leksikal) — siapa pun
(manusia atau LLM) yang tahu definisi 7 kategori ini kemungkinan besar
akan setuju. **Ini memvalidasi bahwa pemetaan skill→KAGAS internal
konsisten dan tidak asal tempel** — bukti bahwa infrastruktur dasarnya
benar. **Ini TIDAK memvalidasi** pertanyaan yang lebih halus yang studi
instruktur manusia (Fleiss' κ, Section V.B) dirancang untuk jawab:
apakah misconception SPESIFIK yang direpresentasikan tiap distractor
terasa alami/masuk akal bagi pelajar sungguhan untuk verb/kata tertentu
itu — itu soal kualitas pedagogis dalam kategori, bukan soal kategori
mana yang benar, dan 30/30 di sini tidak menyentuhnya sama sekali.

**Belum ditambahkan ke teks paper** — ditahan sengaja sampai user
memutuskan apakah proxy ini layak disebut di Section V (sebagai sinyal
tambahan lemah, jelas dibedakan dari studi instruktur) atau cukup jadi
bukti internal/PRD saja. Section V.A/V.B paper tidak diubah sesi ini.

**File:** `src/exaone_provider.py` (`kagas_code`/`primary_distractor_label`
di `SKILL_SPECS` dan `GeneratedItem`). Skrip rating ad-hoc dan hasil mentah
disimpan di scratchpad sesi, belum jadi skrip permanen di `src/` — kalau
studi ini mau diulang rutin (misalnya tiap kali skill baru ditambah),
pertimbangkan jadi skrip permanen `validate_kagas_agreement.py`.

---

### 9.28 `is_mastered()`'s kondisi width — TERBUKTI mustahil secara struktural untuk SEMUA n, dihapus (bukan dikalibrasi ulang)

Lanjutan follow-up yang di-flag di §9.24/perc akapan sebelumnya ("satu
follow-up yang perlu di-flag... width<0.15 juga dipakai is_mastered()").
User eksplisit minta diselesaikan sampai tuntas sebelum ada klaim paper
yang menyentuhnya.

**Temuan awal (sudah dicek giliran lalu)**: untuk n=32, k_threshold=0.8,
width_threshold=0.15 — begitu `k/n≥0.8` (k≥26), `width<0.15` SELALU sudah
benar juga (width(26)=0.1364). Kondisi width tidak pernah menyaring apa
pun di luar yang k/n sudah saring.

**Diperdalam sekarang — ternyata bukan soal kalibrasi angka, tapi
struktural, berlaku untuk SEMUA ukuran urn:** `width(k,n) = 2√(Var Beta(k+.5,
n-k+.5))`, dan `Var ∝ (k+.5)(n-k+.5)` dengan `(k+.5)+(n-k+.5) = n+1`
KONSTAN (tidak bergantung k) — parabola terbuka ke bawah, maksimum di
k=n/2, **monoton turun untuk setiap k≥n/2, untuk SEMUA n**. Dicek langsung
(bukan cuma aljabar) di n=32/64/128: ketiganya monoton turun sempurna di
rentang [n/2, n], dan di ketiganya width pada k=⌈0.8n⌉ SUDAH di bawah 0.15
(0.1364 / 0.0970 / 0.0699). **Konsekuensi: TIDAK ADA pilihan
`width_threshold` yang bisa membuat kondisi width menyaring sesuatu yang
belum disaring `k_threshold` — karena kedua kondisi dibaca dari (k,n) yang
SAMA, dan width monoton di k, kondisi width SELALU bisa direduksi jadi
ambang-k lain.** Ini bukan masalah "angka 0.15 dan 0.8 kebetulan tidak
cocok" (seperti bug `tau=0.2` di §9.24) — ini mustahil secara matematis
untuk kombinasi APA PUN dari k_threshold≥0.5 dan width_threshold.

**Keputusan, konsisten dengan prinsip yang baru disepakati user
(hapus klaim yang tak terdukung, jangan dilunakkan/disembunyikan)**:
kondisi width **dihapus** dari `is_mastered()`, bukan dikalibrasi ulang
(tidak ada kalibrasi yang bisa memperbaikinya). `is_mastered()` sekarang
cuma `proportion() >= k_threshold`. Docstring fungsi dan modul
(`urnings_tracker.py`) ditulis ulang menjelaskan pembuktian ini secara
eksplisit, bukan cuma menghapus diam-diam. Smoke test (`__main__`) tetap
lulus tanpa perubahan — dicek langsung, tidak ada caller (`answer_processing.py`,
`grammar_scheduler.py`) yang pernah memanggil dengan `width_threshold`
custom, jadi perubahan default behavior ini aman.

**Revocable mastery TIDAK rusak oleh penghapusan ini** — fitur "status
hilang otomatis kalau jawaban salah berulang" selalu didorong oleh k/n
turun, bukan oleh width (karena keduanya bergerak bersamaan, width cuma
konfirmasi berlebih yang sekarang terbukti tidak pernah jadi faktor
pembeda). Baris "Mastery criterion" di §2.1 (tabel keputusan) diperbaiki
mengikuti temuan ini.

**Kalau sinyal "seberapa settled" yang genuinely independen tetap
diinginkan nanti**: perlu sumber bukti yang BUKAN snapshot (k,n) yang
sama — misalnya volatilitas tren k dalam M respons terakhir — bukan
ambang kedua di atas urn yang sama. Dicatat sebagai opsi masa depan,
bukan dikerjakan sekarang (di luar scope yang diminta).

**Paper (`src/content.py`) tidak perlu diubah** — dicek, tidak ada klaim
mastery/width di teks paper, jadi perbaikan ini murni internal/PRD.

---

### 9.29 Novelty 2 DIGANTI (bukan direframe lagi): dari "uncertainty-gated escalation" jadi "Urnings robust terhadap prior kesulitan item yang tidak terkalibrasi"

User eksplisit: terlalu banyak reframe di Novelty 2 (escalation) membuatnya
lemah — minta GANTI, bukan reframe lagi. Diputuskan bersama: angkat
temuan §9.13 (robustness Urnings vs Elo terhadap prior yang salah) dari
"design rationale" di Section III-B jadi Novelty 2 resmi, menggantikan
escalation. Escalation TIDAK dihapus dari kode — turun status jadi fitur
aplikasi nyata di Section IV-D (digabung dengan Selection Policy), bukan
novelty riset. Pola yang sama persis dengan "Select" setelah Novelty 1
lama dihapus (§9.18).

**Temuan penting SEBELUM menulis apa pun (investigasi, bukan asumsi)**:
angka "Elo meledak 3-6x" yang saya sebut di awal ternyata dari regime
**pool TUMBUH** (`simulate_item_coldstart_elo.py`) — regime yang justru
Urnings SENDIRI berperilaku non-monoton/membingungkan di situ (§9.13's
Simulasi 2, §9.17's diagnosis artefak saturasi). Mencampur angka Elo dari
regime itu dengan angka Urnings dari regime **pool TETAP**
(`simulate_item_prior.py`) akan jadi perbandingan yang tidak apple-to-apple.

**Dibangun skrip baru `src/simulate_elo_item_prior.py`** — companion
pool-tetap untuk Elo/Parallel Elo (sebelumnya cuma ada untuk Urnings),
dijalankan dengan harness IDENTIK (400 item, 300 learner, 120.000 respons,
kondisi prior sama: korelasi 0.96/0.78/0.47). Hasil (8 replikasi,
mean±SD) — angka netral COCOK PERSIS dengan yang sudah terbit di paper
(Elo 1.722 vs 1.70 lama, Parallel Elo 0.703 vs 0.73 lama, memvalidasi
skrip baru ini setara dengan yang sudah ada):

| Tracker | netral | prior terburuk (corr 0.47) | spread relatif |
|---|---|---|---|
| Urnings | 0.417±0.018 | 0.438±0.016 | +5% |
| Elo | 1.722±0.024 | 2.123±0.040 | +23% |
| Parallel Elo | 0.703±0.022 | 1.133±0.031 | +61% |

**Efeknya jauh kurang dramatis** dari "meledak 3-6x" (itu cuma muncul di
regime pool-tumbuh yang bermasalah sendiri) — tapi BERSIH, monoton,
urutan konsisten di semua 8 replikasi, dan tidak akan pernah butuh
dilunakkan lagi karena regime pool-tetap sudah dipahami penuh ujung ke
ujung (tidak seperti pool-tumbuh yang masih ada artefak belum selesai
dijelaskan). **Keputusan eksplisit, disetujui user**: pakai angka
pool-tetap ini sebagai bukti Novelty 2, JANGAN sebut regime pool-tumbuh
di paper sama sekali (tidak diklaim, tidak dibantah) — cukup dicatat di
sini untuk kejujuran historis.

**Perubahan di `src/content.py`:**
- Abstrak + bullet kontribusi Section I: novelty kedua sekarang "Urnings
  tetap terkalibrasi dari prior yang sengaja tidak terkalibrasi".
- Section III-B: ditambah 1 paragraf jembatan ke Novelty 2 baru.
- Section IV-C: ditulis ulang total jadi "Novelty 2: Robustness to an
  Uncalibrated Item-Difficulty Prior" — menyasar gap G2 (bukan G3 lagi).
- Section IV-D: judul jadi "Application Policies: Item Selection and
  Escalation" — escalation (dulu IV-C) dipindah ke sini, status diturunkan
  jadi fitur, temuan kalibrasi (§9.24/§9.25) dipadatkan jadi beberapa
  kalimat, bukan dihapus — tetap disclosed, cuma tidak lagi jadi beban
  novelty.
- Section V-C: ditulis ulang total jadi validasi Novelty 2 baru (tabel di
  atas, 8 replikasi, mean±SD), termasuk catatan eksplisit soal regime
  pool-tumbuh yang disengaja TIDAK dimasukkan sebagai klaim.
- Persamaan dirapikan ulang: (1) Rasch, (2) seleksi 85%, (3) escalate
  (dulu (2)/(3) tertukar karena urutan section berubah) — semua referensi
  "Eq. N" di teks dicek dan disesuaikan.
- 2 sitasi baru [42]/[43] (Guo et al. 2017, Kuleshov et al. 2018, sudah
  diverifikasi §9.25) sekarang benar-benar dikutip di teks (sebelumnya
  ditahan karena belum ada tempatnya).
- Fig. 3 (escalation) caption diperbarui, bukan lagi berlabel "(Novelty 2)".
  Fig. 4 caption diperjelas "no item-difficulty prior injected" supaya
  tidak tertukar dengan angka prior-injection yang baru.

**Bug ditemukan dan diperbaiki saat verifikasi PDF (bukan cuma percaya
build sukses — workflow yang sudah berulang kali terbukti perlu)**:
`build_docx.py`'s `ANCHORS` mencocokkan prefix teks tetap
("That is, when the system is genuinely uncertain") untuk menyisipkan
Fig. 3 — prefix itu hilang begitu paragraf escalation ditulis ulang, jadi
Fig. 3 **diam-diam tidak pernah disisipkan** di build pertama (sama
persis pola bug ANCHORS yang sudah pernah ditemukan sebelumnya, tercatat
di "Workflow practice worth repeating" di memori project). Diperbaiki:
anchor diganti ke `"This targets gap G3"` (teks yang sekarang benar-benar
ada). Diverifikasi ulang: Fig. 1-5 semua muncul tepat 1x, PDF dibaca
halaman demi halaman (bukan cuma grep), penempatan figure masuk akal.

**File:** `src/content.py`, `src/build_docx.py` (fix ANCHORS),
`src/simulate_elo_item_prior.py` (baru), `paper/KO-DRILL_IEEE.docx`/`.pdf`.

---

### 9.30 Audit lanjutan ("cek lainnya") — 2 ketidaksesuaian klaim-vs-implementasi ditemukan dan diperbaiki, BUKAN soal bukti lemah

User minta cek bagian lain supaya tidak ada yang perlu di-reframe lagi di
masa depan. Beda kategori dari temuan Novelty 1/2 sebelumnya: dua hal ini
bukan "bukti belum cukup kuat" tapi **paper mengklaim sesuatu yang beda
dari yang sungguh dibangun** — lebih serius, harus diperbaiki terlepas
dari apakah hasilnya "melemahkan" kedengarannya, karena ini soal akurasi
faktual bukan soal kekuatan bukti riset.

**Temuan #1: KLUE-RoBERTa diklaim ter-deploy ("fine-tuned for difficulty
regression"), padahal yang sungguh jalan di `answer_processing.py`/
`question_bank.py` adalah `difficulty_heuristic.py` — heuristic
deterministik (jumlah morfem + kiwipiepy), BUKAN KLUE-RoBERTa sama
sekali.** Dicek langsung import statement di kode produksi, bukan
diasumsikan dari nama file. Section III-C/Fig. 1 lama menulis seolah
KLUE-RoBERTa sudah di-fine-tune dan dipakai — PRD sendiri sudah lama
jujur soal ini (§2.3: "benar-benar tidak ada dataset kalibrasi... bukan
soal waktu/usaha") tapi disclosure itu TIDAK PERNAH masuk ke teks paper.

**Diperbaiki, dan ini JUSTRU memperkuat Novelty 2, bukan melemahkannya**:
ditambah 1 paragraf jujur di Section III-C ("Stated plainly, not left for
a reader to discover in the code...") + Fig. 1 caption diperjelas
("target architecture KLUE-RoBERTa; a deterministic placeholder in the
evaluated system") + pembukaan Novelty 2 (IV-C) diperbaiki supaya tidak
lagi bilang "KLUE-RoBERTa's difficulty estimate... not calibrated" (salah
atribusi) jadi "the difficulty-prior estimator this system actually
runs... standing in for KLUE-RoBERTa". Argumennya: skenario "prior yang
tidak terkalibrasi" yang jadi dasar Novelty 2 bukan hipotetis — itu
PERSIS kondisi nyata sistem ini sekarang.

**Temuan #2: Section IV-B (verifier gate) mengklaim "morphological
analyser confirms distractor is not itself a correct answer" seolah
berlaku seragam untuk semua skill — padahal cuma benar untuk 3 dari 11
skill.** Dicek `SKILL_SPECS` langsung: hanya `particle_obj`, `subj_topic`,
`mod_adverb` punya tag correct/distractor yang BEDA (sehingga tag
mismatch benar-benar bisa menolak distractor yang kebetulan juga benar).
8 skill lain (particle_loc, honorific, time_range, tense_past,
irregular_conj, adj_relativizer, verb_choice, noun_counter) correct DAN
distractor-nya **tag SAMA** — keunikan jawaban di situ murni jaminan
desain template (diverifikasi manual saat skill dibangun, §9.9/§9.10/
§9.21/§9.22), BUKAN dicek otomatis per-item. Satu-satunya usaha membuat
cek semantik otomatis untuk ini (`solver_check.py`'s check (b)) sudah
lama diketahui yes-biased dan dikecualikan (§9.16) — tapi Section IV-B
lama tidak pernah mengakui keterbatasan ini, menulis seolah "morphological
analyser" menangani semuanya.

**Diperbaiki**: IV-B ditulis ulang eksplisit membedakan 3-skill
(tag-beda, benar-benar dicek otomatis) vs 8-skill (tag-sama, jaminan
desain + disclosure soal check (b) yang yes-biased). Section V-B (dulu
cuma rencana studi instruktur, belum ada angka sama sekali) ditambah 1
paragraf melaporkan SEMUA angka otomatis yang sudah ada (gen pass 100%,
solver key-accept 66.4%, critic 0 kali terpicu dengan alasannya, check
(b) dikecualikan dengan alasannya) — supaya pembaca tidak perlu menebak
dari PRD, semua ada di paper.

**Verifikasi PDF halaman-demi-halaman (bukan cuma grep)**: Fig. 1-5 tetap
masing-masing muncul tepat 1x setelah build ulang, persamaan (1)(2)(3)
tetap berurutan benar, tidak ada ANCHOR yang pecah lagi.

**File:** `src/content.py` saja (Section III-C, Fig. 1 caption, IV-B,
IV-C pembukaan, V-B). Tidak ada perubahan kode produksi — murni
menyamakan teks paper dengan kode yang sudah ada.

---

### 9.31 Dua pendekatan baru untuk check (b) dicoba secara empiris — KEDUANYA GAGAL, memperkuat kesimpulan §9.16 (bukan cuma belum ketemu prompt yang pas)

User minta investigasi mendalam apakah framing forced-choice bisa memperbaiki
yes-bias check (b), **implementasikan kalau memang terbukti lebih baik**.
Diuji langsung di GPU dengan Qwen3-4B asli (bukan dinalar di atas kertas),
pakai kasus yang kebenarannya sudah diketahui pasti (irregular_conj/
adj_relativizer = ungrammatical pasti, honorific/tense_past = pragmatically
mismatch tapi sintaksis valid, plus sanity check kasus yang harusnya benar):

**Hipotesis awal (salah, tapi worth menguji)**: mungkin yes-bias lama
muncul karena prompt lama ("apakah kalimat ini alami secara gramatikal?")
mencampur gramatikalitas dengan kesesuaian konteks (pragmatik) — dicoba
prompt baru yang eksplisit menyebut petunjuk konteks (penanda waktu,
subjek yang butuh honorifik) dan minta forced-choice "정답/오답".

**Hasil prompt baru (context-anchored forced-choice): 3/7 benar, LEBIH
BURUK dari yang lama** — bahkan membuat kasus sanity-check yang jelas
benar (honorific form yang correct) ikut salah dijawab "오답". Bukan
sekadar tetap bias yes — sekarang noise ke DUA arah, lebih tidak bisa
diprediksi daripada yes-bias yang konsisten.

**Dicoba pendekatan kedua, genuinely berbeda karakter**: perbandingan
berpasangan langsung (tunjukkan DUA kalimat sekaligus — versi benar dan
versi distractor — minta pilih "A saja/B saja/keduanya/tidak satupun").
Non-reasoning: 3/7 (di bawah chance level untuk pilihan 4-arah). Reasoning
mode (budget 300 token): 0/7, SEMUA terpotong di tengah `<think>` sebelum
sempat menjawab — bahkan butuh lebih dari 300 token cuma untuk mulai
menjawab pertanyaan pembanding sederhana ini, mengonfirmasi ulang temuan
lama (§9.16) soal reasoning mode dengan budget token besar (800) juga
tidak menyelesaikan masalah, bukan soal budget yang kurang besar.

**Kesimpulan, 3 framing berbeda sekarang semua gagal dengan pola
kegagalan BERBEDA satu sama lain** (yes-bias konsisten / noise dua-arah /
di-bawah-chance pairwise) — ini memperkuat, bukan sekadar mengulang,
simpulan §9.16: ini keterbatasan kapabilitas genuine Qwen3-4B untuk tugas
penilaian gramatikalitas halus ini, bukan masalah prompt yang belum
ketemu formatnya. **TIDAK diimplementasikan** — sesuai prinsip yang
disepakati user sendiri sesi ini: kalau investigasi menunjukkan memang
tidak ada solusi, jangan dipaksakan/direframe, laporkan jujur.

**Satu catatan ditemukan di tengah pengujian**: kasus irregular_conj
gagal di SEMUA framing termasuk pairwise — ini kemungkinan BUKAN murni
soal yes-bias, tapi tumpang tindih dengan limitasi terpisah yang sudah
didokumentasikan (§9.16/Section V-B paper): Qwen3-4B memang punya blind
spot spesifik untuk pola kontraksi 르-irregular, dikonfirmasi independen
di check (c) juga. Jangan pakai kasus irregular_conj lagi sebagai test
case murni untuk yes-bias di masa depan — kasusnya tercemar oleh
limitasi lain.

**Arah yang belum dicoba, di luar scope sesi ini (dicatat, bukan
dikerjakan)**: cek berbasis frekuensi corpus (bukan LLM-judge sama
sekali) untuk pasangan partikel yang ambigu — butuh akses corpus Korea
berskala besar yang tidak tersedia di lingkungan ini sekarang.

**File:** tidak ada perubahan kode — skrip uji ada di scratchpad sesi,
tidak dipermanenkan karena hasilnya negatif (tidak ada yang perlu
direproduksi).

---

### 9.32 Bug nyata ditemukan dari sisi "user"/personalisasi — kode serving produksi diam-diam pakai target 50%, bukan 85% yang diklaim paper, DIPERBAIKI DI KODE (bukan paper)

User minta audit lanjutan dari sisi lain (distractor, rating system,
question, **answer, user**). Dicek jalur seleksi item sungguhan
(`question_bank.py::BankContentProvider.next_new_grammar_item` — satu-
satunya tempat seleksi adaptif nyata terjadi, `ExaoneGrammarProvider`
tidak pernah memilih di antara kandidat karena generate satu per
panggilan) terhadap klaim Section IV-D Eq. 2 (target sukses 85%, demi
pedagogi, bukan 50% demi presisi pengukuran).

**Ditemukan: kode produksi diam-diam mengimplementasikan target 50%
(mencocokkan item persis ke kemampuan learner, `abs(beta_hat - theta)`,
TANPA offset apa pun), bukan 85%.** Komentar kode sendiri bahkan jujur
menyebut ini "sama persis prinsip target ~50% sukses seperti
`simulate_trackers.pick_adaptive`" — defaultnya memang 0.5. Ini **beda
arah** dari temuan sebelumnya (KLUE-RoBERTa, minimal-pair-probe): di
sana PAPER yang mengklaim sesuatu yang tidak ada di kode; di sini KODE
yang diam-diam melakukan hal berbeda dari yang diklaim paper DAN
divalidasi (`validate_item_discrimination.py` yang saya bangun sendiri
mengukur biaya kebijakan 85% — tapi kebijakan itu TIDAK PERNAH benar-
benar jalan di jalur serving produksi).

**Diperbaiki di KODE, bukan paper** — karena argumen desain 85%
(Wilson et al. [27], "desirable difficulty") sudah benar dan sudah
divalidasi lewat discrimination index (§9.26: masih "good" di 0.34 vs
0.408 maksimal-informasi). Target=50% yang sebelumnya jalan diam-diam
justru persis "classical adaptive testing" yang paper argumentasikan
MELAWAN. `question_bank.py`: tambah `_TARGET_SUCCESS=0.85` +
`_LOGIT_OFFSET=log(0.85/0.15)` sebagai konstanta eksplisit (bukan angka
ajaib tersebar), sort key diubah dari `abs(beta_hat - theta)` jadi
`abs(beta_hat - (theta - _LOGIT_OFFSET))` — formula sama persis dengan
yang sudah divalidasi di `simulate_elo_item_prior.py`'s `pick_adaptive`.

**Smoke test diverifikasi ulang, bukan diasumsikan tetap lulus**:
dihitung manual dulu (logit_estimate tiap urn di `difficulty_spread`)
sebelum ubah kode — ketemu bahwa assersi lama untuk `strong_picks`
(`⊆ {hard-1,hard-2,medium}`) akan PECAH di target 85%, karena learner
kuat sekarang benar-benar diarahkan ke item yang memberi ~85% sukses
BUKAN item sesulit kemampuannya sendiri (yang cuma kasih ~50%) — `hard-2`
digantikan `easy-2` di top-3. Ini bukan bug baru, ini PERSIS perilaku
yang diinginkan Eq. 2: learner kuat tidak lagi diberi item paling sulit
yang tersedia, tapi item yang bikin dia "merasa berhasil" ~85% dari
waktu. Assersi diperbarui sesuai perhitungan, dijalankan ulang
(`question_bank.py`, `daily_drill.py`, `answer_processing.py`), semua
smoke test lulus dengan GPU asli, bukan stub.

**Paper TIDAK perlu diubah** — berbeda dari temuan-temuan sebelumnya,
klaim paper (Eq. 2, Section IV-D, analisis discrimination index) sudah
BENAR dari awal; yang salah adalah implementasinya. Setelah fix ini,
untuk pertama kalinya klaim 85% itu benar-benar match dengan apa yang
diserve ke user sungguhan — sebelumnya cuma benar di simulasi validasi,
sekarang juga benar di jalur produksi.

**File:** `src/question_bank.py` saja.

---

### 9.33 User minta verifikasi: "kalau kode mengikuti paper, apakah paper-nya sendiri sudah benar?" — dicek untuk fix §9.32, ketemu klaim 85% kurang kuat dasarnya, diperbaiki

Pertanyaan tepat setelah saya memperbaiki `question_bank.py` supaya
mengikuti klaim 85% paper (§9.32) — sebelum menganggap paper itu "sudah
benar jadi kode harus ikut", saya verifikasi dulu klaim 85%-nya sendiri,
bukan diasumsikan benar cuma karena sudah tertulis.

**Dicek sitasi Wilson et al. [27] langsung (web search ke paper
aslinya)**: judul/jurnal/tahun/DOI cocok. TAPI — temuan penting —
**penulisnya sendiri eksplisit mengingatkan jangan generalisasi
berlebihan**: 85% itu diturunkan khusus untuk binary classifier berbasis
gradient-descent, bukan untuk pembelajar manusia secara umum, dan Wilson
sendiri bilang "85 percent is [not] some magic number we should be aiming
at all the time." Paper KO-DRILL sebelumnya menulis "Wilson et al. show
that for a broad class of learning systems..." — melebih-lebihkan cakupan
asli temuan itu.

**Lebih parah**: kalimat "the pedagogical literature on desirable
difficulty points the same way" **sama sekali tidak ada sitasinya** —
klaim kosong tanpa rujukan, padahal konsep "desirable difficulties" itu
benar nyata (Bjork, 1994/2011) dan seharusnya mudah disitasi.

**Diperbaiki, bukan dihapus** — karena argumen kualitatifnya (tantangan
moderat lebih baik dari terlalu mudah/sulit) tetap valid dan punya dasar
riil di literatur manusia, cuma butuh sitasi yang benar dan framing yang
jujur soal seberapa jauh Wilson et al. sebenarnya mendukung. Ditambahkan
sitasi baru [44] (Bjork & Bjork 2011, "Making Things Hard on Yourself,
But in a Good Way" — dicek detail publikasi lengkap via web search,
bukan ditebak) dan paragraf Section IV-D ditulis ulang: eksplisit
mengakui Wilson et al. cuma tentang binary classifier gradient-descent
BUKAN pembelajar manusia, dan menyebut 85% sebagai "principled starting
point," BUKAN "value proven optimal for Korean grammar drilling" —
posisi yang sekarang jujur selaras dengan discrimination-index check
yang sudah ada (§9.26) sebagai bukti PENDUKUNG tambahan, bukan bukti
utama.

**Kesimpulan untuk pertanyaan user**: ya, setelah diperiksa, argumen
85%-nya tetap sah dipertahankan (bukan harus diganti/dihapus) — cuma
derajat kepastiannya tadinya dilebih-lebihkan. Fix kode di §9.32 tetap
benar untuk dilakukan.

**Dua temuan tambahan dari audit lanjutan sesi ini, perbaikan kecil**:
1. Section V-A mengklaim "TOPIK-aligned syllabus... 120 discrete
   grammatical points" — dicek `vocab_provider.py` dan seluruh `src/`,
   **tidak ada struktur data level/TOPIK apa pun**, cuma 11 skill yang
   sungguh dibangun. Diperbaiki: dipisahkan eksplisit "target kurikulum"
   (120 poin, aspirational) vs "yang sungguh dibangun & dievaluasi" (11
   skill), supaya pembaca tidak menyimpulkan hasil V-B berlaku untuk
   seluruh 120 poin.
2. Section "Threats to Validity" ditambah 1 poin (jadi "Four threats"):
   feedback bebas-bentuk Qwen3 cuma diperbaiki otomatis untuk SATU jenis
   kesalahan (konjugasi) dan diuji bersih dari 2 kesalahan lain
   (bahasa salah, kebocoran topik dari contoh) di N=4 trial — belum
   diaudit menyeluruh untuk kesalahan faktual lain. Sudah lama
   terdokumentasi di `qwen_feedback.py`'s docstring, belum pernah masuk
   teks paper.

**File:** `src/content.py` saja (Section IV-D, V-A, Threats to Validity,
+1 referensi baru [44]). Diverifikasi di PDF halaman demi halaman, Fig.
1-5 tetap muncul tepat 1x.

---

### 9.34 Cek "not also correct" otomatis genuinely berfungsi ditemukan — bukan LLM, murni skor parsing corpus kiwipiepy — menutup sebagian gap §9.30

User minta implementasikan hal yang masih tertunda. Dicoba 1 arah yang
sebelumnya dicatat "belum dicoba, butuh corpus eksternal" (§9.31) —
ternyata TIDAK butuh corpus eksternal sama sekali: `kiwipiepy`'s
`Kiwi.analyze()` sudah mengembalikan skor (log-probability-like) dari
language model internal yang dilatih di corpus Korea — sinyal
corpus-frequency asli, BUKAN LLM judge, arsitektur sama sekali berbeda
dari 3 percobaan yang sudah gagal (§9.16, §9.31).

**Diuji dulu secara empiris** (bukan diasumsikan) di 4 tipe skill pakai
kasus yang kebenarannya pasti: `irregular_conj` (gap benar vs salah:
+3.0), `adj_relativizer` (+12.3 — gap besar), `particle_loc` (+4.8),
`honorific` (**-1.9, terbalik** — sesuai dugaan: honorific bergantung
konteks sosial yang tidak terwakili di frekuensi lokal).

**Diuji skala N=20/skill di seluruh pasangan terkurasi 8 skill same-tag**
(skill yang sebelumnya TIDAK punya cek otomatis apa pun untuk "distractor
juga tidak kebetulan benar"): HASIL BERSIH TERBELAH — 5 skill reliable
(particle_loc 7/7, time_range 5/5, irregular_conj 3/3, adj_relativizer
6/6, verb_choice 5/5 — semua gap positif konsisten), 3 skill TIDAK
reliable (honorific 1/5, tense_past 2/5, noun_counter 2/5 — gap negatif
rata-rata, sinyal justru terbalik). Pola ini masuk akal: 5 skill reliable
semua soal struktur kalimat LOKAL (partikel vs frasa langsung sekitarnya,
konjugasi), 3 skill tidak reliable semua butuh petunjuk EKSTERNAL
(register sosial honorific, penanda waktu terpisah, konvensi kelas kata
benda arbitrer) yang tidak terwakili di skor frekuensi lokal.

**Diimplementasikan**: `distractor_validator.py` dapat fungsi baru
`lm_plausibility_gap()` + `validate_lm_plausibility(skill_id, ...)` —
secara eksplisit SKIP (valid=True dengan alasan jelas, bukan diam-diam
lolos) untuk skill di luar `LM_PLAUSIBILITY_RELIABLE_SKILLS` (set 5
skill di atas), supaya caller yang cek semua skill seragam tidak diam-
diam mempercayai sinyal yang sudah terbukti salah arah untuk 3 skill
lainnya. Smoke test baru memverifikasi KEDUANYA — skill reliable harus
lolos via gap positif ASLI, skill honorific harus lolos via jalur SKIP
eksplisit (bukan kebetulan benar).

**Skrip validasi permanen baru**: `validate_lm_plausibility.py` (CPU-only
untuk pengecekan sendiri, GPU cuma untuk constructor `ExaoneGrammarProvider`
yang generate item nyata) — dijalankan di N=20/skill untuk 5 skill
reliable, hasil: **26/26 (100%)** — ini SELURUH pasangan terkurasi yang
ada untuk 5 skill itu saat ini (bukan sampel), jadi angka ini akan
berubah KOMPOSISINYA (bukan rate-nya) begitu pasangan baru dikurasi.

**Paper diupdate** (Section IV-B + V-B): sekarang eksplisit menyebutkan
3 framing LLM-judge yang dicoba dan gagal (bukan cuma 1 seperti
sebelumnya) DAN cek non-LLM baru yang berhasil untuk 5/8 skill. Gap
"tidak ada cek otomatis" sekarang menyusut dari 8 skill jadi 3 skill
(honorific, tense_past, noun_counter) yang MEMANG genuinely butuh
konteks eksternal yang tidak bisa diwakili sinyal lokal apa pun —
bukan kegagalan mencari, tapi batasan struktural yang sudah dibuktikan.

**File:** `src/distractor_validator.py` (fungsi baru + smoke test),
`src/validate_lm_plausibility.py` (baru), `src/content.py` (IV-B, V-B).
Diverifikasi di PDF halaman demi halaman, Fig. 1-5 tetap tepat 1x.

---

### 9.35 User: "jangan berhenti karena belum tercatat di paper/PRD, paper harus ikuti diskusi kita" — ditemukan solusi lebih kuat dari §9.34, cakupan naik dari 5/8 jadi 10/11 skill

User eksplisit: kalau sesuatu belum ada di paper/PRD, itu bukan alasan
berhenti — investigasi terus, dan kalau perlu sumber daya tambahan
(corpus eksternal) untuk hasil bagus, go for it. Direspons dengan
investigasi dulu APAKAH corpus eksternal benar-benar yang dibutuhkan
untuk 3 skill yang gagal di §9.34 (honorific/tense_past/noun_counter) —
ternyata BUKAN. Masalahnya bukan "kurang data corpus", tapi kiwipiepy's
scoring model terlalu sederhana (statistik lokal, tidak benar-benar
"mengingat" konteks jauh seperti kata "어제" atau subjek berhormat
beberapa kata sebelumnya).

**Dicoba pendekatan berbeda**: bukan tanya LLM "apakah ini benar?" (yang
3 kali terbukti gagal karena bias — §9.16, §9.31), tapi baca langsung
**log-probability Qwen3-4B sendiri** untuk kalimat yang sudah diisi,
teacher-forced — ini SCORING, bukan JUDGING, jadi acquiescence bias tidak
berlaku sama sekali (tidak ada pertanyaan untuk "disetujui").

**Diuji di 5 kasus known-truth dulu**: SEMUA 5 benar, termasuk 3 yang
gagal di kiwipiepy (honorific, tense_past, noun_counter-style). **Diuji
skala N=20/skill di SEMUA 11 skill**: 49/56 (87.5%) pada run pertama,
dengan pola jelas — `subj_topic` gagal total (0/4, SESUAI DUGAAN, bukan
kegagalan metode — independen mengkonfirmasi ulang §9.11 bahwa itu
konvensi tekstual bukan aturan keras), `noun_counter` lemah (3/5,
diselidiki lebih jauh: counter generik 개 kadang mengalahkan counter
spesifik-tapi-jarang seperti 권/분 murni soal frekuensi mentah — pola
yang bisa dijelaskan, bukan noise).

**Diimplementasikan di `solver_check.py`** (reuse Qwen3-4B yang sudah
dimuat untuk check (c), tidak ada model baru): `sentence_logprob()` +
`log_likelihood_check(skill_id, ...)`, `LOGPROB_RELIABLE_SKILLS` (10 dari
11 skill, `subj_topic` dikecualikan by design) + `LOGPROB_SKILL_NOTES`
untuk mendokumentasikan keterbatasan noun_counter secara eksplisit.
Smoke test baru memverifikasi kedua jalur (skill reliable lolos via gap
asli, subj_topic lolos via skip eksplisit, bukan kebetulan).

**Skrip validasi permanen**: `validate_log_likelihood.py`, dijalankan
N=20/skill di 10 skill: **49/52 (94.2%)** — 9 dari 10 skill nyaris
sempurna, `noun_counter` 3/5 dengan alasan jelas.

**Ini LEBIH KUAT dari §9.34's hasil kiwipiepy (5/8 skill)** — cakupan
naik ke **10/11 skill** (dari sebelumnya cuma 3/11 lewat tag mismatch).
Gap "tidak ada cek otomatis untuk distractor juga-benar" sekarang
tersisa HANYA untuk `subj_topic` — dan itu BUKAN gap yang perlu
ditutup, karena item itu memang secara desain tidak punya satu jawaban
benar yang keras (§9.11).

**Paper diupdate** (Section IV-B + V-B): narasi sekarang menjelaskan
progresi penuh — 3 framing LLM-judge gagal → insight "scoring bukan
judging" → kiwipiepy (5/8) → Qwen3 log-prob (10/11, lebih kuat karena
pemahaman kontekstual lebih dalam dari parser statistik sederhana).

**File:** `src/solver_check.py` (fungsi baru + smoke test),
`src/validate_log_likelihood.py` (baru), `src/content.py` (IV-B, V-B).
Diverifikasi di PDF halaman demi halaman, Fig. 1-5 tetap tepat 1x.

---

### 9.36 "Kenapa tidak pakai EXAONE?" — dicek, EXAONE jauh lebih buruk untuk tugas ini meski Korea-first, pilihan Qwen3 ternyata benar tapi sebelumnya tidak diverifikasi

User tanya kenapa `log_likelihood_check` (§9.35) pakai Qwen3-4B, bukan
EXAONE — padahal paper sendiri (Section III-C) berargumen EXAONE lebih
baik untuk Korea karena Korea-first training. Pertanyaan sah: pilihan
Qwen3 sebelumnya cuma warisan dari `solver_check.py` yang sudah ada
(dipilih untuk alasan independensi dari generator di check (c)), bukan
hasil perbandingan langsung untuk tugas scoring yang baru ini.

**Diuji langsung** — EXAONE-4.0-1.2B dengan metode sama persis (log-
probability teacher-forced) di N=20/skill, 11 skill:

| Model | Overall |
|---|---|
| Qwen3-4B | 49/56 = 87.5% |
| EXAONE-4.0-1.2B | **27/56 = 48.2%** |

**EXAONE jauh lebih buruk** — bahkan DI BAWAH 50% (lebih buruk dari
tebak acak) khusus di `particle_loc` (0/7!). Kemungkinan penyebab:
ukuran model (1.2B vs 4B) lebih dominan daripada keunggulan Korea-first
training untuk tugas nuansa gramatikal halus semacam ini — bukan
assumption, EXAONE memang dilatih Korea-first (per Section III-C), tapi
kapasitas parameter yang jauh lebih kecil kalah telak di tugas ini.

**Kesimpulan: pilihan Qwen3 tetap benar, sekarang dengan bukti, bukan
kebetulan warisan kode lama.** Ditambahkan 1 kalimat jujur di Section
IV-B paper: EXAONE dicek, bukan diasumsikan, dan hasilnya dilaporkan
apa adanya (48.2% vs 87.5%) — termasuk temuan spesifik yang
counter-intuitive (model Korea-first bisa kalah dari model multilingual
lebih besar untuk tugas tertentu) alih-alih diam-diam dipilih tanpa
penjelasan.

**File:** `src/content.py` saja (IV-B). Tidak ada perubahan kode — ini
investigasi perbandingan model, bukan implementasi baru. Skrip uji
ad-hoc di scratchpad, tidak dipermanenkan (hasil negatif untuk EXAONE,
tidak perlu direproduksi rutin).

---

### 9.37 Perplexity sebagai proksi kesulitan item — diuji, GAGAL jelas, tidak diimplementasikan

Lanjutan audit: dicoba pendekatan "scoring bukan judging" (insight yang
baru berhasil di §9.35) untuk masalah prediksi kesulitan item
(difficulty_heuristic.py) yang sebelumnya ditolak kalau basisnya
judgment LLM (§9.36/turn sebelumnya — weak-supervision dari LLM ditolak
karena bias sudah terbukti di literatur [12]). Hipotesis: mean
log-probability per-token (surprisal/perplexity) dari Qwen3 bisa jadi
sinyal kesulitan otomatis yang berbeda — berakar di teori surprisal
psikolinguistik asli (Hale 2001/Levy 2008), bukan dari judgment LLM yang
bias.

**Diuji dengan pasangan kalimat yang urutan kompleksitasnya sudah jelas
secara independen** (jumlah klausa, kehadiran honorific/subordinat/
pasif — bukan ditebak dari LLM): **2 dari 4 pasangan hasilnya TERBALIK**
— kalimat honorific+subordinat kompleks skor "lebih mudah" dari kalimat
sederhana; kalimat panjang skor "lebih mudah" dari kalimat pendek.

**Penjelasan (bukan cuma noise)**: perplexity mengukur predictability
teks bagi MODEL yang dilatih di teks fasih native, bukan kesulitan bagi
PEMBELAJAR. Kalimat formal/honorific yang dikonstruksi benar justru bisa
sangat predictable (pola baku/kolokasi umum sering muncul di corpus),
sama sekali tidak mencerminkan beban belajar strukturnya.

**Tidak diimplementasikan.** Memperkuat kesimpulan sebelumnya:
`difficulty_heuristic.py`'s pendekatan (hitung morfem + kehadiran tag
gramatikal lanjutan) punya logika konseptual yang lebih tepat
(kompleksitas struktural ≈ beban belajar) dibanding 2 pendekatan
berbasis model yang sudah diuji dan gagal (judgment-bias §9.36 sesi
lalu, dan surprisal-tidak-relevan di sini) — bukan kebetulan dipilih,
tapi sekarang punya 2 alternatif yang sudah dibuktikan lebih buruk.

**File:** tidak ada — hasil negatif, skrip uji di scratchpad saja.

---

### 9.38 Separuh trigger eskalasi (Eq. 3) sama sekali tidak terimplementasi — ditemukan dan diperbaiki di kode

Lanjutan audit. Eq. 3 paper: `escalate ⟺ w_k > τ ∨ consecutive(m_j) ≥ 2`
— dicek langsung, `should_escalate()` cuma mengimplementasikan disjunct
PERTAMA (width). Grep `consecutive` di seluruh `src/`: **nol hasil** di
luar yang baru saya tulis. `answer_processing.py` memanggil
`should_escalate(new_learner_urn)` TANPA informasi misconception apa
pun — separuh kondisi OR itu tidak pernah ada tempat untuk dibaca dari
mana pun di kode. Sama kategorinya dengan temuan minimal-pair-probe
(§9.sebelumnya) tapi arahnya beda: ini bukan fitur fiktif yang perlu
dihapus, tapi fitur nyata, implementable, dan sudah dispesifikasi jelas
di paper — jadi diperbaiki di KODE (pola yang sama dengan fix 85% di
§9.32).

**Diimplementasikan**:
- `urnings_tracker.py`: `LearnerSkillProfile` dapat `_consecutive: Dict[skill_id, (kagas_code, count)]`
  + method `record_outcome(skill_id, correct, misconception_code)` (reset
  ke 0 saat benar ATAU saat kagas_code None — misalnya subj_topic yang
  memang `kagas_code=None` by design, §9.11 — jadi tidak pernah salah
  mengakumulasi; increment cuma kalau kode SAMA PERSIS dengan yang
  terakhir, restart ke 1 kalau beda) + `consecutive_misconceptions(skill_id)`.
  `to_dict`/`from_dict` diperbarui ikut menyimpan state ini.
- Fungsi baru `should_escalate_full(learner, consecutive_count, tau)` =
  `should_escalate(learner, tau) OR consecutive_count >= 2` — implementasi
  Eq. 3 LENGKAP, bukan cuma separuh.
- `answer_processing.py`: `submit_grammar_answer` sekarang memanggil
  `skill_profile.record_outcome(item.skill_id, correct, item.kagas_code)`
  (data `kagas_code` ini SUDAH ADA di `GeneratedItem` sejak §9.27 —
  prasyaratnya kebetulan sudah terpasang, tinggal disambungkan) dan pakai
  `should_escalate_full(...)` menggantikan `should_escalate(...)`.

**Smoke test baru** (`urnings_tracker.py`) memverifikasi mekanisme
secara terisolasi dari width: urn SENGAJA dibuat sempit (k=29, width
di bawah τ) supaya width SENDIRI tidak memicu eskalasi — lalu
dikonfirmasi 1 jawaban salah TIDAK eskalasi, 2 jawaban salah berturut-
turut dengan misconception SAMA baru eskalasi, misconception BEDA
me-reset hitungan ke 1 (bukan lanjut), jawaban benar me-reset ke 0, dan
skill dengan `kagas_code=None` (subj_topic) tidak pernah terpicu sama
sekali. Dijalankan dengan GPU asli (`answer_processing.py`'s smoke test
penuh), semua lulus.

**Paper TIDAK perlu diubah** — Eq. 3 dan teksnya sudah akurat
menggambarkan desain yang dimaksud dari awal; sekarang untuk pertama
kalinya juga akurat menggambarkan implementasi.

**File:** `src/urnings_tracker.py`, `src/answer_processing.py`.

---

### 9.39 Klaim "half-life regression" scheduling ambigu — kedengarannya berlaku untuk grammar juga, padahal cuma vocab; grammar pakai mekanisme berbeda yang tidak pernah disebut

Lanjutan audit. Section IV-D (dulu) menulis "Scheduling across sessions
follows a spacing model in the tradition of half-life regression [28]"
tepat setelah membahas kebijakan SELEKSI ITEM GRAMMAR — pembaca wajar
menyimpulkan ini berlaku untuk grammar juga. Dicek `grammar_scheduler.py`
langsung: **grammar SAMA SEKALI tidak pakai FSRS/half-life regression**
— pakai `MASTERED_RECHECK_DAYS=30`, interval TETAP, eksplisit placeholder
yang sudah lama didokumentasikan jujur di modul itu sendiri (dicoba
turunkan dari matematika FSRS, GAGAL transfer — interval meledak jadi
puluhan ribu hari karena FSRS mengasumsikan review terjeda, bukan banyak
respons-benar-berturut di Urnings). Temuan nyata dan substantif ini
**tidak pernah masuk teks paper sama sekali** — cuma ada di komentar
kode dan PRD.

**Diperbaiki**: paragraf ditulis ulang eksplisit memisahkan (a) vocab —
half-life regression [28] genuinely dipakai (`vocab_scheduler.py` benar
wrap `fsrs` library), (b) grammar — mekanisme berbeda (mastery-gated,
interval tetap 30 hari, eksplisit placeholder), plus temuan FSRS-tidak-
transfer dilaporkan sebagai hasil negatif yang jujur, bukan disembunyikan
atau dibiarkan tersirat seolah mekanisme yang sama berlaku untuk
keduanya.

**Tidak ada perubahan kode** — ini murni soal keakuratan teks paper vs
kode yang sudah benar dan sudah lama jujur tentang keterbatasannya
sendiri, cuma belum pernah "naik" ke level paper.

**File:** `src/content.py` saja (Section IV-D).

---

### 9.40 Sitasi Liu et al. [12] salah representasi — paper bilang "divergensi sistematis", aslinya korelasi tinggi dengan data manusia

Melanjutkan pengecekan yang session lokal lain sempat mulai tapi kena
limit sesi di tengah jalan (lihat riwayat §9.39 — transcript yang
dikutip user berhenti tepat di langkah ini). Diverifikasi ulang dari
nol di sini, bukan dipercaya dari transcript: dua pencarian web
independen terhadap paper asli (Liu, Bhandari & Pardos, "Leveraging
LLM-Respondents for Item Evaluation," arXiv 2407.10899 / BJET 2025,
DOI 10.1111/bjet.13570 — versi jurnal sendiri tidak berhasil diambil
langsung karena WebFetch gagal resolve DNS di sandbox ini, jadi
mengandalkan preprint arXiv + ringkasan independen kedua, bukan cuma
satu sumber).

**Klaim di paper kita (sebelum fix), dua tempat:**
- Introduction: *"LLM self-estimates of item difficulty are poorly
  calibrated, systematically biased towards judging items easier than
  they are, and unreliable at the level of the individual item [11],
  [12]."*
- Section III-B: *"Liu et al. ... find systematic divergence from
  human response data [12]. Independent evaluations... report accuracy
  well below what operational use would require, with a consistent
  bias towards rating items as easier than they prove to be."* (kalimat
  kedua ini malah TIDAK punya sitasi sama sekali).

**Temuan asli Liu et al. (dikonfirmasi 2x, konsisten):** parameter item
yang dikalibrasi dari respons LLM punya korelasi TINGGI dengan
kalibrasi manusia (Spearman >0.8 untuk model terbaik yang diuji, naik
ke 0.93 dengan augmentasi resampling) — **bukan divergensi sistematis**.
Keterbatasan sebenarnya jauh lebih sempit: satu LLM punya distribusi
kemampuan yang lebih narrow & kurang variatif dari populasi manusia
asli, butuh ENSEMBLE beberapa model untuk menyamai distribusi itu.
Tidak ada temuan "bias menilai soal lebih mudah" di paper ini sama
sekali — klaim itu sepertinya salah atribusi (mungkin genuinely milik
sumber lain, tidak diverifikasi di sini, atau generalisasi penulis yang
tidak bersumber).

**Fix:** kedua paragraf ditulis ulang supaya akurat ke temuan asli —
korelasi item-level tinggi, keterbatasan di distribusi kemampuan
populasi/variabilitas respons (bukan "divergensi" blanket), dan
argumen diperjelas: yang relevan untuk pipeline SATU-generator seperti
sistem kita bukan "LLM tidak bisa menilai kesulitan", tapi "satu LLM
bukan populasi pembelajar — hasil Liu et al. tidak melisensikan
LLM-tunggal menilai kesulitan soalnya sendiri". Argumen inti paper
(encoder kecil [11] lebih baik dari LLM besar untuk regresi kesulitan)
tidak berubah — itu klaim terpisah, sudah diverifikasi benar
sebelumnya di sesi ini.

**Diverifikasi:** `python3 -m py_compile src/content.py` lolos,
`python3 src/build_docx.py` rebuild `paper/KO-DRILL_IEEE.docx` tanpa
error.

**Belum diverifikasi** (dicatat jujur, bukan diam-diam dilewati): versi
JURNAL (BJET 2025, bukan preprint arXiv 2024) tidak berhasil diakses
langsung karena keterbatasan jaringan sandbox — kemungkinan kecil ada
perbedaan angka antara preprint dan versi final, meski arah temuan
(korelasi tinggi, bukan divergensi) sangat tidak mungkin berbalik total
di proses peer review.

---

### 9.41 Audit lanjutan: 3 sisa inkonsistensi ditemukan setelah branch disinkronkan — nama sitasi, docstring basi, dokumen sekunder basi

User minta audit diteruskan ("jangan karena di paper/PRD tidak ada, malah
tidak diperbaiki... investigasi, kalau meyakinkan go for it"). Hasil
setelah branch disinkronkan ke `jessie-local-work` (§9.40 dst — sebagian
besar temuan session itu sudah solid, bukan reframe lemah, lihat
verifikasi di bawah):

1. **`validate_novelty2.py`'s docstring basi** — klaim `corr(width,
   error) = -0.81/-0.78/-0.78` untuk n=32/64/128 ternyata TIDAK cocok
   lagi dengan output aktual skrip (dijalankan 2x, deterministik,
   seed tetap): **-0.755/-0.849/-0.040**. Kemungkinan besar karena
   `urnings_tracker.py` berubah (dari fix tau + consecutive-misconception
   tracking) setelah docstring ditulis. Klaim di PAPER (-0.76 untuk n=32)
   tetap benar/cocok dengan kode saat ini — cuma dokumentasi internal
   skrip yang basi. Diperbaiki: angka diganti ke hasil aktual, klaim
   "reproduced at n=64/128" dan "+0.09 at n=256" dihapus (tidak lagi bisa
   direproduksi dari skrip ini saat ini — n=256 bahkan tidak diuji skrip
   permanen ini sama sekali), diganti deskripsi jujur: kuat negatif di
   n=32/64, nyaris nol di n=128, bukan tren mulus seperti klaim lama.

2. **Nama penulis "Lin et al." tersisa di body text** meski nomor sitasi
   [29] sudah diarahkan ke paper yang benar (Thomas et al.) oleh session
   sebelumnya — cuma SATU baris di Section II-F (gap G3), tapi tetap
   inkonsistensi nyata (pembaca yang cek nomor [29] di References akan
   lihat "Thomas et al." sementara teksnya bilang "Lin et al."). Diperbaiki.

3. **`docs/literature-review-15.md` ketinggalan jauh** — dokumen kurasi
   terpisah ini masih pakai skema lama sepenuhnya: G1-G4 (4 gap, bukan
   3), "Novelty 1/2/3" versi lama (sebelum Novelty 1 lama dihapus §9.18
   dan Novelty 2 diganti jadi robustness), DAN entri #10 masih salah
   total (metadata "J. Lin... Retrieval-Augmented Lesson Generation"
   padahal deskripsinya tentang temuan 885-learner/2-dari-7-lesson milik
   Thomas et al. — sitasi [29] di paper utama sudah benar, tapi dokumen
   sekunder ini belum ikut diperbaiki). **Ditulis ulang menyeluruh**:
   G1-G4→G1-G3 (gap lama soal "generation/calibration dipisah" dihapus,
   bukan disimpan sebagai entri yatim), semua referensi "Novelty 1/2/3"
   lama diganti ke penomoran saat ini (2 novelty: distractor=1,
   robustness=2; eskalasi bukan novelty), entri #10 diganti metadata
   lengkap ke Thomas et al. dengan catatan koreksi eksplisit, entri #4
   (Liu et al.) diperbaiki sinkron dengan fix §9.40 di `content.py`,
   "Net observation" diperbarui "tiga novelty"→"dua novelty".

**Diverifikasi:** `python3 -m py_compile src/content.py
src/validate_novelty2.py` lolos; `python3 src/build_docx.py` rebuild
`paper/KO-DRILL_IEEE.docx` tanpa error.

**Area lain yang dicek dan ditemukan SOLID, bukan reframe lemah** (dicatat
supaya tidak diperiksa ulang tanpa alasan baru): paragraf 85%-target vs
discrimination-index (angka 0.41 vs 0.34, "18% relative drop" — ada
metodologi simulasi konkret di baliknya, bukan klaim kosong); demosi
eskalasi dari novelty jadi application policy (didukung bukti negatif
langsung, bukan dihindari); caption Fig. 3 sudah konsisten "application
policy" bukan "Novelty 2"; Discussion/Conclusion (Section VI) konsisten
dengan state saat ini; scope-note growing-pool-regime (Section III-B)
jujur soal keterbatasan non-monotonic yang belum dipahami penuh,
bukan ditutupi.

---

### 9.42 Audit reproducibility: 3 dari 5 skrip validasi tidak punya artefak hasil tersimpan — dijalankan & dicommit sekarang, 2 sisanya butuh GPU user

User minta audit untuk menguatkan novelty berdasarkan hasil eval metric.
Dicek: dari 5 skrip validasi permanen (`validate_novelty1.py`,
`validate_novelty2.py`, `validate_item_discrimination.py`,
`validate_lm_plausibility.py`, `validate_log_likelihood.py`), cuma
`validate_novelty1.py` yang pernah punya hasil ter-commit
(`1_validation_results.json`). Tiga lainnya menghasilkan angka yang
dikutip langsung di paper (0.41/0.34 discrimination index, 76-82%
escalation rate, 18-24% hemat biaya, 100% lm_plausibility di 5 skill)
tanpa artefak apa pun untuk diverifikasi ulang — persis jenis gap yang
sudah dua kali ditemukan sebagai masalah nyata (§9.40, §9.41: klaim
paper vs kode/dokumentasi yang diam-diam melenceng seiring waktu).

**Dijalankan dan dicommit sekarang** (2 dari 3 tidak butuh GPU sama
sekali, 1 pakai teknik stub torch/transformers yang sudah terbukti di
§9.21/§9.22 untuk menghindari `ExaoneGrammarProvider.__init__`'s
`.to("cuda")`):
- `item_discrimination_results.json` — murni numpy, langsung jalan.
  Hasil: 50%-target discrimination=0.408, 85%-target=0.335, relative
  drop=17.9% — **cocok persis** dengan klaim paper (0.41/0.34/18%).
- `2_validation_results.json` — murni numpy (Urnings tracker). Hasil
  escalation rate 82.2%, hemat biaya 17.8% (≈18%, tepat di ujung bawah
  rentang klaim paper 76-82%/18-24% — konsisten, bukan penyimpangan).
- `lm_plausibility_results.json` — via stub, generate 5 skill
  (particle_loc, time_range, irregular_conj, adj_relativizer,
  verb_choice) lewat jalur kurasi. Hasil: **100% persis** cocok dengan
  klaim paper soal 5 skill yang reliable untuk check ini.

**Belum bisa diverifikasi di sini** (butuh forward-pass model asli,
bukan cuma instansiasi — teknik stub tidak cukup): `validate_log_likelihood.py`
(94.2%/49-of-52, dan perbandingan Qwen3 87.5% vs EXAONE 48.2%) butuh
`SolverCheckProvider` yang benar-benar menjalankan Qwen3-4B. User perlu
jalankan `python3 src/validate_log_likelihood.py` di mesin GPU dan
commit hasilnya, sama seperti `1_validation_results.json` sebelumnya,
supaya SEMUA angka yang dikutip paper punya artefak ter-verifikasi,
bukan cuma 3 dari 5.

**Update (2026-10-20): gap ke-5 ditutup.** User menjalankan
`validate_log_likelihood.py` sungguhan di mesin GPU (Qwen3-4B asli,
bukan stub) dan commit `log_likelihood_results.json`. Hasil:
**94.2%** (49/52) — cocok persis dengan klaim §9.35. Sekarang **5 dari
5** skrip validasi permanen punya artefak hasil ter-commit
(`1_validation_results.json`, `2_validation_results.json`,
`item_discrimination_results.json`, `lm_plausibility_results.json`,
`log_likelihood_results.json`) — setiap angka eval yang dikutip di
paper sekarang bisa diverifikasi ulang siapa pun tanpa perlu
menjalankan ulang simulasi/model dari nol.

**Masih belum tersimpan** (tercatat eksplisit, bukan lupa): perbandingan
Qwen3 (87.5%) vs EXAONE (48.2%) di §9.36 — PRD sendiri mencatat itu
"skrip ad-hoc di scratchpad, tidak dipermanenkan" karena hasil
negatif untuk EXAONE dianggap tidak perlu direproduksi rutin. Beda
kategori dari 5 skrip di atas (yang permanen by design), jadi tidak
termasuk gap yang perlu ditutup di sini.

---

### 9.43 User minta: kalau eval metric jelek, GANTI metodenya — bukan direframe. 2 pasangan diganti, bukan ditutupi

User eksplisit: kalau ada hasil eval metric yang jelek dan tidak
mendukung novelty, ganti metodenya, jangan terlalu banyak reframe
bahasa sampai argumennya jadi lemah. Diaudit ulang `log_likelihood_results.json`
yang baru di-commit user (§9.42) untuk cari kandidat.

**Ditemukan dua kasus gagal, BUKAN noise — masing-masing punya sebab
spesifik yang bisa diperbaiki di sumbernya:**

1. **`noun_counter` (3/5, bukan "cuma 1 skill lemah" seperti klaim lama
   — lihat poin 3):** 2 dari 2 kegagalan **persis** pasangan yang
   melawan konter 개 (konter paling umum di korpus Korea) dengan konter
   benar yang lebih jarang (권 untuk buku, 분 untuk orang-dihormati).
   3 pasangan lain (명 vs 개, 개 vs 명) lolos bersih dengan margin besar
   (gap 5.5-11.3). Ini confound di METODE pengukuran (frekuensi korpus
   mentah, bukan tata bahasa), bukan soal item-nya salah — keduanya
   tetap valid secara konstruksi (tag NNB sama, diverifikasi deterministik).
   **Diganti** (`exaone_provider.py` SKILL_SPECS): `("선생님","분","개")`
   → `("선생님","분","명")` (lawan sama-sama konter orang, menguji nuansa
   honorifik secara lebih presisi daripada sebelumnya), `("책","권","개")`
   → `("책","권","장")` (lawan konter-lembaran vs konter-buku, kesalahan
   sewajar daripada 권-vs-개 yang nyaris tidak pernah ditebak orang).

2. **`mod_adverb` (4/5):** 1 kegagalan di pasangan (깨끗하다, 청소해요),
   gap kecil (-0.62). Sebab: 청소하다 verba majemuk (청소+하다), jadi
   bentuk salah "깨끗한" bisa menempel ke kata benda "청소" di dalamnya
   dan terbaca semi-natural ("pembersihan yang bersih") — pembuka
   struktural yang verba non-majemuk lain tidak punya. **Diganti**:
   `청소해요` → `씻어요` (verba sederhana, tidak ada kata benda
   tersembunyi untuk bentuk salah menempel), diverifikasi ulang
   pembeda tag EC/ETM masih tegas.

3. **Temuan tambahan sambil audit:** klaim lama "9 of 10 essentially
   clean, only noun_counter weaker" sudah sedikit overclaim SEBELUM
   perbaikan ini — `mod_adverb` juga gagal 1 pasangan (4/5, bukan 5/5),
   jadi seharusnya "8 dari 10 bersih sempurna", bukan 9. Diperbaiki di
   `src/content.py` (dua tempat, Section IV-B dan V-B) sekalian dengan
   perbaikan pasangan di atas.

**Yang SENGAJA tidak diganti** (beda kategori dari dua di atas):
`irregular_conj` dan `subj_topic` (10% masing-masing di check (c) lama,
§9.20/§9.21) — itu bukan confound metode, itu solver/judge genuinely
tidak tahu tata bahasa Korea yang diuji (dikonfirmasi lewat pick-
distribution tracking konten, bukan posisi). Mengganti item di sana
untuk menghindari kegagalan itu = menghindari konten yang justru paling
berharga dibuktikan, sama persis prinsip yang sudah ditegakkan di
§9.20/§9.21 sebelumnya.

**Diverifikasi tanpa GPU** (pola sama seperti §9.21/§9.22): stub
`torch`/`transformers`, jalankan `__main__` via `runpy` — 5/5 sukses di
SEMUA 11 skill dengan pasangan baru. `python3 -m py_compile` lolos,
`build_docx.py` rebuild tanpa error.

**Belum diverifikasi** (dicatat eksplisit, bukan diasumsikan beres):
`log_likelihood_results.json` yang sudah ter-commit (§9.42) masih
menggambarkan PASANGAN LAMA — angka 94.2%/49-of-52 itu valid untuk kode
SEBELUM fix ini, bukan untuk kode sekarang. Teks paper sudah diubah
untuk menyatakan ini eksplisit ("re-measuring the replacements is
recorded as outstanding, not assumed resolved by the fix alone") —
bukan diam-diam dibiarkan klaim lama seolah masih berlaku. User perlu
jalankan ulang `validate_log_likelihood.py` di GPU dan commit hasil
baru supaya artefak dan kode kembali sinkron.

---

### 9.44 Audit sinkronisasi git cloud/local — PDF basi 6 commit, bukan error merge, DIPERBAIKI

User minta cek apakah ada kesalahan saat sinkronisasi git antar sesi
cloud dan local (sesi ini sempat kena limit penggunaan, sesi lain
melanjutkan di cloud sambil ini terputus). Diinvestigasi menyeluruh,
bukan diasumsikan bersih atau diasumsikan rusak:

**Struktur merge sendiri BERSIH**: `git log --merges`, cek conflict
marker (`grep "<<<<<<<"` dkk, nol hasil), `git status` bersih, riwayat
linear sampai titik percabangan (`merge-base` = `080b355`), tidak ada
commit hilang di kedua sisi. Penomoran PRD §9.37-9.39 (sesi ini) vs
§9.40-9.43 (sesi cloud) BERURUTAN TANPA TABRAKAN — sesi cloud secara
eksplisit menyebut "continuing a citation-accuracy check a local
session had started and hit its session limit" di pesan commit
`1c0cbfa`, dan hasil investigasinya mengonfirmasi tepat dugaan saya
sendiri sebelum limit (korelasi tinggi Liu et al., bukan "divergensi
sistematis" — lihat §9.40).

**SATU masalah nyata ditemukan**: `paper/KO-DRILL_IEEE.pdf` basi 6
commit (sejak `c5a15f2`, commit terakhir saya sebelum limit). Dicek
dengan hash biner langsung (bukan tebak): `src/content.py` dan
`paper/KO-DRILL_IEEE.docx` SUDAH benar ter-merge (mengandung semua fix
sesi cloud: sitasi Liu et al. §9.40, nama penulis §9.41, dll — diverifikasi
dengan convert docx→text dan dibandingkan byte). Tapi `.pdf` committed
di HEAD **identik 100% (md5) dengan versi di merge-base `080b355`** —
berarti tidak pernah di-convert ulang di 6 commit manapun sesudahnya,
walau pesan commit-commit itu bilang "rebuilt paper/KO-DRILL_IEEE.docx"
(cuma docx, PDF-nya memang tidak disebut — bukan klaim palsu, cuma
langkah `libreoffice --headless --convert-to pdf` manual yang berulang
kali terlewat lintas beberapa commit/sesi berbeda).

**Akibat nyata**: siapa pun yang membuka file `.pdf` langsung (bukan
`.docx`) akan membaca teks LAMA yang salah (termasuk sitasi Liu et al.
yang sudah terbukti salah representasi di §9.40) — padahal sumber
aslinya (`content.py`) sudah benar. Bukan data hilang, bukan merge
salah ambil sisi (dicek eksplisit: hash `.docx` di HEAD = hash di
commit cloud yang benar) — murni langkah build yang terlewat berulang.

**Diperbaiki**: `python3 src/build_docx.py` + `libreoffice --headless
--convert-to pdf` dijalankan ulang dari `content.py` HEAD saat ini.
Diverifikasi presisi (bukan cuma "build sukses"): ekstrak teks PDF hasil
rebuild dan bandingkan ke ekstraksi teks dari `.docx` yang sudah
dikonfirmasi benar — 0 baris beda. Diff `.docx` yang muncul di
`git status` cuma metadata biner non-deterministik (timestamp
python-docx), dikonfirmasi 0 beda teks terhadap versi HEAD yang sudah
benar — aman di-commit.

**Pelajaran proses untuk sesi-sesi berikutnya**: `build_docx.py` cuma
menghasilkan `.docx`; langkah konversi `.pdf` via LibreOffice itu
terpisah dan manual — kalau sesi mana pun cuma menjalankan yang pertama
dan lupa yang kedua, `.pdf` akan diam-diam basi tanpa error apa pun.
Worth dipertimbangkan menggabungkan kedua langkah jadi satu script
kalau pola ini berulang lagi.

**File:** `paper/KO-DRILL_IEEE.docx` (refresh metadata saja, teks sudah
benar sebelumnya), `paper/KO-DRILL_IEEE.pdf` (fix isi sungguhan).

---

### 9.45 Fitur baru: admin upload kurikulum (.docx) → item di-generate dari situ, bukan hanya 11 skill hand-coded

User minta fitur baru: admin bisa upload dokumen kurikulum (contoh
konkret yang diberikan: `Korean_Grammar_Curriculum.docx` di root
project, 524 poin grammar TOPIK level 1-6), dan soal harus dibuat dari
dokumen itu — bukan hanya dari 11 skill yang sudah di-hand-code di
`exaone_provider.py`. User juga minta kurikulum ini bisa mencakup
"smart factory EV manufacturing" sebagai scope.

Ditanya & diputuskan eksplisit lewat AskUserQuestion sebelum mulai
implementasi (2026-10-20), supaya tidak menebak-nebak scope:
1. **Target versi pertama**: dokumen 524-poin TOPIK yang sudah ada
   dulu, bukan upload generik dari hari pertama (dipilih user).
2. **Makna "smart factory EV manufacturing"**: substitusi
   kosakata/konteks manufaktur di atas grammar TOPIK standar, BUKAN
   kurikulum vokasional terpisah (dipilih user) — jadi grammar tetap
   sama, hanya domain kata benda/konteks kalimat yang nanti diganti
   (belum dibangun, lihat "Belum dikerjakan" di bawah).
3. **Grammar point di luar 7 kategori "kind" yang sudah ter-hand-code**:
   user secara eksplisit memilih **izinkan generasi LLM bebas dengan
   validator longgar**, BUKAN rekomendasi saya (membangun kategori
   terverifikasi satu per satu seperti 11 skill yang ada) — lihat
   §9.46 untuk konsekuensi kualitas nyata dari pilihan ini, diukur
   bukan diasumsikan.

**`src/curriculum_provider.py`** (baru): parser `.docx` → kurikulum
terstruktur. Diverifikasi langsung terhadap dokumen asli, bukan
diasumsikan: body dokumen adalah urutan flat paragraf+tabel; "Heading 1"
= level TOPIK, "Heading 2" = unit, tabel berikutnya dengan header pasti
`["No.", "Grammar", "Meaning / Usage", "Example", "Translation"]` berisi
data (tabel overview pertama, dengan header berbeda, dilewati). Smoke
test (`__main__`) membuktikan parsing benar terhadap isi dokumen
sungguhan: **524/524 entri**, 6 level, 36 unit, nomor unik, beberapa
baris spot-check (`"17"` → `N이/가`, `"40"` → `-았/었/했-`) cocok dengan
yang terlihat langsung di dokumen.

**Temuan kuantitatif yang mendasari keputusan arsitektur berikutnya**:
dari 524 entri, hanya **~53%** yang grammar-nya muncul sebagai literal
substring di kalimat contohnya sendiri (dicek langsung, bukan
diasumsikan) — sisanya butuh pemahaman morfologis untuk menemukan span
mana di kalimat yang merealisasikan grammar point tersebut (misal
`-았/었/했-` terealisasi sebagai `봤어요`, bukan string `-았/었/했-` itu
sendiri). Ini alasan konkret kenapa generasi LLM-assisted (bukan regex
sederhana) dibutuhkan untuk mayoritas entri — lihat §9.46.

---

### 9.46 Generator item dari kurikulum: diuji empiris di sampel representatif, BUKAN sampel pilihan tangan — rate produksi 32%, bukan 100% yang terlihat di sampel awal

**Pilihan model, diuji bukan diasumsikan (konsisten dengan §9.36)**:
EXAONE-4.0-1.2B dicoba PERTAMA (karena jadi default historis di codebase
ini) untuk task ini — mayoritas gagal menemukan span literal di kalimat
contoh sama sekali, kadang malah menjawab dengan deskripsi grammar point
bukan span yang ditemukan. Qwen3-4B dipakai sebagai gantinya, cocok
dengan temuan independen §9.36 (48.2% vs 87.5% pada task penilaian
gramatikalitas yang berbeda) — pola yang sama muncul utuh untuk ketiga
kalinya di task yang berbeda-beda, bukan kebetulan satu task.

**`src/curriculum_item_generator.py`** (baru): untuk entri yang span-nya
tidak bisa ditemukan via substring match (§9.45), model (Qwen3-4B)
diminta (1) menemukan span yang merealisasikan grammar point di kalimat
contoh, (2) membuat satu alternatif salah yang plausibel. Divalidasi
dengan 5 check, EKSPLISIT lebih longgar dari 11 skill hand-coded dan
didisclose demikian (bukan disajikan dengan percaya diri yang sama):
- (a) span "asli" yang disebut model harus benar-benar ada di kalimat
  contoh (sanity check anti-halusinasi) — kalau tidak, item dibuang.
- (b) distractor harus beda dari jawaban benar dan tidak kosong.
- (c) `solver_check.py`'s round-trip check (blind key-acceptance)
  dijalankan dan direkam sebagai SINYAL untuk ditinjau, bukan gate yang
  diam-diam membuang item — sama seperti perlakuan ke 11 skill
  hand-coded.
- (d) **filter gibberish** (ditemukan dari inspeksi manual, bukan
  diasumsikan tidak ada masalah): pada sampel awal 12 entri, 2/7 item
  yang "lolos" check (a)/(b) punya distractor yang bukan kesalahan
  pembelajar sungguhan tapi string rusak (`"합니다니다"`, `"바르요"`) —
  check (a)/(b) tidak bisa menangkap ini karena cuma mengecek string
  beda-dan-tidak-kosong, bukan "apakah ini kata sungguhan". Diperbaiki
  dengan reuse `distractor_validator.lm_plausibility_gap()` (§9.34) pada
  kalimat utuh (bukan kata tunggal): 2 kasus gibberish di atas punya gap
  18.2/9.8, sedangkan 5 distractor baik (kata sungguhan, cuma salah)
  punya gap 0.1–4.1 — pemisahan bersih, diverifikasi bukan ditebak.
  Ambang `gap > 8.0` (titik tengah pemisahan itu).
- (e) **filter konteks-hilang** (ditemukan dari sampel yang lebih besar
  di bawah, bukan terlihat di 12 sampel awal): beberapa grammar point
  level menengah/lanjut (`-(으)며`, `-(으)ㄹ 듯 말 듯 하다`, `-(으)ㄹ
  리(가) 만무하다`) menyambungkan DUA klausa penuh, jadi span yang
  disebut model mencakup hampir seluruh kalimat — stem hasil blanking
  jadi cuma `"___"`, nol konteks tersisa untuk solver maupun pembelajar.
  Dibuang kalau yang tersisa di stem selain blank marker dan tanda baca
  cuma string kosong.

**Hasil pada sampel 12 (pilihan tangan awal, semua level 1-2)**: 7/12
item terproduksi, solver round-trip 7/7 (100%). **Hasil awal ini
menyesatkan** — ditemukan SETELAH diuji di sampel yang representatif
(bukan diasumsikan generalisasi dari sampel kecil).

**Hasil pada sampel 41 (setiap entri ke-13 dari 524, merata di semua 6
level — representatif, bukan dipilih tangan)**: **13/41 item terproduksi
(32%)**, solver round-trip **6/13 (46%)**. Jauh di bawah sampel awal.

**Root cause rate solver yang jauh lebih rendah, diinvestigasi bukan
dibiarkan sebagai angka tanpa penjelasan**: menginspeksi item yang
`key_accepted=False` satu per satu (`-(으)ㄴ/는데도` vs distractor
`"...불구하고"`, `N(이)란` vs `은`, `N에 비하면` vs `에 비해`, `-(으)ㄹ
리(가) 만무하다` vs `리가 없다`, dst) menunjukkan pola konsisten: di
level TOPIK menengah/lanjut, banyak grammar point adalah **sinonim-dekat
satu sama lain** (bukan satu benar-satu-salah secara objektif) —
distractor yang dibangun dari grammar point lain yang genuinely berbeda
seringkali JUGA merupakan frasa natural yang diterima penutur asli,
bukan kesalahan sungguhan. Item seperti ini menguji "textbook pilih
sinonim mana", bukan benar/salah. **Tidak ada check otomatis di atas
yang bisa membedakan ini** — didisclose di docstring modul sebagai
limitasi kualitas yang diketahui, bukan disembunyikan atau direframe
sebagai sukses.

**Kesimpulan untuk keputusan user §9.45 poin 3** (izinkan generasi bebas
+ validator longgar, bukan rekomendasi saya membangun kategori
terverifikasi satu-per-satu): pilihan ini SEKARANG punya data konkret,
bukan cuma trade-off teoretis — mekanismenya BEKERJA (bisa parse
dokumen asli, generate, dan validasi dasar), tapi kualitas konten
menurun tajam begitu melewati grammar point level dasar (particle/ending
tunggal) ke grammar point level lanjut (klausa-ke-klausa, sinonim-dekat).
Rate produksi 32% dan rate agreement-solver 46% pada sampel representatif
harus jadi baseline yang dikutip untuk fitur ini, BUKAN angka 100%/100%
dari sampel 12 yang kebetulan semua level pemula.

**Belum dikerjakan (lanjutan langsung dari sini)**:
- Substitusi kosakata domain "smart factory EV manufacturing" (§9.45
  poin 2) — belum dibangun sama sekali, menunggu pipeline dasar ini
  stabil dulu.
- Integrasi ke `ContentProvider`/`daily_drill.py` — saat ini
  `curriculum_item_generator.py` berdiri sendiri, belum disambungkan ke
  flow aplikasi.
- `kagas_code` sengaja `None` untuk semua item dari modul ini (lihat
  docstring) — pemetaan KAGAS 7-kode yang ada di-hand-derive untuk 11
  skill lama dan tidak otomatis berlaku ke ~500 titik kurikulum lainnya.
- Teks paper BELUM disentuh untuk fitur ini — paper saat ini hanya
  mendeskripsikan 11 skill hand-coded; perlu bagian baru yang jujur
  (prinsip "paper mengikuti apa yang kita temukan", bukan sebaliknya)
  begitu pipeline ini dianggap cukup matang untuk didokumentasikan sebagai
  kontribusi, dengan rate 32%/46% di atas dikutip apa adanya.

**File:** `src/curriculum_provider.py` (baru), `src/curriculum_item_generator.py`
(baru, reuse `distractor_validator.lm_plausibility_gap` dan `solver_check.py`).

---

### 9.47 Re-measurement `mod_adverb` & `noun_counter` setelah fix §9.43 — satu beres (5/5), satu limitasi genuine didisclose (4/5, bukan diganti lagi)

Sebelum lanjut fitur kurikulum, verifikasi dua fix pending dari §9.43
diselesaikan dengan GPU sungguhan (`validate_log_likelihood.py --n-per-skill 20`,
bukan diasumsikan dari teori saja):

**`mod_adverb` (늦다/가요, ganti dari 깨끗하다/씻어요)**: **5/5 (100%)**.
Mengonfirmasi teori §9.46 (lihat komentar di `exaone_provider.py`): akar
masalahnya memang 깨끗하다 sebagai ADJEKTIF (adverb idiomatik "깨끗이"
bersaing dengan bentuk reguler "-하게"), bukan verb pasangannya — ganti
verb saja (percobaan sebelumnya, 청소해요→씻어요) tidak cukup, ganti
adjektif-nya baru menyelesaikan.

**`noun_counter`**: **4/5 (80%)**. Satu-satunya yang gagal: pasangan
`선생님/분/명` (gap **-0.86**, satu-satunya gap negatif dari 5). Ini BUKAN
confound yang sama dengan yang diperbaiki di §9.43 (분 vs 개, dimana 개
adalah counter generik frekuensi-tertinggi) — di sini distraktornya 명,
counter ORANG biasa, bukan catch-all generik. Artinya akar masalahnya
adalah register honorifik 분 itu sendiri yang tidak terlihat oleh sinyal
frekuensi — akar masalah yang SAMA dengan yang sudah terdokumentasi di
`distractor_validator.py` untuk alasan `honorific`/`tense_past`
dikeluarkan dari `LM_PLAUSIBILITY_RELIABLE_SKILLS`.

**Keputusan, konsisten dengan prinsip user (ganti metode kalau jelek,
tapi disclose kalau memang tidak ada solusi)**: TIDAK diganti lagi.
Mengganti kata benda atau counter pasangannya tidak akan mengubah apa pun
karena masalahnya bukan pilihan kata tertentu — setiap kalimat yang
menguji sifat honorifik 분 akan punya blind spot yang sama pada judge
berbasis frekuensi ini, dan menguji sesuatu selain nuansa honorifik akan
menghilangkan tujuan memasukkan 분 sama sekali. 80% keseluruhan dengan
satu limitasi yang dipahami dan terdokumentasi diperlakukan sama seperti
`honorific`/`tense_past`/`irregular_conj`/`subj_topic` — bukan 100% palsu,
bukan juga dihapus/direframe.

**File:** `src/exaone_provider.py` (komentar update untuk kedua skill),
`log_likelihood_results.json` (hasil re-run terbaru, n=20/skill, seed=0,
overall 98.1% across 10 skill — naik dari run sebelumnya karena
`mod_adverb` sekarang penuh dan `noun_counter` 1 kasus lebih baik
dipahami, bukan dihitung ulang secara berbeda).

---

### 9.48 Perbaikan prompt generator kurikulum: solver-agreement naik 46%→64% — dicoba dulu sebelum disclose sebagai limitasi permanen (§9.46)

User minta cek prioritas sebelum lanjut fitur EV/integrasi. Rekomendasi
saya: perbaiki dulu masalah sinonim-dekat §9.46 dengan mencoba perbaikan
prompt (belum pernah dicoba, baru didiagnosis) sebelum menyerah dan
mendisclose sebagai limitasi permanen — konsisten dengan prinsip
"investigasi metode lain, kalau meyakinkan bagus go for it". User
setuju ("mau").

**Perubahan**: `_GEN_PROMPT` di `curriculum_item_generator.py` direvisi
dari sekadar "buat kesalahan yang mungkin dibuat pembelajar" menjadi
instruksi eksplisit yang MELARANG model menukar ke ekspresi lain yang
sama-sama benar, dengan contoh konkret pelanggaran yang dilarang
("-(으)ㄹ 리가 만무하다"→"-(으)ㄹ 리가 없다", "N에 비하면"→"N에 비해") dan
daftar kesalahan pembelajar yang DIPERBOLEHKAN (salah tense, salah level
honorifik, partikel hilang/salah, konjugasi tidak beraturan diterapkan
salah, tertukar dengan grammar lain yang bentuknya mirip tapi beda arti).

**Hasil, diukur pada sweep representatif yang SAMA (41 entri, setiap
ke-13 dari 524) supaya apple-to-apple dengan §9.46**:

| | Sebelum (§9.46) | Setelah (prompt baru) |
|---|---|---|
| Item terproduksi | 13/41 (32%) | 14/41 (34%) |
| Solver round-trip agreement | 6/13 (46%) | **9/14 (64%)** |

**Perbaikan nyata, bukan cuma angka naik tanpa penjelasan**: diperiksa
satu per satu, beberapa item yang SEBELUMNYA lolos tapi gagal solver
(`-아/어 봤자`, `-(으)ㄹ 리(가) 만무하다`, `N을/를 무릅쓰고`,
`-았/었더니`) sekarang malah DITOLAK SENDIRI oleh check (a)/(b)/(d) —
model, dilarang menukar ke sinonim, kalau tidak bisa menemukan kesalahan
genuine lebih memilih gagal format daripada memaksa sinonim sebagai
"jawaban salah". Ini perilaku yang diinginkan: mencegah item cacat
terkirim, bukan cuma menyembunyikan masalah.

**TIDAK sepenuhnya menghilangkan masalah** (jujur, bukan overclaim):
`N에 비하면` vs `에 비해`, `N(이)란` vs `은`, `-(으)ㄹ 때` vs turunannya,
`-느니` vs `-려고` masih lolos sebagai pasangan sinonim-dekat meski sudah
dilarang eksplisit di prompt — beberapa grammar point TOPIK level lanjut
memang tidak punya "tetangga yang salah" untuk dikontraskan, hanya
tetangga yang sama-sama benar. Satu regresi kecil juga terjadi (entri 287,
`-(으)ㄴ/는 만큼`, dulu lolos sekarang di-skip) — dalam batas wajar untuk
perubahan prompt, tidak diinvestigasi lebih lanjut karena bukan pola
yang berulang.

**Keputusan**: cukup satu iterasi ini, tidak lanjut reframe/iterasi
prompt lagi (konsisten dengan prinsip user: jangan terlalu banyak
reframe sehingga menjadi lemah). 64% dengan limitasi near-synonym yang
dipahami dan terdokumentasi lebih baik daripada mengejar 100% dengan
prompt yang makin rumit untuk kasus yang secara linguistik memang tidak
punya jawaban salah yang natural.

**File:** `src/curriculum_item_generator.py` (`_GEN_PROMPT` + docstring
update).

---

### 9.49 Integrasi ke daily_drill/answer_processing: 2 keputusan arsitektur ditanyakan, 4-opsi dibangun, threshold filler dikalibrasi 2x setelah kalibrasi pertama meleset

User minta lanjut ke integrasi (`ContentProvider`/`daily_drill.py`).
Sebelum coding, dua keputusan arsitektur ditanyakan eksplisit ke user
(bukan ditebak) karena menyentuh asumsi yang sudah divalidasi:

1. **Skala tracking mastery**: per grammar point (524 skill Urnings
   terpisah) vs satu skill payung vs di luar Urnings dulu. User pilih
   **per grammar point** (rekomendasi saya) — paling akurat secara
   pedagogis, tapi secara eksplisit DICATAT sebagai belum divalidasi di
   skala ini (§9.12-9.38 semua diukur di ~11 skill).
2. **Jumlah opsi**: item kurikulum cuma 2 opsi, sementara 11 skill lain
   dipaksa 4 opsi karena baseline tebak-acak (25% vs 50%) mempengaruhi
   matematika Urnings/IRT (§9.21, keputusan eksplisit sebelumnya). User
   pilih **buat 4 opsi juga** (rekomendasi saya), dengan mekanisme filler
   generik yang perlu dibangun karena tidak ada candidate pool per-skill
   seperti 11 skill lama.

**Mekanisme filler (check (f) baru di `curriculum_item_generator.py`)**:
reuse pool filler generik yang SUDAH terbukti aman di `exaone_provider.py`
("학교"/"사과"/"저기"/"안녕"), divalidasi per kalimat dengan sinyal
kiwipiepy yang sama dipakai untuk check (d) — filler diterima HANYA jika
skornya jelas lebih rendah dari jawaban benar saat disubstitusi.

**Dua kesalahan kalibrasi ditemukan dan diperbaiki berurutan, bukan
diasumsikan benar dari awal**:
1. Percobaan pertama reuse `lm_plausibility_gap()` mentah (skor SUM,
   sama seperti check (d)) → produksi jatuh drastis dari 14/41 ke
   **5/41**. Diinvestigasi, bukan dibiarkan: skor SUM kiwipiepy
   menyukai kalimat LEBIH PENDEK secara sistematis, tidak peduli
   gramatikalitas — begitu jawaban benar adalah frasa panjang (misal
   "하기가 어려워요"), filler pendek apa pun otomatis menang meski jelas
   omong kosong. Dibuktikan langsung dengan angka (entry 144: gap
   negatif -1.3 sampai -5.5 untuk SEMUA filler, padahal substitusinya
   jelas rusak).
2. Diperbaiki dengan skor ternormalisasi per-token (`_filler_gap()`,
   fungsi baru, terpisah dari `lm_plausibility_gap()` yang dibiarkan
   tidak berubah karena sudah divalidasi di bentuk aslinya untuk
   konteks berbeda). Threshold awal 1.5 (dari 3 contoh kalibrasi) masih
   TERLALU KETAT — diukur lagi di 2 entri baru, margin asli cuma
   0.4-1.4, produksi jatuh lagi ke **7/41**. Diturunkan ke **0.3**
   (comfortably di atas 0, di bawah semua margin aman yang terukur).

**Hasil final, diukur di sweep 41-sampel yang sama**: **13/41 (32%)
produksi, 9/13 (69%) solver-agreement** — sejalan dengan baseline
sebelum check (f) ada (14/41, 64%, §9.48), membuktikan mekanisme filler
itu sendiri TIDAK mengorbankan rate produksi setelah dikalibrasi benar.
Semua item sekarang punya 4 opsi A-D yang valid. Satu anomali didisclose
(bukan disembunyikan): entry "-느니" (352) kedua filler-nya ditandai
`also_acceptable=True` oleh check (b) solver — tapi check (b) SUDAH
terdokumentasi punya yes-bias persisten (lihat `solver_check.py`), jadi
ini dibaca sebagai limitasi lama yang berulang, bukan bukti baru filler
tidak aman (sinyal yang benar-benar menggerbang produksi adalah
`_filler_gap`, sinyal berbeda dan independen dari check (b)).

**`skill_id` format**: `f"curriculum_{entry.number}"` — satu Urnings
track per grammar point, sesuai keputusan user poin 1.

**File baru**: `src/curriculum_grammar_provider.py` — `CurriculumGrammarProvider`
mengimplementasikan `ContentProvider.next_new_grammar_item` protocol yang
sama dengan `ExaoneGrammarProvider`, routing berdasarkan prefix skill_id.

**Dispatcher dibangun dan diverifikasi end-to-end**: `DispatchingGrammarProvider`
baru di `content_provider.py` — routing `next_new_grammar_item` berdasarkan
prefix skill_id (`"curriculum_"` → `CurriculumGrammarProvider`, selain itu →
`ExaoneGrammarProvider`), dengan `.items` sebagai `ChainMap` read-only yang
menggabungkan dict item dari kedua provider tanpa copy. `daily_drill.py` dan
`answer_processing.py` TIDAK PERLU diubah sama sekali — keduanya sudah
duck-typed terhadap `next_new_grammar_item`/`item.skill_id`/`item.options`/dst,
bukan terhadap tipe konkret provider/item tertentu.

**Smoke test end-to-end**: drill dengan skill_ids campuran
(`["particle_loc", "curriculum_40"]`) lewat satu `DispatchingGrammarProvider`
yang sama → `submit_grammar_answer` dipanggil untuk item kurikulum → Urnings
`skill_profile` ter-update dengan benar untuk skill `"curriculum_40"`.

**Satu masalah ditemukan & diperbaiki saat menjalankan smoke test**: GPU OOM
(15.57 GiB, proses makan 15.54 GiB) saat smoke test awalnya memuat 3 model
sekaligus (EXAONE untuk `particle_loc` + Qwen3-4B untuk `CurriculumGrammarProvider`
+ Qwen3-4B TERPISAH LAGI untuk `QwenFeedbackProvider`). Diperbaiki dengan stub
`_FakeFeedbackProvider` di smoke test (bukan di kode produksi) — yang perlu
dibuktikan test ini adalah mekanisme dispatch+Urnings bekerja, bukan menguji
ulang `QwenFeedbackProvider` yang sudah punya smoke test sendiri di
`answer_processing.py`. **Catatan untuk nanti**: kalau provider ini benar-benar
dipakai bersamaan di produksi (bukan cuma smoke test), 3 model GPU sekaligus
bisa jadi masalah nyata, bukan cuma masalah test — belum diselesaikan,
dicatat sebagai limitasi operasional terbuka.

**File:** `src/curriculum_item_generator.py` (check (f) + 2 revisi
kalibrasi), `src/curriculum_grammar_provider.py` (baru),
`src/content_provider.py` (`DispatchingGrammarProvider` baru + smoke test).

---

### 9.50 Validator plausibilitas semantik substitusi domain: 6 metode dicoba, SEMUA gagal — limitasi genuine model 4B, didisclose bukan dipaksakan

**Konteks**: §9.46-9.49 membangun fitur substitusi kosakata domain
EV/smart-factory. User koreksi arsitektur (tabel kata hardcode Claude
diganti jadi keputusan LLM, lihat `domain_vocab_substitution.py`) lalu
minta dibandingkan dengan EXAONE (hasil: EXAONE-4.0-1.2B lebih buruk,
EXAONE-3.5-7.8B tidak bisa dijalankan sama sekali karena bug
inkompatibilitas versi `transformers` di kode custom repo-nya sendiri —
dicoba 3 versi berbeda, 4.46/4.57/5.20, semua gagal dengan error
berbeda, disimpulkan bukan masalah di sisi kita). Ditemukan juga: Qwen3
sendiri menghasilkan substitusi yang gramatikal sempurna tapi semantiknya
rusak ("부품을 먹어요" = makan komponen, "설비를 읽으세요" = membaca alat
berat). User secara eksplisit minta investigasi menyeluruh sebagai
"senior LLM engineer" untuk menemukan validator yang bekerja, bukan
menyerah di percobaan pertama.

**Dataset kalibrasi**: 26-28 pasang (orig, substituted) dari sweep 75
entri kurikulum, dinilai plausibel/tidak secara manual (akal sehat dunia
nyata, bukan gramatikalitas) — dasar untuk menguji setiap sinyal.

**6 metode dicoba, berurutan, masing-masing dengan alasan kenapa metode
sebelumnya gagal dan apa yang dicoba beda**:

1. **kiwipiepy parse score** (reuse sinyal yang sukses di check (d)/(f)
   §9.48-9.49): **GAGAL TOTAL** — PLAUSIBLE dan BAD tercampur rata di
   seluruh rentang gap (-0.24 sampai 2.73), nol separasi. Make sense:
   parser statistik morfologi cuma tahu "kombinasi morfem ini umum atau
   tidak", buta terhadap akal sehat dunia nyata ("pabrik itu enak"
   gramatikal sempurna).
2. **Qwen3 whole-sentence log-probability** (`sentence_logprob`, sinyal
   yang sukses di `solver_check.py` §9.35): **GAGAL TOTAL** — gap
   -0.8 sampai +1.9, PLAUSIBLE dan BAD tercampur di semua rentang. Model
   generatif pun, kalau diukur rata-rata seluruh kalimat, sinyalnya
   terdilusi oleh kata-kata lain yang tetap wajar.
3. **Predicate-conditional-logprob, versi 1** (isolasi token predikat
   via common-suffix token matching, bukan rata-rata seluruh kalimat):
   **SEBAGIAN membaik** — di ekstrem (gap > 0.85) 8/8 BAD terdeteksi
   tanpa false positive, tapi rentang tengah (0.1-0.8) masih campur.
   Precision tinggi di satu ekstrem, recall rendah (~57%) — tidak cukup
   untuk hard gate.
4. **Predicate-conditional-logprob, versi 2** (perbaikan: offset
   karakter presisi dari kiwipiepy + `return_offsets_mapping` tokenizer,
   bukan common-suffix token yang rapuh akibat BPE): **TIDAK lebih baik
   dari versi 1**, malah beberapa kasus PLAUSIBLE bernilai tinggi
   (2.06) di atas kasus BAD — perbaikan presisi teknis tidak menolong
   karena akar masalahnya bukan presisi pengukuran, tapi model memang
   tidak punya sinyal yang relevan di sana.
5. **Forced best-pick** (reuse `solver_check._pick_best`, pola yang
   SUDAH terbukti lebih reliable dari yes/no untuk model ini — dasar
   check (c) di 11 skill): diberi pilihan A/B antara kata baru vs kata
   asli, model diminta pilih mana yang lebih pas. **GAGAL, mendekati
   acak**: kasus BAD yang berhasil menolak kata baru 7/16 (44%), kasus
   PLAUSIBLE yang tetap pilih kata baru 5/12 (42%) — keduanya di bawah
   50%, bukan sekadar lemah tapi genuinely tidak ada sinyal.
6. **Direct judgment dengan reasoning mode** (`enable_thinking=True`,
   budget 600 token, parsing longgar): dimensi yang BELUM dicoba
   (sepanjang sesi ini reasoning mode selalu dimatikan demi kecepatan).
   Hasil awal pada 12-sampel sempat menjanjikan (3/4 verdict jelas
   benar, termasuk MENANGKAP kasus "커피에다가 철분" yang tidak
   terdeteksi metode manapun sebelumnya) — tapi pada tes konfirmasi
   penuh 26-pasang: **correct=15/26, accuracy 57.7%** — cuma sedikit di
   atas tebakan acak (50%), bukan breakthrough yang diharapkan.

**Kesimpulan, sebagai senior-LLM-engineer finding bukan sekadar
"menyerah"**: 6 metode berbeda — 3 varian scoring (kiwipiepy,
whole-sentence Qwen3, predicate-isolation x2), 1 varian diskriminasi
(forced best-pick), 2 varian judgment langsung (dengan/tanpa reasoning)
— semua gagal mencapai level yang bisa diandalkan untuk menggerbang
produksi otomatis. Ini BUKAN karena kurang usaha atau salah
implementasi (setiap metode diverifikasi bekerja secara mekanis,
diukur di data nyata, bukan ditebak) — ini limitasi genuine model 4B
untuk task "akal sehat dunia nyata dalam bahasa Korea", berbeda sifat
dari task-task lain yang BERHASIL di sesi ini (gramatikalitas,
span-finding, scoring vs judging untuk distraktor) yang semuanya
tentang kebenaran LINGUISTIK, bukan kebenaran DUNIA NYATA.

**Belum diputuskan** (butuh arahan user): disclose sebagai limitasi
(fitur substitusi tetap dipakai, kualitas semantik di-flag untuk review
manusia, tidak digerbang otomatis) vs investigasi model API eksternal
(keluar dari arsitektur all-local yang dipakai proyek ini sejauh ini,
perubahan scope besar) vs pendekatan lain yang belum terpikirkan.

**File:** tidak ada perubahan kode (semua eksperimen di atas adalah
skrip investigasi sementara, bukan kode produksi — `domain_vocab_substitution.py`
TIDAK diubah karena tidak ada metode yang cukup baik untuk
diimplementasikan).

---

### 9.51 Model swap ke EXAONE-3.5-7.8B di 3 file (per instruksi eksplisit user, BELUM diverifikasi di GPU)

**Instruksi user**: "pakai exaone 7.8b" → diklarifikasi via AskUserQuestion
mencakup ketiga file sekaligus ("ketiganya") → dikonfirmasi "lanjut".
Target: `src/domain_vocab_substitution.py`, `src/qwen_feedback.py`,
`src/solver_check.py` — ketiganya sebelumnya pakai Qwen3-4B.

**Precondition yang membuat ini sekarang mungkin**: §9.50 mencatat
EXAONE-3.5-7.8B "tidak bisa dijalankan sama sekali" karena bug
inkompatibilitas `transformers` di custom modeling code-nya sendiri.
Commit `ab81d8a` (`src/patch_exaone35_remote_code.py`) memperbaiki bug
itu (dua masalah dalam satu call: `input_embeds`→`inputs_embeds` yang
salah nama, dan parameter `cache_position` yang sudah dihapus dari
`create_causal_mask()`). Tanpa patch itu dijalankan setelah load
pertama, ketiga swap di bawah akan gagal dengan `TypeError` yang sama.

**Catatan penting soal provenance angka**: dalam diskusi sesi ini
sempat disebut hasil "EXAONE-3.5-7.8B 65.4% vs Qwen3-4B 57.7%" pada
task semantic-plausibility (§9.50). Angka itu **TIDAK ditemukan** di
`docs/PRD.md`, file hasil JSON manapun, atau riwayat git saat swap ini
dibuat — kemungkinan berasal dari output terminal sesi lokal paralel
yang dibahas di chat tapi belum pernah dicatat ke PRD sebelum context
di-compact. Angka itu **tidak digunakan sebagai bukti terverifikasi**
di komentar kode yang ditambahkan oleh swap ini; hanya dicatat sebagai
alasan yang disampaikan user, bukan sebagai fakta yang sudah
dikonfirmasi tertulis di repo ini.

**Perubahan per file**, risiko terendah ke tertinggi:

1. **`domain_vocab_substitution.py`** — file ini tidak punya
   `_MODEL_NAME` sendiri (desain aslinya: reuse `generate_fn` dari
   model lain yang sudah di-load, demi hindari OOM 3-model di GPU 15GB,
   lihat §9.49). Ditambahkan `Exaone35Generator`, loader EXAONE-3.5-7.8B
   mandiri, dipakai di `__main__` menggantikan reuse Qwen3-4B milik
   `CurriculumItemGenerator`. Ini **membalikkan** keputusan hindari-OOM
   itu — perlu diuji ulang di GPU asli apakah kombinasi model yang
   berjalan bersamaan sekarang (EXAONE-4.0 generator + EXAONE-3.5-7.8B +
   Qwen3-4B lainnya) masih OOM atau tidak. Belum ada wiring produksi
   nyata untuk modul ini di luar `__main__` (dicek via grep), jadi risiko
   saat ini terbatas pada smoke test, belum pipeline nyata.
2. **`qwen_feedback.py`** — `_MODEL_NAME` diganti langsung. Risiko lebih
   rendah: output modul ini (feedback bahasa bebas ke learner) bukan
   dasar klaim psikometrik apa pun di paper. Catatan teknis belum
   terverifikasi: `enable_thinking` adalah toggle khas Qwen3; EXAONE-3.5
   kemungkinan tidak punya konsep reasoning-mode di chat template-nya,
   jadi `detailed_feedback()` (pakai `enable_thinking=True`) diasumsikan
   menjadi no-op diam-diam (Jinja undefined var), bukan error — **asumsi
   ini belum dikonfirmasi di hardware nyata.**
3. **`solver_check.py`** — **risiko tertinggi dari ketiganya.** Angka
   Qwen3-4B untuk file ini SUDAH DIPUBLIKASIKAN di paper (Section IV-B /
   §9.36: EXAONE-4.0-1.2B 48.2% vs Qwen3-4B 87.5% pada grammaticality-
   judgment; §9.50: Qwen3-4B 57.7% pada semantic-plausibility). Angka itu
   untuk EXAONE-4.0-1.2B, BUKAN EXAONE-3.5-7.8B — swap ini belum
   menggantikan baseline yang dipublikasikan dengan data baru apa pun,
   hanya mengganti kode. **Wajib sebelum dianggap final:** jalankan ulang
   `validate_log_likelihood.py` dan `validate_novelty1.py` di GPU asli
   dengan model baru ini, bandingkan dengan artifact yang sudah di-commit
   (`log_likelihood_results.json`, `1_validation_results.json`). Kalau
   `check(c)` (key-acceptance) atau log-likelihood gap turun dibanding
   Qwen3-4B, kembalikan `_MODEL_NAME` ke `"Qwen/Qwen3-4B"` dan/atau revisi
   bagian paper yang mengutip angka lama — jangan biarkan kode dan
   angka yang dikutip paper diam-diam tidak sinkron.

**Verifikasi yang SUDAH dilakukan**: `py_compile` pada ketiga file
(sintaks valid). **Verifikasi yang BELUM bisa dilakukan** (sandbox ini
tidak punya GPU): loading model nyata, `trust_remote_code` path,
`apply_chat_template` dengan/tanpa `enable_thinking` pada EXAONE-3.5,
dan — paling penting — apakah kualitas hasil (check(c), log-likelihood
gap, feedback quality) berubah dibanding Qwen3-4B.

**File:** `src/domain_vocab_substitution.py`, `src/qwen_feedback.py`,
`src/solver_check.py`.

---

### 9.52 Model swap ke EXAONE-3.5-7.8B diperluas ke exaone_provider.py dan curriculum_item_generator.py (scope diperluas per instruksi user, risiko terbesar sejauh ini)

**Instruksi user**: setelah §9.51, user bilang "EXAONE-3.5-7.8B untuk
semuanya". Ditanya scope persis via AskUserQuestion (karena beberapa
file punya alasan terdokumentasi kenapa TIDAK pakai EXAONE) — user
pilih memasukkan **`curriculum_item_generator.py`** dan
**`exaone_provider.py`** ke dalam scope, sudah diberi tahu di pertanyaan
itu sendiri bahwa keduanya punya temuan empiris sebelumnya yang relevan.

**`curriculum_item_generator.py`** (sebelumnya Qwen3-4B): modul ini
secara eksplisit MENGUJI EXAONE-4.0-1.2B untuk tugas span-finding ini
dan KALAH (gagal menemukan span asli di kalimat, kadang malah
mendeskripsikan grammar point). Temuan itu untuk varian 1.2B, BUKAN
7.8B — jadi tidak diasumsikan masih kalah, tapi juga tidak diasumsikan
sudah beres. UNVERIFIED tanpa run ulang smoke test file ini di GPU
nyata pada sampel representatif dari ~247 entri yang ditangani modul
ini.

**`exaone_provider.py`** (sebelumnya EXAONE-4.0-1.2B) — **swap dengan
blast radius terbesar di seluruh sesi ini**, karena beberapa alasan
sekaligus:
1. Bukan cuma naik ukuran dalam satu keluarga — ganti KELUARGA model
   (3.5, bukan 4.0) sekaligus ukuran (7.8B, bukan 1.2B). Fig. 1,
   Section III-A, dan sitasi [34] (`content.py` baris ~456-468)
   menjelaskan "EXAONE" sambil secara spesifik mengutip EXAONE 4.0 dan
   mode hybrid non-reasoning/reasoning-nya ("its non-reasoning mode is
   sufficient... we do not claim its hybrid design as a property this
   system relies on") — EXAONE-3.5 mendahului desain hybrid itu, jadi
   sitasi [34] sekarang menggambarkan model yang BUKAN yang benar-benar
   jalan. Framing paper itu sendiri masih secara teknis aman (sudah
   di-hedge: "we do not claim..."), tapi sitasinya perlu direvisi kalau
   swap ini jadi final — BELUM diubah di `content.py`, sengaja ditunda
   sampai swap ini terverifikasi di GPU (paper mengikuti investigasi,
   bukan sebaliknya).
2. Modul docstring `exaone_provider.py`'s sendiri (Round 3) mencatat
   EXAONE-4.0-1.2B memilih pasangan verb yang aneh secara semantik dan
   JUSTRU MEMBURUK dengan prompt lebih kompleks — disimpulkan sebagai
   keterbatasan kapasitas 1.2B, bukan sifat keluarga EXAONE. 7.8B bisa
   jadi tidak mengulangi kegagalan itu, bisa juga tidak — UNVERIFIED.
3. Semua 4 artifact validasi yang sudah di-commit
   (`1_validation_results.json`, `item_discrimination_results.json`,
   `lm_plausibility_results.json`, `log_likelihood_results.json`)
   dihasilkan dari item yang digenerate pakai EXAONE-4.0-1.2B. Artifact
   itu TIDAK otomatis invalid (mereka mengukur properti distraktor/item
   yang sudah dihasilkan, bukan "model mana yang menghasilkannya"),
   tapi generasi item BARU apa pun sekarang pakai model berbeda, dan
   klaim kualitas-generasi EXAONE yang baru butuh verifikasi terpisah,
   bukan dibaca dari angka lama.

**Perubahan teknis di kedua file**: `_MODEL_NAME` diganti ke
`"LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"`, `trust_remote_code=True`
ditambahkan di kedua `from_pretrained` call (tokenizer dan model) —
dependensi pada `src/patch_exaone35_remote_code.py` (lihat docstring
skrip itu) tetap berlaku di sini juga.

**Wajib sebelum dianggap final** (lebih berat dari §9.51 karena scope
lebih luas): jalankan smoke test masing-masing file
(`exaone_provider.py`, `curriculum_item_generator.py`) di GPU nyata
dulu untuk pastikan tidak crash/regresi fungsional dasar, LALU jalankan
ulang keempat skrip validasi (`validate_novelty1.py`,
`validate_item_discrimination.py`, `validate_lm_plausibility.py`,
`validate_log_likelihood.py`) dengan item yang digenerate model baru,
bandingkan dengan 4 artifact JSON yang sudah di-commit. Kalau ada
regresi nyata pada kualitas item/distraktor, pertimbangkan kembalikan
`exaone_provider.py` ke EXAONE-4.0-1.2B (file ini adalah generator
utama untuk 11 skill yang mendasari Novelty 1 — risiko regresi di sini
mempengaruhi klaim paper paling langsung dari semua swap yang dibuat
sesi ini).

**Verifikasi yang SUDAH dilakukan**: `py_compile` pada kedua file
(sintaks valid). **Verifikasi yang BELUM bisa dilakukan** (sandbox ini
tidak punya GPU): semua hal yang disebut di atas.

**File:** `src/exaone_provider.py`, `src/curriculum_item_generator.py`.

---

### 9.53 CLI untuk jalankan verifikasi swap EXAONE-3.5-7.8B (`verify_exaone35_swap.py`)

**Konteks**: §9.51/§9.52 masing-masing berakhir dengan daftar "wajib
sebelum dianggap final" yang manual (run smoke test tiap file satu-satu,
urutan tertentu, apply patch kalau kena bug `create_causal_mask`, baru
jalankan 4 skrip validasi dan bandingkan ke baseline). User minta ini
dibuat jadi CLI, bukan daftar langkah manual.

**`src/verify_exaone35_swap.py`** (baru): dua tahap otomatis, jalankan
dari repo root dengan venv aktif:
1. **Smoke test** 5 file yang di-swap, urutan risiko terendah→tertinggi
   (`qwen_feedback.py` → `domain_vocab_substitution.py` →
   `curriculum_item_generator.py` → `solver_check.py` →
   `exaone_provider.py`). Kalau gagal dengan error yang menyebut
   `create_causal_mask`, otomatis jalankan
   `patch_exaone35_remote_code.py` sekali dan coba ulang file itu.
   Berhenti di tahap ini (tidak lanjut ke validasi) kalau ada yang masih
   gagal setelah itu.
2. **Validasi**: jalankan ulang 4 skrip validasi
   (`validate_novelty1.py`, `validate_item_discrimination.py`,
   `validate_lm_plausibility.py`, `validate_log_likelihood.py`) ke file
   `_new_*.json` (TIDAK menimpa baseline yang sudah di-commit), lalu
   diff setiap angka skalar top-level terhadap baseline lama dan tandai
   `<-- regression` kalau turun.

**Belum/TIDAK bisa dilakukan di sandbox ini** (tidak ada GPU): hanya
`py_compile` dan `--help` yang diverifikasi — isi logicnya (subprocess
calls ke 9 skrip lain) belum pernah benar-benar dijalankan end-to-end
karena setiap skrip yang dipanggil butuh model besar + CUDA. Harus
dicoba dulu di GPU nyata user sebelum dipercaya 100% tidak ada bug.

**Usage**:
```
python3 src/verify_exaone35_swap.py                  # full: smoke test + validasi + diff
python3 src/verify_exaone35_swap.py --skip-validate   # cuma smoke test
```

**File:** `src/verify_exaone35_swap.py` (baru).

---

### 9.53 Verifikasi swap EXAONE-3.5-7.8B di GPU nyata: OOM struktural ditemukan & diperbaiki (quantization 4-bit), 2 bug baru ditemukan (1 diperbaiki, 1 BELUM — butuh keputusan)

Sesi ini (beda dari sesi cloud §9.51/9.52, sesi ini PUNYA akses GPU)
menjalankan `src/verify_exaone35_swap.py` yang sebelumnya cuma bisa
`py_compile`-checked oleh sesi cloud. Progres:

**1. 2 bug lama (bukan dari swap) ditemukan & diperbaiki**: assert di
`exaone_provider.py`'s smoke test untuk skill `mod_adverb` masih
menyebut kata lama ("깨끗하게"/"깨끗한") dari SEBELUM perbaikan §9.47
(yang mengganti pasangan ke 늦다/늦게) — assert-nya lupa diupdate waktu
itu. Diperbaiki ke "늦게"/"늦은".

**2. Masalah struktural NYATA ditemukan — OOM, bukan tabrakan proses**:
satu instance EXAONE-3.5-7.8B (bf16) sendirian memakai ~15.6GB dari
15.57GB total GPU. Dampaknya dikonfirmasi lewat eksekusi nyata, bukan
dihitung di atas kertas saja:
- `curriculum_item_generator.py`'s `__main__` memuat generator-nya
  SENDIRI + `SolverCheckProvider()` dalam satu proses — setelah swap
  §9.52, keduanya EXAONE-3.5-7.8B, jadi 2x ~15.6GB dalam satu proses.
  OOM pasti terjadi, dikonfirmasi lewat run nyata (`torch.OutOfMemoryError`,
  "14.89 GiB in use, 682.31 MiB free").
- Lebih penting: `content_provider.DispatchingGrammarProvider` (jalur
  PRODUKSI nyata, §9.49, menggabungkan `ExaoneGrammarProvider` + 
  `CurriculumGrammarProvider`) punya masalah IDENTIK begitu kedua
  provider itu di-swap ke model yang sama — bukan cuma masalah smoke
  test/QA script.

**3. Fix: quantization 4-bit (bitsandbytes), diuji sebelum diadopsi**:
Ditanya ke user, pilihan: investigasi quantization dulu (bukan langsung
turun ke model lebih kecil, bukan juga "terima saja tidak bisa jalan
bersamaan"). Dites langsung: 1 instance EXAONE-3.5-7.8B quantized
4-bit (`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)`)
memakai **7.30GB** (vs ~15.6GB bf16) — kurang dari separuh. **2 instance
sekaligus diuji bersamaan, bukan diasumsikan**: total **14.61GB**,
KEDUANYA tetap generate dengan benar. Diterapkan ke 4 file yang memuat
EXAONE-3.5-7.8B sendiri: `exaone_provider.py`, `curriculum_item_generator.py`,
`solver_check.py`, `domain_vocab_substitution.py`'s `Exaone35Generator`.
`bitsandbytes==0.50.2` ditambahkan ke `requirements.txt`.

**Caveat yang HARUS didisclose, bukan diabaikan**: quantization 4-bit
mengubah precision komputasi model — ini relevan khusus untuk
`solver_check.py`'s `sentence_logprob()` (basis `validate_log_likelihood.py`,
dikutip di paper §9.35/9.36 sebagai 98.1%/87.5% dengan model BF16 TIDAK
di-quantize). Re-run `validate_log_likelihood.py` dengan model baru
(EXAONE-3.5-7.8B, QUANTIZED) mengubah DUA variabel sekaligus (model
beda DAN precision beda) dibanding baseline yang dipublikasikan —
PERBEDAAN HASIL NANTI (kalau ada) tidak bisa langsung diatribusikan ke
salah satu saja. Belum dijalankan ulang validate_*.py sampai smoke test
semua lolos dulu (lihat poin 5).

**4. Setelah quantization diterapkan, re-run smoke test: OOM TERATASI
(dikonfirmasi), tapi 2 kegagalan BARU muncul**:
- `curriculum_item_generator.py` dan `solver_check.py`: **[OK]** — OOM
  benar-benar teratasi, dikonfirmasi lewat run nyata bukan diasumsikan
  dari angka memori saja.
- `domain_vocab_substitution.py`: **[FAIL]** — `ValueError:
  '**모듈**' last character is not a Hangul syllable block`. EXAONE-3.5-7.8B
  kadang membungkus jawabannya dengan markdown bold (`**kata**`),
  sesuatu yang Qwen3-4B (model lama) tidak/jarang lakukan — `_STRIP_CHARS`
  belum mencakup `*`. **Diperbaiki**: `*` ditambahkan ke `_STRIP_CHARS`
  di `domain_vocab_substitution.py` DAN `curriculum_item_generator.py`
  (kerentanan identik, preventif meski belum crash di sana).
- `exaone_provider.py`: **[FAIL] — BELUM diperbaiki, butuh keputusan
  user**. Assert `blind_retries == [None, None, None]` untuk slot
  "해변" (Novelty 2 critic-loop test, PRD §9.14) gagal: EXAONE-4.0-1.2B
  (model lama) terbukti GAGAL TOTAL pada slot ini secara deterministik
  (list `known_failures` di baris ~1060 ditemukan empiris dari
  kegagalan model lama) — tapi EXAONE-3.5-7.8B (lebih besar, lebih
  mampu) sekarang menjawab "노는다" tiga kali, bukan `None`. "노는다"
  BUKAN bentuk kamus verba yang benar (bentuk present-tense 놀다 yang
  benar adalah "논다", bukan "노는다") — tapi `_extract_verb_lemma()`
  cuma mengecek "berakhir dengan 다 dan panjang ≥2", jadi string ini
  LOLOS sebagai valid padahal salah. **Akar masalah**: `_extract_verb_lemma`
  sudah longgar sejak awal (tidak membedakan bentuk kamus asli dari
  konjugasi apa pun yang berakhir 다) — kelemahan ini TERSEMBUNYI
  selama ini karena model lama yang lebih lemah selalu gagal dengan
  cara yang jelas-jelas salah (bukan string yang *terlihat* valid),
  jadi heuristik longgar ini tidak pernah benar-benar diuji oleh kasus
  false-positive. Swap model yang mengungkapnya, bukan menyebabkannya.
  **BELUM DIPUTUSKAN** (ditanya ke user, belum sempat dijawab sebelum
  sesi harus berhenti): perketat `_extract_verb_lemma` (tolak akhiran
  konjugasi present-tense umum seperti -는다/-ㄴ다, bukan cuma -다 polos)
  vs update assert test untuk menerima perilaku baru model vs opsi lain.

**5. Validasi penuh (`validate_novelty1.py` dkk., Step 2/2 skrip
verifikasi) BELUM pernah dijalankan** — `verify_exaone35_swap.py`
berhenti di smoke test (Step 1/2) karena kegagalan di atas, sesuai
desain skripnya sendiri ("stopping before validation (fix the crash
first)"). Urutan kerja berikutnya: selesaikan keputusan poin 4 di atas
→ smoke test semua `[OK]` → jalankan Step 2/2 → bandingkan dengan 4
baseline JSON yang sudah di-commit, dengan caveat poin 3 di atas
tentang quantization.

**File:** `src/exaone_provider.py` (2 fix assert mod_adverb + quantization,
`_extract_verb_lemma` BELUM diubah), `src/curriculum_item_generator.py`
(quantization + fix markdown strip), `src/solver_check.py` (quantization),
`src/domain_vocab_substitution.py` (quantization + fix markdown strip),
`requirements.txt` (+bitsandbytes==0.50.2).

---

### 9.56 Konflik sesi cloud vs sesi lokal diselesaikan: lanjut quantization (bukan revert), qwen_feedback.py yang terlewat diperbaiki, `_extract_verb_lemma` diperketat

**Konteks penting**: sesi cloud ini (§9.51/9.52/9.54) dan sesi lokal
(§9.53, akses GPU nyata) berjalan PARALEL tanpa saling tahu, dan
mengambil keputusan BERLAWANAN pada masalah OOM yang sama — cloud
memilih revert total ke model asli (commit `937504c`), lokal memilih
quantization 4-bit (commit `14a2693`). Keduanya di-push, terjadi merge
conflict, dan merge (`52ba9ad`) memilih **mempertahankan quantization
lokal, membuang revert cloud**. User ditanya ulang secara eksplisit di
sesi ini setelah menemukan konflik ini — jawaban: **lanjut
quantization** (bukan revert lagi), dan **perketat `_extract_verb_lemma`**
untuk isu yang masih pending dari §9.53.

**1. `qwen_feedback.py` yang terlewat quantization, diperbaiki**: saat
fix quantization §9.53 diterapkan ke 4 file, file ini TIDAK disentuh —
masih bf16 polos. Lolos smoke test di run user HANYA karena GPU
kebetulan bersih saat itu (lihat urutan run di chat), bukan karena
aman — `detailed_feedback()`-nya butuh hingga 1200 token baru, KV-cache
lebih besar dari generation file lain, jadi ini yang paling berisiko
OOM lagi. Sekarang pakai `BitsAndBytesConfig(load_in_4bit=True,
bnb_4bit_compute_dtype=torch.bfloat16)`, konsisten dengan 4 file
lainnya.

**2. `_extract_verb_lemma` diperketat, divalidasi empiris dengan
kiwipiepy nyata (bukan ditebak)**: masalah asli — "ends in 다" saja
menerima bentuk KONJUGASI present-tense naratif yang juga berakhir 다
(먹다→먹는다, 가다→간다), bukan hanya bentuk kamus. EXAONE-3.5-7.8B
menghasilkan "노는다" untuk slot "해변" (bahkan BUKAN konjugasi yang
benar dari 놀다 — present tense asli adalah "논다" — tapi cukup mirip
bentuk konjugasi untuk lolos check lama).

Diinstal `kiwipiepy` di sandbox ini (CPU-only, tidak butuh GPU) untuk
menguji hipotesis perbaikan SEBELUM diimplementasikan, bukan asumsi:
```
'읽다': [('읽', 'VV'), ('다', 'EF')]      -- bentuk kamus asli
'간다': [('가', 'VV'), ('ᆫ다', 'EF')]     -- konjugasi (ㄴ fusi batchim)
'노는다': [('놀', 'VV'), ('는다', 'EF')]   -- konjugasi (akhiran literal)
'안다': [('알', 'VV'), ('ᆫ다', 'EF')]     -- AMBIGU: kiwi baca sebagai
                                            konjugasi 알다("tahu"), bukan
                                            kata dasar 안다("memeluk")
```
Temuan penting dari tes ini: kata dasar kamus asli (consonant-stem
verb berakhiran ㄴ-batchim+다, misal 안다 "memeluk", 신다 "memakai
sepatu") SECARA INHEREN ambigu dengan konjugasi vowel/ㄹ-stem — bahkan
kiwipiepy sendiri, parser statistik yang sudah dipakai di seluruh
proyek ini, default membaca "안다" sebagai konjugasi "알다" (tahu)
bukan kata dasar "memeluk", tanpa konteks kalimat. Tidak ada cara
murah untuk membedakan keduanya dari kata tunggal saja.

**Keputusan trade-off, didisclose bukan disembunyikan**: perketat
dengan dua aturan deterministik (Unicode decomposition, bukan model
call) — (a) tolak akhiran literal "-는다" (aman, tidak ambigu, tidak
ada kata dasar asli yang berakhir begini), (b) tolak jika karakter
sebelum "다" punya batchim ㄴ (menangkap pola "-ㄴ다" vowel/ㄹ-stem).
Ini SENGAJA juga menolak kata dasar asli yang kebetulan collision
dengan pola ini (안다, 신다, dan sejenisnya) — diterima sebagai trade-off
sadar: kehilangan kandidat jawaban yang jarang dan ambigu jauh lebih
murah daripada meloloskan konjugasi salah sebagai lemma terverifikasi.

**Diverifikasi** (di sandbox ini, logic murni tanpa GPU): fungsi
diekstrak dan dites standalone terhadap seluruh assertion smoke test
yang sudah ada (읽다→읽다, "travel"→None, 게임하기→None) PLUS kasus bug
baru (노는다→None, 먹는다→None, 간다→None, 논다→None) PLUS kata dasar
asli yang tidak ambigu (가다/먹다/놀다/하다/만나다→diterima apa adanya).
Semua ✓. Efek samping: assert `blind_retries == [None, None, None]`
untuk slot "해변" (§9.53's blocker) seharusnya otomatis kembali lolos
dengan EXAONE-3.5-7.8B juga, karena "노는다" sekarang diekstrak jadi
`None` sama seperti kegagalan model lama -- TIDAK bisa dikonfirmasi di
sini (butuh GPU untuk benar-benar generate), tapi logic ekstraksinya
sudah diverifikasi benar.

**Belum dilakukan** (butuh GPU user): jalankan ulang
`verify_exaone35_swap.py` penuh (smoke test + validasi) untuk konfirmasi
kedua fix ini benar-benar menyelesaikan semua blocker, lalu lanjut ke
Step 2/2 (4 skrip validasi) yang belum pernah jalan sejak swap+quantization,
dengan caveat §9.53 tentang quantization mengubah 2 variabel sekaligus
(model DAN precision) dibanding baseline yang dipublikasikan.

**File:** `src/qwen_feedback.py` (+quantization), `src/exaone_provider.py`
(`_extract_verb_lemma` diperketat).

---

### 9.57 Audit "terlalu hardcoded": gerbang plausibilitas kolokasi (Korean SBERT) untuk fallback LLM verb_lookup, divalidasi empiris di GPU user sebelum diimplementasikan

**Konteks**: user minta audit eksplisit elemen hardcoded di seluruh
codebase ("you are llm engineer senior expert, investigate what can we
try to replace it"). Agent survey menemukan ~55 pasangan kata
tangan-kurasi di `SKILL_SPECS`'s `verb_lookup`/`pairs` sebagai blok
hardcoded terbesar — dipilih di awal proyek karena generasi LLM bebas
menghasilkan pasangan aneh secara semantik ("커피 -> 차다" = "menendang
kopi", module docstring Round 3). 2 opsi perbaikan diajukan: (1)
Korean Sentence-BERT sebagai gerbang plausibilitas untuk generasi LLM
bebas, (2) diversifikasi filler word via LLM+scoring yang sudah
terbukti. User putuskan: skip opsi (2) (dampaknya kecil — filler
sudah terdokumentasi sendiri sebagai "not independently diagnostic",
tidak menyentuh klaim paper manapun — biaya implementasi lebih besar
dari manfaatnya), lanjut opsi (1) saja.

**Validasi empiris SEBELUM implementasi** (bukan diasumsikan bekerja):
tidak bisa dites di sandbox cloud ini (huggingface.co diblokir
kebijakan jaringan sesi — dicek lewat endpoint status proxy, bukan
masalah transient, jadi TIDAK dicoba diakali). Ditulis
`src/investigate_sbert_plausibility.py` untuk dijalankan user di GPU
lokalnya. Hasil nyata: 10 pasangan GOOD (dari verb_lookup sendiri) vs
10 pasangan BAD (termasuk kasus asli "커피+차다") pakai model
`jhgan/ko-sroberta-multitask` — GOOD mean=0.5575, BAD mean=0.3180,
overlap cuma 7/100 pasangan (bad,good) di mana BAD > GOOD. Separasi
jauh lebih baik dari ekspektasi awal (awalnya skeptis berdasarkan
preseden §9.50 yang gagal 6 metode untuk tugas plausibilitas sejenis
— tapi itu untuk substitusi kosakata DOMAIN KHUSUS/teknis
smart-factory, bukan kolokasi sehari-hari seperti di sini; perbedaan
domain inilah alasan prinsipil kenapa hasilnya berbeda, bukan
kontradiksi temuan lama).

**Keterbatasan yang didisclose, bukan disembunyikan**: 1 kasus di data
kalibrasi ("공원"+"마시다" = 0.4689) hampir tidak terpisahkan dari
pasangan GOOD terlemah — nouns yang topically-related (parkir dan
minum sama-sama aktivitas yang berasosiasi) tapi argumen verbanya
sebenarnya salah. Ini kelemahan umum cosine similarity untuk
plausibilitas kolokasi (mengukur keterkaitan topik, bukan kecocokan
argumen-verba sesungguhnya) — bukan bug yang "diperbaiki" dengan
threshold lebih baik, melainkan sifat metode yang diterima dengan
sadar.

**Implementasi** (`src/collocation_plausibility.py`, baru):
- `is_plausible_collocation(noun, verb)` — cosine similarity via
  `jhgan/ko-sroberta-multitask`, threshold `0.43` (dipilih konservatif:
  lebih baik tolak kandidat bagus yang ambigu daripada terima kandidat
  buruk — biaya tolak rendah, caller sudah punya jalur retry/skip).
- **Fail-open secara sengaja**: kalau model/`sentence-transformers`
  tidak tersedia (offline, belum ter-install), gerbang ini BYPASS
  (anggap semua pasangan plausible) — PERSIS perilaku sebelum modul
  ini ada, bukan crash. Satu peringatan dicetak sekali supaya bypass
  ini terlihat, tidak diam-diam. Diverifikasi langsung di sandbox ini
  (tanpa akses huggingface.co): `python3 src/collocation_plausibility.py`
  → bypass terkonfirmasi jalan, tidak crash.
- **Tabel `verb_lookup` yang sudah ada TIDAK disentuh sama sekali** —
  gerbang ini hanya dipasang di jalur fallback `_pick_verb_lemma`
  (dipakai HANYA kalau slot belum ada di tabel,
  `spec["verb_lookup"].get(slot) or self._pick_verb_lemma(...)`) — nol
  risiko regresi ke konten yang sudah terbukti jalan.
- **Critic-loop prompt diperbaiki juga** (bug yang ditemukan sambil
  implementasi, bukan disengaja sejak awal): `_CRITIC_VERB_PROMPT` yang
  lama bilang "ini bukan kata kerja / tidak berakhir 다" ke model —
  pesan itu SALAH kalau alasan tolaknya justru plausibilitas (kata
  kerjanya valid secara gramatikal, cuma aneh maknanya, seperti
  "차다"). Ditambahkan `_CRITIC_VERB_PLAUSIBILITY_PROMPT` terpisah, dan
  `_pick_verb_lemma` sekarang pilih prompt yang BENAR berdasarkan
  kegagalan mana yang sebenarnya terjadi — diverifikasi lewat 4 skenario
  standalone (grammar-fail, plausibility-fail, langsung lolos, dan
  gagal total) sebelum commit, semua lolos.

**Belum diverifikasi** (butuh GPU+internet user, tidak bisa di
sandbox ini): perilaku nyata `is_plausible_collocation` saat benar-benar
dipanggil dari `_pick_verb_lemma` (baru diverifikasi mekanisme kontrol
alirnya secara standalone, bukan end-to-end dengan model nyata), dan
apakah threshold 0.43 masih pas di skala produksi (baru dikalibrasi di
20 pasangan, bukan ratusan).

**File:** `src/collocation_plausibility.py` (baru),
`src/exaone_provider.py` (`_pick_verb_lemma` + 2 prompt kritik),
`requirements.txt` (+sentence-transformers==6.1.0).

---

### 9.58 Investigasi korpus KoLLA v2.0 untuk mining error partikel nyata (persiapan fine-tuning vs kurasi berbasis frekuensi)

**Konteks**: lanjutan audit §9.57 — user tanya spesifik apakah pakai
korpus ini artinya fine-tuning/training model sungguhan, atau cuma
analisis data. Dijelaskan: 2 jalur beda level — (A) analisis frekuensi
murni (bukan training, cuma informasikan kurasi tabel distraktor yang
sudah ada) vs (B) fine-tune classifier kecil (KLUE-RoBERTa) pakai
anotasi error korpus sebagai label asli. User: "mau, apakah menurut
anda itu best to try? apakah ada corpus yang lebih tepat dan cocok?"

**Pencarian alternatif korpus** (websearch, bukan diasumsikan KoLLA
satu-satunya): tidak ditemukan korpus Korea lain yang (a) anotasi error
gramatikal LEBIH LUAS dari partikel saja DAN (b) bisa diunduh langsung
tanpa proses aplikasi/persetujuan. NIKL's "모두의 말뭉치" (Everyone's
Corpus) punya korpus pembelajar Korea, tapi: butuh isi perjanjian
online + persetujuan dulu (bukan unduh langsung), dan yang terkonfirmasi
tersedia adalah anotasi MORFOLOGI (POS tagging), BUKAN anotasi error
gramatikal — tidak jelas apakah anotasi error-nya didistribusikan
resmi sama sekali. **Kesimpulan**: KoLLA v2.0 tetap pilihan PRAKTIS
terbaik saat ini — bukan karena scope-nya ideal (sudah dikonfirmasi
sempit, cuma partikel, §9.57), tapi karena satu-satunya yang
dikonfirmasi bisa diunduh langsung (Zenodo, GPLv3, tanpa aplikasi).

**Download diblokir di sandbox ini juga**: `zenodo.org` ternyata SAMA
diblokir seperti `huggingface.co` (dicek lewat endpoint status proxy —
kebijakan, bukan transient). Ditulis
`src/investigate_kolla_corpus.py` untuk dijalankan user di mesin
dengan akses internet: download via Zenodo API resmi, ekstrak file M2
(format anotasi GEC standar), survei SEMUA tag error_type yang
benar-benar ada (bukan diasumsikan dari deskripsi prosa di paper —
paper cuma bilang "omission/substitution/addition/ordering" tapi tidak
mengonfirmasi string tag M2 yang sesungguhnya dipakai), lalu hitung
instance per jenis substitusi partikel yang relevan ke proyek ini
(이/가, 을/를, 은/는, 에/에서, 과/와, 로/으로).

**Diverifikasi di sandbox ini** (tanpa akses network, logic murni):
regex parser M2 dites terhadap sampel sintetis mengikuti format M2
standar (`S ...` / `A start end|||etype|||correction|||...`) — berhasil
ekstrak etype dan correction dengan benar. **BELUM diverifikasi**:
apakah taksonomi tag KoLLA yang sesungguhnya cocok dengan asumsi
regex ini (butuh file nyata, tidak bisa diunduh di sini) — makanya
skrip SENGAJA mensurvei semua tag yang ada dulu (bukan langsung
filter dengan asumsi), supaya kalau asumsi salah, user/sesi berikutnya
bisa lihat taksonomi asli dan sesuaikan filternya.

**Aturan keputusan yang ditulis eksplisit di skrip**: kalau mayoritas
jenis partikel punya <~30 instance, jalur B (fine-tuning) tidak
realistis (risiko overfitting), jalur A (kurasi berbasis frekuensi,
tanpa training) jadi pilihan yang masuk akal. Belum diputuskan sampai
hasil nyata didapat.

**File:** `src/investigate_kolla_corpus.py` (baru).

---

### 9.59 Bug ekstraksi partikel di `investigate_kolla_corpus.py` ditemukan dari run nyata user, diperbaiki & diverifikasi

**Hasil run pertama user** (GPU/mesin lokal, akses nyata ke Zenodo):
script BERHASIL unduh+parse 4464 anotasi dari `KoLLA_multi-refs.m2`.
Survei taksonomi tag terkonfirmasi: error partikel (tag mengandung
`ADP`) memang BANYAK — `R:NOUN+ADP -> NOUN+ADP`: 152,
`R:ADV+ADP -> ADV+ADP`: 109, `R:VERB+ADP -> VERB+ADP`: 104,
`R:NOUN -> NOUN+ADP`: 197, dst., total ratusan instance. TAPI filter
"project-relevant particle" cuma menangkap **2 instance total** — jelas
bug, bukan berarti datanya tidak ada.

**Akar masalah**: filter v1 mencocokkan field `correction` M2 secara
LITERAL ke daftar partikel pendek (`"에서"` dll.) — tapi `correction`
berisi KATA UTUH termasuk partikelnya (`"학교에서"`, bukan `"에서"`
saja), jadi hampir semua instance nyata tidak pernah cocok.

**Fix v2 pertama (common-prefix diff) JUGA salah**, ditemukan sendiri
lewat tes standalone sebelum dikirim ke user (bukan dari laporan user
lagi): diff `"학교에"` vs `"학교에서"` lewat longest-common-prefix
gagal — karena `"에"` adalah PERSIS prefix literal dari `"에서"`,
seluruh `"학교에"` (3 karakter) ikut dianggap "sama" dengan 3 karakter
pertama `"학교에서"`, menyisakan `before=""` (kosong) → ditolak oleh
pengecekan `not before`. Ini justru menghilangkan kasus 에/에서 yang
PALING relevan untuk `particle_loc` — kegagalan diam-diam yang baru
ketahuan karena ditest eksplisit, bukan diasumsikan jalan.

**Fix v3 (final, diverifikasi)**: ganti pendekatan diff-karakter dengan
pencocokan SUFFIX PARTIKEL YANG DIKETAHUI (`_KNOWN_PARTICLE_SUFFIXES`,
diurutkan terpanjang dulu supaya `"에서"` dicoba sebelum `"에"`) —
lepaskan suffix partikel dari kedua sisi, terima sebagai perubahan
partikel HANYA kalau stem kata bendanya identik di kedua sisi (bukan
ganti kata sekaligus). Diverifikasi lewat 5 assertion standalone,
termasuk kasus bug asli (`학교에`→`학교에서` sekarang benar
menghasilkan `("에","에서")`) dan kasus negatif (ganti kata utuh TIDAK
boleh disalahbaca sebagai ganti partikel) — semua lolos sebelum commit.

**Status**: belum dijalankan ulang oleh user dengan fix ini — hasil
nyata (apakah jumlah instance per pasangan partikel cukup untuk
fine-tuning) masih pending run berikutnya.

**File:** `src/investigate_kolla_corpus.py`.

---

### 9.60 Cleanup codebase: hapus fitur yang tidak pernah terhubung ke paper, hapus script investigasi yang tujuannya sudah tercapai

**Instruksi user**: "sebelum lanjut, saya ingin meng-clean codingan
dulu agar rapi dan ga kompleks, saya hanya mau fokus pada code yang
mendukung paper" — diucapkan tepat setelah hasil run nyata
`investigate_kolla_corpus.py` v2 masuk (329 instance perubahan
partikel relevan proyek, mengonfirmasi opsi kurasi-berbasis-frekuensi
realistis, opsi fine-tuning tidak) — investigasi KoLLA SENGAJA
DIJEDA, bukan dihentikan, akan dilanjutkan setelah cleanup ini.

**Verifikasi sebelum hapus apa pun**: grep eksplisit `content.py`
(teks paper) untuk nama file/konsep fitur kurikulum — HASIL: 0 hit
bermakna (`curriculum_provider.py` dkk. tidak disebut sama sekali;
satu-satunya kemunculan kata "curriculum" generik, tidak berhubungan).
Dikonfirmasi: fitur upload-kurikulum + substitusi domain
EV/smart-factory (dibangun §9.46-9.49, lalu ikut di-swap model di
§9.51-9.56) memang TIDAK PERNAH ditulis ke paper sejak dibuat.

**Scope dikonfirmasi ke user via AskUserQuestion** (2 pertanyaan,
dijawab "no preference" — diinterpretasikan sebagai setuju opsi yang
sudah ditandai "(Recommended)" di pertanyaan itu sendiri, bukan
diasumsikan tanpa dasar):
1. Fitur kurikulum: **dihapus** (bukan dipindah/didiamkan) — paling
   sesuai "fokus ke yang mendukung paper"; riwayat tetap ada di git
   log kalau perlu dikembalikan.
2. `investigate_sbert_plausibility.py`: **dihapus** — tujuannya
   (kalibrasi threshold `collocation_plausibility.py`) sudah tercapai
   dan hasilnya permanen tercatat di docstring modul itu + PRD §9.57.

**File dihapus**: `src/curriculum_provider.py`,
`src/curriculum_item_generator.py`, `src/curriculum_grammar_provider.py`,
`src/domain_vocab_substitution.py`, `src/investigate_sbert_plausibility.py`.

**Dependensi yang diikut-bersihkan** (supaya tidak ada import
menggantung/referensi rusak ke file yang dihapus):
- `content_provider.py`: hapus class `DispatchingGrammarProvider`
  (wiring khusus curriculum) dan blok smoke-test "mixed-source" di
  `__main__` yang meng-import `CurriculumGrammarProvider`.
  `CombinedContentProvider` (vocab+grammar murni, yang benar-benar
  dipakai production/paper) TIDAK disentuh.
- `verify_exaone35_swap.py`: hapus `domain_vocab_substitution.py` dan
  `curriculum_item_generator.py` dari `_SMOKE_TEST_FILES` (file itu
  sudah tidak ada, akan gagal kalau tetap dicoba dijalankan).
- `qwen_feedback.py`, `collocation_plausibility.py`: komentar yang
  menyebut nama file yang sudah dihapus diperbarui supaya tidak
  menunjuk ke file yang tidak ada lagi (riwayat keputusannya tetap
  akurat, cuma referensi filenya diperbaiki).

**Diverifikasi**: `py_compile` SEMUA file di `src/` (bukan cuma yang
diedit) — lolos semua. Grep ulang seluruh `src/` untuk import modul
yang dihapus — nol hasil.

**Belum dihapus, sengaja ditinggalkan**: `Korean_Grammar_Curriculum.docx`
di root repo — ini file yang di-upload user sendiri (data, bukan
kode), bukan kompleksitas kode yang jadi concern user — dibiarkan
kecuali user minta dihapus juga.

**Investigasi yang TIDAK dihentikan** (aktif, sengaja dijeda):
`src/investigate_kolla_corpus.py` tetap ada — ini bukan bagian fitur
kurikulum yang dihapus, melainkan investigasi terpisah untuk
memperkuat skill `particle_loc`/`particle_obj` yang MEMANG didukung
paper, dan user secara eksplisit mau melanjutkannya setelah cleanup
ini.

**File:** `src/content_provider.py`, `src/verify_exaone35_swap.py`,
`src/qwen_feedback.py`, `src/collocation_plausibility.py` (diedit); 5
file dihapus (lihat atas).

---

### 9.61 Investigasi kategori edit-operation KAGAS (SPELL) via data KoLLA yang sudah ada — kandidat skill ke-12

**Konteks**: user tanya kenapa 7 kategori edit-operation KAGAS
(INS/DEL/WS/WO/SPELL/PUNCT/SHORT) tidak diimplementasikan, lalu minta
investigasi apakah bisa "dilatih atau dari corpus/library transformers"
untuk mengisi gap ini.

**Koreksi atas penjelasan sebelumnya**: tidak semua 7 kategori
edit-operation memang tidak cocok format pilihan-ganda-cloze yang
dipakai 11 skill ini — SPELL, WS (spasi), SHORT (pemendekan) secara
struktural BISA jadi item cloze sama seperti yang sudah ada. Yang
benar-benar sulit cuma INS/DEL/WO (soal keberadaan/urutan token, bukan
substitusi satu blank — butuh perubahan format item, bukan cuma
tabel baru).

**Temuan konkret**: survei taksonomi `investigate_kolla_corpus.py` di
§9.59 (run nyata user) sudah menunjukkan `R:SPELL: 1986` — JAUH lebih
banyak dari seluruh kategori partikel digabung (329 total). Ini data
yang SUDAH ADA di mesin user (`kolla_corpus_download/`), tidak perlu
download ulang.

**`src/investigate_kolla_spell.py`** (baru): reuse `parse_m2`/
`download_and_extract` dari `investigate_kolla_corpus.py` (bukan
duplikasi), filter khusus `etype == "R:SPELL"`, laporkan: (a) jumlah
pasangan (asli, koreksi) UNIK, (b) berapa yang BERULANG (count>=2 —
indikasi konfusi sistematis seperti 되/돼, bukan typo acak per-esai),
(c) top-40 pasangan terbanyak, (d) sampel pasangan singleton untuk
menilai apakah long-tail-nya memang typo acak atau masih berpola.

**Aturan keputusan eksplisit di skrip**: beberapa pasangan dengan
count>=10 = kandidat skill baru yang layak (format sama seperti 11
skill lain: cloze blank, bentuk benar vs bentuk salah-umum sebagai
distraktor) — ini Opsi A (kurasi berbasis frekuensi) untuk KATEGORI
BARU, bukan untuk skill yang sudah ada. Kalau 1986 koreksi tersebar di
ratusan pasangan nyaris-unik dengan repetisi rendah per-pasangan →
errornya kemungkinan besar typo acak per-individu, bukan konfusi
sistematis — tidak layak jadi skill baru meski totalnya besar
(jumlah besar tidak otomatis = pola yang bisa dipelajari, perlu dicek
distribusinya, bukan cuma totalnya).

**Diverifikasi**: `py_compile` lolos, parsing `R:SPELL` dites terhadap
sampel M2 sintetis — mekanisme ekstraksi (original dari token-index S
line, dipasangkan dengan correction) terkonfirmasi benar.

**Belum dijalankan**: hasil nyata (apakah SPELL benar-benar punya pola
sistematis yang layak) masih pending run user.

**File:** `src/investigate_kolla_spell.py` (baru).

---

### 9.62 Klaim paper yang KEDALUWARSA ditemukan & diperbaiki: pola non-monoton growing-pool "belum dipahami" padahal SUDAH dijelaskan penuh di §9.17

**Konteks**: user menantang audit sebelumnya (§9.56-9.61) — jangan
terima begitu saja framing "paper sendiri jujur mengakui [X]" sebagai
batas akhir; sebagai senior LLM engineer, cari dulu apakah metode/bukti
yang lebih kuat sudah ada sebelum menyerah pada framing itu.

**Temuan**: `content.py` Section V-C (dekat baris 824, evidensi
Novelty 2) menyatakan hasil non-monoton Urnings di regime growing-pool
"we do not yet have a complete account of" — TAPI PRD §9.17 (ditulis
jauh lebih awal dari paragraf ini, lihat tanggal section) sudah
menyimpulkan secara eksplisit: **"pola non-monoton §9.13 sudah
dijelaskan penuh, bukan lagi open item."** Akar penyebabnya sudah
dikonfirmasi lewat `simulate_item_coldstart_diagnosis.py`, 2 tahap,
bukan dugaan:
1. Prior lemah (noise_std=2.0) mendorong 11.3% item ke k0 ekstrem
   (0/32) vs 0.0% di prior kuat (noise_std=0.3) — diukur dari 2000
   sampel per level.
2. Item yang mentok TIDAK PERNAH dipilih `pick_adaptive` — dikonfirmasi
   lewat pelacakan langsung sepanjang 120.000 respons: 0.00x untuk
   item mentok, vs ~20-23x untuk item normal.
3. Kesimpulan: RMSE akhir kondisi lemah "terlihat lebih baik" bukan
   karena prior tidak informatif itu bermanfaat, tapi karena item yang
   mentok efektif hilang dari peredaran — respons yang sama
   terkonsentrasi ke lebih sedikit item, masing-masing dapat exposure
   lebih banyak, jadi mengkalibrasi lebih presisi terlepas dari titik
   awalnya. Artefak mekanisme seleksi, bukan manfaat nyata prior lemah.

**Ini BUKAN kasus "metode baru ditemukan"** — ini kasus **evidensi yang
sudah ada dan sudah dikonfirmasi penuh, tapi tidak pernah dibawa masuk
ke teks paper** saat paragraf Section V-C ditulis. Paper secara tidak
sengaja UNDERSELLING bukti yang sudah dimiliki proyek ini sendiri.

**Diperbaiki**: paragraf Section V-C ditulis ulang — klaim "we do not
yet have a complete account of" diganti penjelasan lengkap (saturasi
11.3% vs 0.0%, seleksi 0x vs ~20-23x, mekanisme konsentrasi-exposure),
plus implikasi praktis yang sudah tercatat di §9.17 (item dengan
estimasi kesulitan sangat meleset bisa jadi "mati"/tidak pernah
disajikan — perlu monitoring item mati, bukan cuma akurasi kalibrasi).
Regime fixed-pool tetap jadi headline result (lebih simpel, memisahkan
kualitas prior dari interaksi saturasi-seleksi ini) — growing-pool
dilaporkan untuk kelengkapan dan karena temuannya sendiri relevan
secara operasional, bukan karena mengubah klaim inti Novelty 2.

**Diverifikasi**: `py_compile` pada `content.py` lolos. `build_docx.py`
dijalankan ulang → `paper/KO-DRILL_IEEE.docx` ter-generate. Konversi
PDF via `libreoffice --headless --convert-to pdf` → `paper/KO-DRILL_IEEE.pdf`.
Dicek langsung lewat ekstraksi teks PDF (`pdfplumber`): teks baru
("saturated starting urn", "selected 0 times") terkonfirmasi muncul
di PDF yang di-build ulang, bukan diasumsikan berhasil.

**Catatan untuk audit lanjutan**: ini mengindikasikan kemungkinan ada
klaim "belum dipahami/belum diselesaikan" LAIN di paper yang juga
sudah kedaluwarsa relatif terhadap temuan PRD yang lebih baru — belum
disurvei secara sistematis, baru 1 kasus ini yang dicek dan
dikonfirmasi.

**File:** `src/content.py`, `paper/KO-DRILL_IEEE.docx`, `paper/KO-DRILL_IEEE.pdf`.

---

### 9.63 Regresi nyata dari swap EXAONE-3.5-7.8B dilaporkan user: `qwen_feedback.py` balas pakai Korea, bukan English — direvert khusus file ini

**Laporan user**: feedback/explanation yang seharusnya Bahasa Inggris
keluar dalam Bahasa Korea.

**Diagnosis** (dari docstring `qwen_feedback.py` sendiri, bukan
tebakan): modul ini punya riwayat kegagalan PERSIS seperti ini —
"Qwen3-1.7B, asked to answer in Indonesian directly, ignored the
instruction and answered in Korean instead" — diperbaiki waktu itu
lewat teknik one-shot example (2 kalimat kontras, menghindari model
mengkopi topik contoh) yang **diuji dan divalidasi khusus untuk
Qwen3**. Swap ke EXAONE-3.5-7.8B (§9.51, `_MODEL_NAME` diganti,
mekanisme prompt lain dibiarkan sama) TIDAK PERNAH menguji ulang
kepatuhan bahasa untuk model baru ini — risiko yang sudah diberi
peringatan generik ("NOT confirmed on real hardware") tapi tidak
spesifik soal bahasa. Laporan user mengonfirmasi: risiko itu
benar-benar terjadi, EXAONE-3.5-7.8B mengulang kegagalan asli Qwen3-1.7B.

**Keputusan**: revert `qwen_feedback.py` SAJA (bukan ketiga file)
kembali ke `Qwen/Qwen3-4B`, bf16 (tanpa quantization — ukuran asli
~8GB nyaman di budget 16GB, tidak perlu quantization yang baru
dipasang §9.55). File ini dipilih untuk direvert lebih dulu karena
riwayat kepatuhan-bahasanya SECARA SPESIFIK sudah tervalidasi untuk
Qwen3, bukan model lain — mencoba rekayasa ulang teknik serupa untuk
EXAONE berisiko lebih tinggi dan lebih lama dibanding kembali ke
solusi yang sudah terbukti bekerja.

**Keputusan `exaone_provider.py`/`solver_check.py` BELUM diubah** —
masih menunggu hasil Step 2/2 (`verify_exaone35_swap.py`) yang user
sedang jalankan di GPU-nya, terpisah dari masalah ini.

**Diverifikasi**: diff eksplisit antara versi saat ini dan versi
pre-swap (`git show ab81d8a:src/qwen_feedback.py`) dicek dulu untuk
pastikan hanya `_MODEL_NAME`+kode loading yang berbeda (bukan ada
perubahan lain yang ikut hilang kalau direvert) — dikonfirmasi bersih,
cuma 2 blok (konstanta model+komentar, cara load model). `py_compile`
lolos setelah revert.

**File:** `src/qwen_feedback.py` (dikembalikan ke `Qwen/Qwen3-4B`, bf16).

---

### 9.64 Coba perbaiki EXAONE dulu sebelum revert permanen ke Qwen3 (user menantang §9.63) — fix ditulis, BELUM diuji di GPU

**Konteks**: user menolak revert langsung (§9.63) — minta coba
perbaiki EXAONE-3.5-7.8B dulu, konsisten dengan prinsip proyek ini
sendiri ("re-test sebelum menyimpulkan infeasible", persis pelajaran
yang sudah tertulis di docstring file ini soal Qwen3-1.7B→4B).

**Diagnosis root-cause yang lebih presisi**: path Bahasa Inggris di
`qwen_feedback.py` **tidak pernah punya one-shot example sama sekali**
— cuma instruksi bare "Answer ONLY in English". Path Indonesia SUDAH
punya `_ID_FAST_EXAMPLE`/`_ID_DETAILED_EXAMPLE` sebagai pengaman sejak
dulu (teknik yang sama persis yang memperbaiki Qwen3-1.7B). Qwen3-4B
rupanya tidak pernah butuh pengaman itu untuk Inggris (bahasa
high-resource, representasi kuat di model manapun) — tapi EXAONE yang
Korean-first (per framing paper Section III-C) kemungkinan butuh
pengaman yang sama yang selama ini cuma diberikan ke path Indonesia.

**Fix**: dikembalikan dulu ke EXAONE-3.5-7.8B (revert §9.63 dibatalkan
sementara), ditambahkan `_EN_FAST_EXAMPLE`/`_EN_DETAILED_EXAMPLE` —
teknik identik (2 topik berbeda dalam 1 contoh, menghindari topic-leak)
yang sudah terbukti untuk path Indonesia, sekarang diterapkan juga ke
path Inggris.

**Gap pengujian yang juga ditemukan sambil memperbaiki**: smoke test
`__main__` SELAMA INI cuma cek topic-leak ("학교"/"공부" tidak boleh
bocor), TIDAK PERNAH benar-benar cek apakah output sungguh dalam
bahasa yang diminta — persis bug yang baru dilaporkan user bisa lolos
smoke test tanpa terdeteksi. Ditambahkan `_hangul_char_count()` +
assertion batas karakter Hangul (longgar, karena contoh kalimat Korea
yang disematkan sebagai ilustrasi itu legitimate/diharapkan, bukan
pelanggaran) untuk kedua path (fast/detailed) di kedua bahasa.

**BELUM diverifikasi di GPU** — tidak bisa dites di sandbox ini (tidak
ada GPU). `py_compile` lolos, itu saja yang bisa dikonfirmasi di sini.
Threshold Hangul-count (`_FAST_HANGUL_LIMIT=15`, `_DETAILED_HANGUL_LIMIT=40`)
adalah TEBAKAN AWAL, belum dikalibrasi dari data nyata — mungkin perlu
disesuaikan setelah run pertama.

**Keputusan yang masih terbuka**: kalau fix ini TERBUKTI bekerja di
GPU user (smoke test lolos termasuk assertion Hangul baru) → EXAONE
dipertahankan untuk ketiga file. Kalau TETAP gagal (masih jawab
Korea) → baru revert permanen ke Qwen3 (seperti §9.63), kali ini
dengan bukti sudah benar-benar dicoba, bukan diasumsikan infeasible.

**File:** `src/qwen_feedback.py` (EXAONE-3.5-7.8B + one-shot example
Inggris + assertion kepatuhan-bahasa baru).

---

### 9.65 Fix §9.64 TERBUKTI berhasil di GPU nyata — EXAONE dipertahankan untuk qwen_feedback.py, threshold Hangul dikalibrasi ulang dari data nyata

**Hasil run user**: `fast [en]` dan `detailed [en]` KEDUANYA sepenuhnya
dalam Bahasa Inggris, benar, dan berkualitas tinggi — one-shot example
Inggris (§9.64) berhasil memperbaiki kepatuhan bahasa EXAONE-3.5-7.8B,
PERSIS seperti yang sudah terbukti untuk Qwen3 dulu. Tidak ada
kebocoran topik, partikel dikutip dengan tepat.

**Tapi smoke test tetap gagal** — bukan karena bahasanya salah,
melainkan threshold `_DETAILED_HANGUL_LIMIT=40` yang saya tulis
sendiri sebagai "tebakan awal" (§9.64) ternyata terlalu ketat: jawaban
`detailed [en]` yang BENAR dan berkualitas tinggi itu punya 89 karakter
Hangul (4 kalimat contoh Korea yang memang diminta prompt — "one more
example sentence contrasting both uses" — plus kutipan partikel
berulang untuk kontras). False positive dari threshold yang belum
dikalibrasi, bukan bug di fix bahasanya.

**Diperbaiki**: `_DETAILED_HANGUL_LIMIT` dinaikkan dari 40 → 150,
berdasarkan margin nyata di atas 89 yang terobservasi, bukan ditebak
ulang.

**Keputusan yang sebelumnya terbuka (§9.64) sekarang terjawab**: EXAONE
TERBUKTI bisa diperbaiki, bukan infeasible — dipertahankan untuk
`qwen_feedback.py`, TIDAK direvert ke Qwen3. Prinsip proyek ini
("re-test sebelum menyimpulkan infeasible") terbukti benar lagi.

**Belum diverifikasi**: path `[id]` (Bahasa Indonesia) — run user
berhenti di assertion gagal sebelum sempat mencapai iterasi "id" di
loop. Perlu run ulang untuk konfirmasi penuh kedua bahasa.

**File:** `src/qwen_feedback.py` (threshold Hangul dikalibrasi ulang).

---

### 9.66 Siapkan 3 pengujian tertunda sekaligus: unblock smoke test exaone_provider.py, buat script head-to-head qwen_feedback.py vs Qwen3

**Konteks**: user minta ketiga item tertunda diuji sekaligus (Step 2/2
untuk `exaone_provider.py`/`solver_check.py`, perbandingan head-to-head
`qwen_feedback.py` vs Qwen3). Tidak bisa dijalankan di sandbox ini
(tidak ada GPU) — jadi fokus sesi ini: siapkan SEMUA tool-nya supaya
user tinggal jalankan berurutan.

**1. Unblock smoke test `exaone_provider.py`**: assertion keras untuk
'교도소' (§9.66 sebelumnya ditemukan, belum diputuskan: "수감하다"
kemungkinan salah transitivitas, bukan fix sungguhan) diubah dari
`assert` keras jadi laporan lunak (print peringatan, lanjut) — supaya
Step 1/2 tidak lagi macet di pertanyaan yang belum terjawab, bisa
lanjut ke Step 2/2. Keputusan transitivitas '교도소' TETAP terbuka,
cuma tidak lagi memblokir pengujian lain.

**2. `src/compare_feedback_models.py`** (baru): head-to-head
EXAONE-3.5-7.8B (config saat ini) vs Qwen3-4B (config asli yang sudah
tervalidasi) pada 4 item test (lebih dari 1 kasus yang dipakai smoke
test sendiri), kedua bahasa (en/id), kedua jalur (fast/detailed) —
total 16 perbandingan. **Sengaja TIDAK reuse `QwenFeedbackProvider.__init__`
untuk sisi Qwen3** — constructor itu sekarang hardcode
`trust_remote_code=True` + quantization 4-bit (ditambahkan untuk
EXAONE), yang akan mengubah konfigurasi Qwen3-4B dari yang sudah
tervalidasi (bf16 polos, ~8GB). Tiap model di-load dengan konfigurasi
MASING-MASING yang benar (`object.__new__` + set `.tok`/`.model` manual),
tapi tetap reuse method `fast_feedback`/`detailed_feedback`/`_generate`
yang sama persis — supaya teks prompt yang dibandingkan identik, tidak
ada logic yang didup likasi dan bisa out-of-sync.

**Bukan verdict otomatis** — kualitas teks feedback itu prosa bebas,
proyek ini tidak punya automated grader untuk itu (beda dari
solver-check/log-likelihood yang bisa diskor). Script ini menghasilkan
"reading pack" — user baca berdampingan, nilai: kebenaran faktual,
kejelasan, apakah ada yang salah/menyesatkan (kegagalan paling
penting untuk dicek).

**Diverifikasi**: `py_compile` lolos untuk kedua file yang diubah/dibuat.
Tidak bisa dites end-to-end (tidak ada GPU di sandbox ini).

**Urutan yang disarankan untuk user jalankan**:
1. `python3 src/verify_exaone35_swap.py` — sekarang seharusnya lolos
   Step 1/2 (termasuk peringatan lunak '교도소', bukan crash), lanjut
   ke Step 2/2 (4 skrip validasi + diff vs baseline paper).
2. `python3 src/compare_feedback_models.py` — baca hasil perbandingan
   16 kasus, laporkan pola yang terlihat (atau campur aduk).

**File:** `src/exaone_provider.py` (known_failures jadi soft-report),
`src/compare_feedback_models.py` (baru).

---

### 9.67 Hasil Step 2/2 + perbandingan feedback: `solver_check.py` DIREVERT (regresi nyata), `qwen_feedback.py` DIPERTAHANKAN (scope Inggris saja), `exaone_provider.py` BELUM diputuskan (butuh isolasi ulang)

**Hasil Step 2/2 nyata** (user jalankan di GPU):
```
overall_solver_key_accept_rate: 0.6636 -> 0.5500  (turun 11.4 poin, REGRESI)
log_likelihood overall_pass_rate: 0.9808 -> 0.8462  (turun 13.5 poin, REGRESI)
relative_discrimination_drop: tidak berubah (metrik ini memang tidak bergantung model)
lm_plausibility overall_pass_rate: tidak berubah (pakai kiwipiepy, bukan LLM)
```
2 metrik yang DIKUTIP LANGSUNG di paper (87.5%/94.2%) turun signifikan
dengan EXAONE-3.5-7.8B sebagai solver. Ini PERSIS skenario yang sudah
ditulis sebagai kondisi revert di komentar kode `solver_check.py`
sendiri sejak swap awal (§9.51): *"If EXAONE-3.5-7.8B regresses either
check(c) key-acceptance or the log-likelihood gap, revert _MODEL_NAME
to Qwen/Qwen3-4B"* — kondisi itu sekarang terpenuhi, bukan hipotetis
lagi.

**Hasil `compare_feedback_models.py`** (16 perbandingan: 4 item x 2
bahasa x 2 jalur): Bahasa Inggris — EXAONE dan Qwen3 sama bagus di
8/8 kasus, tidak ada kesalahan fakta. **Bahasa Indonesia — EXAONE
bermasalah serius**: `fast_feedback[id]` terpotong di tengah kalimat
di SEMUA 4 kasus, mengandung kata tidak nyata ("nenyawa", "dalyang",
"saatane"); `detailed_feedback[id]` kasus 읽은/읽는 EXAONE menulis
**"Jawaban Benar: 읽는"** — TERBALIK dari jawaban sebenarnya
(`correct='읽은'`), kesalahan fakta yang bisa menyesatkan pembelajar.
Qwen3 bersih di semua 16 kasus termasuk Indonesia.

**Keputusan user**: target produk saat ini CUMA Bahasa Inggris —
jalur Indonesia TETAP ADA di kode (tidak dihapus) tapi TIDAK AKTIF
dipakai pengguna sekarang. Jadi:
- **`solver_check.py` → DIREVERT ke `Qwen/Qwen3-4B`** (bf16, tanpa
  quantization). Regresinya tidak terkait bahasa sama sekali (soal
  akurasi menilai partikel Korea), jadi tidak terpengaruh keputusan
  scope-Inggris — tetap direvert.
- **`qwen_feedback.py` TETAP EXAONE-3.5-7.8B** — Inggris (satu-satunya
  yang dipakai aktif) terbukti bagus di semua 8 kasus. Masalah kualitas
  Indonesia (terpotong, kata ngaco, 1 kasus jawaban terbalik) DICATAT
  sebagai known issue, bukan diperbaiki sekarang — harus diperbaiki
  SEBELUM jalur Indonesia diaktifkan ke pengguna nyata, jangan
  diasumsikan aman.
- **`exaone_provider.py` BELUM diputuskan** — `overall_generation_pass_rate`-nya
  sendiri tidak berubah (100%→100%), tapi `overall_solver_key_accept_rate`
  (metrik gabungan generator+solver) ikut terpengaruh regresi
  `solver_check.py` yang lama (saat itu KEDUANYA masih EXAONE) — efek
  generator SENDIRI belum bisa dipisahkan dari run sebelumnya. User
  minta diuji ULANG sekarang dengan `solver_check.py` sudah balik ke
  Qwen3, supaya `overall_solver_key_accept_rate` yang baru murni
  mengukur kualitas generator EXAONE dinilai oleh solver yang SUDAH
  terbukti baik (Qwen3) — isolasi yang lebih bersih.

**Diverifikasi**: diff eksplisit vs versi pre-swap (`ab81d8a`) dicek
dulu (cuma konstanta model + kode loading yang beda, konsisten dengan
revert `qwen_feedback.py` sebelumnya). `py_compile` lolos.

**Langkah berikut untuk user**: `git pull` lalu
`python3 src/verify_exaone35_swap.py` lagi — sekarang `solver_check.py`
pakai Qwen3, `exaone_provider.py` masih EXAONE. Kalau
`overall_solver_key_accept_rate` SEKARANG mendekati/menyamai baseline
asli (0.6636), itu konfirmasi generator EXAONE sendiri baik-baik saja,
regresi sebelumnya murni dari solver. Kalau MASIH jauh di bawah,
berarti generator EXAONE juga berkontribusi ke masalah, perlu
dipertimbangkan revert juga.

**File:** `src/solver_check.py` (dikembalikan ke `Qwen/Qwen3-4B`, bf16).

---

### 9.68 Review eksternal (2026-10-21): perbaikan cepat Q1–Q6 + harness evaluasi, dan satu temuan besar yang BELUM diperbaiki

**Konteks:** review kode vs PRD dan daftar keterbatasan (`/mnt/project-files/reviews/` di project ini). User setuju mulai dari perbaikan cepat dan harness evaluasi.

**Diperbaiki:**
- **Q1 salah tanda target 85%:** `validate_item_discrimination.py` memakai `beta = -logit(0.15)` = +1.73 (item 15%-sukses, `avg_success_rate` 19%). Sekarang `beta = -logit(0.85)` = −1.73, dengan assert bahwa tingkat sukses item ≈ target. Hasil ulang: diskriminasi 0.343 (sebelumnya 0.335), drop relatif **16%** (sebelumnya 18%). Rumus di paper diperbaiki ("β=θ̂−logit(0.85)"), docx/pdf di-build ulang.
- **Q2:** `collocation_plausibility.py` sekarang **fail-closed** secara default. Fail-open hanya kalau `KO_DRILL_ALLOW_PLAUSIBILITY_BYPASS=1`.
- **Q3:** parser huruf dan ya/tidak di `solver_check.py` dipindah ke `llm_parsing.py` (bisa dites tanpa GPU). Sekarang hanya token pertama yang dibaca, jadi "예시" tidak lagi dianggap "ya". Ditambah `pick_by_logprob()` (skoring cloze, tanpa huruf) dan `pick_best_rotated()` (4 rotasi urutan opsi) untuk menghilangkan bias posisi.
- **Q4:** `QuestionBank.submit_answer()` menulis balik `updated_item_urn` ke bank (sebelumnya tidak ada pemanggil yang melakukannya).
- **Q5:** README disinkronkan dengan model yang benar-benar dipakai.
- **Q6:** urutan kandidat filler diacak per item (sebelumnya selalu "학교"/"사과" yang sama).
- **Harness:** `eval_harness.py` + `docs/eval-protocol.md` (draft). Hasil dari file yang sudah ada: solver key acceptance **38/55 item unik = 69.1% (CI 95% 56–80%)**; log-likelihood **51/52 = 98.1% (CI 90–99.7%)**; kiwipiepy 26/26 (CI 87–100%). Akurasi solver bergantung posisi kunci: 75% saat kunci di A vs 53% di B.

**Temuan besar, BELUM diperbaiki (butuh keputusan user):** konvensi tanda kesulitan item Urnings **terbalik**. Di kernel `update()` (dan `run_urnings`), urn item berperilaku sebagai *kesulitan*: logit(k_i) berkorelasi **+0.94** dengan β sebenarnya (dicek langsung, 60.000 respons, seleksi acak). Tapi seleksi adaptif (`simulate_trackers.run_urnings`, `simulate_item_prior.py`, `simulate_item_coldstart*.py`, `validate_novelty2.py`, `BankContentProvider`) memakai `beta_hat = -logit(k_i)`, dan seeding prior (`seed_item_from_encoder`, `SIG(-mu_hat)`) juga terbalik. Akibatnya simulasi Urnings tidak benar-benar adaptif. Dengan tanda yang benar (seleksi sungguh adaptif, |P−0.5| rata-rata turun dari 0.29 ke 0.10), Urnings di setup Section III-B yang sama memberi **SD 2.87, RMSE 1.95**, lebih buruk dari Elo (2.69/1.70), bukan 1.11/0.43. Ini menyentuh klaim inti paper (Urnings stabil di bawah seleksi adaptif) dan Novelty 2 (robust terhadap prior: prior yang diinjeksikan ke Urnings selama ini selalu terbalik). Perlu dicek juga apakah kernel butuh koreksi MH untuk seleksi adaptif seperti yang dibahas di literatur Urnings.

### 9.69 Urnings diperbaiki (2026-10-21): tanda urn item + koreksi seleksi adaptif, di simulasi DAN di kode produksi

**Keputusan user:** "Perbaiki + tambah koreksi adaptif" (kartu keputusan, menjawab temuan besar §9.68).

**Diperbaiki:**
- **Tanda:** urn item adalah *kesulitan* (logit(k_i) ≈ +β). Semua seleksi dan seeding sekarang memakai `+logit(k_i)` dan `SIG(+mu_hat)` (`simulate_*.py`, `validate_novelty2.py`, `urnings_tracker.seed_item_from_encoder`, `question_bank`). Assert arah di smoke test `answer_processing.py` ikut dibalik: menjawab benar item yang lebih sulit sekarang menaikkan urn pelajar lebih banyak, sesuai model.
- **Koreksi seleksi adaptif:** item dipilih dengan kernel Gaussian pada jarak logit (σ=0.5, Gergely et al. 2025 Eq. 17), dan rasio penerimaan MH dikali p(item | state usulan) / p(item | state sekarang) (Bolsinova et al. 2022; Gergely et al. 2025 Eq. 4). Di simulasi: `simulate_trackers.urnings_step`. **Di produksi** (baru): `urnings_tracker.selection_probs` + argumen `selection_prob` di `update()`; `BankContentProvider` sekarang *sampling* dari kernel dengan target 85% (bukan "salah satu dari 3 terdekat"), dan `QuestionBank.submit_answer` meneruskan probabilitas seleksi ke update. Item hasil generate langsung (tidak dipilih dari urn) tidak perlu koreksi.

**Hasil ulang (Section III-B, 120.000 respons, 300 pelajar, 400 item):** Elo SD 2.32 / RMSE 1.35 (masih naik); Parallel Elo 1.55 / 0.59; **Urnings 1.00 / 0.41** (SD asli 1.04). Tanpa koreksi MH, Urnings 2.63 / 1.73, lebih buruk dari Elo. Sapuan ukuran urn (16/32/64/128): RMSE 0.59 / 0.35 / 0.32 / 0.36; urn 128 belum menyebar (SD 0.72) dalam ~400 respons per pelajar. Paper diperbarui (abstrak, III-B, Fig. 4, Diskusi).

**Catatan jujur:** di simulasi statis, Glicko masih jauh lebih akurat (RMSE 0.11). Di simulasi drift (`simulate_drift.py`), RMSE@100%: Urnings 0.903, Parallel Elo 0.904, Glicko 1.027, Elo 1.227, jadi alasan menolak Glicko (§2.1) masih berlaku tapi selisihnya lebih kecil dari sebelumnya (0.86 vs 1.05).

### 9.70 Novelty 2 dihapus; Novelty 1 (A1) dibangun ulang dengan metode generatif + model pilihan miskonsepsi

**Novelty 2 dihapus (keputusan user, 2026-10-21):** setelah §9.69, klaim "Urnings kebal terhadap prior buruk" tidak berlaku lagi. Prior lemah tetap merugikan Urnings (RMSE 0.42 → 0.68; Elo 1.34 → 1.70; Parallel Elo 0.60 → 0.90, `simulate_item_prior.py`, `simulate_elo_item_prior.py`). Ketahanan lama ternyata artefak bug. Dibandingkan SOTA, isinya juga bukan hal baru (Urnings + koreksi seleksi sudah ada di Bolsinova 2022 dan Gergely 2025). User: "noveltynya dibandingkan dengan existing research / sota … apakah di hapus saja?" Dihapus dari paper sebagai novelty (Section IV-C dan V-C lama hilang). Hasil prior dipertahankan satu paragraf di III-B sebagai catatan desain. Ditambah kondisi baru: ukuran urn mengikuti σ² encoder yang benar (σ²=noise²) mengurangi kerugian prior lemah dari 0.68 ke 0.59, jadi encoder harus melaporkan varians yang terkalibrasi.

**A1, metode baru (user: "saya mau powerful metodenya dan hasilnya bagus"; tanpa konten hardcoded):**
- **`misconception_tracker.py`** (baru): model pilihan multinomial-logit (Luce) di antara distraktor yang ditawarkan (nested logit, Suh & Bolt 2010), di-update online dengan satu langkah SGD per jawaban salah (λ=0.4), mirip hubungan Elo dengan Bradley-Terry. Disimpan per pelajar di `UserDrillState.misconceptions`.
- **Simulasi** (`simulate_misconceptions.py`, `misconception_results.json`; 6 miskonsepsi, 3 ditawarkan per item, ketersediaan Zipf, 300 pelajar × 20 replikasi). Akurasi top-1 setelah 10 / 40 jawaban salah:
  - stabil: online 58.1% / 73.1%; MLE semua riwayat 59.4% / 78.0%; rate 60.0% / 77.3%; count 45.9% / 51.4%; peluang 16.7%;
  - setelah remediasi: online **53.2% / 73.7%**; MLE 40.1% / 59.3%; rate 40.2% / 59.7%; count 34.6% / 43.3%.
  - Online kalah ~5 poin kalau miskonsepsi tidak pernah berubah, tapi menang ~14 poin setelah remediasi. Varian urn per miskonsepsi sempat dicoba, lalu dibuang (lebih buruk saat stabil).
- **`kagas_annotator.py`** (baru, menggantikan validator aturan kiwipiepy): menyelaraskan morfem kalimat benar vs salah (alomorf 을/를 dkk. digabung), lalu mengklasifikasi edit per kategori KAGAS. Edit kelas tertutup (partikel/akhiran/counter) menghasilkan kode stabil, mis. `PART:에서>에`. Keterbatasan: kiwipiepy kadang memotong distraktor yang ambigu secara berbeda (mis. "공원이 책을" → 이/MM), jadi distraktor seperti itu ditolak (aman, tapi bisa menurunkan tingkat lolos).
- **`item_foundry.py`** (baru, menggantikan `exaone_provider.py`):
  - codebook miskonsepsi per skill ditulis EXAONE dan diverifikasi annotator, lalu dibekukan di `data/misconception_codebook.json`;
  - item ditulis EXAONE (kalimat, span kunci, 3 distraktor, masing-masing satu miskonsepsi berbeda; satu harus miskonsepsi dominan pelajar kalau sudah diketahui);
  - verifikasi tanpa LLM-judge: satu edit di kategori skill, kode cocok dengan codebook, tiga kode berbeda, dan kunci harus punya log-prob Qwen3-4B lebih tinggi dari tiap distraktor (dilewati untuk subj_topic);
  - critic: alasan gagal dikirim balik ke EXAONE, maksimal 2 revisi.
- **Personalisasi:** `daily_drill` meneruskan `target_misconception` = miskonsepsi dominan ke provider. Generator mewajibkan satu distraktor merealisasikannya; `BankContentProvider` memilih item bank yang menawarkan miskonsepsi itu. Trigger eskalasi `consecutive(m_j)` sekarang memakai kode distraktor yang **dipilih**, bukan kode level item.
- **Dihapus:** `exaone_provider.py` (slot, `verb_lookup`, pasangan tetap, filler, `primary_distractor_label`), `collocation_plausibility.py` (SBERT), `distractor_validator.py` (fungsi `correct_korean_conjugation` dipindah ke `qwen_feedback.py`), skor parse kiwipiepy (`validate_lm_plausibility.py` + hasilnya), `validate_novelty1.py`, `validate_log_likelihood.py`, `verify_exaone35_swap.py`. `1_validation_results.json` dan `log_likelihood_results.json` dipertahankan sebagai catatan item kurasi lama dan kalibrasi gerbang log-prob (51/52).
- **Validasi GPU, BELUM dijalankan** (sandbox tanpa GPU): `python3 src/validate_item_foundry.py --per-skill 20` membangun codebook, meminta 20 item per skill (setengah ditargetkan), lalu melaporkan tingkat lolos (Wilson CI), alasan penolakan, cakupan kode, dan solver rotasi sebagai pembanding. Semua item disimpan ke `item_foundry_results.json` untuk review instruktur. Logika pipeline sudah dites di CPU dengan model pengganti (`python3 src/item_foundry.py`).
- **Paper:** Section IV-B dan V-B ditulis ulang, Eq. 2–3 baru (model pilihan), seleksi jadi kernel (Eq. 4), eskalasi Eq. 5, Fig. 1–2 diperbarui. Referensi ditambah: KAGAS (Yoon et al. 2023; sebelumnya KAGAS salah dikutip sebagai [22], KoLLA) dan Suh & Bolt (2010).
- **Inkonsistensi lama:** paper menyebut Qwen3 sebagai explainer, padahal `qwen_feedback.py` memakai EXAONE-3.5-7.8B. Diselesaikan di §9.71.

### 9.71 Run codebook pertama (GPU), perbaikan parser + annotator, benchmark checker, explainer kembali ke Qwen3-4B

**Hasil run `--codebook-only` (EXAONE-3.5-7.8B, 3 balasan × 11 skill, 6 entri per balasan):** parsing JSON sudah berhasil (32/33 balasan), tapi hanya **3 entri** lolos (honorific 1, noun_counter 2). Penyebab utama, dibaca dari `codebook_log`: EXAONE hampir selalu menambahkan romanisasi dan terjemahan Inggris di belakang kalimat Korea ("공원에서 산책했어요. (Gongwon-eseo ...) - I walked ..."). Annotator menghitung bagian Latin itu sebagai edit tambahan, jadi pasangan yang sebenarnya minimal ditolak sebagai "not-minimal". Satu balasan gagal parse karena komentar `//` di dalam JSON.

**Diperbaiki:**
- `item_foundry.korean_only()`: buang segmen romanisasi/terjemahan dalam kurung dan setelah " - "; entri yang masih berisi huruf Latin di dalam kalimatnya ditolak dengan alasan `format`, tidak diam-diam diubah. Prompt codebook sekarang meminta Hangul saja.
- `_loads_lenient`: komentar `//` setelah nilai JSON dibuang.
- `kagas_annotator` langkah 4: bentuk predikat yang rusak dan tidak bisa dianalisis kiwi (먹았어요 → `먹았/NNG`) dulu jadi `MULTI`. Sekarang, kalau hanya satu kata predikat yang berbeda, edit dibaca dari permukaan kata itu (먹[었>았]어요 → `END:었>았`), atau `CONJ` kalau stem benarnya irregular (tag `-I`, mis. 아름다워요). Ini kesalahan harmoni vokal 았/었 yang justru miskonsepsi nyata untuk `tense_past`.
- `validate_item_foundry.py --from-log FILE`: verifikasi ulang balasan yang sudah tercatat tanpa memanggil EXAONE lagi (tanpa GPU bersama `--codebook-only`).

**Hasil verifikasi ulang atas balasan yang sama** (`codebook_reverified_results.json`): **3 → 32 entri**, 7 dari 11 skill punya ≥3 entri (cukup untuk 3 distraktor): particle_loc 3, honorific 3, tense_past 5, irregular_conj 4, adj_relativizer 3, subj_topic 3, noun_counter 7. Masih kurang: particle_obj 1, time_range 1, mod_adverb 2, verb_choice 0.

**Temuan jujur soal kualitas:** verifier memeriksa BENTUK (satu edit, kategori benar), bukan apakah bentuk salahnya miskonsepsi pelajar yang masuk akal. Sebagian entri yang lolos jelas bagus (에서>에, 었>았, 권>개, 대>개, 한> dihilangkan), sebagian sampah (말씀하시공셨습니다, 읽어요어요, 아름답께, 이것은 책을이에요), dan satu contoh "benar" sendiri salah (스무 년). Jadi rating manusia pada codebook (2 rater, kappa) wajib, dan generator EXAONE-7.8B kemungkinan terlalu lemah untuk menulis codebook. Opsi penguatan diajukan ke user (lihat thread).

**Benchmark checker (`src/benchmark_checker.py`, baru, butuh GPU):** kandidat Qwen3-4B (sekarang), Qwen3-4B-Base, Qwen3-8B / 8B-Base (4-bit), Kanana-1.5-8B-base (4-bit), Gemma-3-12B-pt (4-bit, gated). Aturan sama dengan gate: log-prob kalimat benar > kalimat salah. Tiga set: 52 pasangan kurasi, kalimat pelajar KoLLA dengan tepat satu error (anotator 0) vs koreksinya, dan contoh codebook. Melaporkan akurasi + Wilson CI, margin rata-rata, VRAM puncak, detik per pasangan. `SolverCheckProvider` sekarang bisa dimuat 4-bit. EXAONE sengaja tidak jadi kandidat (generator tidak boleh menilai outputnya sendiri).

**Explainer kembali ke Qwen3-4B (bf16)** atas persetujuan user: `detailed_feedback` butuh reasoning mode yang tidak ada di EXAONE-3.5; §9.67 menemukan EXAONE menulis penjelasan Indonesia yang terpotong, berisi kata tidak nyata, dan sekali membalik jawaban benar, sedangkan Qwen3 bersih di 16/16. Satu model Qwen3-4B dipakai untuk checker dan explainer. Inkonsistensi paper vs kode di §9.70 selesai.

**File:** `src/item_foundry.py`, `src/kagas_annotator.py`, `src/validate_item_foundry.py`, `src/solver_check.py`, `src/qwen_feedback.py`, `src/benchmark_checker.py` (baru), `data/misconception_codebook.json`, `codebook_reverified_results.json` (baru), README.

### 9.72 Hasil benchmark checker: Qwen3-4B tetap dipakai

**Run pertama** (400 pasangan KoLLA): semua model ~57%. Ini bug di `benchmark_checker.py`, bukan kelemahan model: set itu memuat error "kata hilang/berlebih" (M:/U:), di mana kalimat koreksi lebih panjang, dan total log-prob selalu memenangkan kalimat yang lebih pendek. Diperbaiki: hanya penggantian satu kata (R:, 1 token), dikelompokkan partikel / ejaan / lainnya.

**Run ulang** (52 kurasi, 246 KoLLA, 32 codebook; aturan gate: log-prob kalimat benar > salah):

| Model | Kurasi | KoLLA (CI 95%) | Codebook | VRAM |
|---|---|---|---|---|
| Qwen3-4B (bf16) | 51/52 = 98,1% | 215/246 = 87,4% (82,7–91,0) | 29/32 | 8,2 GB |
| Qwen3-4B-Base | 96,2% | 217 = 88,2% (83,6–91,7) | 30/32 | 8,2 GB |
| Qwen3-8B (4-bit) | 98,1% | 209 = 85,0% (80,0–88,9) | 30/32 | 6,5 GB |
| Qwen3-8B-Base (4-bit) | 94,2% | 217 = 88,2% (83,6–91,7) | 30/32 | 6,5 GB |
| Kanana-1.5-8B-base (4-bit) | 94,2% | 211 = 85,8% (80,9–89,6) | 30/32 | 6,2 GB |
| Gemma-3-12B-pt (4-bit) | 50/52 = 96,2% | 218 = 88,6% (84,0–92,0) | 30/32 | 8,5 GB |

Per jenis error KoLLA (partikel n=114 / ejaan n=131): Qwen3-4B 96 = 84,2% (CI 76,4–89,8) / 118 = 90,1%; Qwen3-4B-Base 96 / 120; Qwen3-8B 92 / 116; Qwen3-8B-Base 96 / 120; Kanana 90 / 120; Gemma 99 = 86,8% (79,4–91,9) / 118.

**Kegagalan partikel Qwen3-4B, dibaca satu per satu (17 dari 18 tercatat):** 10 adalah partikel yang *dihilangkan* pelajar (선생님 좋아요 → 선생님이 좋아요, 아버지 거실에 계세요 → 아버지는 ...). Dalam bahasa Korea lisan, penghilangan partikel sering wajar, jadi model tidak sepenuhnya salah, dan sebagian anotasi KoLLA bersifat preskriptif. Hanya 6 yang merupakan substitusi partikel (친구들이 → 친구들을, 공원은 → 공원을, 부모님이 → 부모님께서, 에 → 의), dan 1 partikel berlebih. Jenis error yang dihasilkan item foundry adalah substitusi, yang paling jarang gagal. Artinya, ~84% adalah batas bawah untuk kasus drill.

**Keputusan:** semua CI tumpang tindih, jadi tidak ada model yang terbukti lebih baik. Qwen3-4B dipertahankan karena tertinggi di kurasi, dan bobot yang sama dipakai untuk explainer (dua model di GPU, bukan tiga). **Temuan untuk paper:** gate kuat pada distraktor buatan (98%) tapi melewatkan ~1 dari 8 error pelajar nyata (87%), jadi rating instruktur tetap diperlukan. Paper Section IV-B diperbarui dengan angka ini.

**File:** `src/benchmark_checker.py`, `gemma_benchmark_results.json`, `src/content.py`.

### 9.73 Disederhanakan: KoLLA dihapus, tanpa sampling, codebook disetujui instruktur

**Instruksi user:** "saya maunya simple, namun hasilnya bagus"; KoLLA dipakai untuk semua skill atau dihapus.

**Keputusan:**
- **KoLLA dihapus.** Skema anotasinya hanya mencakup partikel dan ejaan (tidak ada error akhiran verba dalam 400 sampel benchmark), jadi paling banyak menutup 4 dari 11 skill. Dihapus: `investigate_kolla_corpus.py`, `investigate_kolla_spell.py`, `benchmark_checker.py`, dan kedua file hasil benchmark (file itu juga memuat kalimat korpus GPLv3). Hasil benchmark §9.72 tetap tercatat di sini sebagai dasar keputusan checker, tapi tidak dikutip di paper. Referensi Song et al. di tinjauan pustaka tetap ada.
- **Self-consistency sampling dan benchmark generator dibatalkan** sebelum di-commit, karena menambah waktu run (±10 balasan × 11 skill × beberapa model) dan kompleksitas. Catatan untuk paper: replay atas 3 balasan yang tercatat menunjukkan filter itu memang membuang entri sampah. Jadi ini bisa disebut sebagai future work.
- **Alur codebook yang dipakai (sederhana):** EXAONE mengusulkan, verifier (`korean_only` + annotator KAGAS) menyaring bentuknya, lalu **instruktur menyetujui atau menghapus** entri di `data/misconception_codebook.json` satu kali sebelum item dibuat. Codebook kecil (≤6 entri × 11 skill) dan dibekukan, jadi review ini cepat. Review ini juga validasi yang memang dibutuhkan paper (rating kode miskonsepsi, V-B). `SolverCheckProvider` kembali tanpa opsi 4-bit.

**Langkah berikut (GPU):** `python3 src/validate_item_foundry.py --codebook-only` dengan prompt baru (Hangul saja), lalu review instruktur atas codebook, lalu `--per-skill 20 --reuse-codebook`.

### 9.74 Codebook dihapus: label miskonsepsi = kode edit annotator KAGAS

**Keputusan user** ("ok", atas rekomendasi): codebook tidak wajib. Yang dibutuhkan novelty hanyalah label yang konsisten antar soal, dan annotator sudah menghasilkannya secara otomatis.

**Sekarang:**
- EXAONE menulis kalimat, span kunci, dan 3 distraktor ("tiga kesalahan khas yang berbeda"). Formatnya `"distractors": ["...", "...", "..."]`, tanpa nomor miskonsepsi.
- Label setiap distraktor = `kagas_annotator.annotate(kalimat kunci, kalimat distraktor)`, misalnya `PART:에서>에` atau `END:었>았`. Kesalahan yang sama selalu dapat kode yang sama di soal mana pun (alomorf 을/를 digabung). Untuk open-class (verb_choice), labelnya pasangan leksikal (`VERB:마시>먹`), yang lebih jarang berulang. Ini keterbatasan yang ditulis jujur.
- Verifikasi: satu edit di kategori skill, tiga kode berbeda, kode target ada (kalau diminta), dan log-prob Qwen3-4B. Personalisasi: kode dominan diterjemahkan ke kalimat untuk prompt (`describe_code`: "'에' written where '에서' is needed").
- `validate_item_foundry.py`: tanpa tahap codebook. Permintaan bertarget memakai kode yang sudah muncul di item skill itu. Laporan menambah jumlah kode berbeda dan berapa yang berulang (≥2 item).
- Dihapus: `build_codebook`, `MisconceptionEntry`, `load/save_codebook`, `_slug`, `data/misconception_codebook.json`, `item_foundry_results.json` dan `codebook_reverified_results.json` lama.
- Paper: IV-B "Naming the misconceptions" ditulis ulang (label dihitung, bukan ditulis). Verifier sekarang tiga cek, V-B (protokol dan rating instruktur) disesuaikan, Fig. 2 diperbarui. Rating instruktur sekarang menilai apakah kode tiap distraktor adalah kesalahan pelajar yang nyata.

**Langkah berikut (GPU):** `python3 src/validate_item_foundry.py --per-skill 20`.

### 9.75 Run item foundry pertama (tanpa codebook): OOM dan acceptance rendah, diperbaiki

**Hasil sebelum OOM** (5 dari 11 skill, 20 permintaan per skill): particle_loc 8/20, particle_obj 0/20, honorific 3/20, time_range 0/20, tense_past 1/20. Alasan penolakan terbanyak: `wrong-category`, `not-minimal`, dan `target`. Untuk particle_obj dan time_range, ±1 dari 3 balasan gagal di `target`. Partikel pendek seperti 을/를/까지 sering muncul lebih dari sekali di kalimat, sehingga syarat "target muncul tepat sekali" gagal walaupun kalimatnya bagus. Kolom solver (pilih huruf) 0/12, padahal kolom itu tidak dipakai sebagai gate.

**Penyebab OOM:** riwayat percakapan critic terus bertambah (prompt + semua balasan + semua alasan), dengan `max_new_tokens=1500`, di GPU yang sama dengan Qwen3-4B bf16 (15,25 GB terpakai).

**Diperbaiki:**
- EXAONE sekarang menandai bagian yang diuji dengan kurung siku di dalam kalimat (`동생이 공원[에서] 축구를 해요.`). Posisinya jadi pasti, walaupun teks yang sama muncul di tempat lain. Field `target` dihapus.
- Critic hanya mengirim balik balasan terakhir; `max_new_tokens` 400 (balasan JSON satu soal cukup pendek); `torch.cuda.empty_cache()` setelah setiap generate.
- Pengecekan solver dengan pilih huruf dihapus dari validasi (bukan gate, judge yang lemah, dan hasilnya 0/12). Paper V-B disesuaikan.
- `FoundryStats.rejected` menyimpan 10 balasan pertama yang ditolak beserta alasannya. File hasil ditulis ulang setelah setiap skill. Ada opsi `--skills` untuk cek cepat.

### 9.76 Cek cepat (--per-skill 4): 0/4 di particle_loc, penyebabnya bentuk balasan EXAONE

Dari 10 balasan yang ditolak dan tersimpan, EXAONE menulis distraktor dalam tiga bentuk: (a) teks pengganti saja (yang diminta), (b) kata utuh yang memuat bagian itu (`카페에서` untuk `카페[에]`), dan (c) kalimat utuh, kadang masih dengan kurung siku. Kode lama selalu menempelkan distraktor di posisi kurung, jadi bentuk (b) dan (c) menghasilkan kalimat rusak (`친구가 카페카페에서 ...`) dan ditolak sebagai not-minimal atau wrong-category. Selain itu EXAONE kadang memberi 4 distraktor.

**Diperbaiki:** `_wrong_sentence` menerima ketiga bentuk itu dan mengubahnya menjadi kalimat salah utuh. Annotator tetap yang memutuskan apakah kalimat itu satu edit dari kunci. `_shared_blank` memakai span dalam kurung kalau semua distraktor hanya berbeda di situ, dan melebar ke satu kata utuh kalau perlu. Kalau ada lebih dari 3 distraktor yang valid, yang diambil 3 (distraktor target diutamakan). Prompt menegaskan "only the replacement text, no comments".

**Replay offline** atas balasan yang tercatat: bentuk (b) sekarang diterima (`카페[에]` → 에서/로/와). Sisanya ditolak karena memang buruk: mengganti kata benda (집 untuk 학교), atau menghapus subjek. Beberapa kunci yang ditulis EXAONE juga salah (카페[에] 책을 읽고 있어요; seharusnya 에서). Kunci seperti itu ditangkap oleh gate log-prob Qwen3-4B, karena kalimat dengan 에서 skornya lebih tinggi.

### 9.77 Cek cepat kedua: 2/44, generator tetap EXAONE, prompt ditulis ulang

Run ini masih memakai kode sebelum §9.76. Selain itu EXAONE sering keluar dari aturan: kurung siku di akhir atau di luar kalimat, lebih dari satu kurung, bagian yang salah dikurung, kunci salah (`사과[을]`), komentar bahasa Inggris, dan skill lain ikut menguji partikel. Dua item yang diterima tidak layak (opsi berupa frasa, misalnya `크게읽어요`). jc memutuskan generator tetap EXAONE-3.5-7.8B.

**Diperbaiki:**
- Verifier: kurung siku sesudah akhir kalimat ditolak, kunci dan distraktor harus satu kata atau kurang, markdown `**` dibuang.
- Prompt ditulis ulang. Aturan tugas (7 aturan bernomor dan format output) ada di system prompt; pesan user hanya berisi grammar point, target miskonsepsi, contoh, dan kalimat yang harus dihindari. Contoh format tetap (`공원[에서]`) hanya dipakai sampai skill punya item yang diterima; sesudah itu contohnya item terakhir yang diterima di skill itu, jadi tidak menarik skill lain ke partikel.
- Alasan penolakan untuk critic ditulis dengan kalimat biasa ("it makes a particle mistake, but this question tests a verb or adjective ending mistake"), bukan kode `MULTI`/`PART`.
- Temperature 0.8 → 0.6.
- `--generator gemma-3-12b` tersedia untuk perbandingan, default tetap EXAONE.
- Annotator (langkah 4) diperluas ke kata benda + partikel. Partikel yang salah bentuk (`학교으로`, `학교이`) dipecah kiwi menjadi morfem aneh dan diberi label MULTI. Akibatnya distraktor partikel yang bagus ditolak sebagai wrong-category, dan ini penyebab penolakan terbanyak. Sekarang kategorinya diambil dari morfem benar yang diganti (`PART:에>으로`).
- Sebaliknya, kata isi yang ditempel ke kata lain (`늦[게]` → `늦일찍`, `천천히빠르게읽어요`) tidak lagi dibaca sebagai satu kesalahan ending. Sebelumnya dua item seperti itu lolos di cek cepat.

### 9.78 Generator hanya menulis kalimat; bagian kosong dihitung oleh kode

Cek cepat dengan prompt §9.77: 6/44 diterima, tetapi hampir semuanya tidak layak. Contohnya: kunci di posisi aneh (`책을 [에서] 읽었습니다`); soal honorific yang menguji 습니다/어요, bukan -(으)시-; opsi rusak (`보러갔어요`). Jawaban yang ditolak menunjukkan EXAONE-7.8B tidak bisa memakai kurung siku dengan benar (`[을/를]`, kata benda dikurung untuk honorific). Selain itu, EXAONE terus menulis distraktor sebagai kalimat utuh walaupun diminta potongan saja. jc memutuskan generator tetap EXAONE.

**Keputusan:** tugasnya disesuaikan dengan kemampuan EXAONE. Generator menulis `{"sentence": kalimat benar, "wrong_sentences": [3 salinan, masing-masing dengan satu kesalahan]}`, tanpa kurung siku dan tanpa potongan. Kode yang menghitung bagian kosong (`_blank`): span tempat ketiga kalimat salah berbeda dari kalimat benar, diperlebar ke morfem utuh, atau ke kata utuh kalau ada opsi yang kosong. Tanda baca akhir tidak ikut. Kalau kesalahan ada di kata yang berbeda, soal ditolak. Verifikasi lainnya tetap: annotator (satu edit, kategori skill), kode berbeda, gate log-prob, critic. `_wrong_sentence`/`_shared_blank` dihapus.

### 9.79 Arah baru: pipeline berbasis model/library, langkah 1 = mengukur komponen

jc ingin metode seperti Duolingo/Eedi: semuanya model, library atau prompt, tanpa heuristik buatan sendiri (rekomendasi lengkap ada di Claude Doc "KO-DRILL: metode personalisasi berbasis model"). Run 6d41db3 (format kalimat utuh, §9.78): 11/44 diterima (25%). particle_loc dan noun_counter bagus, tetapi honorific, irregular_conj, adj_relativizer dan mod_adverb menguji akhiran lain (습니다/어요).

Komponen pengganti yang diusulkan:
- KAGAS resmi menggantikan `kagas_annotator.py` dan `_blank`.
- KoBART GEC `Soyoung97/gec_kr` dan log-prob Qwen3 mengecek kalimat benar dan salah.
- Qwen3-4B menjadi classifier skill (pilih 1 dari 11).
- Outlines (JSON schema) menggantikan parser.

`src/benchmark_components.py` mengukur ketiganya pada 52 pasangan kurasi sebelum dipasang.

**KAGAS resmi, diukur di sandbox:**
- Satu edit: 52/52.
- Tipe sesuai skill: 47/52. Gagalnya ada di tiga tempat:
  - `작는` diberi label SPELL;
  - `부르어요` diberi label SHORTEN;
  - 3 penggantian kata kerja (verb_choice) diberi label CONJUGATION.

Kesimpulan: KAGAS andal untuk "tepat satu edit" dan posisinya. Span-nya per kata (eojeol), jadi bagian kosong menjadi satu kata utuh. Cocok-tidaknya dengan skill lebih baik diputuskan oleh classifier.

**Instalasi:** `cyhunspell` yang dipatok repo tidak bisa di-build lagi. Yang jalan adalah `pip install hunspell` (pyhunspell), ditambah `soylemma`, konlpy dan Java, di Python 3.11.

GEC dan classifier butuh GPU dan Hugging Face, jadi dijalankan jc.

**Hasil run jc (83908c3), 52 pasangan kurasi:**
- KoBART GEC: kalimat benar dibiarkan 42/52 (81%); kalimat salah dikembalikan ke kunci 28/52 (54%). Model ini sering menulis ulang kalimat (`가르쳐요` → `가르쳐 주셨어요`, `책은` → `이 책은`) dan tidak membetulkan `부르어요`. Di bawah target 90%, jadi **tidak dipakai**.
- Classifier skill Qwen3-4B (stem + kunci + satu distraktor): 37/52 (71%). Kesalahannya sistematis: setiap stem yang mengandung 을/를 diberi label particle_obj. Pada 11 item hasil generator, hasil classifier sama dengan penilaian manual (§9.79): 7 item sesuai skill dipertahankan, 4 item di luar skill ditolak. Prompt diubah supaya membandingkan dua kalimat utuh (benar vs salah) dan perlu diukur ulang.
- Classifier v2 (membandingkan dua kalimat, d1f023a): 36/52 (69%), tidak lebih baik. Bias kuat ke particle_obj (opsi B), dan pada item generator hanya 5/11 cocok. Skor huruf tanpa reasoning ternyata terlalu lemah untuk model 4B. Langkah berikut: mode reasoning Qwen3 (`--thinking`).
- Classifier mode reasoning (`--thinking`, 052155b): 43/52 (83%), bias particle_obj hilang. Sisa kesalahan ada di honorific (4/5) dan irregular_conj (dilabel verb_choice). Pada 11 item generator, hasilnya sama dengan penilaian manual: 6 dipertahankan, 5 di luar skill ditolak. Jadi bias tadi berasal dari prompt atau skor huruf, bukan hanya ukuran model.
- Temuan: label KAGAS resmi per kata (eojeol) dengan tipe kasar (PARTICLE/ENDING). Itu tidak cukup sebagai label miskonsepsi yang muncul ulang di banyak soal (`에서>에`). Keputusan label diserahkan ke jc.

### 9.80 Versi 2: pipeline tanpa heuristik (hanya model/LLM/library)

jc ingin metode seperti Duolingo/Eedi: semuanya LLM, model terlatih, atau library, tanpa aturan buatan sendiri. Rancangan lengkapnya ada di Claude Doc, bagian "Versi 2". Kurikulum tidak lagi 11 skill tetap, tetapi data: konteks (misalnya smart factory / EV manufacturing) dan tujuan belajar.

Pipeline:
1. EXAONE melalui Outlines (JSON schema: stem dengan satu `___`, kunci satu kata, 3 distraktor + `mistake`).
2. Gate log-prob Qwen3 untuk kunci.
3. Classifier tujuan belajar oleh Qwen3 mode reasoning.
4. Miskonsepsi = embedding kalimat `mistake` (sentence encoder multilingual), tanpa kode.
5. Model pilihan distraktor softmax(θ·e).

Yang akan dihapus setelah terbukti: annotator, blank, filter, dan parser.

`src/benchmark_v2.py` mengukur empat hal: format, gate kunci, gate tujuan, dan apakah embedding mengelompokkan kesalahan yang sama. Kode KAGAS dipakai hanya sebagai kunci evaluasi.


### 9.81 Repo cleanup (2026-10-21)

Atas persetujuan jc, file yang sudah tidak dipakai dihapus: `src/validate_novelty2.py` + `2_validation_results.json` (Novelty 2 dihapus, §9.70), `src/compare_feedback_models.py` + `src/patch_exaone35_remote_code.py` (perbandingan explainer selesai, Qwen3 dipilih), `src/benchmark_components.py` + `benchmark_components_results.json` (kesimpulan di §9.79; classifier pindah ke `benchmark_v2.py`), dan `eval_report.json` (output yang dibuat ulang tiap run, sekarang di `.gitignore`). `Korean_Grammar_Curriculum.docx` dan PDF literatur di `docs/` tetap disimpan (keputusan jc). `requirements.txt` ditambah `outlines`, `pydantic`, `sentence-transformers`. Menunggu hasil v2 sebelum menghapus `kagas_annotator.py`, heuristik di `item_foundry.py`, `llm_parsing.py`, fungsi pilih-huruf di `solver_check.py`, dan `difficulty_heuristic.py` (yang terakhir juga mengubah teks paper Section III-C). Opsi `--generator gemma-3-12b` juga dihapus (EXAONE tetap generator, §9.78); model yang sudah diuji dan tidak dipakai (Qwen3-4B-Base, Qwen3-8B/-Base, Kanana-1.5-8B-base, Gemma-3-12B, EXAONE-4.0-1.2B, KoBART `gec_kr`) sudah tidak dirujuk kode, jadi boleh dihapus dari cache Hugging Face lokal. Audit lengkap: `/mnt/project-files/cleanup/audit-repo.md`.


### 9.82 Hasil v2 pertama dan perbaikan v2.1 (2026-10-21)

**Hasil run jc** (`benchmark_v2_results.json`, 6 item × 11 objective, kurikulum EV smart factory): format 66/66 (Outlines bekerja), diterima 13/66 = 19,7% (CI 11,9–30,8%). Embedding `mistake` (multilingual-e5-base): nearest neighbour sama-mistake 15/42 = 36%, cosine sama 0,900 vs beda 0,886 — hampir tidak terpisah.

**Dibaca per item, hampir semua 13 item yang lolos tetap tidak layak dipakai:**
1. **Blank di tempat yang salah.** Diminta langsung menulis stem dengan `___`, EXAONE menaruh blank di posisi acak: `공장의 ___ 에서` dengan jawaban `에서` (jadi "에서 에서"), `을___`, `동기화되었어요었어요`. Key gate log-prob menolak sebagian besar, tapi beberapa tetap lolos.
2. **Distraktor yang sebenarnya benar.** Untuk adj_relativizer, mod_adverb, verb_choice dan noun_counter, distraktornya kata lain yang juga benar (`신중하게` vs `빠르게`, `다섯` vs `세`). Key gate hanya mengecek bahwa kunci paling mungkin, bukan bahwa distraktor salah.
3. **Kalimat terlalu panjang dan teknis untuk TOPIK I:** sering dua kalimat, karena topik kursus dimasukkan utuh.

**Perbaikan v2.1 (`src/benchmark_v2.py`), tetap tanpa heuristik:**
- Skema: EXAONE menulis dulu satu kalimat benar (maksimal 40 karakter, satu kalimat), lalu `answer` = kata dalam kalimat itu. Stem = kalimat dengan answer diganti `___`; kalau answer tidak muncul tepat sekali di kalimat, item gagal format. Urutan ini sama dengan format kalimat-utuh §9.78 yang memang ditulis LLM dengan baik.
- Prompt: satu contoh kurasi per objective (dari `log_likelihood_results.json`, data, bukan aturan), "maksimal 8 kata, satu kata sehari-hari dari topik", dan distraktor harus membuat kalimat salah, bukan kata lain yang benar.
- Gate baru `unique_ok`: Qwen3-4B mode reasoning diberi stem dan 4 opsi (diacak), lalu menyebut semua opsi yang benar; item lolos hanya kalau jawabannya tepat kunci saja. Diterima = key_ok ∧ unique_ok ∧ objective_ok.
- Embedding dibandingkan dua encoder: multilingual-e5-base vs Qwen3-Embedding-0.6B (encoder berbasis instruksi).


### 9.83 Kelas miring (5 mahir / 500 pemula) dan difficulty dari responden LLM (2026-10-21)

**Pertanyaan jc:** bagaimana kalau hanya 5 orang yang mahir dan 500 lainnya pemula?

**Simulasi** (`src/simulate_skewed_population.py`, CPU): Rasch, seleksi adaptif dengan target 85%, Urnings dengan koreksi MH, 505 pelajar, 60.000 respons (sekitar 120 per pelajar), 300 item, 4 replikasi. Kelas *balanced* θ~N(0,1) dibandingkan dengan kelas *skewed*: 500 pemula θ~N(−1; 0,5) dan 5 mahir θ~N(2,5; 0,3). Ada dua pool: *wide* β~U(−3,3), dan *matched* (difficulty sesuai kebutuhan kelas, seperti generator yang menulis item untuk pelajarnya). Ada dua prior item: *flat* (n=32), dan *model* (μ̂ = β + N(0; 0,8), korelasi sekitar 0,9, urn diukur dari varians).

| kelas | pool | prior | RMSE kemampuan lainnya | RMSE kemampuan mahir | RMSE item mudah | RMSE item sulit | respons/item sulit |
|---|---|---|---|---|---|---|---|
| balanced | wide | flat | 0,56 | 0,80 | 0,65 | 1,55 | 38,5 |
| balanced | wide | model | 0,45 | 0,99 | 0,48 | 0,80 | 9,8 |
| skewed | wide | flat | 1,15 | 0,94 | 1,12 | 1,62 | 18,5 |
| skewed | wide | model | 0,91 | 0,86 | 0,84 | 0,77 | 2,7 |
| skewed | matched | flat | 1,63 | 0,70 | 1,64 | 0,49 | 48,2 |
| skewed | matched | model | 0,90 | 1,16 | 0,94 | 0,27 | 5,0 |

**Temuan:**
1. **5 pelajar mahir tidak dirugikan:** RMSE 0,86–0,94 di pool wide, setara kelas balanced (0,80–0,99). Angka mahir paling bising, karena hanya 5 orang × 4 replikasi.
2. **Item sulit hanya mendapat sedikit respons** (2,7–18,5 per item, vs 38,5 di kelas balanced). Tanpa prior, difficulty-nya tetap salah (RMSE 1,62); dengan prior model turun ke 0,77. Prior paling berguna tepat di bagian ini.
3. **Yang justru terdampak adalah 500 pemula:** RMSE 1,15 vs 0,56 di kelas balanced. Dengan pool yang cocok untuk kelas tetapi prior flat, RMSE naik ke 1,63, karena semua item mulai di β=0 sementara item sebenarnya sekitar −2,7, sehingga seluruh skala bergeser. Prior model menahan skala itu (0,90). Jadi pada kelas yang miring, prior item juga menjadi jangkar skala, bukan hanya mempercepat konvergensi.
4. **Batasan:** prior yang disimulasikan berkorelasi sekitar 0,9 dengan difficulty sebenarnya. Korelasi prior LLM sungguhan belum diketahui sampai ada data pelajar; prior yang lemah (r=0,44) sudah diukur di §9.69 / paper Section III-B.

**`src/difficulty_llm.py` (baru), pengganti `difficulty_heuristic.py`:** Qwen3-0.6B, 1.7B dan 4B dipakai sebagai responden dengan kemampuan berbeda (Liu et al., 2025, BJET, PDF ada di `docs/`). p(kunci) = softmax log-prob kalimat dari 4 opsi. Fit Rasch least-squares logit p = θ_m − β_i (mean β = 0). μ̂ = β distandarkan; σ̂² = SE² + 0,5 (`LINK_VARIANCE`, karena skala responden belum di-link ke skala pelajar). Diukur di `benchmark_v2.py` (akurasi responden harus naik dengan ukuran model; korelasi dengan heuristik morfem). Belum dipasang ke `question_bank` / `answer_processing`; itu dilakukan bersama penghapusan grup B.

---

*Dokumen ini hidup — update setiap ada keputusan baru atau keputusan lama
yang direvisi. Jangan biarkan basi.*
