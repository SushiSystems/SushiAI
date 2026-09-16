# Dokümantasyon Ağacı ve Kök Dizin Standartlaştırılması Tasarım Belgesi

**Tarih:** 2026-09-16  
**Durum:** Taslak (Kullanıcı İncelemesinde)  
**Kapsam:** SushiAI deposunun SushiStack ve SushiSkills standartlarına (`repository-layout`, `documentation`, `cpp-code-style`) uyumlu hale getirilmesi.

---

## 1. Amaç ve Kapsam

Bu çalışmanın amacı, SushiAI kod tabanındaki kök dizin kirliliğini gidermek, eksik temel dosyaları (`README.md`, `CLAUDE.md`, `AGENTS.md`, `.clang-format`, `.editorconfig`) eklemek ve `docs/` hiyerarşisini `documentation` becerisinin belirlediği kurallı dizin ağacına dönüştürmektir.

Bu aşama (Faz 1), sonraki aşamalarda eklenecek olan `project-tools` denetleyicileri (`check_docs_layout.py`, `check_source_comments.py`, `check_changelog.py`) ve olası kod refactor'ı için sağlam ve hatasız bir zemin hazırlar.

---

## 2. Kök Dizin Değişiklikleri

`repository-layout` kuralı gereğince kök dizin yalnızca şu girdileri barındırabilir:
`README.md`, `LICENSE`, `CMakeLists.txt`, `AGENTS.md`, `CLAUDE.md`, `.clang-format`, `.editorconfig`, `.gitignore`, `.gitattributes`, `cmake/`, `cli/`, `tools/`, `modules/`, `tests/`, `applications/`, `third_party/`, `docs/`.

### 2.1. README.md
- Mevcut `docs/README.md` dosyasının kapsamlı içeriği doğrudan kök dizindeki `README.md` dosyasına taşınır.
- Yönlendirme yapılmaz; projenin mimarisi, özellikleri, gereksinimleri, derleme adımları ve gösterim çıktıları doğrudan GitHub ana sayfasında sunulur.

### 2.2. CLAUDE.md ve AGENTS.md
- `docs/CLAUDE.md` kök dizine taşınır. Kök `CLAUDE.md`, yapay zekâ asistanlarının uyması gereken SushiStack becerilerini (`skills:`), öncelik sıralamasını ve zorunlu CLI kullanım kuralını belgeler.
- Kök dizine `AGENTS.md` eklenir; ajanlar için mimari sınırlar, tek kapı olarak `sa` CLI kuralı ve kalite ilkeleri bildirilir.

### 2.3. .clang-format ve .editorconfig
- Kök dizine SushiStack standartlarını kodlayan `.clang-format` eklenir (Allman blok parantezleri, 4 boşluk girinti, 100 sütun genişliği, sınıf içi erişim belirteçleri bir seviye girintili).
- Kök dizine `.editorconfig` eklenir (UTF-8, LF sonlandırma, trim trailing whitespace, insert final newline).

### 2.4. Doxyfile ve CLI Uyarlaması
- Kök dizindeki `Doxyfile`, kökten kaldırılarak `docs/Doxyfile` konumuna taşınır.
- `cli/sushiai/services/project.py` içerisindeki `doxygen` servisi, öncelikle `docs/Doxyfile` yolunu arayacak, geriye uyumluluk amacıyla kök dizini yedek kontrol edecek şekilde güncellenir.

### 2.5. .gitignore ve docs/api-site
- `docs/api-site/` derleme çıktısı git takibinden çıkarılır ve silinir.
- `.gitignore` dosyasına `/docs/api-site/` girdisi eklenir.

---

## 3. Dokümantasyon Ağacı (`docs/`) Yeniden Yapılandırması

`docs/` altındaki dizin yapısı `documentation` becerisi kuralına tam uyumlu hale getirilir:

```text
docs/
    README.md                     # El kitabı ana fihristi
    CONTRIBUTING.md               # Katkı ve değişiklik kuralları
    getting_started/
        TUTORIAL.md               # Adım adım başlangıç rehberi
    architecture/
        ARCHITECTURE.md           # Sistem mimarisi ve bileşen detayları
    guides/
        CLI_GUIDE.md              # sa CLI komut kılavuzu
        INTEGRATION.md            # Tüketici projeler için entegrasyon rehberi
    reference/
        CHANGELOG.md              # Sürüm ve değişiklik günlüğü (standart madde biçimi)
        CODE_OF_CONDUCT.md        # Davranış kuralları
        SECURITY.md               # Güvenlik politikası
    design/
        README.md                 # Tasarım belgeleri konu haritası
        REMAINING_WORK.md         # Tekil iş ve eksikler listesi (backlog)
        ROADMAP.md                # Taşınan slop/make_sushiai_great_again.md
    archive/
        superpowers/              # Eski superpowers plan ve tasarım belgeleri
```

### 3.1. Dosya Taşımaları ve Yeniden Düzenlemeler
1. `docs/TUTORIAL.md` -> `docs/getting_started/TUTORIAL.md`
2. `docs/ARCHITECTURE.md` -> `docs/architecture/ARCHITECTURE.md`
3. `docs/CLI_GUIDE.md` -> `docs/guides/CLI_GUIDE.md`
4. `docs/INTEGRATION.md` -> `docs/guides/INTEGRATION.md`
5. `docs/CHANGELOG.md` -> `docs/reference/CHANGELOG.md`
   - Başlıklar `## Unreleased` altında toplanır.
   - Her madde en fazla 240 karakter, tek cümle ve geçmiş zaman fiiliyle başlayan standart SushiStack biçimine (`- YYYY-MM-DD — scope: Past-tense verb...`) dönüştürülür.
6. `docs/CODE_OF_CONDUCT.md` ve `docs/SECURITY.md` -> `docs/reference/`
7. `docs/slop/make_sushiai_great_again.md` -> `docs/design/ROADMAP.md`
   - Başlığına `**Status:** Open — phase 2 of 4` satırı eklenir.
   - `docs/slop/` dizini silinir.
8. `docs/superpowers/plans/` ve `docs/superpowers/specs/` -> `docs/archive/superpowers/` altına arşivlenir.
9. `docs/design/README.md` oluşturulur: Konu ile tasarım belgesi eşlemesi yapılır.
10. `docs/design/REMAINING_WORK.md` oluşturulur: Tekil iş listesi ve eksikler burada listelenir.
11. `docs/README.md` yenilenir: El kitabı ana fihristi olarak tüm alt belgelere kategorik bağlantı verir.

---

## 4. Bağlantı ve Referans Bütünlüğü

Taşınan dosyalar sebebiyle bağlantı kopmalarını önlemek için:
- Kök `README.md` içindeki `docs/` bağlantıları güncellenir.
- `docs/architecture/ARCHITECTURE.md` içindeki dosya bağlantıları (`slop/...` yerine `design/ROADMAP.md`, `TUTORIAL.md` yerine `getting_started/TUTORIAL.md` vb.) düzeltilir.
- C++ kaynak dosyalarındaki Doxygen `@file` ve açıklama bloklarında geçen dokümantasyon yolları yeni konumlara güncellenir.

---

## 5. Doğrulama ve Kabul Kriterleri

1. Kök dizinde `Doxyfile`, `docs/api-site` veya yetkisiz dosya kalmayacaktır.
2. `sa doxygen` komutu sorunsuz çalışacak ve dokümantasyon çıktısını başarıyla üretecektir.
3. Kök dizinde `README.md`, `CLAUDE.md`, `AGENTS.md`, `.clang-format` ve `.editorconfig` bulunacaktır.
4. `docs/` ağacında `slop/` ve `superpowers/` klasörleri kalmayacak, tüm belgeler `documentation` kuralındaki kurallı alt dizinlerde yer alacaktır.
5. Tüm belgeler arası iç bağlantılar hatasız çözümlenecektir.
6. `sa test` komutu eksiksiz geçecektir (kod bütünlüğü bozulmayacaktır).
