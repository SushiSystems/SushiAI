# Kaynak Kod Yorumları ve Kod Stili Standartlaştırma Tasarım Belgesi

**Tarih:** 2026-09-16  
**Durum:** Taslak (Kullanıcı İncelemesinde)  
**Kapsam:** `include/` ve `src/` ağacındaki C++ başlık ve kaynak dosyalarının `source-comments`, `cpp-code-style` ve `project-tools` standartlarına uyumlu hale getirilmesi.

---

## 1. Amaç ve Kapsam

Bu çalışmanın amacı, SushiAI kütüphanesinin çekirdek arayüz (`include/SushiAI/`) ve kaynak kod (`src/`) dosyalarındaki 631 adet `source-comments` kural ihlalini tamamen temizlemek ve C++17 kod stili standartlarını sağlamaktır.

Yapılacak tüm denetim ve doğrulamalar `tools/documentation/check_source_comments.py` statik denetleyicisi ile yürütülecektir. Hiçbir derleme veya derleyici ikilisi çalıştırılmayacaktır.

---

## 2. İhlal Türleri ve Düzeltme Kuralları

`tools/documentation/check_source_comments.py` tarafından tespit edilen 5 temel ihlal türü bulunmaktadır:

### 2.1. Dosya Başlıkları (`rule_file_header` - 92 dosya)
- **Kural:** Her dosya Apache 2.0 kutulu lisans bloğu ile başlar. Ardından tam bir boş satır bırakılır ve en fazla 6 satırdan oluşan Doxygen `@file` bloğu gelir:
  ```cpp
  /**************************************************************************/
  /* checkpoint.hpp                                                         */
  /**************************************************************************/
  /*                          This file is part of:                         */
  /*                                 SushiAI                                */
  /*                https://github.com/SushiSystems/SushiAI                 */
  /*                         https://sushisystems.io                        */
  /**************************************************************************/
  /* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
  /* ... Apache-2.0 notice ...                                              */
  /**************************************************************************/

  /**
   * @file checkpoint.hpp
   * @brief Binds checkpoint records to module parameters.
   * @author Mustafa Garip
   */

  #pragma once
  ```
- Lisans kutusunun ilk satırında dosyanın adı yer alır.
- `@file` etiketinde dosya adı tekrarlanır.
- `@brief` fiil ile başlayan tek bir yalın cümle içerir.
- `@author` olarak yazar adı belirtilir.
- `@date`, sürüm veya geçmiş blokları kesinlikle bulunmaz.
- Başlık bloklarındaki uzun anlatım paragrafları (`@section` vb.) kaldırılır veya ilgili tasarım belgesine aktarılır.
- Başlık dosyalarında `#pragma once` yönergesi `@file` bloğunun hemen ardından konumlandırılır.

### 2.2. Sembol Yorum Tavanı (`rule_block_ceiling` - 217 ihlal)
- **Kural:** Fonksiyon, sınıf, üye ve yapıları belgeleyen `/** ... */` ve ardışık `///` blokları en fazla 8 satır olabilir.
- 8 satırı aşan bloklar, yalnızca zorunlu etiketler (`@brief`, `@param`, `@return`, `@pre`) kalacak biçimde özetlenir.
- Açıklama cümleleri tek satırda tutulur, gereksiz boşluk ve tekrarlar sadeleştirilir.

### 2.3. Ardışık Satır Yorumları (`rule_comment_runs` - 250 ihlal)
- **Kural:** Gövde içerisinde iki veya daha fazla `//` satırının ardışık gelmesi yasaktır (paragraf oluşturur).
- Fonksiyon gövdesi içindeki `//` yorumu yalnızca tek bir satırda, kodun dayandığı bir değişmezi (invariant) ifade edebilir.
- Açıklayıcı çok satırlı `//` blokları ya tek satırlık özlü bir değişmeze indirgenir ya da sembol başındaysa kurallı `/** ... */` Doxygen bloğuna dönüştürülür.

### 2.4. Ayraç Çizgileri (`rule_separators` - 71 ihlal)
- **Kural:** Lisans bloğu haricinde `// -------------------`, `// ===================` veya `/* ******** */` gibi ayraç çizgileri kullanılamaz.
- Tüm ayraç çizgileri koddan temizlenir.

### 2.5. Geçmiş ve TODO Etiketleri (`rule_history` - 1 ihlal)
- **Kural:** Kaynak kodda `TODO`, `FIXME`, `HACK`, `XXX` veya `previously`, `fixed in` gibi geçmiş ifadeleri bulunamaz.
- Bu ifadeler koddan çıkarılır; açık kalan görevler `docs/design/REMAINING_WORK.md` dosyasına taşınır.

---

## 3. Modül Dağılımı ve İş Paketleri

Kaynak kodlar modül sınırları doğrultusunda 6 ayrı görev paketine ayrılır:

1. **Paket 1: Çekirdek ve Kök Dosyalar (Top-level & IO)**
   - Dosyalar: `include/SushiAI/SushiAI.h`, `include/SushiAI/version.hpp`, `src/version.cpp`, `include/SushiAI/io/*`, `src/io/*`
   - Toplam ihlal: 16
2. **Paket 2: Veri ve Yapılandırma (Data & Config)**
   - Dosyalar: `include/SushiAI/data/*`, `src/data/*`, `include/SushiAI/config/*`, `src/config/*`
   - Toplam ihlal: 50
3. **Paket 3: Temel ve Eğitim (Core & Train)**
   - Dosyalar: `include/SushiAI/core/*`, `src/core/*`, `include/SushiAI/train/*`, `src/train/*`
   - Toplam ihlal: 85
4. **Paket 4: Sinir Ağı ve Optimizasyon (NN & Optim)**
   - Dosyalar: `include/SushiAI/nn/*`, `src/nn/*`, `include/SushiAI/optim/*`, `src/optim/*`
   - Toplam ihlal: 120
5. **Paket 5: Otomatik Türev (Autograd)**
   - Dosyalar: `include/SushiAI/autograd/*`, `src/autograd/*`
   - Toplam ihlal: 72
6. **Paket 6: Hesaplama Grafı ve İndirgeme (Graph)**
   - Dosyalar: `include/SushiAI/graph/*`, `src/graph/*`
   - Toplam ihlal: 288

---

## 4. Doğrulama Ölçütleri

1. Her paket tamamlandığında `python tools/documentation/check_source_comments.py <ilgili modül yolları>` çalıştırılır ve 0 bulgu teyit edilir.
2. Tüm paketler bittiğinde:
   `python tools/documentation/check_source_comments.py include src` komutunun çıktısı 0 hata ile sonlanmalıdır.
3. Dokümantasyon ve değişiklik günlüğü denetleyicileri:
   `python tools/documentation/check_docs_layout.py .` ve `python tools/documentation/check_changelog.py .` temiz kalmalıdır.
