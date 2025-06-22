# SushiAI

**Fizik Bilgili AI Framework'ü**

---

## Proje Hakkında

SushiAI, fiziksel sistemleri simüle eden derin öğrenme temelli **surrogate modeller** ile simulasyon programlarında veya bir bilgisayarda **anında** çalıştırmayı amaçlayan, C++ tabanlı bir çekirdek üzerine kurulu bir projedir. Python ile eğitim ve veri hazırlama ve Unity (C#) ile gerçek zamanlı kullanım sunması planlanmıştır.

**Vizyon:** "Fizik kurallarını göz ardı etmeden, simulasyonlarda derin öğrenme yardımıyla gerçek zamanlı performans sunmak."
**Misyon:** "Sushi Systems'in Virtual World projesi olan Project: RL'yi bit-tabanlı bilgisayarlarda maksimum performans ile oynatabilmek."

---

## Temel Özellikler

* **Surrogate Modeling**
  CFD, FEM, ısı transferi, soft-body ve dalga simülasyonları gibi ağır fiziksel hesaplamaların derin öğrenme ile çok hızlı ve doğru bir biçimde tahmini.

* **Modüler Tensor ve Autograd Sistemi**
  NumPy benzeri çok boyutlu tensörler desteklenir. Otomatik türev alma sistemi (autograd) zincir kuralına (chain rule) göre çalışır. Eğitim sırasında gradyanların doğru şekilde hesaplanmasını ve ileri/geri geçişlerin yapılmasını sağlar.

    Element-wise işlemlerde broadcasting.

    MLP/CNN için ağırlık güncellenebilir parametreler.

    Gradient caching + backward() zincir sistemi.

* **GNN (Graf Sinir Ağı)**
  Düzensiz geometrilere sahip mesh verilerinde topolojik bağıntıları öğrenir. Mesh üzerindeki her düğüm (node), kenar (edge) ve fiziksel özellik GNN ile işlenir.

    GCN, GAT, MPNN tabanlı katman altyapısı

    MeshGraphNet gibi gerçek CFD örneklerine uygun yapı

* **PINN (Fizik Bilgili Sinir Ağı)**
  Kısmi diferansiyel denklemleri (örneğin Navier-Stokes, Fourier, Laplace) doğrudan loss fonksiyonuna entegre ederek, veri az olsa bile fizik kurallarıyla tutarlı modellerin eğitilmesini sağlar.

    Fiziksel loss bileşenleri (PDE residual, boundary loss, initial condition loss)

    Otomatik türev destekli PDE değerlendirme

    Gerçek dünya fizik senaryolarında uygulanabilir (ısı, akışkan, deformasyon)

* **Python Arayüzü (Yakında)**
  Pybind11 tabanlı bağlayıcılar ile Python üzerinden eğitim, doğrulama ve veri işleme işlemlerine olanak sağlayan sezgisel bir API sunar. Bu arayüz, “PySushi” adıyla kullanıma açılacaktır.

* **Unity (C#) Arayüzü (Yakında)**
  SushiAI çekirdeği, .DLL biçiminde derlenerek Unity oyun motoruna kolayca entegre edilir. Bu sayede Unity içindeki bir sahne içinde fiziksel işlemlerin çıktıları (inference) gerçek zamanlı olarak istenilen komponent'e uygulanır.

---

## Mimariye Genel Bakış

```text
                +-----------------------------+
                |      SushiAI Core (C++)     |
                |-----------------------------|
                | - Tensor & Autograd         |
                | - NN Katmanları (MLP, CNN)  |
                | - GNN Katmanları            |
                | - PINN Lossları             |
                | - Dataset Çözücüleri        |
                +--------------+--------------+
                               |
           +-------------------+-------------------+
           |                                       |
       [ Python ]                             [ Unity (C#) ]
    Eğitim & Analiz                        Gerçek Zamanlı Fizik
                                              Simülasyonları
```

---

## Planlanan Kullanım Alanları

1. **Soft-Body Fizikler**

   * Soft-body davranışları, çarpışma ve deformasyon gibi fiziksel etkiler altında karmaşık şekil değişimlerini içerir. SushiAI, bu davranışları veri temelli öğrenerek, geleneksel fizik motorlarına kıyasla çok daha hızlı ve gerçek zamanlı deformasyon tahminleri sunacak.

2. **Okyanus, Deniz ve Dalga Modelleme**

   * Deniz yüzeyi dinamikleri; rüzgâr etkisi, kıyı yansıması ve gemi etkileşimleri gibi çoklu fizik süreçlerini içerir. SushiAI, bu sistemleri Graf Sinir Ağı (GNN) tabanlı modellerle temsil ederek gerçek zamanlı, fiziksel olarak anlamlı dalga tahminleri sağlayacak.

3. **Akışkanlar Dinamiği (CFD)**

   * Hava veya sıvı akışlarının yüzeylerle etkileşimi, genellikle hesaplamalı akışkanlar dinamiği (CFD) ile çözülür. SushiAI, geçmiş CFD verilerinden öğrenerek, aerodinamik kuvvetleri ve akış alanlarını yüksek doğrulukta ve düşük gecikmeyle tahmin edecek.

4. **Termal Analiz**

   * Zamana bağlı ısı iletimi problemleri mühendislik sistemlerinde kritik öneme sahiptir. SushiAI, PINN mimarisi aracılığıyla sıcaklık dağılımını ve termal davranışları fiziksel denklemlerle uyumlu biçimde hızlıca hesaplayabilme kabiliyetine sahip olacak.

5. **Yapısal Gerilme-Şekil Değiştirme (FEM)**

   * Yük altında şekil değiştiren yapıların gerilme, yer değiştirme ve deformasyon analizleri geleneksel olarak FEM ile yapılır. SushiAI, bu tür analizleri çok daha düşük işlem maliyetiyle ve gerçek zamanlı kullanım senaryolarına uygun şekilde gerçekleştirebilecek.

---

## Kurulum

* (Planlanıyor, takipte kalın.)

---

## Yol Haritası

* (Planlanıyor, takipte kalın.)

---

## Lisans ve İletişim

* **Lisans:** Business Source License 1.1
* **Projenin Sorumlusu:** Mustafa Garip
* **E-Posta:** [mustafagarip@sushisystems.io](mailto:mustafagarip@sushisystems.io)
* **GitHub:** [https://github.com/sushimg/SushiAI](https://github.com/sushimg/SushiAI)
* **LinkedIn:** [https://www.linkedin.com/in/mustafgarip](https://www.linkedin.com/in/mustafgarip)

---

*Bu proje, gerçek fizik kurallarına bağlı kalarak yapay zeka ile gerçek zamanda kompleks fizik hesaplarının yapılmasını amaçlar. Eğer proje hakkında sorunuz olursa "Lisans ve İletişim" kısmındaki linkler aracılığıyla iletişime geçebilirsiniz. Müsait olduğum her zaman dönerim. Geribildirimlerinizi bekliyorum, Hoşça kalın!*
