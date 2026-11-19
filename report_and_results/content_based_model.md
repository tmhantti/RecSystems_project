# Content-based-suositin

## Malli ja toteutus

Content-based-malli käyttää MovieLens-elokuvien genrejä ja julkaisuvuotta. Elokuvista muodostetaan genre- ja vuosipiirteet. Käyttäjäprofiili muodostetaan hänen koulutusdatan arvioistaan: arvioita keskitetään käyttäjän keskiarvon ympärille ja yhdistetään elokuvien piirrevektoreihin. Suosittelemattomat eli vielä näkemättömät elokuvat pisteytetään kosinisamankaltaisuudella, minkä jälkeen käyttäjälle palautetaan kymmenen parasta.

Algoritmi on toteutettu projektin omassa `ContentRecommender`-luokassa AI-avusteisesti. Toteutus käyttää NumPya ja Pandasia sekä scikit-learnin `cosine_similarity`-apufunktiota, mutta ei valmista recommender-mallia. Vertailussa oli kaksi saman content-based-mallin varianttia: `content_all` käyttää kaikkia arvioita ja `content_positive` vain positiivisia arvioita.

Valmiita kirjastopohjaisia ratkaisuja tutkittiin ja alustavasti kokeiltiin, mutta päädyttiin tekemään malli käytännössä itse, vaikkei content based mallin tarkoituksena ollut täyttää itsetehty vaatimusta. 

## Viritys ja arviointi

Aineistosta suodatettiin 9 725 vuorovaikutusta. Aikajärjestykseen perustuvassa jaossa oli 967 988 koulutus-, 11 248 validointi- ja 11 248 testivuorovaikutusta. Hyperparametrit valittiin vain validointidatan NDCG@10:n perusteella; testituloksia ei käytetty valintaan.

Kummallekin variantille arvioitiin 20 asetusta. `content_all`-mallissa `year_weight` sai 20 tasavälistä arvoa väliltä 0–1. `content_positive`-mallissa kokeiltiin viittä vuosipainoa `{0, 0.1, 0.3, 0.5, 1}` ja neljää positiivisuusrajaa `{3, 3.5, 4, 4.5}`. Testissä käytettiin positiivisten osumien rajana arvoa 4. Malli on deterministinen. NDCG@10:n 95 %:n luottamusvälit laskettiin 2 000 käyttäjäbootstrap-otoksella (siemen 42).

| Variantti | Valitut parametrit | Precision@10 | Recall@10 | NDCG@10 (95 % CI) | Catalogue coverage |
|---|---|---:|---:|---:|---:|
| `content_all` | `year_weight=0.0` | 0.001450 | 0.009352 | 0.005205 [0.0038, 0.0067] | 0.816291 |
| `content_positive` | `year_weight=0.1`, `positive_threshold=3.5` | 0.002128 | 0.014262 | 0.007455 [0.0058, 0.0092] | 0.665983 |

`content_positive` oli parempi kaikissa osuvuusmittareissa: sen NDCG@10 oli noin 43 % korkeampi kuin `content_all`-variantin. Molempien osuvuus oli silti matala. `content_positive` kattoi pienemmän osan luettelosta. Arvioiden ollessa kokonaislukuja raja-arvot 3.5 ja 4 valitsevat samat positiiviset arviot, joten niiden viritystulokset eivät ole aidosti eri asetuksia.

Aiemmassa yhteisajossa item-kNN-baseline saavutti testissä NDCG@10-arvon 0.038955 (95 % CI [0.0348, 0.0431]), selvästi enemmän kuin `content_positive`-mallin 0.007455. Paritettu Wilcoxon-testi osoitti eron item-kNN:n hyväksi (molemmissa content-vertailuissa p < 10⁻⁴²). Tämä kNN-tulos on Part 1 -baseline; kNN arvioidaan nykyisin omassa tiedostossaan. Tulokset eivät osoita, että content-malli olisi kilpailukykyinen yleisessä osuvuudessa.

## Tulokset ryhmittäin

Ryhmien rajat laskettiin koulutusdatasta: käyttäjän aktiivisuuden mediaani oli 100 koulutusarviota ja elokuvan suosion mediaani 142 koulutusarviota. Alla olevat luvut ovat testidatasta. Ryhmäkohtainen tuloste raportoi Recall@10:n, NDCG@10:n ja NDCG:n 95 %:n luottamusvälin; ryhmäkohtaista Precision@10-arvoa ja catalogue coveragea ei tulostettu.

Tulokset excel tiedostossa.
| Variantti | Ulottuvuus | Ryhmä (koulutusdataan perustuva raja) | Käyttäjiä | Recall@10 | NDCG@10 (95 % CI) |
|---|---|---|---:|---:|---:|
| `content_all` | Käyttäjän aktiivisuus | Matala: <100 | 2261 | 0.010394 | 0.005959 [0.0039, 0.0082] |
| `content_all` | Käyttäjän aktiivisuus | Korkea: >=100 | 2016 | 0.008185 | 0.004361 [0.0025, 0.0065] |
| `content_all` | Elokuvan suosio | Suosittu: >=142 | 4001 | 0.008873 | 0.004931 [0.0035, 0.0064] |
| `content_all` | Elokuvan suosio | Long-tail: <142 | 548 | 0.012774 | 0.006006 [0.0022, 0.0110] |
| `content_positive` | Käyttäjän aktiivisuus | Matala: <100 | 2261 | 0.017249 | 0.009870 [0.0072, 0.0129] |
| `content_positive` | Käyttäjän aktiivisuus | Korkea: >=100 | 2016 | 0.010913 | 0.004746 [0.0031, 0.0067] |
| `content_positive` | Elokuvan suosio | Suosittu: >=142 | 4001 | 0.013372 | 0.007043 [0.0053, 0.0089] |
| `content_positive` | Elokuvan suosio | Long-tail: <142 | 548 | 0.021898 | 0.009051 [0.0044, 0.0146] |

`content_positive` suoriutui tässä testissä paremmin vähäisen aktiivisuuden käyttäjillä kuin aktiivisilla käyttäjillä. Long-tail-ryhmän NDCG oli myös korkeampi kuin suosittujen elokuvien ryhmän, mutta long-tail-ryhmän otos oli pienempi. Nämä havainnot rajaavat yleistä johtopäätöstä: mallilla voi olla hyötyä tietyissä ryhmissä, vaikka sen kokonaistason osuvuus on heikko.

## 2.5 Sparsity ablation

Koulutusvuorovaikutuksista poimittiin ilman palautusta 50 % ja 20 %; validointi- ja testidata pidettiin samoina. Täyden datan perusteella valitut hyperparametrit pidettiin kiinteinä. Harvennetut ajot tehtiin siemenillä 0–9, ja alla ovat niiden keskiarvot.

| Variantti | Koulutusdata | Precision@10 | Recall@10 | NDCG@10 | Catalogue coverage |
|---|---:|---:|---:|---:|---:|
| `content_all` | 100 % | 0.001450 | 0.009352 | 0.005205 | 0.816291 |
| `content_all` | 50 % | 0.001272 | 0.007996 | 0.004301 | 0.809933 |
| `content_all` | 20 % | 0.001099 | 0.007283 | 0.003909 | 0.820481 |
| `content_positive` | 100 % | 0.002128 | 0.014262 | 0.007455 | 0.665983 |
| `content_positive` | 50 % | 0.001863 | 0.012333 | 0.006459 | 0.746909 |
| `content_positive` | 20 % | 0.001590 | 0.010638 | 0.005459 | 0.855992 |

`content_positive` säilytti paremman NDCG:n kaikilla datamäärillä. Sen NDCG laski täydestä datasta 20 %:iin siirryttäessä noin 27 %, ja `content_all`-variantin noin 25 %. Coverage-arvon nousu `content_positive`-mallilla 20 %:n kohdalla ei tarkoita parempaa osuvuutta: harvempi koulutusdata jättää enemmän elokuvia käyttäjälle näkemättömiksi. Aiemman yhteisajon rankingissa item-kNN oli paras täydellä ja 50 %:n datalla, mutta `content_positive` nousi kNN:n ohi 20 %:n datalla; `content_all` jäi kolmanneksi kaikilla tasoilla.

## Johtopäätös ja rajoitukset

Positiivisiin arvioihin perustuva content-variantti oli omista vaihtoehdoista paras, ja ryhmittely viittaa mahdolliseen hyötyyn long-tail-elokuville ja vähän aktiivisille käyttäjille. Kokonaisosuvuus jäi kuitenkin heikoksi, eikä yksittäisten ryhmien tuloksia pidä yleistää ilman ryhmäkohtaista epävarmuusarviota. Lisäksi ryhmätaulukosta puuttuvat Precision@10 ja catalogue coverage, joten se ei vielä täytä kaikkia tulospohjan sarakkeita. Content-varianttien keskinäistä paritettua merkitsevyystestiä ei ole tässä tulosteessa; niiden tulokset esitetään siksi kuvailevana vertailuna.

Malli ei ota huomioon monimuotoisuuden ja suosioharhan hallintaa, joka voi heikentää tuloksia.
