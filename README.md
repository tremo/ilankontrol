# Interactive Character Simulation MVP

State-driven, parametrik ve asamali ilerleyen bir karakter simule edicisi.

## Ozellikler

- Lokal calisan Streamlit UI
- Tek JSON state objesi (`scenario`, `characters`, `relationships`, `current_stage`, `tension`, `risk`, `history`)
- Numeric trait sistemi (`base`, `current`, `volatility`, `dynamic_enabled`)
- Runtime trait drift (tension/risk ve emotional inertia etkili)
- 6-stage progression + freeze stage + manual override
- Relationship matrix runtime guncellemesi
- Planner -> Generator -> Validator -> Rewrite loop akisi
- Save/Load JSON
- Import paneli ile uzun metinden otomatik scenario/character/history mapleme
- Coklu provider secimi: OpenAI, Claude (Anthropic), Gemini, OpenAI-compatible, rule_based

## Mimari

```text
Streamlit UI
  -> Simulation Controller (streamlit_app + SimulationEngine)
    -> State Manager (JSON)
    -> Planner (LLM)
    -> Generator (LLM)
    -> Validator (rule + optional LLM)
    -> Rewrite Loop
    -> State Update
```

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Calistirma

```bash
streamlit run streamlit_app.py
```

## Cift Tiklama Ile Calistirma (macOS)

Proje klasorundeki `run_simulation.command` dosyasina cift tiklayin.

- Ilk calistirmada otomatik `.venv` olusturur ve bagimliliklari kurar
- Sonraki calistirmalarda direkt uygulamayi acar

## LLM Secenekleri

Sidebar'dan `Provider` secilir:

- `rule_based`: API cagrisi olmadan deterministic/fallback uretim
- `openai`: OpenAI API
- `anthropic`: Claude API
- `gemini`: Gemini API
- `openai_compatible`: OpenAI-compatible endpoint (lokal veya uzak) kullanir
  - `Base URL` girilebilir

Ayni panelden:

- model preset veya custom model name secilebilir
- OpenAI, Claude ve Gemini API key alanlari ayridir
- `Auto-fetch models from provider API` ile provider'dan guncel model listesi cekilebilir
- `Refresh model list now` ile manuel yenilenebilir
- SSL sorunlari icin:
- `Verify SSL certificates` kapatilabilir (guvensiz)
- `CA bundle path` ile kurumsal/ozel sertifika zinciri verilebilir
- `Remember LLM settings on this machine` aciksa secimler ve API key'ler lokal dosyada saklanir: `.sim_ui_settings.json`
  - Not: Bu dosya plain text'tir; hassas ortamda kapali tutun veya `Forget saved LLM settings` ile silin.

## Scenario Import

- `Import` sekmesine uzun formatli senaryo metnini yapistir
- `Import (Replace State)` ile temiz state'e aktar
- `Import (Merge Into Current)` ile mevcut state ustune aktar
- `Treat detected chat lines as initial history` secenegi ile chat satirlarini runtime history'ye dahil etmeyi ac/kapat

Importer su alanlari otomatik doldurur:

- `scenario.environment_description`, `safe_word`, initial tension/risk, stage
- karakter bloklari (`* Isim: ...`) -> `characters`
- chat satirlari:
  - baslangic marker'inden once -> `scenario.style_examples`
  - baslangic marker'inden sonra -> `history`
- relationship matrix baslangic degerleri (heuristic)

## Simulation Flow

- `Start Simulation (NPC Opening)` ile sahne NPC tarafindan acilir
- Ortam satiri + NPC acilis hamleleri otomatik uretilir
- Sonrasinda `Simulate Turn` aktif olur ve kullanici (Mert) turunda yazar
- Her turdan sonra ayarlanabilir sayida NPC otomatik cevap verir (`NPC responses per turn`)

## Dosya Yapisi

- `streamlit_app.py`: UI + panel kontrolleri
- `app/default_state.py`: stage/trait sabitleri ve default state
- `app/state_manager.py`: normalize, save/load, JSON islemleri
- `app/prompting.py`: system/planner/generator prompt olusturma
- `app/llm_client.py`: LLM provider adaptoru
- `app/scenario_importer.py`: uzun metin -> state importer
- `app/validator.py`: ihlal kontrolu + rewrite kurallari
- `app/simulation_engine.py`: drift/stage/relationship + turn pipeline

## Not

Bu MVP davranis odakli simule motorudur. LLM sadece metin katmanidir; ana davranis numeric state tarafindan belirlenir.

## Ilan Takip Otomasyonu (Gunluk, Ucretsiz)

Bu repo'ya eklenen:

- `monitor_ilan.py`: hedef ilan sayfasini kontrol eder
- `.github/workflows/ilan-monitor.yml`: her gun calistirir

Calisma prensibi:

- Ilk kosuda sayfadaki mevcut ilanlari `state.json` dosyasina kaydeder (baseline)
- Sonraki kosularda yeni ilan varsa Telegram bildirimi gonderir
- Guncel `state.json` dosyasini repoya geri commit eder

Gerekli GitHub Secrets:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Kurulum adimlari:

1. Telegram'da `@BotFather` ile bir bot olusturup token alin.
2. Kendi `chat_id` degerinizi ogrenin.
3. GitHub repo -> `Settings` -> `Secrets and variables` -> `Actions` altina yukaridaki iki secret'i ekleyin.
4. `Actions` sekmesinden `ilan-monitor` workflow'unu `Run workflow` ile bir kez manuel calistirin.

Not:

- Planlanan cron saati `06:00 UTC`'dir (Turkiye saati ile `09:00`).
- Hedef URL workflow dosyasindaki `TARGET_URL` env degiskeninde tanimlidir.
