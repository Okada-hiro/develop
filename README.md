# develop

Step 1 の開始点として、`whisper-large-v3` を使う ASR スクリプトを追加しています。

## 使い方

依存関係がまだ入っていなければ、リポジトリ直下で次を実行します。

```bash
bash develop/environment.sh
```

ASR の実行例:

```bash
python3 develop/asr_whisper_large_v3.py /path/to/audio.wav --language ja --word-timestamps
```

トークン候補の確率も欲しい場合:

```bash
python3 develop/asr_whisper_large_v3.py /path/to/audio.wav --language ja --word-timestamps --token-topk 5
```

text から `initial_prompt` 用の単語リストを作る:

```bash
python3 develop/summary_to_prompt_terms.py --text-file /path/to/context.txt
```

Gemini を使う場合は `GEMINI_API_KEY` または `GOOGLE_API_KEY` を設定してください。
設定されていれば `gemini-2.5-flash-lite` を優先し、失敗時は既存の正規表現ベース抽出にフォールバックします。

`develop/output/` に以下を保存します。

- `*.txt`: 全文テキスト
- `*.json`: `segments` を含む詳細結果

`--token-topk 5` を付けた場合、各 `segment` に `token_probs` を追加します。
各トークンについて、選ばれたトークンの確率と top-k 候補を保存します。

## pyannote

`pyannote.audio` を使う処理は Hugging Face のトークンが必要です。
事前に対象モデルの利用条件に同意したうえで、`HUGGINGFACE_HUB_TOKEN` を設定してください。

Colab では次のどちらかで渡せます。

```python
import os
os.environ["HUGGINGFACE_HUB_TOKEN"] = "hf_xxx"
```

または Colab Secrets を使う場合:

```python
import os
from google.colab import userdata

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

VAD で 30 秒前後のチャンクを作る:

```bash
python3 develop/pyannote_vad_segmenter.py /path/to/audio.wav
```

話者 diarization を出す:

```bash
python3 develop/pyannote_diarize.py /path/to/audio.wav
```

Whisper の `segments` に話者ラベルを重ねる:

```bash
python3 develop/pyannote_diarize.py /path/to/audio.wav --whisper-json develop/output/audio.json
```

VAD で分割してから Whisper で転写する:

```bash
python3 develop/asr_whisper_large_v3.py /path/to/audio.wav --language ja --word-timestamps --vad-chunk-transcribe
```

このモードでは `pyannote/segmentation-3.0` で音声区間を検出し、30 秒前後のチャンクごとに Whisper を回してから結果を結合します。

audio のみで、2 通りの VAD 分割と話者認識つきで回す:

```bash
python3 develop/dual_pass_whisper_pipeline.py /path/to/audio.wav --language ja
```

text と audio から、text を `initial_prompt` 化して回す:

```bash
python3 develop/dual_pass_whisper_pipeline.py /path/to/audio.wav --text-file /path/to/context.txt --language ja
```

このスクリプトは次をまとめて実行します。

- text があれば単語リストを作る
- Gemini が使えれば `gemini-2.5-flash-lite` で `initial_prompt` を作る
- Gemini が使えなければ既存の正規表現ベース抽出にフォールバックする
- VAD の `greedy` と `sliding` の 2 通りで文字起こしする
  - `greedy`: 30 秒前後
  - `sliding`: 20 秒前後
- pyannote で話者認識する
- 上位トークン候補と低信頼アラートを JSON に残す
- 2 通りの転写結果が異なる箇所を `pass_mismatch` アラートとして残す
- 内部判断用に、セグメント単位の差分も `segment_diff_candidates` に残す
- `greedy` と `sliding` の transcription を `*.greedy.txt` と `*.sliding.txt` に保存する
- 話者ごとに見やすくした transcription を JSON の `greedy_by_speaker` / `sliding_by_speaker` に入れ、`*.greedy.by_speaker.txt` と `*.sliding.by_speaker.txt` にも保存する
- アラート一覧を `*.alerts.txt` に保存し、実行時にも表示する
- `prompt_generation` を JSON に保存し、実行時にも表示する

## 次工程への接続

- 話者認識: `segments` の時間情報を使って diarization 結果と結合
- VAD 分割: セグメント単位または VAD 区間単位で再転写
- LLM 後処理: `json` の低信頼箇所を入力に使う

## 現時点のメモ

- この Whisper 本体には明示的な VAD 分割は入っていません。30 秒窓と `no_speech_threshold` はありますが、独立した VAD ではありません。
- `pyannote.audio` は導入済みですが、学習済みモデルの取得には Hugging Face へのアクセスが必要です。
