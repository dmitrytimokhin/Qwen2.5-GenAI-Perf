# README: Нагрузочное тестирование модели Qwen2.5-0.5B-Instruct с использованием Triton Inference Server

Этот проект демонстрирует процесс настройки и запуска нагрузочного тестирования модели **Qwen2.5-0.5B-Instruct** с использованием **Triton Inference Server** и инструмента **genai-perf**. Работа проводилась на платформе **Windows** с использованием **Windows PowerShell**. Основные характеристики компьютера:

- GPU: NVIDIA RTX 3060 12GB
- RAM: 32 ГБ
- CPU: Intel Core i5-12400F

## Содержание

1. [Требования](#требования)
2. [Установка и настройка](#установка-и-настройка)
3. [Запуск Triton Inference Server](#запуск-triton-inference-server)
4. [Запуск нагрузочного теста](#запуск-нагрузочного-теста)
5. [Описание шагов](#описание-шагов)

---

## Требования

Для успешного выполнения проекта необходимо:

1. Установленный **Docker Desktop** с поддержкой GPU.
2. Доступ к интернету для скачивания необходимых зависимостей.
3. Минимум 32 ГБ оперативной памяти и видеокарта с поддержкой CUDA.

---

## Установка и настройка

### Шаг 1: Запуск Docker контейнера с Triton Inference Server

Откройте **Windows PowerShell** и выполните следующие команды:

```powershell
docker run -it --rm --gpus=all --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 -p 8000:8000 -p 8001:8001 -p 8002:8002 nvcr.io/nvidia/tritonserver:24.11-trtllm-python-py3
```

Эта команда запускает контейнер с **Triton Inference Server**, предоставляя доступ к GPU и открывая порты для взаимодействия.

---

### Шаг 2: Обновление системы и установка Python

Внутри контейнера выполните:

```bash
apt-get update && apt-get install -y \
software-properties-common && \
add-apt-repository ppa:deadsnakes/ppa && \
apt-get update && apt-get install -y \
python3.10 python3.10-venv python3.10-dev python3-pip && \
apt-get clean && rm -rf /var/lib/apt/lists/*
```

Эти команды обновляют систему и устанавливают Python версии 3.10, который необходим для работы с библиотеками.

---

### Шаг 3: Установка необходимых библиотек

1. Установите `triton_cli`:
   ```bash
   pip install git+https://github.com/triton-inference-server/triton_cli.git@0.0.11
   ```

2. Клонируйте репозиторий TensorRT-LLM:
   ```bash
   git clone https://github.com/NVIDIA/TensorRT-LLM.git
   ```

3. Установите необходимые версии библиотек:
   ```bash
   pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830
   pip install tensorrt-llm==0.15.0
   ```

---

### Шаг 4: Подготовка модели

1. Склонируйте модель Qwen2.5-0.5B-Instruct:
   ```bash
   git clone https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct ./tensorrt/tmp/Qwen2.5/0.5B-Instruct
   ```

2. Конвертируйте чекпоинт модели:
   ```bash
   python3.10 TensorRT-LLM/examples/qwen/convert_checkpoint.py --model_dir ./tensorrt/tmp/Qwen0.5/1.5B-Instruct --output_dir ./tensorrt/checkpoints/tllm_checkpoint_1gpu_fp16 --dtype float16
   ```

3. Соберите TensorRT engine:
   ```bash
   trtllm-build --checkpoint_dir ./tensorrt/checkpoints/tllm_checkpoint_1gpu_fp16 --output_dir ./tensorrt/engines/Qwen2.5/0.5B-Instruct/trt_engines/fp16/1-gpu --gpt_attention_plugin float16 --gemm_plugin float16
   ```

---

### Шаг 5: Настройка модели в Triton

1. Клонируйте репозиторий TensorRT-LLM Backend:
   ```bash
   git clone https://github.com/triton-inference-server/tensorrtllm_backend.git --branch v0.12.0 ./tensorrt/tensorrtllm_backend
   ```

2. Создайте директорию для модели:
   ```bash
   mkdir -p tensorrt/models/qwen-2.5-0.5B/
   ```

3. Скопируйте конфигурационные файлы:
   ```bash
   cp tensorrt/tensorrtllm_backend/all_models/inflight_batcher_llm/* tensorrt/models/qwen-2.5-0.5B/ -r
   ```

4. Настройте конфигурационные файлы с помощью скрипта `fill_template.py`:
   ```bash
   python3.10 tensorrt/tensorrtllm_backend/tools/fill_template.py -i tensorrt/models/qwen-2.5-0.5B/preprocessing/config.pbtxt tokenizer_dir:tensorrt/tmp/Qwen2.5/0.5B-Instruct/,tokenizer_type:llama,triton_max_batch_size:2048,preprocessing_instance_count:4,stream:True
   python3.10 tensorrt/tensorrtllm_backend/tools/fill_template.py -i tensorrt/models/qwen-2.5-0.5B/postprocessing/config.pbtxt tokenizer_dir:tensorrt/tmp/Qwen2.5/0.5B-Instruct/,tokenizer_type:llama,triton_max_batch_size:2048,postprocessing_instance_count:4,stream:True
   python3.10 tensorrt/tensorrtllm_backend/tools/fill_template.py -i tensorrt/models/qwen-2.5-0.5B/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:2048,decoupled_mode:True,bls_instance_count:4,accumulate_tokens:True,stream:True
   python3.10 tensorrt/tensorrtllm_backend/tools/fill_template.py -i tensorrt/models/qwen-2.5-0.5B/ensemble/config.pbtxt triton_max_batch_size:2048,stream:True
   python3.10 tensorrt/tensorrtllm_backend/tools/fill_template.py -i tensorrt/models/qwen-2.5-0.5B/tensorrt_llm/config.pbtxt triton_max_batch_size:2048,decoupled_mode:True,max_beam_width:1,engine_dir:tensorrt/engines/Qwen2.5/0.5B-Instruct/trt_engines/fp16/1-gpu/,max_tokens_in_paged_kv_cache:40960,max_attention_window_size:40960,kv_cache_free_gpu_mem_fraction:0.9,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_batching,max_queue_delay_microseconds:300,triton_backend:tensorrtllm,encoder_engine_dir:,decoding_mode:gpt,stream:True,streaming:True,batch_scheduler_policy:max_utilization
   ```

5. Запустите сервер Triton:
   ```bash
   python3.10 ./tensorrt/tensorrtllm_backend/scripts/launch_triton_server.py --model_repo=tensorrt/models/qwen-2.5-0.5B
   ```

---

## Запуск нагрузочного теста

Откройте второе окно **Windows PowerShell** и выполните:

```powershell
docker run -it --rm --gpus=all -p 8003:8001 nvcr.io/nvidia/tritonserver:24.03-py3-sdk
```

Внутри контейнера запустите нагрузочный тест:

```bash
genai-perf -m ensemble --service-kind triton --output-format trtllm --input-type synthetic --num-of-output-prompts 100 --random-seed 123 --input-tokens-mean 2500 --input-tokens-stddev 250 --streaming --expected-output-tokens 150 --concurrency 1 --measurement-interval 7200 --profile-export-file my_profile_export.json --url host.docker.internal:8001
```

<img width="1137" alt="Снимок экрана 2025-02-17 в 14 48 21" src="https://github.com/user-attachments/assets/d730c468-e029-4f81-9a57-e2e8836f56f1" />


---

## Описание шагов

1. **Запуск Triton Inference Server**: Создается контейнер с поддержкой GPU, который будет выполнять инференс модели.
2. **Обновление системы и установка Python**: Устанавливается Python 3.10 и необходимые зависимости.
3. **Установка библиотек**: Устанавливаются библиотеки для работы с моделями и Triton.
4. **Подготовка модели**: Модель загружается, конвертируется и собирается в формат TensorRT.
5. **Настройка модели в Triton**: Конфигурируются параметры модели для работы в Triton.
6. **Запуск нагрузочного теста**: Используется инструмент `genai-perf` для оценки производительности модели.

---

## Важные замечания

- Убедитесь, что Docker Desktop настроен на использование GPU.
- Если возникают ошибки с памятью, попробуйте уменьшить размер батча или количество токенов.
- Для анализа результатов используйте файл `my_profile_export.json`.
