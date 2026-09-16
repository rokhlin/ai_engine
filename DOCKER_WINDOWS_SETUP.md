# Руководство по установке и запуску Media Cataloger в Docker на Windows

Данный документ содержит полное руководство со всеми необходимыми командами для установки, настройки и запуска **Media Cataloger (AI Engine)** и **Media Cataloger Web UI** в среде Docker на операционной системе Windows, а также варианты получения файлов из репозитория GitHub.

---

## 1. Системные требования и подготовка Windows

### 1.1. Требования
- **ОС:** Windows 10 (Build 19044+) или Windows 11 (64-bit).
- **Виртуализация:** Включена в BIOS/UEFI (Intel VT-x или AMD-V).
- **WSL2:** Установлен компонент Windows Subsystem for Linux 2.
- **Docker Desktop:** Установлен Docker Desktop для Windows.

### 1.2. Проверка и запуск Docker Desktop
Откройте терминал **PowerShell** от имени администратора или текущего пользователя:

```powershell
# 1. Проверка статуса WSL2
wsl --status

# 2. Проверка установленных дистрибутивов WSL
wsl -l -v

# 3. Запуск службы и приложения Docker Desktop (если не запущены)
Start-Service "com.docker.service" -ErrorAction SilentlyContinue
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
```

> [!TIP]
> Дождитесь, пока в системном трее Windows иконка Docker Desktop перестанет мигать (зеленый статус "Engine running").

Проверьте доступность Docker:
```powershell
docker version
docker compose version
```

Если вывод отображает обе секции (`Client` и `Server`) — Docker готов к работе.

### 1.3. Настройка общего доступа к дискам в Docker Desktop
Если вы планируете монтировать папки с дисков `C:`, `D:`, сетевых дисков `Z:`:
1. Откройте окно **Docker Desktop**.
2. Перейдите в **Settings (шестеренка)** -> **Resources** -> **File sharing**.
3. Убедитесь, что нужные диски (например, `C:\`, `D:\`) добавлены в список.
4. Нажмите **Apply & restart**.

---

## 2. Получение и скачивание файлов с GitHub

Выберите один из вариантов загрузки проекта на вашу Windows-машину:

### Вариант А: Клонирование репозиториев через Git (Рекомендуется)
Так как репозиторий является приватным, `git clone` — самый быстрый и удобный способ, использующий сохраненные учетные данные Windows (Git Credential Manager):

```powershell
# Создание папки проектов
New-Item -ItemType Directory -Path "C:\Projects\media_cataloger" -Force
cd "C:\Projects\media_cataloger"

# 1. Клонирование AI Engine (Backend)
git clone https://github.com/rokhlin/media_cataloger.git

# 2. Клонирование Web UI (Frontend & Gateway)
git clone https://github.com/rokhlin/media_cataloger_web.git

# Переход в папку каталогера
cd media_cataloger
```

#### Как переключиться на последнюю версию или конкретный релиз:
```powershell
# Обновить ветку main до актуального состояния:
git checkout main
git pull

# Или переключиться на последний релизный тег (например, v0.7.0):
git fetch --tags
git checkout v0.7.0
```

---

### Вариант Б: Скачивание ZIP-архива из приватного репозитория GitHub
> [!NOTE]
> Так как репозиторий приватный, прямая анонимная ссылка возвращает ошибку `404 Not Found`.

#### Способ 1 (через браузер):
1. Откройте в браузере страницу репозитория: `https://github.com/rokhlin/media_cataloger`
2. Нажмите зеленую кнопку **Code** -> **Download ZIP**.
3. Распакуйте архив в нужную папку на Windows.

#### Способ 2 (через PowerShell с GitHub Token):
```powershell
# Подставьте ваш Personal Access Token (PAT):
$Token = "ghp_ваш_токен_здесь"
$Headers = @{ 
    "Authorization" = "Bearer $Token"
    "Accept"        = "application/vnd.github.v3+json" 
}

# Скачивание архива ветки main:
$Url = "https://api.github.com/repos/rokhlin/media_cataloger/zipball/main"
Invoke-WebRequest -Uri $Url -Headers $Headers -OutFile "media_cataloger.zip"

# Распаковка архива
Expand-Archive -Path "media_cataloger.zip" -DestinationPath "." -Force
```

---

### Вариант В: Скачивание через официальный GitHub CLI (`gh`)
Если у вас установлен GitHub CLI:

```powershell
cd "C:\Projects\media_cataloger"

# Скачивание исходных кодов последнего релиза
gh release download --repo rokhlin/media_cataloger --archive=zip

# Распаковка
Expand-Archive -Path "*.zip" -DestinationPath "." -Force
```

---

### Вариант Г: Загрузка готовых предсобранных Docker-образов из GitHub Packages (GHCR)
Если в репозитории настроен GitHub Actions CI/CD и образы публикуются в реестре `ghcr.io`:

```powershell
docker pull ghcr.io/rokhlin/media_cataloger:latest
docker pull ghcr.io/rokhlin/media_cataloger_web:latest
```

---

## 3. Настройка конфигурации (`.env`) для Windows

В проекте уже подготовлена структура каталогов. Создайте конфигурационный файл на основе примера:

```powershell
# Находясь в папке media_cataloger:
Copy-Item ".\data\config\.env.example" ".\data\config\.env"
```

Откройте файл `.\data\config\.env` в любом редакторе (например, `notepad .\data\config\.env` или VS Code).

### Ключевые параметры для Windows:

```ini
# =====================================================================
# ПУТИ МОНТИРОВАНИЯ ДЛЯ WINDOWS
# =====================================================================

# Локальная папка или диск Windows с фото и видео:
# Допустимы варианты:
# MEDIA_INPUT=C:\Users\username\Pictures
# MEDIA_INPUT=D:\MediaArchive
# MEDIA_INPUT=Z:\               (подключенный сетевой диск)
# MEDIA_INPUT=./media_input     (локальная подпапка в проекте)
MEDIA_INPUT=./media_input

# Папка для базы данных SQLite, распознанных лиц и метаданных sidecar:
# MEDIA_OUTPUT=C:\Users\username\MediaCatalogOutput
# MEDIA_OUTPUT=./media_output
MEDIA_OUTPUT=./media_output

# Папка с файлами конфигурации и настроек:
CONFIG_PATH=./data/config

# =====================================================================
# AI ПРОВАЙДЕРЫ (GEMINI / LOCAL)
# =====================================================================
MODEL_PROVIDER=local

# Google Gemini (при использовании)
GEMINI_API_KEY=ваш_ключ_здесь
GEMINI_MODEL=gemini-3.6-flash

# =====================================================================
# ПОДКЛЮЧЕНИЕ К LM STUDIO НА ХОСТЕ WINDOWS
# =====================================================================
# ВАЖНО: Изнутри Docker контейнера Windows доступен по адресу host.docker.internal!
LOCAL_API_BASE=http://host.docker.internal:1234/v1
LOCAL_MODEL_NAME=qwen2.5-vl-7b-instruct
FALLBACK_TO_LOCAL=true

# =====================================================================
# WHISPER (РАСПОЗНАВАНИЕ РЕЧИ)
# =====================================================================
WHISPER_MODEL=base
# Для Docker Desktop на Windows без явного проброса NVIDIA GPU используйте cpu:
WHISPER_DEVICE=cpu
WHISPER_FALLBACK_TO_CPU=true

# =====================================================================
# БЕЗОПАСНОСТЬ И ПОРТЫ
# =====================================================================
API_PORT=8001
API_HOST=0.0.0.0
AI_SERVICE_USER=admin
AI_SERVICE_PASSWORD=admin
JWT_SECRET=super_secret_jwt_key_2026_windows_setup
```

> [!IMPORTANT]
> **Связь контейнера с Windows хостом (`host.docker.internal`):**
> Если LM Studio или другая локальная LLM запущена на Windows на порту `1234`, в параметре `LOCAL_API_BASE` укажите `http://host.docker.internal:1234/v1`. Также в самом приложении LM Studio перейдите во вкладку **Developer** и убедитесь, что сервер включен и слушает адрес `0.0.0.0` (All interfaces), а не только `127.0.0.1`.

---

## 4. Сборка и запуск контейнеров

### Сценарий 1: Запуск только AI Engine (`media-cataloger`)

Находясь в папке `media_cataloger`:

```powershell
# Вариант 1.1: С использованием Docker Compose напрямую
docker compose up -d --build

# Вариант 1.2: С использованием встроенного PowerShell скрипта
.\run.ps1 up

# Вариант 1.3: С использованием утилиты manage.py (требуется python)
python manage.py up
```

Контейнер `media-cataloger` соберется, запустится в фоне и будет доступен на порту **8001**.

---

### Сценарий 2: Запуск Web UI (`media-cataloger-web`)

Перейдите в папку веб-интерфейса `media_cataloger_web`:

```powershell
cd ..\media_cataloger_web

# Запуск Web UI контейнера
docker compose up -d --build
```

Веб-интерфейс будет доступен в браузере по адресу: **http://localhost:8000**

---

### Сценарий 3: Единый запуск Full-Stack (AI Engine + Web UI)

Если обе папки (`media_cataloger` и `media_cataloger_web`) расположены рядом в одной директории:

```powershell
cd C:\Projects\media_cataloger\media_cataloger

# Запуск обоих сервисов одной командой
docker compose -f docker-compose.all.yml up -d --build
```

---

## 5. Проверка работоспособности и логи

### 5.1. Проверка статуса запущенных контейнеров
```powershell
docker ps
```
Вы должны увидеть запущенные контейнеры со статусом `Up`:
- `media-cataloger` (порт `0.0.0.0:8001->8001/tcp`)
- `media-cataloger-web` (порт `0.0.0.0:8000->8000/tcp`)

### 5.2. Просмотр логов в реальном времени
```powershell
# Логи AI Engine:
docker logs -f media-cataloger

# Логи Web UI:
docker logs -f media-cataloger-web
```

### 5.3. Проверка через браузер
- **AI Engine Swagger API:** Откройте [http://localhost:8001/docs](http://localhost:8001/docs)
- **Проверка здоровья AI Engine:** Откройте [http://localhost:8001/health](http://localhost:8001/health)
- **Web UI & Family Tree Dashboard:** Откройте [http://localhost:8000](http://localhost:8000)

---

## 6. Остановка и перезапуск

```powershell
# Остановка контейнеров
docker compose down

# Перезапуск контейнеров
docker compose restart

# Остановка с удалением временных образов/томов при необходимости
docker compose down --volumes
```

---

## 7. Устранение неполадок на Windows (Troubleshooting)

| Проблема / Ошибка | Причина | Решение |
| :--- | :--- | :--- |
| `open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified` | Служба Docker Desktop не запущена | Запустите приложение **Docker Desktop** из меню Пуск и дождитесь зеленого статуса. |
| `service cataloger has no build section` | В старой версии `docker-compose.yml` не был указан контекст сборки | Примените обновленный `docker-compose.yml`, в котором прописан блок `build: context: .` |
| `unable to open database file` в логах контейнера | Путь к базе данных был указан как сетевой путь Windows (например, `\\ZIMABOARD\...`) | Внутри контейнера пути нормализуются автоматически. Убедитесь, что том смонтирован в `/app/media_output`, а в `.env` указано `MEDIA_OUTPUT=./media_output` или локальный диск хоста. |
| `Connection refused: host.docker.internal:1234` | LM Studio на Windows не принимает подключения извне | В настройках LM Studio во вкладке **Local Server** включите **Serve on Local Network (0.0.0.0)** и включите **CORS**. |
| `Permission denied` при обращении к диску `D:\` или `C:\` | В Docker Desktop не предоставлен доступ к диску | Откройте **Docker Desktop** -> **Settings** -> **Resources** -> **File sharing**, добавьте диск и нажмите **Apply & restart**. |
| Порт `8001` или `8000` уже занят | Другой процесс использует порт | Измените порт в `.env`: например, `CATALOGER_PORT=8005` и `PORT=8080`. |
