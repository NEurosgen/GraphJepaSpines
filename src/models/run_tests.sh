#!/bin/bash

# Скрипт для запуска всех тестов модуля models

# Получаем абсолютный путь к директории, где находится этот скрипт (src/models)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Корень проекта (GIT_Graph_refactor), чтобы импорты `src.models...` работали
PROJECT_ROOT="$( cd "$DIR/../.." && pwd )"

export PYTHONPATH="$PROJECT_ROOT"

echo "================================================="
echo "Запуск тестов модуля models..."
echo "Директория с тестами: $DIR/test_models"
echo "Корень проекта: $PROJECT_ROOT"
echo "================================================="

# Запускаем pytest через conda с нужным окружением
conda run -n torch_5060 pytest "$DIR/test_models" -v
