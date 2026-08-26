#!/usr/bin/env python3
"""
Генератор HTML-страниц по текстовому описанию (LangChain цепочка)
Запуск: python main.py
"""

import os
import json
import re
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List

from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

# Загружаем переменные окружения из .env
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _lc_to_openai_messages(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    """Преобразует сообщения LangChain в формат OpenAI."""
    role_map = {
        SystemMessage: "system",
        HumanMessage: "user",
        AIMessage: "assistant",
    }
    converted: List[Dict[str, str]] = []
    for m in messages:
        role = None
        for cls, r in role_map.items():
            if isinstance(m, cls):
                role = r
                break
        if role is None:
            role = "user"
        content = m.content if isinstance(m.content, str) else str(m.content)
        converted.append({"role": role, "content": content})
    return converted


def get_llm() -> RunnableLambda:
    """Создаёт Runnable для вызова OpenAI Chat Completions API."""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("TEMPERATURE", "0.2"))

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY не найден в .env файле")

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    logger.debug(f"Инициализация OpenAI client: model={model}, temperature={temperature}")

    def _invoke(messages_or_value: Any) -> str:
        if hasattr(messages_or_value, "to_messages"):
            lc_messages = messages_or_value.to_messages()
        else:
            lc_messages = messages_or_value

        if not isinstance(lc_messages, list):
            raise TypeError("Ожидается список сообщений для входа LLM")

        oa_messages = _lc_to_openai_messages(lc_messages)
        resp = client.chat.completions.create(
            model=model,
            messages=oa_messages,
            temperature=temperature,
        )
        content = resp.choices[0].message.content if resp.choices and resp.choices[0].message else ""
        return content or ""

    return RunnableLambda(_invoke)


def build_analysis_chain(llm):
    """Chain 1: Анализ запроса пользователя."""
    system = (
        "Ты — опытный веб-дизайнер и HTML-верстальщик. Проанализируй описание страницы и выдели ключевую информацию.\n"
        "Верни строгий JSON со следующими полями:\n"
        "- title: заголовок страницы (H1)\n"
        "- description: краткое описание (до 2 предложений)\n"
        "- items: список пунктов/товаров/услуг (массив строк, 3-5 пунктов)\n"
        "- contacts: контактная информация (массив строк)\n"
        "- style: стиль страницы (например, 'уютный', 'минималистичный', 'современный')\n"
        "- colors: массив цветов (например, ['#f5e6d3', '#d4a373', '#5c3d2e'])\n"
        "Верни ТОЛЬКО валидный JSON. Без пояснений."
    )
    human = "Описание страницы: {task}"
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", human),
    ])
    return prompt | llm | StrOutputParser()


def build_tools_chain(llm):
    """Chain 2: Подбор инструментов и компонентов."""
    system = (
        "Ты — веб-архитектор. На основе анализа страницы предложи, какие HTML-элементы и стили понадобятся.\n"
        "Верни строгий JSON со следующими полями:\n"
        "- html_elements: список нужных HTML-элементов (например, ['header', 'h1', 'p', 'ul', 'footer'])\n"
        "- css_features: список CSS-фич (например, ['flexbox', 'border-radius', 'shadow', 'gradient'])\n"
        "- fonts: рекомендации по шрифтам (1-2 названия, например, 'Georgia, serif')\n"
        "- layout: тип макета (например, 'одноколоночный', 'двухколоночный')\n"
        "Верни ТОЛЬКО валидный JSON. Без пояснений."
    )
    human = "Анализ страницы: {analysis}"
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", human),
    ])
    return prompt | llm | StrOutputParser()


def build_generation_chain(llm):
    """Chain 3: Генерация HTML+CSS кода."""
    system = (
        "Ты — опытный HTML/CSS разработчик. Сгенерируй полноценный HTML-документ с встроенными CSS-стилями.\n"
        "Требования:\n"
        "- Вся страница должна быть в одном HTML-файле.\n"
        "- Используй современные CSS-фичи (flexbox, border-radius, тени, градиенты).\n"
        "- Стили должны точно соответствовать описанию и анализу.\n"
        "- Страница должна быть адаптивной (хотя бы минимально).\n"
        "- Добавь базовые hover-эффекты для кнопок и ссылок.\n"
        "Верни ТОЛЬКО HTML-код. Без пояснений и Markdown-разметки."
    )
    human = (
        "Описание: {task}\n"
        "Анализ (JSON): {analysis}\n"
        "Компоненты и стили: {tools}\n"
        "Сгенерируй готовую HTML-страницу."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", human),
    ])
    return prompt | llm | StrOutputParser()


def build_review_chain(llm):
    """Chain 4: Проверка и улучшение кода."""
    system = (
        "Ты — опытный HTML-аудитор. Проверь сгенерированный HTML-код и исправь проблемы:\n"
        "- Валидность HTML (закрытые теги, правильная структура).\n"
        "- Соответствие цветам и стилю из анализа.\n"
        "- Визуальная читаемость и контрастность.\n"
        "Верни ИСПРАВЛЕННУЮ полную HTML-страницу. Без пояснений."
    )
    human = (
        "Анализ (JSON): {analysis}\n"
        "Исходный HTML: {code}\n"
        "Исправь код, если нужно. Верни финальную версию."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", human),
    ])
    return prompt | llm | StrOutputParser()


def parse_json(text: str) -> Dict[str, Any]:
    """Парсит JSON из текста, даже если текст содержит лишние символы."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Пытаемся извлечь JSON-объект
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"Не удалось распарсить JSON из текста: {text[:100]}...")


def strip_code_fences(text: str) -> str:
    """Убирает markdown-обёртку вокруг кода."""
    pattern = r'^\s*```[a-zA-Z]*\s*\n|\n\s*```\s*$'
    return re.sub(pattern, '', text, flags=re.MULTILINE)


def run_chain(task: str, out_dir: Path) -> Path:
    """Запускает полную цепочку."""
    logger.info("🚀 Запуск цепочки генерации HTML-страницы")
    logger.info(f"📝 Описание: {task}")

    llm = get_llm()

    # Шаг 1: Анализ
    logger.info("🔍 Шаг 1: Анализ запроса...")
    analysis_chain = build_analysis_chain(llm)
    analysis_text = analysis_chain.invoke({"task": task})
    logger.info("✅ Анализ получен")
    analysis = parse_json(analysis_text)
    analysis_json = json.dumps(analysis, ensure_ascii=False)

    # Шаг 2: Подбор инструментов
    logger.info("🔧 Шаг 2: Подбор компонентов...")
    tools_chain = build_tools_chain(llm)
    tools_text = tools_chain.invoke({"analysis": analysis_json})
    logger.info("✅ Компоненты подобраны")
    tools = parse_json(tools_text)
    tools_json = json.dumps(tools, ensure_ascii=False)

    # Шаг 3: Генерация кода
    logger.info("💻 Шаг 3: Генерация HTML-кода...")
    generation_chain = build_generation_chain(llm)
    code_text = generation_chain.invoke({
        "task": task,
        "analysis": analysis_json,
        "tools": tools_json
    })
    code_text = strip_code_fences(code_text).strip()
    logger.info("✅ HTML-код сгенерирован")

    # Шаг 4: Проверка
    logger.info("🔎 Шаг 4: Проверка и исправление...")
    review_chain = build_review_chain(llm)
    final_code = review_chain.invoke({
        "analysis": analysis_json,
        "code": code_text
    })
    final_code = strip_code_fences(final_code).strip()
    logger.info("✅ Проверка завершена")

    # Сохраняем результат
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "index.html"
    out_path.write_text(final_code, encoding="utf-8")
    logger.info(f"💾 HTML-страница сохранена: {out_path}")

    # Сохраняем промежуточные данные (для отладки)
    with open(out_dir / "analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    with open(out_dir / "tools.json", "w", encoding="utf-8") as f:
        json.dump(tools, f, ensure_ascii=False, indent=2)

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Генератор HTML-страниц через LangChain")
    parser.add_argument(
        "--task",
        type=str,
        help="Описание страницы (в кавычках)",
        default="Создай одностраничный сайт-визитку для небольшой пекарни «Домашний хлеб». На странице должны быть: логотип (можно текстовый) и название. Короткое описание: «Печём хлеб и булочки с душой». Список из 3–4 популярных позиций (например: «Бородинский хлеб», «Круассан с шоколадом», «Пирожок с капустой»). Телефон и адрес для связи. Стиль: уютный, тёплый, натуральный (как домашняя выпечка). Используй цвета: бежевый, терракотовый, тёмно-коричневый."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="output",
        help="Папка для сохранения результата"
    )
    args = parser.parse_args()

    try:
        out_file = run_chain(args.task, Path(args.out_dir))
        print(f"\n✅ Готово! Страница сохранена: {out_file}")
        print(f"📂 Открой файл в браузере: {out_file.absolute()}")
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
