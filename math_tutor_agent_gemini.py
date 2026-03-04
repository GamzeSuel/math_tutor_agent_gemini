import json
import os
import re #Regular Expression
from dataclasses import dataclass, field
from typing import List, Literal, Type, TypeVar
from json_repair import repair_json

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

import ollama  # pip install ollama

load_dotenv()  # .env içindeki ayarları yükler (opsiyonel)


# ---------------------------
# 1) Structured Schemas (JSON output)
# ---------------------------

class LessonOutput(BaseModel):
    explanation: str = Field(..., description="Konuyu açık ve anlaşılır anlat.")
    connections: str = Field("", description="Konunun bağlı olduğu konuları ve nerede kullanıldığını anlat.")
    intuition: str = Field("", description="Sembollerin anlamı, nereden geldiği, iç mantık/sezgi.")
    examples: List[str] = Field(default_factory=list, description="1-2 kısa çözümlü örnek.")
    quiz: str = Field("", description="Tek soruluk mini quiz.")
    rubric_key_points: List[str] = Field(default_factory=list, description="Quiz cevabını değerlendirirken beklenen ana noktalar.")

class EvalOutput(BaseModel): #Bu sınıf, veri doğrulaması yapan bir şablondur. Yani bu modele uymayan veri gelirse hata verir.
    score: int = Field(..., ge=0, le=10) #field bu alan boş geçemez. 0'dan küçük 10'dan büyük olamaz.
    missing_points: List[str]
    incorrect_points: List[str]
    feedback: str
    next_step: Literal["re-explain", "more-examples", "harder-quiz", "connect-topics"] #literal ile sadece [] içi seçimnler geçerli.



# ---------------------------
# 2) Agent State
# ---------------------------

@dataclass
class TutorProfile:
    topic: str
    level: Literal["middle_school", "high_school", "college"] = "high_school"
    language: Literal["tr"] = "tr"

@dataclass
class TutorState:
    profile: TutorProfile
    weak_points: List[str] = field(default_factory=list)

class FollowUpOutput(BaseModel):  #Bu bir LLM response contract’ı.
    diagnosis: str = Field(..., description="Öğrencinin nerede takıldığını kısa söyle.")
    questions: List[str] = Field(..., description="Öğrenciye sorulacak 2-4 kısa teşhis sorusu.")
    micro_explain: str = Field(..., description="Eksik noktaya yönelik çok kısa tekrar anlatım (max 5 cümle).")
    new_example: str = Field(..., description="Tek kısa örnek + çözüm.")
    next_quiz: str = Field(..., description="Bir sonraki TEK mini soru.")
    rubric_key_points: List[str] = Field(..., description="Yeni quiz için 3-5 rubrik maddesi.")


# ---------------------------
# 3) Prompts
# ---------------------------

LESSON_SYSTEM = (
    "Sen bir matematik öğretmeni ajanısın. Türkçe anlat. Ezber değil anlam öğret.\n"
    "ÇIKTI SADECE GEÇERLİ JSON OLSUN. Kod bloğu yok, açıklama metni yok.\n"
    "AŞAĞIDAKİ ŞEMAYI AYNEN KULLAN ve SADECE BU ALANLARI DÖNDÜR:\n\n"
    "{\n"
    '  "explanation": "Kısa ve net anlatım (en fazla 6 cümle)",\n'
    '  "connections": "En fazla 4 madde/cümle",\n'
    '  "intuition": "En fazla 5 cümle",\n'
    '  "examples": ["Örnek 1 (kısa çözümlü)", "Örnek 2 (kısa çözümlü)"],\n'
    '  "quiz": "Tek soru (çoktan seçmeli değil).",\n'
    '  "rubric_key_points": ["Madde1", "Madde2", "Madde3"]\n'
    "}\n\n"
    "Kurallar:\n"
    "- explanation/intuition: kısa\n"
    "- examples: en fazla 2 adet\n"
    "- quiz: TEK soru, tek satır\n"
    "- explanation/connections/intuition içinde asla çift tırnak (\") kullanma.\n"
    "- Matematik sembollerini düz yaz: integral işareti yerine 'integral' yaz.\n"
    "ZORUNLU: 6 alanın hepsi dolu olacak. Hiçbiri boş geçilemez.\n"
    "explanation en fazla 3 cümle olacak. connections 3 madde olacak. intuition 3 cümle olacak.\n"
    "examples TAM 2 örnek olacak. rubric_key_points TAM 3 madde olacak.\n"
    "ZORUNLU: 6 alanın hepsi DOLU olacak (boş string yok).\n"
    "explanation TAM 2-3 cümle.\n"
    "connections TAM 3 madde (kısa).\n"
    "intuition TAM 3 cümle.\n"
    "examples TAM 2 örnek (her biri 2-3 satır).\n"
    "rubric_key_points TAM 3 madde.\n"
)

FOLLOWUP_SYSTEM = (
  "Sen gerçek bir matematik öğretmenisin. Öğrenci anlamadı.\n"
  "Sokratik ol: önce kısa teşhis soruları sor, sonra mikro tekrar anlat, sonra tek örnek, sonra yeni tek quiz ver.\n"
  "ÇIKTI SADECE GEÇERLİ JSON.\n"
  "Alanlar: diagnosis, questions, micro_explain, new_example, next_quiz, rubric_key_points\n"
)


def lesson_user_prompt(topic: str, level: str) -> str:
    return (
        f"Konu: {topic}\n"
        f"Seviye: {level}\n\n"
        "Konu anlatımı + bağlantılar + sezgi + örnek + quiz + rubrik üret."
    )

EVAL_SYSTEM = (
    "Sen bir değerlendirme (grader) ajanısın. Rubriğe göre öğrencinin cevabını puanla.\n"
    "ÇIKTI SADECE GEÇERLİ JSON OLSUN. Kod bloğu yok, açıklama metni yok.\n"
    "Türkçe geri bildirim ver. Adil ol. Somut eksikleri söyle.\n"
    "next_step alanı sadece şunlardan biri olsun:\n"
    "- re-explain\n- more-examples\n- harder-quiz\n- connect-topics\n"
    "ZORUNLU: JSON şu alanları içermeli: score, missing_points, incorrect_points, feedback, next_step."
)

def eval_user_prompt(question: str, rubric: List[str], answer: str) -> str:
    payload = {
        "question": question,
        "rubric_key_points": rubric,
        "user_answer": answer
    }
    return json.dumps(payload, ensure_ascii=False) #Python sözlüğünü JSON formatındaki string’e çevirir.


# ---------------------------
# 4) Ollama Client Wrapper
# ---------------------------

T = TypeVar("T", bound=BaseModel) #T herhangi bir tip olabilir ama mutlaka BaseModel’den türemiş olmalı.

def _extract_json(text: str) -> str:
    """
    Ollama bazen JSON dışı karakter basarsa diye:
    - Önce direkt parse denenecek
    - Olmazsa ilk '{' ile son '}' arasını kırpıp döndürür
    """
    text = text.strip()
    # hızlı yol: zaten JSON gibi
    if text.startswith("{") and text.endswith("}"):
        return text

    # kırpma: ilk { ... son }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]

    # hiç bulamazsa olduğu gibi döndür (sonraki parse hata verecek)
    return text

class OllamaTutorAgent:  #config-aware agent tasarımı.
    def __init__(self, model: str | None = None, host: str | None = None):
        """
        model: örn "gemma3:1b" (default env OLLAMA_MODEL ya da gemma3:1b)
        host:  örn "http://localhost:11434" (opsiyonel, env OLLAMA_HOST da olur)
        """
        self.model = model or os.getenv("OLLAMA_MODEL") or "gemma3:1b"
        if host:
            os.environ["OLLAMA_HOST"] = host

    def _chat_json(self, system: str, user: str, schema: Type[T], temperature: float = 0.2, retries: int = 1) -> T: #Bu metod sınıf içinde kullanılmak üzere yazılmıştır (private niyet).
        last_err: Exception | None = None #retry sürecinde son hatayı saklamak için kullanılan tip güvenli bir değişken.    

        # Pydantic şemasını Ollama'ya ver
        json_schema = schema.model_json_schema()

        for _ in range(retries + 1):
            resp = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                format=json_schema,   # format="json" yerine şema
                options={
                    "temperature": temperature,
                    "num_predict": 520,   #Maksimum üretilecek token sayısı.(kısa tut)
                },
            )

            content = resp["message"]["content"]
            json_text = _extract_json(content)

            #JSON Parse Etme
            try:
                data = json.loads(json_text)
            except json.JSONDecodeError:
                fixed = repair_json(json_text)
                data = json.loads(fixed)

            #hangi Pydantic modeli istendiyse ona göre default alanlar ekleniyor.
            if schema is EvalOutput:
                data.setdefault("missing_points", [])
                data.setdefault("incorrect_points", [])
                data.setdefault("feedback", "")
                data.setdefault("next_step", "re-explain")
                data.setdefault("score", 0)

            if schema is FollowUpOutput:
                data.setdefault("diagnosis", "")
                data.setdefault("questions", [])
                data.setdefault("micro_explain", "")
                data.setdefault("new_example", "")
                data.setdefault("next_quiz", "")
                data.setdefault("rubric_key_points", [])

            if schema is LessonOutput:
                data.setdefault("connections", "")
                data.setdefault("intuition", "")
                data.setdefault("examples", [])
                data.setdefault("quiz", "")
                data.setdefault("rubric_key_points", [])

            return schema.model_validate(data)

        raise RuntimeError(f"Ollama JSON parse/validate başarısız. Son hata: {last_err}\nHam çıktı:\n{content}")

    def generate_lesson(self, topic: str, level: str) -> LessonOutput:
        return self._chat_json(
            system=LESSON_SYSTEM,
            user=lesson_user_prompt(topic, level),
            schema=LessonOutput,
            temperature=0.4,
        )

    def evaluate(self, question: str, rubric: List[str], answer: str) -> EvalOutput:
        return self._chat_json(
            system=EVAL_SYSTEM,
            user=eval_user_prompt(question, rubric, answer),
            schema=EvalOutput,
            temperature=0.2,
        )

    def generate_followup(self, topic: str, level: str, lesson: LessonOutput, user_answer: str, ev: EvalOutput) -> FollowUpOutput:
        payload = {
            "topic": topic,
            "level": level,
            "lesson_summary": {
                "explanation": lesson.explanation,
                "connections": lesson.connections,
                "intuition": lesson.intuition,
                "examples": lesson.examples,
                "quiz": lesson.quiz,
                "rubric_key_points": lesson.rubric_key_points,
            },
            "student_answer": user_answer,
            "evaluation": ev.model_dump(),
            "instruction": "Öğrenciye 2-4 kısa teşhis sorusu sor. Sonra eksik noktaya odaklı mikro anlatım ver. 1 kısa örnek çöz. Sonra yeni tek quiz yaz."
        }
        return self._chat_json(
            system=FOLLOWUP_SYSTEM,
            user=json.dumps(payload, ensure_ascii=False),
            schema=FollowUpOutput,
            temperature=0.2,
        )

# ---------------------------
# 5) Simple CLI loop
# ---------------------------

def main():
    print("=== Math Tutor Agent (Ollama Local) ===")
    topic = input("Konu (örn: logaritma, türev): ").strip() or "logaritma"
    level = input("Seviye (middle_school/high_school/college): ").strip() or "high_school"

    # Model seçimi:
    # örn: "llama3.1:8b", "qwen2.5:7b-instruct", "mistral:7b" (sende hangisi yüklüyse)
    agent = OllamaTutorAgent(model="gemma3:1b")

    lesson = agent.generate_lesson(topic, level)

    print("\n--- AÇIKLAMA ---\n", lesson.explanation)
    print("\n--- BAĞLANTILAR ---\n", lesson.connections)
    print("\n--- SEZGİ / İÇ MANTIK ---\n", lesson.intuition)

    print("\n--- ÖRNEKLER ---")
    for i, ex in enumerate(lesson.examples, 1):
        print(f"{i}) {ex}")

    print("\n--- MİNİ QUIZ ---\n", lesson.quiz)
    ans = input("\nCevabın: ").strip()

    ev = agent.evaluate(lesson.quiz, lesson.rubric_key_points, ans)
    print("\n--- DEĞERLENDİRME ---")
    print("Skor:", ev.score, "/10")
    print("Eksik noktalar:", ev.missing_points)
    print("Hatalı noktalar:", ev.incorrect_points)
    print("Geri bildirim:", ev.feedback)
    print("Sonraki adım:", ev.next_step)

if __name__ == "__main__":
    main()