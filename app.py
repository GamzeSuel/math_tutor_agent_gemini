import streamlit as st
from math_tutor_agent_gemini import OllamaTutorAgent

st.set_page_config(page_title="Math Tutor AI", layout="wide")
st.title("📘 Math Tutor AI (Local Ollama)")
st.write("Gemma3:1b ile çalışan matematik öğretmeni")

@st.cache_resource  #Bu fonksiyonun oluşturduğu “ağır” kaynağı cache’le (önbelleğe al) ve tekrar tekrar üretme.
def get_agent():
    return OllamaTutorAgent(model="gemma3:1b")

agent = get_agent()

topic = st.text_input("Konu gir:", "integral")
level = st.selectbox("Seviye seç:", ["middle_school", "high_school", "college"])

#Eğer session_state içinde "lesson"-"ev"-"followup"-"attempt" anahtarı yoksa, oluştur ve değerini None yap.
if "lesson" not in st.session_state:
    st.session_state.lesson = None
if "ev" not in st.session_state:
    st.session_state.ev = None
if "followup" not in st.session_state:
    st.session_state.followup = None
if "attempt" not in st.session_state:
    st.session_state.attempt = 0

if st.button("Dersi Oluştur"):
    st.session_state.ev = None
    st.session_state.followup = None
    st.session_state.attempt = 0
    with st.spinner("Model düşünüyor..."):
        st.session_state.lesson = agent.generate_lesson(topic, level)

lesson = st.session_state.lesson
if lesson:
    st.subheader("📖 Açıklama")
    st.write(lesson.explanation)

    st.subheader("🔗 Bağlantılar")
    st.write(lesson.connections)

    st.subheader("🧠 Sezgi")
    st.write(lesson.intuition)

    st.subheader("📝 Örnekler")
    for ex in lesson.examples:
        st.write("-", ex)

    st.subheader("🎯 Mini Quiz")
    st.write(lesson.quiz)

    user_answer = st.text_area("Cevabını yaz", key="user_answer")

    if st.button("Cevabı Değerlendir"):
        with st.spinner("Değerlendiriliyor..."):
            st.session_state.ev = agent.evaluate(
                lesson.quiz,
                lesson.rubric_key_points,
                user_answer
            )
            st.session_state.attempt += 1

            # score düşükse follow-up üret (agent'te generate_followup varsa)
            if st.session_state.ev.score < 7 and hasattr(agent, "generate_followup"):
                st.session_state.followup = agent.generate_followup(
                    topic, level, lesson, user_answer, st.session_state.ev
                )
            else:
                st.session_state.followup = None

    if st.session_state.ev:
        ev = st.session_state.ev
        st.success(f"Skor: {ev.score}/10")
        st.write("Geri Bildirim:", ev.feedback)
        st.write("Eksikler:", ev.missing_points)
        st.write("Sonraki Adım:", ev.next_step)

    #dinamik bir öğrenci geri bildirim paneli.
    if st.session_state.followup:
        fu = st.session_state.followup
        st.subheader("🧑‍🏫 Öğretmen Takibi (Follow-up)")
        st.write("Teşhis:", fu.diagnosis)
        st.write("Sana sorularım:")
        for q in fu.questions:
            st.write("-", q)
        st.write("Kısa tekrar:", fu.micro_explain)
        st.write("Yeni örnek:", fu.new_example)
        st.write("Yeni quiz:", fu.next_quiz)